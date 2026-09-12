"""Character metadata extraction phase (PAI 1.1 library — no __main__).

Extracts character mentions from a project's PAI scene files via an LLM
and writes stub entries into `kb/on_scene/characters.json` (preserves
user edits — only fills blank fields and adds new entries).

Entry point: `pai kb characters extract` → studio_server
              POST /api/kb/characters/extract → run().

This module **does not** train LoRAs — that's a separate ML pipeline.
It produces the metadata stubs (anchor, costumes, lora.path placeholder,
director_notes) so downstream code can reference characters by id and the
user knows what LoRAs need to be trained.

Three reference-image strategies (mirroring build_props.py):
  user-only   (default; safest)
              Only writes stubs. The user supplies actual reference photos
              for LoRA training in a separate step.
  flux        Synthesizes a reference photo via Flux T2I from the character
              anchor. Useful as a "starting point" for LoRA training datasets
              when no real photos exist.
  prompt-only Stubs only — same as user-only but more explicit.

Never overwrites existing character entries; merges new fields in-place.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pace_core.llm_client import strip_fences

PIPELINE_DIR = Path(__file__).resolve().parent
from pace_core.paths import MODELS_FILE, paths_for  # noqa: E402

CHAR_REFS    = PIPELINE_DIR / "character_refs"

LOCAL_PORT_COMFY  = 15000

EXTRACTION_SYSTEM_PROMPT = """You are a casting director and production assistant. Read the scene breakdown and extract every CHARACTER who appears, even briefly.

A character is a named human (or human-like entity) the camera frames. Skip:
  - Anonymous extras / background crowds (described but not named)
  - Animals (unless they have dialogue or a name)
  - Voices off-screen with no body

Return one JSON object: {"characters": [...]}. For each character:
  id                stable lowercase identifier with underscores (e.g. "jia", "elder_brother")
  name              human-readable label (e.g. "Jia", "Elder Brother")
  trigger           short LoRA trigger string (e.g. "jia_woman", "wei_man") — 1-3 tokens
  anchor            30-50 word Flux prompt fragment describing physical appearance
                    (age, build, ethnicity, distinguishing features). NO emotion words,
                    NO clothing — those go in costumes / per-shot prompt.
  age_estimate      integer
  gender            "male" | "female" | "nonbinary" | "unknown"
  scenes            list of scene_id values where the character appears
  director_notes    one sentence: what role they play in the story / why they matter
  costumes          object: {<look_name>: "description"}; at least one entry like
                    {"default": "<costume description>"} per visible look in the script

Output JSON only — no prose, no markdown fences."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ─────────────────────────  LLM extraction  ──────────────────────────────


def extract_characters_from_scenes(scenes_breakdown: dict, model_key: str) -> tuple[list[dict], float]:
    """Returns (characters, cost_usd). Uses llm_client.call_model so
    api-key resolution flows through the studio's llm_tokens store
    (same path the rest of the pipeline uses)."""
    from pace_core.llm_client import call_model  # type: ignore

    models = json.loads(MODELS_FILE.read_text())
    if model_key not in models:
        raise ValueError(f"unknown model '{model_key}'. options: {list(models)}")
    cfg = models[model_key]

    user_prompt = (
        "Extract characters from this scene breakdown. Return JSON only.\n\n"
        f"{json.dumps(scenes_breakdown, ensure_ascii=False)}"
    )
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    raw, cost = call_model(cfg, messages, api_key=None)
    print(f"  [llm] extraction model={model_key} cost=${cost:.4f}", file=sys.stderr)
    raw = strip_fences(raw)
    data = json.loads(raw)
    chars = data["characters"] if isinstance(data, dict) else data
    return chars, cost


# ─────────────────────────  Registry merge  ──────────────────────────────


SCHEMA_KEYS = {"_schema_version", "_purpose", "_field_definitions", "ensemble_rules"}


def merge_into_characters_json(extracted: list[dict], chars_file: Path) -> tuple[int, int]:
    """Returns (created_count, updated_count). Never overwrites existing fields."""
    if chars_file.exists():
        registry = json.loads(chars_file.read_text())
    else:
        registry = {"_schema_version": "0.1"}

    created = 0
    updated = 0
    for c in extracted:
        cid = c["id"]
        if cid in SCHEMA_KEYS:
            continue
        existing = registry.get(cid)
        costumes = c.get("costumes") or {"default": ""}
        if existing is None:
            registry[cid] = {
                "trigger":   c.get("trigger", cid),
                "anchor":    c.get("anchor", ""),
                "costumes":  costumes,
                "lora": {
                    "path":   None,         # to be filled after LoRA training
                    "weight": 1.0,
                    "_note":  "Train via separate Flux LoRA pipeline; "
                              "drop the .safetensors path here when ready.",
                },
                "animation": {"anchor": c.get("anchor", ""), "_note": "Optional animation-mode anchor"},
                "director_notes":  c.get("director_notes", ""),
                "_extracted": {
                    "name":          c.get("name", cid),
                    "age_estimate":  c.get("age_estimate"),
                    "gender":        c.get("gender"),
                    "scenes":        c.get("scenes", []),
                    "extracted_at":  _now_iso(),
                },
            }
            created += 1
        else:
            # Only fill blank fields
            scenes_existing = (existing.get("_extracted") or {}).get("scenes", [])
            scenes_new = sorted(set(scenes_existing + c.get("scenes", [])))
            existing.setdefault("_extracted", {})["scenes"] = scenes_new
            if not existing.get("anchor"):
                existing["anchor"] = c.get("anchor", existing.get("anchor", ""))
            existing_costumes = existing.get("costumes") or {}
            for look, desc in costumes.items():
                existing_costumes.setdefault(look, desc)
            existing["costumes"] = existing_costumes
            if scenes_new != scenes_existing:
                updated += 1

    chars_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    return created, updated


# ─────────────────────────  Optional Flux reference synth  ─────────────────


# The Flux 1 reference-synth path was removed with the flux1-dev-fp8 unet it
# hard-coded. Nothing in the registry is a Flux 1 workflow any more -- all 15
# registered workflows are flux2 or custom -- so this builder and the two
# synthesize_reference_image() functions on top of it were the last things
# naming that model, and they named it in a hand-built graph rather than
# through the workflow registry every other path goes through.
#
# The replacement is POST /api/images (backend flux_comfyui), which dispatches
# by channel against the registered flux2 workflows and writes its output where
# the rest of the pipeline expects it. The Reference Image node's Generate
# source is the same route.


# ─────────────────────────  Entry point  ──────────────────────────────


def run(*, project: str = "main",
        model: str = "claude-sonnet-5-rb",
        reference_strategy: str = "user-only",
        dry_run: bool = False) -> dict[str, Any]:
    """Library entry point for `pai kb characters extract` (HTTP-callable).

    Args mirror the old CLI flags exactly; returns a structured result
    dict instead of printing+exiting.

    Raises ValueError on bad input (caller — studio_server — translates
    to HTTP 400). Other exceptions bubble as 500.
    """
    if reference_strategy not in ("user-only", "flux", "prompt-only"):
        raise ValueError(f"reference_strategy must be one of "
                         f"user-only/flux/prompt-only, got {reference_strategy!r}")

    from pace_core.breakdown import scenes_kb
    scenes_data = scenes_kb.load_all_scenes(project=project)
    scenes = scenes_data.get("scenes", scenes_data)
    if not scenes:
        raise ValueError(f"project {project!r} has no scenes — split_script first")

    print(f"== Phase 1: extracting characters from {len(scenes)} scenes via {model}")
    extracted, cost = extract_characters_from_scenes({"scenes": scenes}, model_key=model)
    print(f"   -> {len(extracted)} characters identified: {[c['id'] for c in extracted]}")

    chars_file = paths_for(project).chars_file
    created, updated = merge_into_characters_json(extracted, chars_file)
    print(f"== Phase 2: registry merge → {created} new, {updated} updated, "
          f"written to {chars_file}")

    result: dict[str, Any] = {
        "project":            project,
        "model":              model,
        "extracted_count":    len(extracted),
        "extracted_ids":      [c.get("id") for c in extracted],
        "registry_created":   created,
        "registry_updated":   updated,
        "llm_cost_usd":       round(cost, 4),
        "reference_strategy": reference_strategy,
        "flux_refs":          {"ok": [], "fail": []},
    }

    if dry_run or reference_strategy in ("user-only", "prompt-only"):
        return result

    # Optional Phase 3: Flux reference synth
    registry = json.loads(chars_file.read_text())
    for c in extracted:
        cid = c["id"]
        entry = registry.get(cid)
        if not entry:
            continue
        result["flux_refs"]["fail"].append({
            "id": cid,
            "error": "flux reference synth is retired — it drove a hand-built "
                     "Flux 1 graph. Use POST /api/images (or the Reference "
                     "Image node's Generate source) instead."})
    return result
