"""Character metadata extraction phase (PACE library — no __main__).

Extracts character mentions from a project's PACE scene files via an LLM
and writes stub entries into `kb/on_scene/characters.json` (preserves
user edits — only fills blank fields and adds new entries).

Entry point: `run()`, called by a host CLI or server.

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

from pace_core.llm_client import default_model
from pace_core.llm_client import strip_fences

PIPELINE_DIR = Path(__file__).resolve().parent
from pace_core import paths
from pace_core.paths import paths_for  # noqa: E402

CHAR_REFS    = PIPELINE_DIR / "character_refs"

LOCAL_PORT_COMFY  = 15000

EXTRACTION_SYSTEM_PROMPT = """You are a casting director and production assistant. Read the scene breakdown and extract every CHARACTER who appears, even briefly.

A character is any individually framed agent: human, animal, creature, robot,
or animated object. Include unnamed animals and silent characters. Reuse every
character_id already declared in setup.subjects exactly; do not rename it.
Skip unindividuated background crowds and off-screen voices with no body.

Return one JSON object: {"characters": [...]}. For each character:
  id                stable lowercase identifier with underscores (e.g. "jia", "elder_brother")
  name              human-readable label (e.g. "Jia", "Elder Brother")
  entity_type       "human" | "animal" | "creature" | "robot" | "unknown"
  species           source-supported species, or null; do not invent one
  trigger           short LoRA trigger string (e.g. "jia_woman", "wei_man") — 1-3 tokens
  anchor            30-50 word Flux prompt fragment describing physical appearance
                    (age, build, ethnicity, distinguishing features). NO emotion words,
                    NO clothing — those go in costumes / per-shot prompt.
  age_estimate      integer or null when unknown/inapplicable
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

    models = json.loads(paths.MODELS_FILE.read_text())
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


def declared_characters(scenes: list[dict]) -> list[dict]:
    """Source-only entries for every typed subject, including panel overrides."""
    from pace_core.pai_compat import resolve_shot

    entries = {}
    for scene in scenes:
        for shot in scene.get("shots") or []:
            for panel in [None, *(shot.get("panels") or [])]:
                resolved = resolve_shot(scene, shot, panel)
                for subject in (resolved.get("setup") or {}).get("subjects") or []:
                    cid = subject.get("character_id")
                    if not isinstance(cid, str) or not cid or cid in SCHEMA_KEYS:
                        continue
                    entry = entries.setdefault(cid, {"id": cid, "name": cid, "entity_type": "unknown",
                        "scenes": [], "metadata_status": "source_only"})
                    sid = scene.get("scene_id")
                    if sid and sid not in entry["scenes"]:
                        entry["scenes"].append(sid)
    return list(entries.values())


def seed_declared_characters(*, project: str) -> dict:
    """Repair missing registry identities from typed shots without a model call."""
    from pace_core.breakdown.scenes_kb import load_all_scenes

    entries = declared_characters(load_all_scenes(project=project)["scenes"])
    created, updated = merge_into_characters_json(entries, paths_for(project).chars_file)
    return {"declared_ids": [c["id"] for c in entries], "registry_created": created,
            "registry_updated": updated}


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
                    "metadata_status": c.get("metadata_status", "extracted"),
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
        for key in ("entity_type", "species", "body_proxy_3d"):
            if key == "entity_type" and existing is not None and c.get("metadata_status") == "source_only":
                continue  # Source seeding must not reclassify an existing artist entry.
            if c.get(key) and not registry[cid].get(key):
                registry[cid][key] = c[key]
        if c.get("metadata_status") != "source_only":
            registry[cid].setdefault("_extracted", {})["metadata_status"] = "extracted"

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
        model: str = default_model("writer"),
        reference_strategy: str = "user-only",
        dry_run: bool = False) -> dict[str, Any]:
    """Library entry point for `pai kb characters extract` (HTTP-callable).

    Args mirror the old CLI flags exactly; returns a structured result
    dict instead of printing+exiting.

    Raises ValueError on bad input (caller — a host server — translates
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
    declared = declared_characters(scenes)
    missing = [c for c in declared if c["id"] not in {item["id"] for item in extracted}]
    extracted = [*extracted, *missing]
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
        "source_only_ids":    [c["id"] for c in missing],
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
