"""Locations creation phase.

Takes `kb/on_scene/scenes_breakdown.json`, finds locations that don't yet have a bible
file, asks an LLM to write the bible (anchor, lighting plan, camera positions,
variable states, AND a `primitives_spec` for greybox geometry) — then runs
Blender headless to materialize the .blend file from that spec.

The LLM is constrained: it can only emit primitives from the small fixed
vocabulary in `primitives.PRIMITIVE_SCHEMAS`. Anything more complex (vehicles,
hero set pieces) must be described as `imported_mesh` referencing an existing
.glb (the user can populate those via `build_props.py` or `image_to_3d.hunyuan_mv`).

When the LLM cannot describe a location with this vocabulary, it sets
`needs_human_authoring: true` in the bible — pipeline.py will then fall back
to whatever .blend the user hand-authors.

Usage:
  ANTHROPIC_API_KEY=sk-... python3 pipeline/build_locations.py
  python3 pipeline/build_locations.py --model gpt-4o --dry-run
  python3 pipeline/build_locations.py --location street    # rebuild specific
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pace_core.llm_client import strip_fences

PIPELINE_DIR     = Path(__file__).resolve().parent
from pace_core.paths import paths_for, MODELS_FILE, BOX  # noqa: E402
BLENDER_BIN      = BOX.blender_bin    # default: $PATH lookup

# Import the schemas without loading bpy
from pace_core.breakdown.primitives import PRIMITIVE_SCHEMAS  # noqa: E402


SYSTEM_PROMPT = """You are a production designer. You will see a film scene and must produce a "Location Bible" — the document a real PD writes for one shooting location.

Output ONE JSON object with these fields:

  id                  stable lowercase identifier; matches the scene's location_ref
  name                short human label
  type                "INT" or "EXT"
  build_type          one of: fully_synthetic | photo_reference | greenscreen_plate | hybrid_real_modified
  anchor              30-60 word Flux prompt fragment describing the location visually
  spatial_layout      {scale_meters: {length, width, height_open}, key_landmark, ground_treatment, background_layer}
  lighting_plan       {key, fill, effect}     each a one-line description of what kind of light source
  camera_positions    {<position_name>: <description>}     several named camera presets matching the scene's needs
  variable_states     {<state_name>: {options, default, constancy_rule}}    things that may differ shot-to-shot vs that must stay constant
  production_notes    one sentence on the trickiest thing to keep consistent
  primitives_spec     {primitives: [...]}     greybox geometry for ControlNet depth pass — see PRIMITIVE_SCHEMAS below

The `primitives_spec` is the most important field for downstream automation. RULES:

  1. Every primitive must use a `kind` from the schemas below. Do NOT invent kinds.
  2. Use the smallest set of primitives that adequately blocks out the scene for ControlNet.
     For most scenes 4-8 primitives is enough. Don't model details — model VOLUMES.
  3. Coordinate frame: +X = right, +Y = forward (away from default camera), +Z = up.
     Default camera is at (0, -8, 1.6) looking at origin; place geometry so it's framed.
  4. Heights in meters. Walls/grounds/ceilings: realistic dimensions.
     Ground typical [10, 10]; ceiling height 2.7-3m for interiors; walls match the room.
  5. For complex hero objects (vehicle, machinery, statue, anything organic) use kind="imported_mesh"
     with a placeholder glb_path like "assets/3d_models/<id>.glb" — the user will produce the .glb
     via build_props.py separately. Do NOT try to model them as boxes.
  6. If the scene cannot be expressed with this vocabulary AT ALL (non-Euclidean, abstract, etc.),
     set primitives_spec = {"primitives": [], "needs_human_authoring": true} and explain why
     in production_notes.

Return JSON only — no prose, no markdown fences."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_user_prompt(location_id: str, scenes_in_loc: list[dict]) -> str:
    parts = [
        f"## LOCATION ID: {location_id}",
        "",
        "## SCENES THAT TAKE PLACE HERE",
        json.dumps(scenes_in_loc, ensure_ascii=False, indent=2),
        "",
        "## PRIMITIVE SCHEMAS (your only allowed `kind` values)",
        json.dumps(PRIMITIVE_SCHEMAS, ensure_ascii=False, indent=2),
        "",
        "Return JSON only.",
    ]
    return "\n".join(parts)


def _llm_call(messages: list, model_key: str) -> tuple[str, float]:
    """Returns (raw_text, cost_usd). Uses llm_client.call_model which
    routes api-key resolution through llm_tokens.py (same path the rest
    of the pipeline uses), so studio_server doesn't need env vars."""
    from pace_core.llm_client import call_model  # type: ignore

    models = json.loads(MODELS_FILE.read_text())
    if model_key not in models:
        raise ValueError(f"unknown model '{model_key}'. options: {list(models)}")
    cfg = models[model_key]
    raw, cost = call_model(cfg, messages, api_key=None)
    print(f"  [llm] model={model_key} cost=${cost:.4f}", file=sys.stderr)
    return raw, cost


class BibleParseError(ValueError):
    """The model answered and was billed, and the answer was not JSON.

    A ValueError subclass so existing `except ValueError` handlers keep
    catching it, carrying `.cost` so the spend is not lost with the result.
    """
    cost: float = 0.0


def generate_bible(location_id: str, scenes_in_loc: list[dict],
                   model_key: str) -> tuple[dict, float]:
    """LLM call → (location bible dict, cost_usd)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_prompt(location_id, scenes_in_loc)},
    ]
    raw, cost = _llm_call(messages, model_key)
    raw = strip_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # The call happened and was billed. Carry the cost on the exception so
        # a location that fails to parse is still counted -- scene_09 of Zheng
        # cost $0.1276 and was reported in a total that did not include it.
        err = BibleParseError(
            f"LLM did not return valid JSON for {location_id}: {e}\n"
            f"first 500 chars: {raw[:500]}")
        err.cost = cost
        raise err
    return data, cost


def validate_bible(bible: dict) -> list[str]:
    """Return a list of warnings (non-fatal)."""
    warnings = []
    spec = bible.get("primitives_spec", {})
    prims = spec.get("primitives", []) if isinstance(spec, dict) else spec
    for i, p in enumerate(prims):
        kind = p.get("kind")
        if kind not in PRIMITIVE_SCHEMAS:
            warnings.append(f"primitive #{i}: unknown kind {kind!r}")
            continue
        if "name" not in p:
            warnings.append(f"primitive #{i} ({kind}): missing 'name'")
        if kind == "imported_mesh" and "glb_path" not in p:
            warnings.append(f"primitive #{i} (imported_mesh): missing 'glb_path'")
    if not bible.get("anchor"):
        warnings.append("missing 'anchor' (Flux prompt won't have any reference for this location)")
    if not bible.get("camera_positions"):
        warnings.append("missing 'camera_positions' (shot author/LLM will have nothing to pick from)")
    return warnings


def render_blend_from_spec(bible: dict, location_id: str, project: str) -> Path | None:
    """Run Blender headless to materialize bible.primitives_spec → .blend.
    Output lands in <project>/blender_scenes/<location_id>.blend."""
    spec = bible.get("primitives_spec", {})
    if not isinstance(spec, dict):
        return None
    if spec.get("needs_human_authoring"):
        print(f"  [{location_id}] flagged needs_human_authoring; skipping Blender build")
        return None
    if not spec.get("primitives"):
        print(f"  [{location_id}] no primitives; skipping Blender build")
        return None

    blend_out_dir = paths_for(project).blender_scenes_dir
    blend_out_dir.mkdir(parents=True, exist_ok=True)
    out_blend = blend_out_dir / f"{location_id}.blend"
    spec_tmp  = Path(f"/tmp/__location_spec_{location_id}.json")
    spec_tmp.write_text(json.dumps(spec, ensure_ascii=False))

    cmd = [
        BLENDER_BIN, "--background",
        "--python", str(PIPELINE_DIR / "primitives.py"),
        "--", str(spec_tmp), str(out_blend),
    ]
    print(f"  [blender] {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"  [blender] FAIL exit={r.returncode}\n{r.stderr[-1000:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [{location_id}] Blender timed out")
        return None
    spec_tmp.unlink(missing_ok=True)
    return out_blend if out_blend.exists() else None


def run(*, project: str = "main",
        model: str = "claude-sonnet-5-rb",
        location: str | None = None,
        overwrite: bool = False,
        dry_run: bool = False) -> dict:
    """Library entry point for `pai kb locations build` (HTTP-callable).

    Args mirror the old CLI flags; returns a structured result dict.
    `location=None` ⇒ generate every missing bible (skip ones that exist
    unless `overwrite=True`). `location="<id>"` ⇒ generate just that one
    (always regenerates).

    Raises ValueError on bad input (translates to HTTP 400 in studio_server).
    """
    from pace_core.breakdown import scenes_kb
    scenes_data = scenes_kb.load_all_scenes(project=project)
    scenes = scenes_data.get("scenes", scenes_data)
    if not scenes:
        raise ValueError(f"project {project!r} has no scenes — split_script first")

    # Group scenes by location_ref
    by_loc: dict[str, list[dict]] = {}
    no_loc: list[str] = []
    for s in scenes:
        loc = s.get("location_ref")
        if not loc:
            no_loc.append(s.get("scene_id", "?"))
            continue
        by_loc.setdefault(loc, []).append(s)

    if location:
        if location not in by_loc:
            raise ValueError(f"location {location!r} not found in project {project!r}")
        targets = {location: by_loc[location]}
    else:
        targets = by_loc

    locations_dir = paths_for(project).kb_dir / "locations"
    locations_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    total_cost = 0.0
    for loc_id, loc_scenes in targets.items():
        bible_path = locations_dir / f"{loc_id}.bible.json"
        if bible_path.exists() and not overwrite and not location:
            summary.append({"location": loc_id, "status": "skipped",
                            "detail": "already exists"})
            continue

        print(f"\n== Generating bible for '{loc_id}' ({len(loc_scenes)} scenes)")
        try:
            bible, cost = generate_bible(loc_id, loc_scenes, model)
        except ValueError as e:
            # Print it: `generate_bible` builds a diagnostic with the first 500
            # characters of what came back, and it used to reach only the
            # returned summary -- which the studio's status endpoint did not
            # pass on, so a paid call failed and no surface anywhere said so.
            print(f"  [FAILED] {loc_id}: {e}", file=sys.stderr)
            total_cost += getattr(e, "cost", 0.0)
            summary.append({"location": loc_id, "status": "failed", "detail": str(e)})
            continue
        total_cost += cost

        bible.setdefault("id", loc_id)
        bible.setdefault("_meta", {})["generated_at"] = _now_iso()
        bible["_meta"]["model_used"] = model
        bible["_meta"]["scene_refs"] = [s.get("scene_id") for s in loc_scenes]

        warnings = validate_bible(bible)
        bible["_meta"]["warnings"] = warnings
        for w in warnings:
            print(f"  [warn] {w}", file=sys.stderr)

        if dry_run:
            summary.append({"location": loc_id, "status": "dry-run",
                            "bible_preview": bible, "warnings": warnings})
            continue

        bible_path.write_text(json.dumps(bible, indent=2, ensure_ascii=False))
        print(f"  wrote {bible_path}")
        out_blend = render_blend_from_spec(bible, loc_id, project)
        summary.append({
            "location": loc_id, "status": "ok" if out_blend else "bible-only",
            "bible_path": str(bible_path),
            "blend_path": str(out_blend) if out_blend else None,
            "warnings":   warnings,
        })

    return {
        "project":          project,
        "model":            model,
        "scenes_no_location": no_loc,
        "results":          summary,
        "llm_cost_usd":     round(total_cost, 4),
    }