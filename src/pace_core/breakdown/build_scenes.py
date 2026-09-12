"""
build_scenes.py — Materialize Blender greybox .blend files from Location Bibles.

Usage: blender --background --python build_scenes.py

Reads every kb/on_scene/scenes/*.json and produces the matching .blend in
pipeline/scenes/ by:
  1. Dispatching the bible's `primitives_spec` through `primitives.build_from_spec`
     (handles the 7 greybox primitive kinds + imported_mesh delegation).
  2. Adding Subject_A / Subject_B procedural humanoids at the bible's
     `subject_defaults` positions (blender_render.py overrides per shot).
  3. Adding the default camera from `bible.camera_default`.

Replaces the old hand-coded build_street() / build_cafe() / build_apartment()
functions; the per-location geometry now lives in the bibles themselves.
For the bootstrap of an entirely new location, use `build_locations.py` (LLM
writes the bible) followed by this script (Blender materializes it).
"""

import bpy
import json
import math
import os
from pathlib import Path

from pace_core.breakdown.primitives        import build_from_spec        # noqa: E402
from pace_core.breakdown.build_characters  import build_humanoid         # noqa: E402
from pace_core.paths             import SCENES_DIR, BLENDER_SCENES_DIR as SCENES_OUT, FILM_REPO_ROOT, project_scenes_dir

# Project-as-namespace: PAI_STUDIO_PROJECT env var overrides SCENES_DIR to
# point at any project's directory. Default = SCENES_DIR (= splits/main).
# Blender invocation: PAI_STUDIO_PROJECT=father_pass1 blender --background \
#     --python pipeline/build_scenes.py
# .blend output (SCENES_OUT) is NOT per-project — .blend files are shared
# production assets used by every project that references the same location.
_project_override = os.environ.get("PAI_STUDIO_PROJECT")
if _project_override:
    SCENES_DIR = project_scenes_dir(_project_override)

SCENES_OUT.mkdir(parents=True, exist_ok=True)


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _add_camera(camera_default: dict):
    loc = camera_default.get("location", [3, 0, 1.65])
    rot_deg = camera_default.get("rotation_deg", [90, 0, 90])
    bpy.ops.object.camera_add(location=tuple(loc))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = tuple(math.radians(r) for r in rot_deg)
    bpy.context.scene.camera = cam
    return cam


def _add_subjects(subject_defaults: dict):
    """subject_defaults: {Subject_A: {origin, character, pass_index}, Subject_B: {...}}"""
    for name, cfg in subject_defaults.items():
        if name.startswith("_"):  # skip _note / _comment
            continue
        build_humanoid(
            name,
            cfg.get("character", "jia"),
            origin=tuple(cfg.get("origin", [0, 0, 0])),
            pass_index=cfg.get("pass_index", 0),
        )


def build_one(bible_path: Path) -> Path | None:
    bible = json.loads(bible_path.read_text())
    loc_id = bible.get("id") or bible_path.stem
    spec = bible.get("primitives_spec")
    if not spec or spec.get("needs_human_authoring"):
        print(f"  [{loc_id}] skipped: no primitives_spec (or flagged needs_human_authoring)")
        return None

    print(f"== building {loc_id} ==")
    _clear_scene()
    build_from_spec(spec, glb_root=FILM_REPO_ROOT)

    subj = bible.get("subject_defaults") or {}
    if subj:
        _add_subjects(subj)

    cam = bible.get("camera_default") or {}
    _add_camera(cam)

    out_blend = SCENES_OUT / f"{loc_id}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"  wrote {out_blend}")
    return out_blend


def main():
    bibles = sorted(SCENES_DIR.glob("*.json"))
    if not bibles:
        print(f"  no scene files found in {SCENES_DIR}")
        return
    built = []
    for b in bibles:
        try:
            out = build_one(b)
            if out:
                built.append(out)
        except Exception as e:
            print(f"  [{b.stem}] FAILED: {e}")
    print(f"\nAll scenes built → {SCENES_OUT}  ({len(built)}/{len(bibles)})")


if __name__ == "__main__":
    main()
