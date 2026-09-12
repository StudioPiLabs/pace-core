#!/usr/bin/env python3
"""Stamp the director's beat groups onto the shots they cover.

`kb/shot_design.json` says what each stretch of the film is doing and what
camera that asks for. Nothing reads it until it reaches the shots: the prompt
projection admits `camera._design.intent` as grounds-not-words, the
camera-movement deriver classifies against it, and `design_check` has nothing
to hold a shot against without it. Derived and unstamped, it is a document
about the film that the film does not contain.

Two things are written, and they are not the same kind of claim.

`camera._design` -- the group id and its intent prose -- is stamped on every
shot the group covers. It is provenance: it says which reading of the theme
this shot was built for, and it is never compiled into a prompt.

The typed values are applied only where the shot holds the SPLITTER'S
FALLBACK. `split_script` returns "medium" when no shot size can be parsed from
the action text and "eye_level" when no angle can; on Zheng that is 78 of 78
shots for both, because a Chinese shooting script does not name its coverage.
A fallback is the absence of a decision, so replacing it with the director's
is strictly better. A value the action text actually produced is a decision,
and is left alone -- three of Zheng's shots carry a `pull_out` or an `arc`
read off their own prose, and the group's movement does not overwrite them.

    uv run python -m pace_core.breakdown.apply_shot_design --project <slug>
    uv run python -m pace_core.breakdown.apply_shot_design --project <slug> --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.qc.design_check import _FRAME_MOVES, _GEAR_MOVES   # noqa: E402
from pace_core.paths import iter_canonical_scene_files, paths_for      # noqa: E402

#: What `split_script` writes when the action text settles nothing. Replacing
#: one of these is filling a blank, not overruling an author.
FALLBACK_SHOT_SIZE = "medium"
FALLBACK_ANGLE = "eye_level"


def _groups_by_scene(design: dict) -> dict[str, dict]:
    return {s: g for g in design.get("groups") or [] for s in g.get("scenes") or []}


def apply_to_scene(scene: dict, group: dict) -> tuple[dict, list[str]]:
    """(scene, notes). Pure: the caller decides whether to write."""
    notes: list[str] = []
    want_size = group.get("shot_size")
    want_angle = group.get("angle")
    want_move = group.get("camera_movement")

    for shot in scene.get("shots") or []:
        sid = shot.get("shot_id")
        cam = shot.setdefault("camera", {})
        cam["_design"] = {"group": group.get("id"), "intent": group.get("intent")}

        ci = cam.setdefault("creative_intent", {})
        if want_size and ci.get("shot_size") in (None, "", FALLBACK_SHOT_SIZE):
            if ci.get("shot_size") != want_size:
                ci["shot_size"] = want_size
                notes.append(f"{sid}: shot_size -> {want_size}")

        ex = cam.setdefault("extrinsics", {})
        if want_angle and ex.get("angle") in (None, "", FALLBACK_ANGLE):
            if ex.get("angle") != want_angle:
                ex["angle"] = want_angle
                notes.append(f"{sid}: angle -> {want_angle}")

        traj = cam.setdefault("trajectory", {})
        has_move = bool(traj.get("movement_3d") or traj.get("movement_2d"))
        if want_move in _GEAR_MOVES:
            # Rig behaviour, not a framing change: it belongs in `gear`, which
            # is where this corpus family lost `handheld` once already by
            # putting it in movement_3d, where it is not a legal value.
            if traj.get("gear") != want_move:
                traj["gear"] = want_move
                notes.append(f"{sid}: gear -> {want_move}")
        elif want_move in _FRAME_MOVES and not has_move:
            traj["movement_3d"] = [_FRAME_MOVES[want_move]]
            traj["static"] = False
            notes.append(f"{sid}: movement_3d -> {_FRAME_MOVES[want_move]}")
        elif want_move in _FRAME_MOVES and has_move:
            notes.append(f"{sid}: kept its own move "
                         f"{traj.get('movement_3d') or traj.get('movement_2d')}")
    return scene, notes


def run(*, project: str, write: bool = False) -> dict:
    p = paths_for(project)
    dfile = Path(p.shot_design_file)
    if not dfile.exists():
        raise ValueError(f"{project!r} has no kb/shot_design.json — derive one first")
    by_scene = _groups_by_scene(json.loads(dfile.read_text(encoding="utf-8")))

    stamped, skipped, all_notes = [], [], []
    for f in iter_canonical_scene_files(Path(p.scenes_dir)):
        doc = json.loads(f.read_text(encoding="utf-8"))
        group = by_scene.get(doc.get("scene_id"))
        if not group:
            skipped.append(doc.get("scene_id"))
            continue
        new, notes = apply_to_scene(doc, group)
        all_notes += notes
        if write:
            f.with_suffix(".json.pre-design.bak").write_text(
                json.dumps(json.loads(f.read_text(encoding="utf-8")),
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            f.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        stamped.append(doc.get("scene_id"))

    return {"project": project, "scenes_stamped": stamped,
            "scenes_in_no_group": skipped, "changes": len(all_notes),
            "notes": all_notes, "written": write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--write", action="store_true",
                    help="write the scenes, with a .pre-design.bak beside each")
    a = ap.parse_args()
    out = run(project=a.project, write=a.write)
    for n in out["notes"][:12]:
        print("  " + n)
    if len(out["notes"]) > 12:
        print(f"  … {len(out['notes']) - 12} more")
    print(json.dumps({k: v for k, v in out.items() if k != "notes"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
