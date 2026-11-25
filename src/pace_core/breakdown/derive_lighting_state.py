#!/usr/bin/env python3
"""Carry a power loss the breakdown declares into the shots it darkens.

A beat that says "all the panels lose power, plunging the cabin into pitch black" sits in a scene whose defaults
light it by sunlight, and nothing wrote the shot's `lighting.condition`, so the compiler stated both in one prompt.
The schema has the value (`blackout`, which the compiler substitutes for the natural source); this pass fills it.

The state is read off each shot's own action text against a short, closed lexicon, as `world_state.STATE_WORDS`
is, so a reviewer can argue with it. It persists to the scene's later shots until an action restores the light,
since a state lasts until an event changes it (`world_state`). The shot that loses the power is dark too: its
action asserts the darkness, not the moment before it. A condition someone authored is never overwritten.

    uv run python -m pace_core.breakdown.derive_lighting_state --project <slug>
    uv run python -m pace_core.breakdown.derive_lighting_state --project <slug> --write
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pace_core.paths import iter_canonical_scene_files, paths_for

LIGHT_LOST = re.compile(
    r"\b(?:los(?:e|es|ing)|cuts?) (?:all )?(?:its |their )?power\b|\bpower (?:goes|went|is) (?:out|off|cut)\b"
    r"|\bpower (?:cut|failure|outage)\b|\blights? (?:go|goes|went) out\b|\bblackout\b|\bpitch[- ]black\b"
    r"|\bplung\w* (?:\w+ ){0,3}into (?:total |complete )?dark(?:ness)?\b|断电|停电|一片漆黑|陷入黑暗", re.I)
LIGHT_BACK = re.compile(
    r"\bpower (?:returns|comes back|is restored)\b|\blights? (?:come|comes|came) back\b|\bpower(?:s)? back (?:on|up)\b"
    r"|恢复供电|来电|灯(?:重新)?亮", re.I)


def _action_text(shot: dict) -> str:
    return " ".join(a.get("description_en") or a.get("standalone") or a.get("description_zh") or ""
                    for a in (shot.get("events") or {}).get("actions") or [])


def derive(scene: dict) -> list[str]:
    """Shot ids whose `lighting.condition` the scene's own actions make `blackout`; fills them in place."""
    dark, filled = False, []
    for shot in scene.get("shots") or []:
        text = _action_text(shot)
        if LIGHT_LOST.search(text):
            dark = True
        elif LIGHT_BACK.search(text):
            dark = False
        if dark and not (shot.get("lighting") or {}).get("condition"):
            shot["lighting"] = {**(shot.get("lighting") or {}), "condition": "blackout"}
            filled.append(shot.get("shot_id"))
    return filled


def run(*, project: str, write: bool = False) -> dict:
    out = {}
    for f in iter_canonical_scene_files(Path(paths_for(project).scenes_dir)):
        before = f.read_text(encoding="utf-8")
        doc = json.loads(before)
        if filled := derive(doc):
            out[doc.get("scene_number")] = filled
            if write:
                f.with_suffix(".json.pre-lighting.bak").write_text(before, encoding="utf-8")
                f.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"project": project, "blackout": out, "written": write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    out = run(project=a.project, write=a.write)
    for scene, shots in sorted(out["blackout"].items(), key=lambda kv: kv[0] or 0):
        print(f"  scene {scene}: blackout on {', '.join(shots)}")
    print("written" if a.write else "dry run; --write applies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
