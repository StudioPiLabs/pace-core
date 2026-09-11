#!/usr/bin/env python3
"""Restage a tight shot the camera cannot deliver, or split it into singles.

A declared close-up of two people side by side is over-constrained: the band a
close-up names crops the frame vertically, the cast's width decides the
distance, and the frame comes back as loose as a medium -- the greybox reports
it as `shot_size_bound_by: width`. An operator reaches for two remedies, in
order:

1. Restage: bring the others in toward the focus and stagger them in depth,
   so the group is narrow enough for the declared size to bind.
2. Split: give each subject a single of their own -- the shot / reverse-shot
   coverage a tight exchange is normally shot as.

Every candidate is built before it is kept (`make_panel_greybox` on the scene
document in memory), and a restage is written only when the declared size
binds on it and nobody ends up hidden behind anybody else. A split gives the
focus subject the existing panel and every other subject a new one after it,
numbered the way the studio's insert-panel route numbers them; the new panels
carry a reaction beat read off that subject's own declared pose and gaze, and
no emotion cue, because the shot's cue belongs to whoever acts in it.

Writes go to the scene documents in place, with a `.pre-restage.bak` beside
each one this touches.

    uv run python -m pace_core.breakdown.restage_tight_shots --project <slug>
    uv run python -m pace_core.breakdown.restage_tight_shots --project <slug> --write
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.pai_compat import primary_focus_of, resolve_shot   # noqa: E402
from pace_core.paths import paths_for                                     # noqa: E402

TIGHT_SIZES = ("extreme_close_up", "close_up", "medium_close_up")
# Screen share between the focus and each other subject once restaged: a
# shoulder's overlap at a close-up, not a seat apart.
RESTAGE_SEPARATION = 0.12
# Keep restaged subjects this far inside the frame edges.
_EDGE = 0.08

# (scene document, scene id, panel id) -> {"ok", "bound_by", "overlap"}
Probe = Callable[[dict, str, str], dict]


def _x(s: dict) -> float:
    x = (s.get("screen_position") or {}).get("x")
    return 0.5 if x is None else float(x)


def tight_multi_panels(scene: dict) -> list[tuple[dict, dict]]:
    """(shot, panel) pairs that declare a tight size on two or more subjects.

    Over-the-shoulder is left alone: it frames two people by design, with the
    lens behind one of them, and is not solved by fitting the group's width.
    """
    out = []
    for shot in scene.get("shots") or []:
        for panel in shot.get("panels") or []:
            r = resolve_shot(scene, shot, panel)
            cam = r.get("camera") or {}
            ci = cam.get("creative_intent") or {}
            subs = (r.get("setup") or {}).get("subjects") or []
            if ci.get("shot_size") not in TIGHT_SIZES or len(subs) < 2:
                continue
            if ci.get("framing") == "ots" or (cam.get("extrinsics") or {}).get("position") == "ots":
                continue
            out.append((shot, panel))
    return out


def focus_of(panel: dict, shot: dict, subjects: list[dict]) -> str:
    """The subject the panel is about: its declared focus, else the most central."""
    pf = primary_focus_of(panel, shot)
    ids = [s.get("character_id") for s in subjects]
    if pf.get("type") == "character" and pf.get("ref") in ids:
        return pf["ref"]
    return min(subjects, key=lambda s: abs(_x(s) - 0.5)).get("character_id")


def restaged_subjects(subjects: list[dict], focus_id: str) -> list[dict]:
    """The same cast drawn in around the focus and staggered behind it.

    The focus keeps its declared x and comes to the foreground; everyone else
    keeps their side of it and their order, one RESTAGE_SEPARATION apart, in
    the midground.
    """
    subs = copy.deepcopy(subjects)
    focus = next(s for s in subs if s.get("character_id") == focus_id)
    fx = _x(focus)
    focus.setdefault("screen_position", {})
    focus["screen_position"].update(x=fx, depth="foreground")
    others = [s for s in subs if s is not focus]
    left = sorted((s for s in others if _x(s) < fx), key=_x, reverse=True)
    right = sorted((s for s in others if _x(s) >= fx), key=_x)
    for sign, group in ((-1, left), (1, right)):
        for k, s in enumerate(group, start=1):
            x = min(max(fx + sign * k * RESTAGE_SEPARATION, _EDGE), 1 - _EDGE)
            s.setdefault("screen_position", {})
            s["screen_position"].update(x=round(x, 3), depth="midground")
    return subs


def _registry_name(cid: str, characters: dict) -> str:
    """How the prose names a character: its registered English name, else its key."""
    for key, e in characters.items():
        if not isinstance(e, dict):
            continue
        if cid == key or cid == e.get("name") or cid in (e.get("aliases") or []):
            return e.get("name_en") or key.replace("_", " ").title()
    return cid


def reaction_action(subject: dict, characters: dict) -> dict:
    """A reaction beat for a subject given a single of their own.

    Built only from what the subject already declares -- whom it looks at,
    and its pose -- so the split adds coverage and no story.
    """
    cid = subject.get("character_id")
    name = _registry_name(cid, characters)
    gaze = subject.get("gaze") or {}
    target = gaze.get("target_ref") if gaze.get("target_type") == "character" else None
    en = f"{name} watches {_registry_name(target, characters)}" if target else f"{name} reacts"
    if subject.get("pose"):
        en += f", {subject['pose']}"
    zh = f"{cid}注视{target}" if target else f"{cid}的反应"
    return {"description_en": en + ".", "description_zh": zh, "temporal": "atomic",
            "foreground": "focal", "background": False}


def _max_panel_n(scene: dict) -> int:
    vals = [int(m.group(1)) for sh in scene.get("shots") or []
            for p in sh.get("panels") or []
            if (m := re.search(r"_panel_(\d+)$", str(p.get("id") or "")))]
    return max(vals or [0])


def _single(panel: dict) -> None:
    ov = panel.get("camera_override") or {}
    ci = dict(ov.get("creative_intent") or {}, framing="single")
    panel["camera_override"] = {**ov, "creative_intent": ci}


def split_panel(scene: dict, scene_id: str, shot: dict, panel: dict,
                subjects: list[dict], focus_id: str, characters: dict) -> list[str]:
    """Give the focus this panel and every other subject a single after it."""
    seed = copy.deepcopy(panel)
    by_id = {s.get("character_id"): s for s in subjects}
    panel["setup_override"] = {**(panel.get("setup_override") or {}),
                               "subjects": [copy.deepcopy(by_id[focus_id])]}
    _single(panel)
    at = shot["panels"].index(panel) + 1
    new_ids = []
    for s in sorted((s for s in subjects if s.get("character_id") != focus_id), key=_x):
        n = max(int(scene.get("_max_panel_n") or 0), _max_panel_n(scene)) + 1
        pid = f"{scene_id}_{shot['shot_id']}_panel_{n:04d}"
        p = copy.deepcopy(seed)
        p.update(id=pid, scene_id=scene_id, shot_id=shot["shot_id"])
        p.pop("render", None)
        p["primary_focus"] = {**(seed.get("primary_focus") or {}),
                              "type": "character", "ref": s.get("character_id")}
        p["setup_override"] = {"subjects": [copy.deepcopy(s)]}
        _single(p)
        p["events_override"] = {"actions": [reaction_action(s, characters)], "emotions": []}
        shot["panels"].insert(at, p)
        at += 1
        scene["_max_panel_n"] = n
        scene.setdefault("compile_hints", []).append(
            {"panel_id": pid, "flux": {}, "gpt_image_2": {}, "wan_i2v": {}})
        new_ids.append(pid)
    for i, p in enumerate(shot["panels"], start=1):
        p["panel_number"] = i
    return new_ids


def restage_scene(scene: dict, scene_id: str, probe: Probe, characters: dict,
                  max_overlap: float) -> list[dict]:
    """Restage or split every tight multi-subject panel of one scene, in place."""
    rows = []
    for shot, panel in tight_multi_panels(scene):
        pid = panel.get("id")
        subs = (resolve_shot(scene, shot, panel).get("setup") or {}).get("subjects") or []
        before = probe(scene, scene_id, pid)
        if before.get("bound_by") != "width":
            rows.append({"panel": pid, "action": "kept", "bound_by": before.get("bound_by")})
            continue
        focus_id = focus_of(panel, shot, subs)
        trial = copy.deepcopy(scene)
        tshot = next(s for s in trial["shots"] if s.get("shot_id") == shot.get("shot_id"))
        tpanel = next(p for p in tshot["panels"] if p.get("id") == pid)
        tpanel["setup_override"] = {**(tpanel.get("setup_override") or {}),
                                    "subjects": restaged_subjects(subs, focus_id)}
        after = probe(trial, scene_id, pid)
        overlap = after.get("overlap")
        if after.get("bound_by") == "height" and (overlap is None or overlap <= max_overlap):
            panel["setup_override"] = tpanel["setup_override"]
            rows.append({"panel": pid, "action": "restaged", "bound_by": "height",
                         "overlap": overlap})
            continue
        new_ids = split_panel(scene, scene_id, shot, panel, subs, focus_id, characters)
        rows.append({"panel": pid, "action": "split", "focus": focus_id,
                     "new_panels": new_ids,
                     "restage_tried": {"bound_by": after.get("bound_by"), "overlap": overlap}})
    return rows


def build_probe(project: str) -> Probe:
    """Build the panel from the document in memory and read what the camera did."""
    from pace_core.qc.greybox_gate import _mask
    from pace_core.node.panel_greybox import make_panel_greybox

    def probe(scene: dict, scene_id: str, panel_id: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            r = make_panel_greybox(project, scene_id, panel_id,
                                   Path(tmp) / f"{panel_id}_90001_.png", scene=scene)
            if not r.get("ok"):
                return {"ok": False, "bound_by": None, "overlap": None,
                        "error": r.get("error")}
            masks = [m for m in (_mask(p) for p in (r.get("body_mattes") or {}).values())
                     if m is not None and m.any()]
            worst = 0.0
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    inter = float((masks[i] & masks[j]).sum())
                    worst = max(worst, inter / masks[i].sum(), inter / masks[j].sum())
            return {"ok": True, "bound_by": r.get("shot_size_bound_by"),
                    "overlap": round(worst, 4)}
    return probe


def run(project: str, scenes: list[str] | None = None, write: bool = False,
        probe: Probe | None = None) -> dict:
    from pace_core.qc.greybox_gate import MAX_SUBJECT_OVERLAP

    p = paths_for(project)
    probe = probe or build_probe(project)
    try:
        characters = json.loads(Path(p.chars_file).read_text())
    except (OSError, ValueError):
        characters = {}
    report = {"project": project, "write": write, "scenes": {}}
    for f in sorted(Path(p.scenes_dir).glob("scene_*.json")):
        if f.name.count(".") != 1:
            continue
        text = f.read_text()
        scene = json.loads(text)
        sid = scene.get("scene_id") or f.stem
        if scenes and sid not in scenes:
            continue
        rows = restage_scene(scene, sid, probe, characters, MAX_SUBJECT_OVERLAP)
        if not rows:
            continue
        report["scenes"][sid] = rows
        if write and any(r["action"] != "kept" for r in rows):
            f.with_suffix(".json.pre-restage.bak").write_text(text)
            f.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", action="append", help="scene_id(s); default all")
    ap.add_argument("--write", action="store_true",
                    help="write the scenes, with a .pre-restage.bak beside each")
    a = ap.parse_args()
    print(json.dumps(run(a.project, a.scene, a.write), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
