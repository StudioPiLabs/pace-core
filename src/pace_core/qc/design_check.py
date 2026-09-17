#!/usr/bin/env python3
"""Does the corpus still say what the production design says?

Two documents state things about a film that no scene file can state for
itself, and until now nothing read either of them.

`kb/shot_design.json` is the director's 核心主题 -- the theme, written as four
beat groups with an `intent` paragraph each ("the camera keeps its
distance, like a stranger at the window"). A person read it once and hand-applied its
(shot size, angle, movement) triples into the scene documents. After that the
theme left the machine: nothing could check that a shot still served it, and a
regenerated shot could not rederive it.

`kb/location_stubs.json` holds the 世界观 -- one prose `setting` per location
plus era/region/culture, which every scene that names that location copies
into its own `shot_defaults`. Five stubs became eleven copies, and a copy can
drift from its source in silence.

Six rules, each from a divergence a production actually carried:

  shot_size / angle    the shot's own value disagrees with its beat group's.
  movement_absent      the group asks for a movement no shot carries. The
                       design asks for handheld; the shots carry
                       `gear: tripod` and handheld nowhere. It was there once -- stored inside
                       `movement_3d`, where it is not a legal value (see
                       densify_panels) -- and the cleanup that moved it into
                       `gear` normalised it to tripod instead.
  movement_conflict    a shot whose trajectory says it moves while the DSL
                       beside it says hold. All four crane shots.
  world_drift          a shot's backdrop CONTRADICTS the stub of the location
                       it names: a shot naming a car interior while
                       carrying the highway's region and culture.
  world_unresolved     the stub knows a field and the shot does not carry it,
                       so the prompt is compiled without it.
  prop_link_unhonoured a prop's own `linked_scenes` names this scene and no
                       shot in it declares the prop,
                       including a prop with only one scene it can appear
                       in. (`linked_shots` is filled on 0 of 15 -- a second
                       field the registry declares and nothing writes.)
  location_ref_drift   `narrative_meta.location_ref` names a different
                       location than the backdrop does: a scene headed
                       EXT. SUPER-HIGHWAY that refs a car interior.

    uv run python -m pace_core.qc.design_check --project <slug>
    uv run python -m pace_core.qc.design_check --project <slug> --strict

Reports; never edits. Which side of a divergence is right -- the design or the
scene -- is a directorial decision, and a checker that guessed would quietly
overwrite the film with its own opinion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.pai_compat import resolve_prop_key, resolve_shot   # noqa: E402
from pace_core.paths import PAI_PROJECTS_ROOT               # noqa: E402

#: Rig behaviour. Describes how the camera is held, not where the frame goes,
#: and so lives in `trajectory.gear` -- the distinction densify_panels.py
#: documents, and one a breakdown has got wrong.
_GEAR_MOVES = {"handheld", "steadicam"}

#: Framing moves, in the schema's own vocabulary. The design writes a
#: direction ("crane_up"); Movement3D does not carry one.
_FRAME_MOVES = {
    "crane_up": "crane", "crane_down": "crane", "crane": "crane",
    "push_in": "push_in", "pull_out": "pull_out", "tracking": "tracking",
    "orbit": "orbit",
}

#: The world fields a scene copies from the location stub it names, compared
#: strictly. `setting` is deliberately NOT among them: a shot is entitled to
#: specialise its location's prose, and an enrichment that writes a per-shot
#: setting for every shot would be flagged on every shot. A shot's prose that
#: adds "the wreck must not overlap the subject" is later and better than the
#: stub, not a divergence from it. The typed fields have no such licence: a
#: shot naming a car interior while carrying the highway's region is simply
#: wrong.
_WORLD_FIELDS = ("era", "region", "culture")


def _groups_by_scene(design: dict) -> dict[str, dict]:
    """scene_id -> the beat group that claims it."""
    out = {}
    for g in design.get("groups") or []:
        for s in g.get("scenes") or []:
            out[s] = g
    return out


def check_theme(design: dict, scene: dict) -> list[tuple[str, str, str]]:
    """(shot_id, rule, detail) for one scene against its beat group."""
    group = _groups_by_scene(design).get(scene.get("scene_id"))
    if not group:
        return []
    # A design that was derived and never stamped onto the shots disagrees
    # with every one of them, which is one fact about the project rather than
    # N faults in the scene. Reported once, so a real drift in a scene that
    # HAS been applied is not buried under it.
    if not any(((sh.get("camera") or {}).get("_design") or {}).get("group")
               for sh in scene.get("shots") or []):
        return [("-", "design_not_applied",
                 f"group {group.get('id')!r} covers this scene; no shot "
                 f"carries camera._design")]
    out = []
    for shot in scene.get("shots") or []:
        sid = str(shot.get("shot_id"))
        r = resolve_shot(scene, shot)
        cam = r.get("camera") or {}
        ci, ex = cam.get("creative_intent") or {}, cam.get("extrinsics") or {}
        traj = cam.get("trajectory") or {}

        for key, got in (("shot_size", ci.get("shot_size")),
                         ("angle", ex.get("angle"))):
            want = group.get(key)
            if want and got and got != want:
                out.append((sid, key, f"{got} (group asks {want})"))

        want_move = group.get("camera_movement")
        m3d = list(traj.get("movement_3d") or [])
        m2d = list(traj.get("movement_2d") or [])
        if want_move in _GEAR_MOVES:
            if (traj.get("gear") or "") != want_move:
                out.append((sid, "movement_absent",
                            f"gear={traj.get('gear')!r} (group asks {want_move})"))
        elif want_move in _FRAME_MOVES:
            if _FRAME_MOVES[want_move] not in m3d:
                out.append((sid, "movement_absent",
                            f"movement_3d={m3d} (group asks {want_move})"))

        # A trajectory that moves, beside a DSL that holds. The two are
        # written by different steps and neither reads the other.
        moving = [m for m in m3d + m2d if m != "static"]
        dsl = str(traj.get("movement_dsl") or "")
        if moving and dsl.startswith("hold"):
            out.append((sid, "movement_conflict", f"{moving} vs dsl {dsl!r}"))
    return out


def check_props(props: dict, scene: dict) -> list[tuple[str, str, str]]:
    """(shot_id, rule, detail) for one scene against the prop registry.

    No keyword matching: the registry states the link itself. Every prop
    carries `linked_scenes`, written when it was extracted, so a prop whose
    own record names this scene while no shot in it declares the prop is a
    gap the registry proves rather than one a heuristic guesses at.

    A prop that names the only scene it can appear in while no shot there
    declares it is not background
    dressing the location bible already covers; it is the object the beat is
    about.
    """
    sid = scene.get("scene_id")
    # Resolved to registry keys, not compared as spelled. The enrichment
    # invents an identifier per shot -- `computer_monitor`, `u_pan` -- so a
    # raw comparison reports the monitor unhonoured in all four scenes that
    # declare one. This rule was written to find exactly that kind of gap and
    # was making it: half its findings were the checker's own.
    declared = set()
    for sh in scene.get("shots") or []:
        for p in ((resolve_shot(scene, sh).get("setup") or {}).get("props") or []):
            if not isinstance(p, dict):
                continue
            ref = p.get("prop_id")
            declared.add(resolve_prop_key(ref, props) or ref)
    out = []
    for pid, rec in (props or {}).items():
        if pid.startswith("_") or not isinstance(rec, dict):
            continue
        if sid in (rec.get("linked_scenes") or []) and pid not in declared:
            out.append(("-", "prop_link_unhonoured",
                        f"{pid} names this scene; no shot declares it"))
    return out


def check_world(stubs: dict, scene: dict) -> list[tuple[str, str, str]]:
    """(shot_id, rule, detail) for one scene against the location stubs."""
    out = []
    nm = scene.get("narrative_meta") or {}
    dflt_loc = (((scene.get("shot_defaults") or {}).get("setup") or {})
                .get("backdrop") or {}).get("location")
    ref = nm.get("location_ref")
    if ref and dflt_loc and ref != dflt_loc:
        out.append(("-", "location_ref_drift",
                    f"narrative_meta says {ref!r}, backdrop says {dflt_loc!r}"))

    for shot in scene.get("shots") or []:
        sid = str(shot.get("shot_id"))
        b = ((resolve_shot(scene, shot).get("setup") or {}).get("backdrop")) or {}
        stub = stubs.get(b.get("location")) or {}
        if not stub:
            continue
        # A shot that CONTRADICTS its stub and one that never states the
        # field at all are different faults. The first is a wrong answer; the
        # second is a prompt that will carry no era at all, because a
        # compiler reads the shot's backdrop and inherits nothing from the
        # stub. Reporting both as "drift" hides which one you are looking at.
        absent = [k for k in _WORLD_FIELDS
                  if stub.get(k) is not None and not b.get(k)]
        differs = [k for k in _WORLD_FIELDS
                   if stub.get(k) is not None and b.get(k) and b.get(k) != stub.get(k)]
        if differs:
            out.append((sid, "world_drift",
                        f"{b.get('location')}: {', '.join(differs)}"))
        if absent:
            out.append((sid, "world_unresolved",
                        f"{b.get('location')}: {', '.join(absent)} not on the shot"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when anything is flagged")
    a = ap.parse_args()

    kb = (a.projects_root or PAI_PROJECTS_ROOT) / a.project / "kb"
    scenes = sorted((kb / "scenes").glob("scene_*.json"))
    if not scenes:
        print(f"no scenes for {a.project!r}", file=sys.stderr)
        return 1

    dfile, sfile = kb / "shot_design.json", kb / "location_stubs.json"
    pfile = kb / "props.json"
    props = json.loads(pfile.read_text(encoding="utf-8")) if pfile.exists() else {}
    props = props.get("props", props)
    design = json.loads(dfile.read_text(encoding="utf-8")) if dfile.exists() else {}
    stubs = (json.loads(sfile.read_text(encoding="utf-8")).get("stubs") or {}
             if sfile.exists() else {})
    if not design:
        print(f"note: no shot_design.json under {kb} — theme rules skipped",
              file=sys.stderr)

    by_rule: dict[str, int] = {}
    shots = flagged = 0
    for sp in scenes:
        doc = json.loads(sp.read_text(encoding="utf-8"))
        shots += len(doc.get("shots") or [])
        found = (check_theme(design, doc) + check_world(stubs, doc)
                 + check_props(props, doc))
        for sid, rule, detail in found:
            by_rule[rule] = by_rule.get(rule, 0) + 1
            flagged += 1
            print(f"  {doc.get('scene_id')}/{sid:8} {rule:20} {detail}")

    print(f"\n  {flagged} findings across {shots} shots in {len(scenes)} scenes")
    for r, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"    {r:20} {n}")
    return 1 if (a.strict and flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
