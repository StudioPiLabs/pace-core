"""Give a moving shot the second panel it structurally needs.

`split_script.py` emits exactly one panel per shot. That is correct for a held
frame and malformed for a moving one: a panel carries the composition target
(`screen_position`), and a camera that translates has two different framings --
where it starts and where it ends. With one panel the end framing is simply
unstated, so nothing downstream can render it, and the composition measurement
has no second target to check against.

What counts as "moving" is the schema's own vocabulary, not a guess. Movement2D
(pan/tilt/zoom) and Movement3D (push_in, tracking, crane, ...) change framing.
Gear does not: `handheld` and `steadicam` describe how the rig behaves, not
where the frame goes. This matters in practice -- AutomaticDrive stores
`handheld` inside `movement_3d`, where it is not a legal value, and 16 of its
27 shots therefore look like moving shots while holding frame.

The end panel is a clone of the start, which encodes a *reframing* move: the
subject stays at the same screen position while the camera travels, so the
frame follows the subject. The alternative (a revealing move, where the subject
drifts across frame as the camera goes) is a different shot and would need its
own end position authored; it is deliberately not guessed at here.

A clone alone states the START framing twice, which left the end framing as
unstated as it was with one panel. On the video path that went unnoticed,
because `plan_camera_track` interpolates between the pair; on the still path
each panel renders on its own, and the two came back as one image — same
prompt, same greybox, nothing to tell them apart. So a move that changes how
much of the subject is in frame also writes the size it arrives at, one step
along the ShotSize ladder, as the end panel's `camera_override`.

That is still a composition target rather than a camera transform: it says
where the frame ends up, not how the camera gets there. The transform between
the two remains `plan_camera_track`'s job. A pan or a crane reframes without
changing size and gets no step, because for those the clone already states the
end framing correctly.

    uv run python -m pace_core.breakdown.densify_panels --project <slug>          # dry run
    uv run python -m pace_core.breakdown.densify_panels --project <slug> --apply
"""
from __future__ import annotations

import copy
import typing

from pace_core.pai_compat import dig, movement_of
from pace_core.types_v1 import Movement2D, Movement3D, ShotSize

# Every movement literal that changes what is in frame, in both the current
# spelling and the v1.0 spellings the scene documents on disk actually carry
# (`crane` is the literal; the data says `crane_up`).
FRAMING_MOVES: set[str] = (
    set(typing.get_args(Movement2D)) | set(typing.get_args(Movement3D)) |
    {"crane_up", "crane_down", "pedestal_up", "pedestal_down",
     "dolly_in", "dolly_out", "zoom_in", "zoom_out"} |
    # …and the forms movement_of() actually emits. It canonicalises the
    # schema's pan_left/pan_right into the planner's pan_rl/pan_lr and can
    # emit orbit, none of which are Movement2D/Movement3D literals — so this
    # set, built from the schema, could never match a pan and a panning shot
    # silently never got its second panel. No shot in either corpus declares
    # one today, so this is a latent gap rather than a live defect, but it
    # would bite the first pan anyone authors.
    {"pan_lr", "pan_rl", "orbit"}
)
# Present in the same lists but not framing changes: rig character, framing
# relationships, and the explicit no-move token.
NOT_A_MOVE: set[str] = {"static", "handheld", "steadicam", "from_behind",
                        "reverse_shot", "tripod"}


def framing_moves(shot: dict) -> list[str]:
    """The moves in this shot that change the frame, canonicalised."""
    out = []
    for m in movement_of(shot) or []:
        if m in NOT_A_MOVE:
            continue
        if m in FRAMING_MOVES and m not in out:
            out.append(m)
    return out


# Moves that change how much of the subject is in frame, and which way along
# the ShotSize ladder they travel. A pan or a crane reframes without changing
# size, so it gets no step: the clone already states that end framing, which is
# the same size at the same screen position.
_TIGHTENS = {"push_in", "push_in_slow", "dolly_in", "zoom_in"}
_LOOSENS  = {"pull_out", "dolly_out", "zoom_out"}

# Widest to tightest, taken from the schema rather than restated. It is the
# order the Literal declares and the order the greybox's own framing bands use,
# so a step here is a step there.
_SIZE_LADDER: tuple[str, ...] = typing.get_args(ShotSize)


def end_shot_size(start: str | None, moves: list[str]) -> str | None:
    """The size the frame arrives at, or None when the move doesn't change it.

    One step per shot, not one per move: a shot that declares both `push_in`
    and `dolly_in` is describing one movement two ways, and stepping twice
    would turn a medium into a close-up on a vocabulary artefact. A move that
    would step off either end of the ladder stays where it is — a push-in on an
    extreme close-up ends on an extreme close-up, and saying so is better than
    inventing a tighter size the schema does not have.
    """
    if not start or start not in _SIZE_LADDER:
        return None
    tighten = any(m in _TIGHTENS for m in moves)
    loosen  = any(m in _LOOSENS for m in moves)
    if tighten == loosen:          # neither, or a shot that declares both
        return None
    i = _SIZE_LADDER.index(start)
    j = min(i + 1, len(_SIZE_LADDER) - 1) if tighten else max(i - 1, 0)
    return _SIZE_LADDER[j] if j != i else None


# A crane or pedestal changes the camera's height, not how much of the subject
# is in frame, so its end framing differs in ANGLE rather than in size. Only
# the rungs a vertical move actually travels are listed: `dutch` is a roll,
# `continuous` is not a position, and the body-part rungs (shoulder/hip/knee)
# describe a subject-relative height this move does not address.
_ANGLE_LADDER: tuple[str, ...] = ("ground", "low", "eye_level", "high",
                                  "overhead", "aerial")
# The documents on disk spell two of these rungs with a suffix the Angle
# Literal does not have: 5 shots say `low_angle` and 4 say `high_angle` across
# this corpus. Read them rather than fail to recognise the angle a shot plainly
# declares; the schema spelling is what gets written back.
_ANGLE_ALIASES = {"low_angle": "low", "high_angle": "high",
                  "eye": "eye_level", "eyelevel": "eye_level"}
_RISES  = {"crane_up", "pedestal_up", "tilt_down"}
_FALLS  = {"crane_down", "pedestal_down", "tilt_up"}


def end_angle(start: str | None, moves: list[str]) -> str | None:
    """The angle the frame arrives at, or None when the move doesn't change it.

    A camera that rises ends up looking further down, so `crane_up` steps
    toward `high`; a tilt down does the same thing from a fixed height, which
    is why it sits with the rises. Same one-step-per-shot rule and same
    refusal to run off the ladder as `end_shot_size`.
    """
    start = _ANGLE_ALIASES.get(start or "", start) or "eye_level"
    if start not in _ANGLE_LADDER:
        return None                    # dutch, continuous, or a body-part rung
    rise = any(m in _RISES for m in moves)
    fall = any(m in _FALLS for m in moves)
    if rise == fall:
        return None
    i = _ANGLE_LADDER.index(start)
    j = min(i + 1, len(_ANGLE_LADDER) - 1) if rise else max(i - 1, 0)
    return _ANGLE_LADDER[j] if j != i else None


def _next_panel_n(scene_doc: dict) -> int:
    """Panel numbering is scene-wide, not per shot -- scene_04's shot_02 holds
    panel_0002. Advance the scene counter so a new panel cannot collide with
    one in a sibling shot."""
    known = [int(str(p.get("id", "")).rsplit("_", 1)[-1] or 0)
             for s in scene_doc.get("shots") or []
             for p in s.get("panels") or []
             if str(p.get("id", "")).rsplit("_", 1)[-1].isdigit()]
    return max([int(scene_doc.get("_max_panel_n") or 0)] + known + [0]) + 1


def densify_scene(scene_doc: dict, *, apply: bool = False) -> dict:
    """Add the missing end panel to every moving shot in one scene."""
    scene_id = scene_doc.get("scene_id") or "scene"
    added, skipped = [], []
    # Allocated within this run, not read back from the document: on a dry run
    # nothing is appended, so re-deriving the counter per shot handed every
    # shot in a scene the same panel number and reported colliding ids.
    next_n = _next_panel_n(scene_doc)
    for shot in scene_doc.get("shots") or []:
        moves = framing_moves(shot)
        panels = shot.get("panels") or []
        if not moves:
            skipped.append({"shot": shot.get("shot_id"), "why": "frame is held"})
            continue
        # A shot that declares `static: true` says its frame is held. When it
        # also lists a movement, the two statements contradict each other and
        # this module is not the place to decide which one the director meant:
        # the panel it would add is a second framing for a shot that says it
        # has only one.
        #
        # That is not hypothetical. Every 2-panel shot in AutomaticDrive came
        # from exactly this pair of statements — 4 shots declaring static
        # alongside a crane, and 0 shots genuinely moving — and their panels
        # rendered identically because there was no second framing to state.
        # Reported, not resolved, for the same reason the beat checker reports:
        # which half is wrong is an authoring decision.
        if (dig(shot, "camera", "trajectory", "static") or False):
            skipped.append({"shot": shot.get("shot_id"),
                            "why": f"declares static AND {'+'.join(moves)} — "
                                   f"contradiction, not densified"})
            continue
        if len(panels) >= 2:
            skipped.append({"shot": shot.get("shot_id"),
                            "why": f"already has {len(panels)} panels"})
            continue
        if not panels:
            skipped.append({"shot": shot.get("shot_id"), "why": "no panel to clone"})
            continue

        n, next_n = next_n, next_n + 1
        end = copy.deepcopy(panels[-1])
        end["id"] = f"{scene_id}_{shot.get('shot_id')}_panel_{n:04d}"
        end["panel_number"] = len(panels) + 1
        # Reframing move: screen_position is carried over unchanged, which is
        # what the clone already does. The note is what tells a reader (and the
        # storyboard HTML) that this is the end of a move rather than a beat.
        end["notes"] = f"end framing of {'+'.join(moves)}"
        # …and say it in a field, not only in prose. A clone states the START
        # framing twice, so the end framing stayed unstated — which is the gap
        # this module exists to close. The pair rendered as one image: same
        # prompt, same greybox, nothing to tell them apart.
        #
        # This states a composition target, not a camera transform. The
        # transform between the two remains plan_camera_track's job.
        start_size = dig(shot, "camera", "creative_intent", "shot_size")
        start_angle = dig(shot, "camera", "extrinsics", "angle")
        end_size = end_shot_size(start_size, moves)
        end_ang = end_angle(start_angle, moves)
        override: dict = {}
        if end_size:
            override["creative_intent"] = {"shot_size": end_size}
        if end_ang:
            override["extrinsics"] = {"angle": end_ang}
        if override:
            end["camera_override"] = override
        record = {"shot": shot.get("shot_id"), "moves": moves,
                  "start_panel": panels[-1].get("id"), "end_panel": end["id"],
                  "ends_at": (f"size {start_size} -> {end_size}" if end_size else "")
                             + (f" angle {start_angle or 'eye_level'} -> {end_ang}"
                                if end_ang else "")
                             or "no framing change this move can state"}
        added.append(record)
        if apply:
            panels.append(end)
            shot["panels"] = panels
            scene_doc["_max_panel_n"] = n
            # Every panel owns a compile-hints slot; a panel without one is
            # invisible to the per-backend prompt overrides.
            hints = scene_doc.setdefault("compile_hints", [])
            if not any(h.get("panel_id") == end["id"] for h in hints):
                hints.append({"panel_id": end["id"], "flux": {},
                              "gpt_image_2": {}, "wan_i2v": {}})
    return {"scene_id": scene_id, "added": added, "skipped": skipped}


# No LLM beat pass lives here, and that is deliberate. split_script.py already
# runs a model over the screenplay and extracts its beats -- in AutomaticDrive,
# 27 key_actions became 27 shots, exactly 1:1 in every scene. A second pass
# asking a model for "more visual beats" either restates what the first already
# captured or invents action the screenplay does not contain. If beat density
# is ever genuinely too coarse, the fix belongs in the splitter, not in a
# parallel implementation here.
#
# What was actually missing needed no model: a moving shot has two framings and
# was given one panel. That is derivable from movement_3d, above.


def main() -> int:
    """Walk a project's scenes and report (or write) the end panels they need.

    Dry run by default: this rewrites authored KB documents, and it keeps a
    `.json.pre-densify.bak` beside each one it changes. The suffix names the
    change rather than being a plain `.bak`, because a plain one is overwritten
    by the next tool to touch the same scene and a two-step repair then has no
    first rollback point.
    """
    import argparse
    import json
    import shutil
    import sys
    from pathlib import Path
    from pace_core.paths import PAI_PROJECTS_ROOT

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", default=None, help="limit to one scene id")
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    root = a.projects_root or PAI_PROJECTS_ROOT
    scenes = sorted((Path(root) / a.project / "kb" / "scenes").glob("*.json"))
    if a.scene:
        scenes = [p for p in scenes if p.stem == a.scene]
    if not scenes:
        print(f"no scenes under {Path(root) / a.project}", file=sys.stderr)
        return 1

    added = held = 0
    for sp in scenes:
        doc = json.loads(sp.read_text(encoding="utf-8"))
        r = densify_scene(doc, apply=a.apply)
        held += sum(1 for s in r["skipped"] if s["why"] == "frame is held")
        for rec in r["added"]:
            added += 1
            print(("  " if a.apply else "  [dry] ")
                  + f"{r['scene_id']}/{rec['shot']:8} {'+'.join(rec['moves']):12} "
                    f"{rec['start_panel']} -> {rec['end_panel']}")
        if a.apply and r["added"]:
            shutil.copy2(sp, sp.with_suffix(".json.pre-densify.bak"))
            sp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{added} end panel(s) {'added' if a.apply else 'would be added'}; "
          f"{held} shot(s) hold frame and correctly stay at one panel"
          + ("" if a.apply else "  — re-run with --apply"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
