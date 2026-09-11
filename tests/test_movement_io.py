"""Regression tests for pipeline/movement_io.py — the canonical reader,
writer, planner adapter, and Wan motion compiler for pai-1.0 shot dicts.

Run: python3 pipeline/tests/test_movement_io.py
"""
from __future__ import annotations
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent

from pace_core.camera.movement_io import (
    read_movement, write_movement, shot_to_planner_dict,
    compile_wan_motion, compile_h3_motion, shot_to_narrative,
)


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def _shot_with_traj(*, movement_2d=None, movement_3d=None, gear=None,
                    easing="linear", static=False, position=None,
                    shot_size=None, angle=None,
                    focal_length_mm=None, t_stop=None) -> dict:
    """Construct a minimal pai-1.0 shot dict for tests."""
    cam: dict = {
        "trajectory": {
            "movement_2d": list(movement_2d or []),
            "movement_3d": list(movement_3d or []),
            "easing":      easing,
            "static":      static,
        },
        "extrinsics": {},
        "intrinsics": {},
        "creative_intent": {},
    }
    if gear is not None:
        cam["trajectory"]["gear"] = gear
    if position is not None:
        cam["extrinsics"]["position"] = position
    if angle is not None:
        cam["extrinsics"]["angle"] = angle
    if shot_size is not None:
        cam["creative_intent"]["shot_size"] = shot_size
    if focal_length_mm is not None:
        cam["intrinsics"]["focal_length_mm"] = focal_length_mm
    if t_stop is not None:
        cam["intrinsics"]["t_stop"] = t_stop
    return {"camera": cam}


# ── read_movement ─────────────────────────────────────────────────────


def test_read_3d_push_in(fails):
    """SCINE Movement3D `push_in` → planner Movement `push_in`."""
    m, e = read_movement(_shot_with_traj(movement_3d=["push_in"], easing="ease_out"))
    _expect(m == ["push_in"], f"movements: {m}", fails)
    _expect(e == "ease_out",  f"easing: {e}", fails)


def test_read_2d_pan_right_maps_to_pan_lr(fails):
    """SCINE Movement2D `pan_right` → planner `pan_lr` (left-to-right)."""
    m, _ = read_movement(_shot_with_traj(movement_2d=["pan_right"]))
    _expect(m == ["pan_lr"], f"movements: {m}", fails)


def test_read_arc_maps_to_orbit(fails):
    """SCINE Movement3D `arc` → planner `orbit`."""
    m, _ = read_movement(_shot_with_traj(movement_3d=["arc"]))
    _expect(m == ["orbit"], f"movements: {m}", fails)


def test_read_composite_3d_plus_2d(fails):
    """movement_3d + movement_2d compose into one flat list.
    `pai_compat.movement_of` walks movement_2d first, then movement_3d."""
    m, _ = read_movement(_shot_with_traj(movement_3d=["push_in"], movement_2d=["pan_right"]))
    _expect(set(m) == {"push_in", "pan_lr"}, f"composite (order-agnostic): {m}", fails)


def test_read_handheld_gear_appended(fails):
    """gear=handheld appended to the movement list."""
    m, _ = read_movement(_shot_with_traj(movement_3d=["push_in"], gear="handheld"))
    _expect("handheld" in m, f"handheld present: {m}", fails)


def test_read_behind_position_maps_to_from_behind(fails):
    """extrinsics.position=behind appends `from_behind`."""
    m, _ = read_movement(_shot_with_traj(position="behind"))
    _expect("from_behind" in m, f"from_behind present: {m}", fails)


def test_read_static_true(fails):
    """trajectory.static=true emits the planner `static` token."""
    m, _ = read_movement(_shot_with_traj(static=True))
    _expect(m == ["static"], f"static: {m}", fails)


def test_read_empty(fails):
    """Missing / empty → ([], 'linear')."""
    m, e = read_movement({})
    _expect(m == [] and e == "linear", f"empty: m={m} e={e}", fails)


def test_read_unknown_easing_falls_back_to_linear(fails):
    """Unknown easing → silently clamps to linear."""
    _, e = read_movement(_shot_with_traj(movement_3d=["push_in"], easing="bouncy"))
    _expect(e == "linear", f"easing should clamp: {e}", fails)


# ── write_movement ────────────────────────────────────────────────────


def test_write_creates_trajectory_block(fails):
    """write_movement creates camera.trajectory on an empty shot."""
    shot = {}
    write_movement(shot, ["push_in"], "ease_out")
    traj = shot["camera"]["trajectory"]
    _expect(traj["movement_3d"] == ["push_in"], f"movement_3d: {traj['movement_3d']}", fails)
    _expect(traj["easing"]      == "ease_out",  f"easing: {traj['easing']}", fails)
    _expect(traj["movement_2d"] == [],          f"movement_2d empty: {traj['movement_2d']}", fails)


def test_write_pan_lr_lands_in_2d(fails):
    """Planner `pan_lr` → SCINE Movement2D `pan_right`."""
    shot = {}
    write_movement(shot, ["pan_lr"], "linear")
    traj = shot["camera"]["trajectory"]
    _expect(traj["movement_2d"] == ["pan_right"], f"movement_2d: {traj['movement_2d']}", fails)
    _expect(traj["movement_3d"] == [],            f"movement_3d empty: {traj['movement_3d']}", fails)


def test_write_handheld_lands_in_gear(fails):
    """Planner `handheld` → SCINE Camera.trajectory.gear."""
    shot = {}
    write_movement(shot, ["handheld"], "linear")
    _expect(shot["camera"]["trajectory"].get("gear") == "handheld",
            f"gear: {shot['camera']['trajectory'].get('gear')}", fails)


def test_write_from_behind_lands_in_extrinsics_position(fails):
    """Planner `from_behind` → SCINE Camera.extrinsics.position=behind."""
    shot = {}
    write_movement(shot, ["from_behind"], "linear")
    _expect(shot["camera"]["extrinsics"].get("position") == "behind",
            f"position: {shot['camera']['extrinsics'].get('position')}", fails)


def test_write_clamps_easing(fails):
    """write_movement clamps unknown easing to linear."""
    shot = {}
    write_movement(shot, ["push_in"], "bouncy")
    _expect(shot["camera"]["trajectory"]["easing"] == "linear",
            f"easing: {shot['camera']['trajectory']['easing']}", fails)


def test_write_returns_shot_for_chaining(fails):
    """write_movement returns the same shot for chained calls."""
    shot = {}
    ret = write_movement(shot, ["push_in"])
    _expect(ret is shot, "write should return shot", fails)


# ── shot_to_planner_dict ──────────────────────────────────────────────


def test_planner_adapter_flattens_pillar_to_planner_camera(fails):
    """pai-1.0 puts shot_size/angle in camera.creative_intent/extrinsics;
    planner reads them flat. Adapter does the flattening.

    shot_size arrives as the schema's own ShotSize Literal. It used to be
    translated to a Chinese table key on the way through, which is why this
    once expected "特写" for a close_up."""
    shot = _shot_with_traj(
        movement_3d=["push_in"], easing="ease_out",
        shot_size="close_up", angle="low",
        focal_length_mm=35, t_stop=2.8,
    )
    p = shot_to_planner_dict(shot, scene_id="scene_01")
    _expect(p["camera"]["shot_size"] == "close_up", f"shot_size: {p['camera']['shot_size']}", fails)
    _expect(p["camera"]["angle"]     == "low",     f"angle: {p['camera']['angle']}", fails)
    _expect(p["camera"]["lens_mm"]   == 35,        f"lens_mm: {p['camera']['lens_mm']}", fails)
    _expect(p["frame"]["movement"]   == ["push_in"], f"movement: {p['frame']['movement']}", fails)
    _expect(p["frame"]["movement_easing"] == "ease_out", f"easing: {p['frame']['movement_easing']}", fails)
    _expect(p["scene_ref"] == "scene_01", f"scene_ref: {p['scene_ref']}", fails)


def test_planner_adapter_prefers_scene_location_ref(fails):
    """When a scene with narrative_meta.location_ref is passed, scene_ref
    pulls from there."""
    shot = _shot_with_traj()
    scene = {"narrative_meta": {"location_ref": "cafe_interior"}}
    p = shot_to_planner_dict(shot, scene=scene, scene_id="scene_99")
    _expect(p["scene_ref"] == "cafe_interior",
            f"scene_ref from narrative_meta: {p['scene_ref']}", fails)


# ── compile_wan_motion ────────────────────────────────────────────────


def test_wan_motion_empty(fails):
    _expect(compile_wan_motion([]) == "", "empty should be empty", fails)


def test_wan_motion_single(fails):
    s = compile_wan_motion(["push_in"])
    _expect("push-in" in s.lower(), f"single push_in: {s}", fails)


def test_wan_motion_composite(fails):
    s = compile_wan_motion(["push_in", "pan_lr"])
    _expect("combined with" in s, f"composite: {s}", fails)
    _expect("push-in" in s.lower() and "left to right" in s.lower(),
            f"both phrases: {s}", fails)


def test_wan_motion_easing_suffix(fails):
    s = compile_wan_motion(["push_in"], easing="ease_in_out")
    _expect("smoothly" in s or "easing" in s, f"ease_in_out suffix: {s}", fails)


def test_wan_motion_unknown_kind_dropped(fails):
    s = compile_wan_motion(["totally_invented_move", "push_in"])
    _expect("push-in" in s.lower(), f"push_in still present: {s}", fails)
    _expect("totally_invented_move" not in s, f"unknown dropped: {s}", fails)


# ── compile_h3_motion ────────────────────────────────────────────────

def test_h3_motion_empty(fails):
    _expect(compile_h3_motion([]) == "", "empty should be empty", fails)


def test_h3_motion_single_no_intensity_no_easing(fails):
    """No intensity/easing given → no amplitude/speed/ease clause at all,
    not a guessed default — H3's guide says omit for medium/normal."""
    s = compile_h3_motion(["push_in"])
    _expect(s == "The camera pushes in.", f"bare sentence: {s}", fails)


def test_h3_motion_dramatic_intensity_sets_amplitude_and_speed(fails):
    s = compile_h3_motion(["push_in"], intensity="dramatic")
    _expect("large amplitude" in s and "fast speed" in s, f"dramatic: {s}", fails)


def test_h3_motion_subtle_intensity_sets_amplitude_and_speed(fails):
    s = compile_h3_motion(["pull_out"], intensity="subtle")
    _expect("small amplitude" in s and "slow speed" in s, f"subtle: {s}", fails)


def test_h3_motion_multiple_movements_use_gerund_after_the_first(fails):
    """'pans right while tracks' is ungrammatical; the second-and-later
    verb must be gerund form."""
    s = compile_h3_motion(["pan_lr", "tracking"])
    _expect(s == "The camera pans right while tracking alongside the subject.",
           f"gerund join: {s}", fails)


def test_h3_motion_easing_clause_only_when_no_intensity(fails):
    """Amplitude+speed already states the pacing — stacking an easing
    clause on top would talk about speed twice, so intensity wins and
    easing is dropped when both are given."""
    with_ease = compile_h3_motion(["tilt_up"], easing="ease_in_out")
    _expect("eases in and out smoothly" in with_ease, f"easing alone: {with_ease}", fails)
    with_both = compile_h3_motion(["tilt_up"], easing="ease_in_out", intensity="dramatic")
    _expect("eases in and out" not in with_both,
           f"intensity suppresses easing clause: {with_both}", fails)


def test_h3_motion_static_dropped_when_a_real_movement_also_present(fails):
    """static + crane_up together is a contradiction some corpus
    shots actually carry on disk (stale data, not a real combined move)
    -- 'holds a static shot while pedestaling up' is nonsense regardless
    of which token is the leftover, so static loses to any real movement."""
    s = compile_h3_motion(["static", "crane_up"])
    _expect(s == "The camera pedestals up.", f"static dropped: {s}", fails)


def test_h3_motion_static_alone_still_produces_a_phrase(fails):
    s = compile_h3_motion(["static"])
    _expect(s == "The camera holds a static shot.", f"static alone: {s}", fails)


def test_h3_motion_unknown_kind_dropped(fails):
    s = compile_h3_motion(["totally_invented_move", "push_in"])
    _expect(s == "The camera pushes in.", f"unknown dropped: {s}", fails)


def test_h3_motion_uses_natural_sentence_not_bracket_tags(fails):
    """H3's own skill guide asks for camera motion as natural English
    action embedded in the shot, not the older Hailuo `[Pan left]`
    bracket-tag convention — this is the whole reason compile_h3_motion
    exists as a function distinct from compile_wan_motion's phrase style."""
    s = compile_h3_motion(["pan_lr"])
    _expect("[" not in s and "]" not in s, f"no bracket tags: {s}", fails)
    _expect(s.startswith("The camera "), f"natural sentence: {s}", fails)


def test_shot_to_narrative(fails):
    # full shot: action beat + framing context, en preferred over zh.
    shot = {
        "camera": {
            "creative_intent": {"shot_size": "extreme_close_up", "framing": "single"},
            "extrinsics": {"angle": "eye_level"},
            "trajectory": {},
        },
        "events": {"actions": [
            {"description_en": "lips quivering in a slow last-breath whisper",
             "description_zh": "嘴唇翕动", "intensity": "dramatic"},
        ]},
    }
    n = shot_to_narrative(shot)
    _expect("lips quivering" in n, f"beat (en) missing: {n}", fails)
    _expect("extreme_close_up" in n and "eye_level" in n, f"framing context missing: {n}", fails)
    _expect("dramatic intensity" in n, f"intensity missing: {n}", fails)
    _expect("嘴唇翕动" not in n, f"should prefer en over zh: {n}", fails)
    # zh fallback when no en
    shot2 = {"camera": {"creative_intent": {"shot_size": "medium_close_up"}, "extrinsics": {}, "trajectory": {}},
             "events": {"actions": [{"description_zh": "缓缓转身"}]}}
    _expect("缓缓转身" in shot_to_narrative(shot2), "zh fallback failed", fails)
    # empty shot → empty string (caller decides)
    _expect(shot_to_narrative({}) == "", "empty shot should give empty narrative", fails)
    # enrichment: scene setting (location/INT-EXT/time) + prior-shot continuity
    scene = {"narrative_meta": {"location_ref": "dying_chamber", "interior_exterior": "INT", "time_of_day": "night"}}
    prev = {"camera": {"creative_intent": {"shot_size": "wide"}}}
    ne = shot_to_narrative(shot, scene=scene, prev_shot=prev)
    _expect("dying_chamber" in ne and "night" in ne and "INT" in ne, f"scene setting missing: {ne}", fails)
    _expect("follows a wide shot" in ne, f"prior-shot continuity missing: {ne}", fails)
    _expect("lips quivering" in ne, "beat should still lead the enriched narrative", fails)

# The hand-rolled runner that used to close this file -- a TESTS list, a main()
# looping it, and in one case a wrapper generator -- is gone. It reimplemented
# what pytest already does: collection (these are plain test_* functions taking
# the `fails` fixture from conftest), fixture injection, and pass/fail
# reporting. Nothing referenced it, no doc or CI invoked the file directly, and
# the wrapper generator was outright dead -- it published 29 module attributes
# named pytest_test_*, which pytest does not collect, so every one of those
# wrappers had never run. Run these with: uv run pytest tests/
