"""Regression test for pipeline/camera_planner.py.

Asserts that:
  1. plan_camera() static path matches the expected geometry for representative
     shots, mirroring the math in blender_render._position_camera_for_shot.
  2. plan_camera() movement path emits start/end positions consistent with the
     declared movement kind.
  3. Lens, aperture, target_pos fields are present and well-formed.

Run: python3 pipeline/tests/test_camera_planner.py
"""
from __future__ import annotations
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent

from pace_core.camera.camera_planner import plan_camera  # noqa: E402


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def test_static_close_up_eye_level_cafe(fails: list[str]) -> None:
    """Cafe close-up: distance 0.7m, eye level, seated subject (1.2m)."""
    shot = {
        "id": "s1", "scene_ref": "cafe_interior",
        "camera": {"shot_size": "close_up", "angle": "eye_level",
                   "movement": "static", "lens_mm": 85, "aperture": "f/1.2"},
    }
    plan = plan_camera(shot, bible=None)
    _expect(_approx(plan["position"][0], 0.7), f"close_up cafe x: got {plan['position'][0]}", fails)
    _expect(_approx(plan["position"][2], 1.2), f"close_up cafe z (eye_level seated): got {plan['position'][2]}", fails)
    _expect(plan["lens_mm"] == 85, f"lens_mm passthrough: got {plan['lens_mm']}", fails)
    _expect(plan["aperture"] == "f/1.2", f"aperture passthrough: got {plan['aperture']}", fails)
    _expect("movement" not in plan, "static shot should have no movement block", fails)


def test_static_wide_low_angle_street(fails: list[str]) -> None:
    """Street wide low-angle: distance 5m, mid_body 0.9m, low_angle tilt -25 / h_off -0.5."""
    shot = {
        "id": "s2", "scene_ref": "street",
        "camera": {"shot_size": "wide", "angle": "low_angle", "movement": "static"},
    }
    plan = plan_camera(shot, bible=None)
    _expect(_approx(plan["position"][0], 5.0),  f"wide x: got {plan['position'][0]}", fails)
    _expect(_approx(plan["position"][2], 0.4),  f"wide z (mid_body 0.9 + h_off -0.5): got {plan['position'][2]}", fails)
    # 115, not 65. Blender's rx is 90 at level and DECREASES as the camera
    # tilts DOWN — measured with a subject standing at the origin and the
    # camera above it: 61% of the frame at rx=65, 1% at rx=90, 0% at rx=115.
    # This test's own camera sits at z=0.4, BELOW the subject's mid-body of
    # 0.9, so it has to tilt up to see them, which is rx > 90. The old
    # expectation of 65 restated the `90 + tilt` formula rather than the
    # geometry, and that formula aimed every high-angle shot at the sky.
    _expect(_approx(plan["rotation_deg"][0], 115.0), f"low_angle tilt -25 -> rot_x = 115: got {plan['rotation_deg'][0]}", fails)


def test_high_angle_looks_down_and_top_down_looks_down_hardest(fails: list[str]) -> None:
    """The other direction, which nothing pinned. A high angle puts the camera
    above the subject looking down; top_down is the extreme of the same move.
    Both must land below 90, and top_down below high_angle."""
    def rx(angle: str) -> float:
        return plan_camera({"id": "s", "scene_ref": "street",
                            "camera": {"shot_size": "wide", "angle": angle,
                                       "movement": "static"}}, bible=None)["rotation_deg"][0]

    _expect(rx("eye_level") == 90.0, f"eye_level should be level: got {rx('eye_level')}", fails)
    _expect(rx("high_angle") < 90.0, f"high_angle should look DOWN: got {rx('high_angle')}", fails)
    _expect(rx("low_angle") > 90.0, f"low_angle should look UP: got {rx('low_angle')}", fails)
    _expect(rx("top_down") < rx("high_angle"),
            f"top_down should look down harder than high_angle: "
            f"{rx('top_down')} vs {rx('high_angle')}", fails)
    _expect(0.0 <= rx("top_down") <= 30.0,
            f"top_down should be near straight down: got {rx('top_down')}", fails)


def test_lens_default(fails: list[str]) -> None:
    """No lens specified → default 50mm."""
    shot = {"id": "s3", "scene_ref": "street",
            "camera": {"shot_size": "medium", "angle": "eye_level"}}
    plan = plan_camera(shot, bible=None)
    _expect(plan["lens_mm"] == 50, f"default lens 50: got {plan['lens_mm']}", fails)


def test_movement_push_in(fails: list[str]) -> None:
    """push_in: end_position should be 0.8m closer than start (smaller X)."""
    shot = {"id": "s4", "scene_ref": "street",
            "camera": {"shot_size": "medium", "angle": "eye_level",
                       "movement": "push_in", "lens_mm": 50}}
    plan = plan_camera(shot, bible=None)
    _expect("movement" in plan, "push_in should produce movement block", fails)
    if "movement" in plan:
        m = plan["movement"]
        _expect(m["kind"] == "push_in", f"kind: {m['kind']}", fails)
        _expect(_approx(m["end_position"][0] - m["start_position"][0], -0.8),
                f"push_in delta: got {m['end_position'][0] - m['start_position'][0]}", fails)


def test_movement_zoom_in(fails: list[str]) -> None:
    """zoom_in: end_lens > start_lens (no position change)."""
    shot = {"id": "s5", "scene_ref": "street",
            "camera": {"shot_size": "medium", "angle": "eye_level",
                       "movement": "zoom_in", "lens_mm": 35}}
    plan = plan_camera(shot, bible=None)
    if "movement" not in plan:
        fails.append("zoom_in should produce movement block")
        return
    m = plan["movement"]
    _expect(m["start_lens_mm"] == 35, f"start_lens: {m['start_lens_mm']}", fails)
    _expect(m["end_lens_mm"] > m["start_lens_mm"], f"zoom_in lens grows: {m['start_lens_mm']} -> {m['end_lens_mm']}", fails)
    _expect(m["end_position"] == m["start_position"], "zoom_in should not move camera position", fails)


def test_unknown_movement_degrades_to_static(fails: list[str]) -> None:
    """Unknown movement label → degrade to static + warning."""
    shot = {"id": "s6", "scene_ref": "street",
            "camera": {"shot_size": "medium", "angle": "eye_level",
                       "movement": "rocket_launch"}}
    plan = plan_camera(shot, bible=None)
    _expect("movement" not in plan, "unknown movement should not produce movement block", fails)
    _expect("_movement_warning" in plan, "unknown movement should set warning", fails)


def test_bible_override(fails: list[str]) -> None:
    """If bible provides camera_planning.eye_level_m, planner uses it."""
    shot = {"id": "s7", "scene_ref": "alien_temple",
            "camera": {"shot_size": "close_up", "angle": "eye_level"}}
    bible = {"camera_planning": {"eye_level_m": 2.4, "mid_body_m": 1.5}}
    plan = plan_camera(shot, bible=bible)
    _expect(_approx(plan["position"][2], 2.4),
            f"bible override eye_level=2.4: got {plan['position'][2]}", fails)


def test_plan_cameras_single(fails: list[str]) -> None:
    """Legacy single-camera shot → list of one plan with cam_id='A'."""
    from pace_core.camera.camera_planner import plan_cameras
    shot = {"id": "s10", "scene_ref": "street",
            "camera": {"shot_size": "medium", "angle": "eye_level"}}
    plans = plan_cameras(shot, bible=None)
    _expect(len(plans) == 1, f"expected 1 plan, got {len(plans)}", fails)
    _expect(plans[0]["cam_id"] == "A", f"cam_id should default to 'A', got {plans[0].get('cam_id')}", fails)


def test_plan_cameras_multi(fails: list[str]) -> None:
    """Multi-camera shot → one plan per camera, ids preserved."""
    from pace_core.camera.camera_planner import plan_cameras
    shot = {
        "id": "s11", "scene_ref": "street",
        "cameras": [
            {"id": "wide",  "shot_size": "wide",     "angle": "eye_level", "lens_mm": 35},
            {"id": "close", "shot_size": "close_up", "angle": "eye_level", "lens_mm": 85},
            {                "shot_size": "medium",   "angle": "eye_level", "lens_mm": 50},  # no id
        ],
    }
    plans = plan_cameras(shot, bible=None)
    _expect(len(plans) == 3, f"expected 3 plans, got {len(plans)}", fails)
    _expect(plans[0]["cam_id"] == "wide",  f"first cam_id 'wide', got {plans[0]['cam_id']}", fails)
    _expect(plans[1]["cam_id"] == "close", f"second cam_id 'close', got {plans[1]['cam_id']}", fails)
    _expect(plans[2]["cam_id"] == "C",     f"third cam_id auto-default 'C', got {plans[2]['cam_id']}", fails)
    _expect(plans[0]["lens_mm"] == 35 and plans[1]["lens_mm"] == 85 and plans[2]["lens_mm"] == 50,
            "lens_mm values should match per-camera", fails)


# Schema-vocab coverage tests removed 2026-05-23 along with the
# legacy pai_lang.types module. The planner's distance/angle/movement
# tables are keyed on its own internal vocab (Chinese ShotSize labels,
# planner Movement Literals) which bridges to SCINE English via
# pai_compat reverse maps — not a 1:1 invariant.


# ─── camera movement MVP — composite + easing + N-frame track ────────


def test_movement_via_frame_path(fails: list[str]) -> None:
    """PAI 0.3 path: shot.frame.movement = ['push_in'] should reach the planner."""
    shot = {
        "scene_ref": "cafe_interior",
        "camera":    {"shot_size": "medium", "angle": "eye_level"},
        "frame":     {"movement": ["push_in"]},
    }
    plan = plan_camera(shot)
    _expect("movement" in plan, "frame.movement should produce movement block", fails)
    m = plan["movement"]
    _expect(m["kinds"] == ["push_in"], f"kinds: {m.get('kinds')}", fails)
    _expect(m["kind"] == "push_in", f"kind (back-compat): {m.get('kind')}", fails)


def test_movement_composite_pan_plus_zoom(fails: list[str]) -> None:
    """Pan+zoom: both deltas should compose into the end keyframe."""
    shot = {
        "scene_ref": "cafe_interior",
        "camera":    {"shot_size": "medium", "angle": "eye_level", "lens_mm": 50},
        "frame":     {"movement": ["pan_left", "zoom_in"]},
    }
    plan = plan_camera(shot)
    m = plan["movement"]
    # pan_left: yaw_delta_deg = -10  → rotation_deg[2] = 90 - 10 = 80
    _expect(_approx(m["end_rotation_deg"][2], 80.0),
            f"end yaw: {m['end_rotation_deg'][2]}", fails)
    # zoom_in: lens_zoom_pct = 40  → 50 * 1.4 = 70mm
    _expect(m["end_lens_mm"] == 70, f"end lens: {m['end_lens_mm']}", fails)
    _expect(m["kinds"] == ["pan_left", "zoom_in"],
            f"kinds: {m.get('kinds')}", fails)


def test_movement_easing_field_default(fails: list[str]) -> None:
    """Movement block carries an `easing` field — defaults to linear."""
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"},
            "frame":  {"movement": ["push_in"]}}
    plan = plan_camera(shot)
    _expect(plan["movement"].get("easing") == "linear",
            f"default easing: {plan['movement'].get('easing')}", fails)


def test_movement_easing_explicit(fails: list[str]) -> None:
    """frame.movement_easing = ease_out should propagate into the plan."""
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"},
            "frame":  {"movement": ["push_in"], "movement_easing": "ease_out"}}
    plan = plan_camera(shot)
    _expect(plan["movement"].get("easing") == "ease_out",
            f"explicit easing: {plan['movement'].get('easing')}", fails)


def test_stylistic_only_handheld(fails: list[str]) -> None:
    """Stylistic-only movements (handheld) produce zero geometric delta but
    DO record the movement so prompts can pick it up."""
    from pace_core.camera.camera_planner import plan_camera
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"},
            "frame":  {"movement": ["handheld"]}}
    plan = plan_camera(shot)
    m = plan.get("movement")
    _expect(m is not None, "handheld should produce a movement block", fails)
    _expect(m and m.get("_stylistic_only") is True,
            "handheld should be flagged stylistic_only", fails)
    _expect(m and m["start_position"] == m["end_position"],
            f"handheld start==end: {m and (m['start_position'], m['end_position'])}", fails)


def test_plan_camera_track_static(fails: list[str]) -> None:
    """plan_camera_track on a static shot returns N identical keyframes."""
    from pace_core.camera.camera_planner import plan_camera_track
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"}}
    track = plan_camera_track(shot, n_frames=8)
    _expect(len(track) == 8, f"track length: {len(track)}", fails)
    first = (track[0]["position"], track[0]["lens_mm"])
    last  = (track[-1]["position"], track[-1]["lens_mm"])
    _expect(first == last, f"static track should not change: {first} != {last}", fails)
    _expect(track[0]["frame_index"] == 0 and track[-1]["frame_index"] == 7,
            "frame_index should span 0..N-1", fails)


def test_plan_camera_track_linear_push_in(fails: list[str]) -> None:
    """Linear easing: a push_in track linearly interpolates position[0]."""
    from pace_core.camera.camera_planner import plan_camera_track
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level", "lens_mm": 50},
            "frame":  {"movement": ["push_in"]}}
    track = plan_camera_track(shot, n_frames=5, easing="linear")
    _expect(len(track) == 5, f"len: {len(track)}", fails)
    # push_in: distance_delta_m = -0.8, start_x = 2.0 (medium)
    # midpoint should be at 2.0 + (-0.8)*0.5 = 1.6
    _expect(_approx(track[2]["position"][0], 1.6, tol=1e-3),
            f"midframe x: {track[2]['position'][0]}", fails)
    # endpoint should be at 1.2
    _expect(_approx(track[-1]["position"][0], 1.2, tol=1e-3),
            f"endframe x: {track[-1]['position'][0]}", fails)


def test_plan_camera_track_ease_out_shifts_midpoint(fails: list[str]) -> None:
    """ease_out: fast start, slow end → midframe further along than linear."""
    from pace_core.camera.camera_planner import plan_camera_track
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"},
            "frame":  {"movement": ["push_in"]}}
    linear = plan_camera_track(shot, n_frames=5, easing="linear")
    eased  = plan_camera_track(shot, n_frames=5, easing="ease_out")
    # ease_out at t=0.5 evaluates to 0.75 (1 - 0.25)
    # linear x at midpoint = 1.6, ease_out x at midpoint = 2.0 + (-0.8)*0.75 = 1.4
    _expect(eased[2]["position"][0] < linear[2]["position"][0] - 0.05,
            f"ease_out midpoint should be further than linear: "
            f"linear={linear[2]['position'][0]}, ease_out={eased[2]['position'][0]}",
            fails)
    # Endpoints must match regardless of easing
    _expect(_approx(linear[-1]["position"][0], eased[-1]["position"][0]),
            f"endpoints differ: linear={linear[-1]['position'][0]}, ease_out={eased[-1]['position'][0]}",
            fails)


def test_legacy_string_movement_still_works(fails: list[str]) -> None:
    """Old-style shot.camera.movement = 'push_in' (string) still produces a plan."""
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level",
                       "movement": "push_in"}}
    plan = plan_camera(shot)
    _expect("movement" in plan, "legacy string path should still work", fails)
    _expect(plan["movement"]["kind"] == "push_in",
            f"legacy kind: {plan['movement']['kind']}", fails)


def test_composition_absent_is_unchanged(fails: list[str]) -> None:
    """No shot.composition key -> identical plan to before this field existed."""
    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"}}
    plan = plan_camera(shot)
    _expect("composition_solved" not in plan,
            "no composition key should mean no solve attempted", fails)
    _expect(_approx(plan["rotation_deg"][2], 90.0),
            f"default dead-center yaw unchanged: got {plan['rotation_deg'][2]}", fails)


def test_composition_off_center_target_solves_and_converges(fails: list[str]) -> None:
    """shot.composition present -> rotation_deg is adjusted, and the solve
    actually converges (round-tripped via composition_solver.project_to_screen).

    This shot is loose enough (medium at 2.0 m, cafe anchors 0.7/1.2 m ->
    0.5 m span inside a 0.81 m frame) that the planner picks the
    silhouette-midpoint anchor, so the round-trip check re-projects both
    ends of composition_solved's subject_span_z and averages them —
    matching what the solver actually optimized for — rather than
    re-projecting plan["target_pos"] (their 3D midpoint) alone, which is
    not, in general, the same point under a nonlinear pinhole projection."""
    from pace_core.setup.composition_solver import project_to_screen

    shot = {"scene_ref": "cafe_interior",
            "camera": {"shot_size": "medium", "angle": "eye_level"},
            "composition": {"subject_xy": [0.0, 0.4],
                             "target_x": 1 / 3, "target_y": 1 / 3}}
    plan = plan_camera(shot)
    solved = plan.get("composition_solved")
    _expect(solved is not None, "composition_solved should be present", fails)
    _expect(bool(solved and solved["converged"]),
            f"solve should converge: {solved}", fails)
    _expect(bool(solved and solved["anchor"] == "silhouette_midpoint"),
            f"loose framing should use the two-point anchor: {solved}", fails)
    _expect(plan["rotation_deg"][2] != 90.0,
            "yaw should move off the dead-center default", fails)

    z_foot, z_head = solved["subject_span_z"]
    got_foot = project_to_screen(plan["position"], plan["rotation_deg"],
                                  plan["lens_mm"], [0.0, 0.4, z_foot])
    got_head = project_to_screen(plan["position"], plan["rotation_deg"],
                                  plan["lens_mm"], [0.0, 0.4, z_head])
    _expect(got_foot is not None and got_head is not None,
            "solved rotation should keep the subject's full span in front of camera", fails)
    if got_foot is not None and got_head is not None:
        mid_x = (got_foot[0] + got_head[0]) / 2.0
        mid_y = (got_foot[1] + got_head[1]) / 2.0
        _expect(_approx(mid_x, 1 / 3, tol=1e-3) and _approx(mid_y, 1 / 3, tol=1e-3),
                f"rendered foot/head midpoint should match target: got ({mid_x}, {mid_y})", fails)


def test_composition_tight_framing_keeps_single_anchor(fails: list[str]) -> None:
    """A framing too tight to show the whole eye_level/mid_body span must
    keep the single-height aim, and put THAT point on the target.

    Measured on scene_08's real panels (medium_close_up, 1.2 m): the
    two-point silhouette-midpoint objective is satisfiable with both
    endpoints off-screen in opposite directions, so applying it here tilts
    the camera down ~15 degrees and fills the frame with the subject's
    torso, head out of frame. It scores ~1-3% on a mask-centroid metric
    and is a worse shot — the metric, not the aim, is what breaks down at
    this magnification. Measured once by a script since removed; the
    figures are in the PACE paper's composition section."""
    from pace_core.setup.composition_solver import project_to_screen

    shot = {"scene_ref": "street",   # 1.65 / 0.9 anchors -> 0.75 m span
            "camera": {"shot_size": "medium_close_up", "angle": "eye_level"},
            "composition": {"subject_xy": [0.0, 0.5],
                             "target_x": 0.5, "target_y": 0.5}}
    plan = plan_camera(shot)
    solved = plan.get("composition_solved")
    _expect(bool(solved and solved["anchor"] == "single_height"),
            f"tight framing should keep the single anchor: {solved}", fails)
    _expect(bool(solved and solved["converged"]), f"should converge: {solved}", fails)
    # 0.75 m span against a 1.2 m * 2 * (18/50)/(16/9) = 0.486 m frame.
    _expect(bool(solved and solved["frame_height_m"] < 0.75),
            f"frame should be shorter than the span: {solved}", fails)
    got = project_to_screen(plan["position"], plan["rotation_deg"],
                            plan["lens_mm"], plan["target_pos"])
    _expect(got is not None and _approx(got[0], 0.5, tol=1e-3)
            and _approx(got[1], 0.5, tol=1e-3),
            f"single-anchor aim point should land on target: got {got}", fails)


def test_composition_extreme_close_up_still_converges(fails: list[str]) -> None:
    """The tightest framing in the shot-size table must still produce a
    converged solve — an unconverged rotation is applied anyway today, so
    silent non-convergence would ship a miscomposed frame."""
    shot = {"scene_ref": "street",
            "camera": {"shot_size": "extreme_close_up", "angle": "eye_level"},
            "composition": {"subject_xy": [0.0, 0.5],
                             "target_x": 0.5, "target_y": 0.5}}
    solved = plan_camera(shot).get("composition_solved")
    _expect(bool(solved and solved["converged"]),
            f"extreme_close_up solve should converge: {solved}", fails)

# The hand-rolled runner that used to close this file -- a TESTS list, a main()
# looping it, and in one case a wrapper generator -- is gone. It reimplemented
# what pytest already does: collection (these are plain test_* functions taking
# the `fails` fixture from conftest), fixture injection, and pass/fail
# reporting. Nothing referenced it, no doc or CI invoked the file directly, and
# the wrapper generator was outright dead -- it published 29 module attributes
# named pytest_test_*, which pytest does not collect, so every one of those
# wrappers had never run. Run these with: uv run pytest tests/
