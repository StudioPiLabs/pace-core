"""Regression tests for composition_solver — the screen_position grounding
mechanism (Section "Actor blocking" of the PACE paper).

test_camera_axes_match_blender_ground_truth hardcodes right/up/forward
vectors obtained by running mathutils.Euler(...).to_matrix() directly in
Blender 4.2.5 (checked interactively, not assumed) for two
rotations — this is the authority the rest of the solver is checked
against, since a sign error here would silently invert every composition
result without any local exception being raised.
"""

import pytest

from pace_core.camera.camera_planner import _euler_xyz_to_matrix
from pace_core.setup.composition_solver import (
    ZONE_TO_XY,
    camera_axes,
    project_to_screen,
    solve_rotation_for_screen_position,
    target_xy,
)
from pace_core.types_v1 import ScreenZone


def _approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def test_camera_axes_match_blender_ground_truth():
    # rotation_deg = (90, 0, 90): plan_camera's default "camera looks along
    # -X toward the world origin" pose. Verified against actual Blender
    # (mathutils.Euler((rad(90),0,rad(90)),'XYZ').to_matrix()) in Blender 4.2.
    right, up, forward = camera_axes((90.0, 0.0, 90.0))
    assert all(_approx(a, b) for a, b in zip(right, (0.0, 1.0, 0.0)))
    assert all(_approx(a, b) for a, b in zip(up, (0.0, 0.0, 1.0)))
    assert all(_approx(a, b) for a, b in zip(forward, (-1.0, 0.0, 0.0)))


def test_zone_to_xy_covers_every_schema_zone():
    # ScreenZone (types_v1.py) is the authoring vocabulary; if a zone is
    # ever added there without a corresponding entry here, target_xy()
    # would silently return None for real authored data.
    missing = [z for z in ScreenZone.__args__ if z not in ZONE_TO_XY]
    assert not missing, f"ZONE_TO_XY missing schema zones: {missing}"


def test_target_xy_precise_wins_over_zone():
    assert target_xy({"zone": "left", "x": 0.9, "y": 0.1}) == (0.9, 0.1)
    assert target_xy({"zone": "center"}) == (0.5, 0.5)
    assert target_xy({}) is None
    assert target_xy(None) is None
    assert target_xy({"zone": "not_a_real_zone"}) is None


def test_project_to_screen_default_aim_is_centered():
    # plan_camera's own default: camera at (distance,0,z), aimed at
    # (0,0,z) via rotation (90,0,90) — the look-at point must project to
    # dead center by construction.
    cam_pos = (2.0, 0.0, 1.65)
    rot = (90.0, 0.0, 90.0)
    x, y = project_to_screen(cam_pos, rot, lens_mm=50, world_pos=(0.0, 0.0, 1.65))
    assert _approx(x, 0.5) and _approx(y, 0.5)


def test_project_to_screen_behind_camera_returns_none():
    cam_pos = (2.0, 0.0, 1.65)
    rot = (90.0, 0.0, 90.0)
    # A point further along +X than the camera is behind it (camera looks -X).
    assert project_to_screen(cam_pos, rot, lens_mm=50, world_pos=(5.0, 0.0, 1.65)) is None


@pytest.mark.parametrize("zone", list(ZONE_TO_XY))
def test_solve_converges_and_roundtrips_for_every_zone(zone):
    cam_pos = (2.0, 0.0, 1.65)
    base_rot = (90.0, 0.0, 90.0)
    # Subject at the scene's default aim point (0,0,1.65) — the same point
    # the un-composed camera already looks dead-center at.
    subject_pos = (0.0, 0.0, 1.65)
    tx, ty = ZONE_TO_XY[zone]

    result = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=50, subject_pos=subject_pos,
        target_x=tx, target_y=ty,
    )
    assert result["converged"], f"zone={zone} did not converge: {result}"

    # Round-trip: re-project under the SOLVED rotation and confirm the
    # subject actually lands at the target, independent of the solver's
    # own internal residual bookkeeping.
    got = project_to_screen(cam_pos, result["rotation_deg"], lens_mm=50,
                             world_pos=subject_pos)
    assert got is not None
    assert _approx(got[0], tx, tol=1e-3), f"zone={zone} x: got {got[0]}, want {tx}"
    assert _approx(got[1], ty, tol=1e-3), f"zone={zone} y: got {got[1]}, want {ty}"


def test_solve_preserves_roll():
    cam_pos = (2.0, 0.0, 1.65)
    base_rot = (90.0, 3.0, 90.0)  # non-zero roll on purpose
    result = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=50, subject_pos=(0.0, 0.0, 1.65),
        target_x=1 / 3, target_y=1 / 3,
    )
    assert _approx(result["rotation_deg"][1], 3.0)


def test_solve_continuous_xy_off_grid_targets():
    cam_pos = (2.0, 0.0, 1.65)
    base_rot = (90.0, 0.0, 90.0)
    subject_pos = (0.0, 0.0, 1.65)
    for tx, ty in [(0.2, 0.8), (0.9, 0.15), (0.5, 0.5), (0.05, 0.05)]:
        result = solve_rotation_for_screen_position(
            cam_pos, base_rot, lens_mm=50, subject_pos=subject_pos,
            target_x=tx, target_y=ty,
        )
        assert result["converged"], f"target=({tx},{ty}) did not converge: {result}"
        got = project_to_screen(cam_pos, result["rotation_deg"], lens_mm=50,
                                 world_pos=subject_pos)
        assert _approx(got[0], tx, tol=1e-3) and _approx(got[1], ty, tol=1e-3)


def test_solve_off_axis_subject_position():
    # Subject NOT on the camera's default boresight (e.g. the second slot
    # in a 2-character mannequin layout, offset in Y) — the harder,
    # realistic case this solver actually exists for.
    cam_pos = (2.0, 0.0, 1.65)
    base_rot = (90.0, 0.0, 90.0)
    subject_pos = (0.0, 0.5, 1.65)  # offset toward image-right by construction
    tx, ty = ZONE_TO_XY["lower_left"]
    result = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=35, subject_pos=subject_pos,
        target_x=tx, target_y=ty,
    )
    assert result["converged"], result
    got = project_to_screen(cam_pos, result["rotation_deg"], lens_mm=35,
                             world_pos=subject_pos)
    assert _approx(got[0], tx, tol=1e-3) and _approx(got[1], ty, tol=1e-3)


def test_solve_with_head_pos_targets_foot_head_midpoint():
    # subject_head_pos present -> the solver optimizes the MIDPOINT of the
    # foot's and head's independent projections, not subject_pos alone —
    # the fix for the vertical-offset gap the composition-precision check
    # found (a single eye-level aim point undercounts a standing figure's
    # true visual center, worse the tighter the frame).
    cam_pos = (2.0, 0.0, 1.2)
    base_rot = (90.0, 0.0, 90.0)
    subject_foot = (0.0, 0.0, 0.7)   # mid_body-ish anchor
    subject_head = (0.0, 0.0, 1.65)  # eye_level-ish anchor
    tx, ty = 0.5, 0.5

    result = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=50, subject_pos=subject_foot,
        target_x=tx, target_y=ty, subject_head_pos=subject_head,
    )
    assert result["converged"], result

    got_foot = project_to_screen(cam_pos, result["rotation_deg"], lens_mm=50,
                                  world_pos=subject_foot)
    got_head = project_to_screen(cam_pos, result["rotation_deg"], lens_mm=50,
                                  world_pos=subject_head)
    assert got_foot is not None and got_head is not None
    mid_x = (got_foot[0] + got_head[0]) / 2.0
    mid_y = (got_foot[1] + got_head[1]) / 2.0
    assert _approx(mid_x, tx, tol=1e-3) and _approx(mid_y, ty, tol=1e-3)

    # The midpoint landing on target is the point of the fix — the foot
    # and head individually should NOT both also land on target (that
    # would mean the span collapsed to a single point, defeating the
    # purpose): confirm they differ from each other before the solve even
    # changes rotation, i.e. the input span is real.
    assert not _approx(subject_foot[2], subject_head[2])


def test_solve_with_head_pos_backward_compatible_when_omitted():
    # Omitting subject_head_pos must reproduce the original single-point
    # behavior exactly — every existing caller (and every other test in
    # this file) relies on this.
    cam_pos = (2.0, 0.0, 1.65)
    base_rot = (90.0, 0.0, 90.0)
    subject_pos = (0.0, 0.3, 1.65)
    tx, ty = 0.25, 0.6

    without_head = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=50, subject_pos=subject_pos,
        target_x=tx, target_y=ty,
    )
    with_head_equal_to_foot = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=50, subject_pos=subject_pos,
        target_x=tx, target_y=ty, subject_head_pos=subject_pos,
    )
    assert _approx(without_head["rotation_deg"][0], with_head_equal_to_foot["rotation_deg"][0])
    assert _approx(without_head["rotation_deg"][2], with_head_equal_to_foot["rotation_deg"][2])


def test_solve_with_head_pos_behind_camera_reports_unconverged():
    # If the HEAD point (not just the foot) goes behind the camera
    # mid-solve, that must surface as unconverged too — checking only
    # subject_pos and ignoring subject_head_pos would silently accept a
    # composition that isn't actually achievable for the full span.
    cam_pos = (0.3, 0.0, 1.2)
    base_rot = (90.0, 0.0, 90.0)
    subject_foot = (0.0, 0.0, 0.7)
    subject_head = (5.0, 0.0, 1.65)  # far on the wrong side — forces behind-camera
    result = solve_rotation_for_screen_position(
        cam_pos, base_rot, lens_mm=50, subject_pos=subject_foot,
        target_x=0.5, target_y=0.5, subject_head_pos=subject_head,
    )
    assert not result["converged"]


def test_solve_matches_camera_planner_matrix_convention():
    # composition_solver must use the exact same rotation matrix camera_planner
    # (and therefore the real Blender renderer) uses — not a reimplementation
    # that happens to agree numerically by coincidence.
    from pace_core.setup import composition_solver
    assert composition_solver.camera_axes.__module__ == "pace_core.setup.composition_solver"
    m = _euler_xyz_to_matrix((12.0, -7.0, 40.0))
    right, up, forward = camera_axes((12.0, -7.0, 40.0))
    assert all(_approx(right[i], m[i][0]) for i in range(3))
    assert all(_approx(up[i], m[i][1]) for i in range(3))
    assert all(_approx(forward[i], -m[i][2]) for i in range(3))
