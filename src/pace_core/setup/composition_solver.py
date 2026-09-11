"""Composition solver — grounds PAI's screen_position field (rule-of-thirds
zone or normalized (x, y), Section "Identity & Staging" in the PAI schema)
in the same explicit 3D scene the camera trajectory is already compiled
into, instead of leaving it as authored-but-unconsumed metadata.

Given a shot's already-planned camera (position + rotation from
camera_planner.plan_camera) and a subject's real world position, this
solves for a pan/tilt (yaw/pitch) adjustment so the subject's projection
lands at the specified screen position. The camera does not move — only
its aim changes, matching how an operator recomposes on a locked-off
tripod rather than repositioning the subject (subject world position is
shared across every shot in a scene; only framing should vary per-shot).

Pure functions, no I/O — mirrors camera_planner.py's own design rule.
Reuses camera_planner._euler_xyz_to_matrix so this solver's notion of
"right"/"up"/"forward" is provably identical to what the Blender renderer
actually applies (cross-checked directly against Blender's own
mathutils.Euler.to_matrix, not just derived on paper — see
tests/test_composition_solver.py's test_camera_axes_match_blender_ground_truth).

solve_rotation_for_screen_position accepts an optional subject_head_pos
so a caller can target the midpoint of a two-point span's projections
rather than a single fixed-height point — the silhouette-centroid
reading of "where the subject sits in the frame", appropriate when the
whole subject is visible. It is deliberately opt-in per call rather than
always-on: driving a whole-body centroid to the target in a close-up
tilts the camera down until the head leaves the frame, which scores
better on a mask-centroid metric and is a worse shot. camera_planner
.plan_camera decides which regime applies from the frame's actual
vertical extent at the subject's range; see its Composition paragraph.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from pace_core.camera.camera_planner import _euler_xyz_to_matrix

# Full-frame sensor width in mm.
SENSOR_WIDTH_MM = 36.0

# Rule-of-thirds zone -> normalized (x, y). x: 0.0 = left edge, 1.0 = right
# edge; y: 0.0 = top edge, 1.0 = bottom edge (matches ScreenPosition's own
# docstring in core/types_v1.py). Grid lines at the standard 1/3, 2/3 marks.
ZONE_TO_XY: dict[str, tuple[float, float]] = {
    "left":         (1 / 3, 0.5),
    "center":       (0.5,   0.5),
    "right":        (2 / 3, 0.5),
    "upper":        (0.5,   1 / 3),
    "lower":        (0.5,   2 / 3),
    "upper_left":   (1 / 3, 1 / 3),
    "upper_center": (0.5,   1 / 3),
    "upper_right":  (2 / 3, 1 / 3),
    "lower_left":   (1 / 3, 2 / 3),
    "lower_center": (0.5,   2 / 3),
    "lower_right":  (2 / 3, 2 / 3),
    "center_left":  (1 / 3, 0.5),
    "center_right": (2 / 3, 0.5),
}


def target_xy(screen_position: Optional[dict]) -> Optional[tuple[float, float]]:
    """Resolve a ScreenPosition dict ({zone: ...} or {x, y}) to a concrete
    normalized (x, y). Precise (x, y) wins when both are present. None if
    the spec has neither a recognized zone nor explicit coordinates."""
    if not screen_position:
        return None
    x, y = screen_position.get("x"), screen_position.get("y")
    if x is not None and y is not None:
        return float(x), float(y)
    zone = screen_position.get("zone")
    if zone in ZONE_TO_XY:
        return ZONE_TO_XY[zone]
    return None


def _fov_half_tan(lens_mm: float, sensor_mm: float) -> float:
    """tan(half-FOV) for one axis, given that axis's sensor extent."""
    return (sensor_mm / 2.0) / lens_mm


def camera_axes(rotation_deg: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
    """World-space (right, up, forward) unit vectors for a camera at this
    rotation_deg = [rx, ry, rz]. Blender camera-local convention: -Z is
    forward, +Y is up, +X is right."""
    m = _euler_xyz_to_matrix(rotation_deg)
    right = [m[0][0], m[1][0], m[2][0]]
    up = [m[0][1], m[1][1], m[2][1]]
    forward = [-m[0][2], -m[1][2], -m[2][2]]
    return right, up, forward


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _project_point(cam_pos: Sequence[float], right: Sequence[float],
                    up: Sequence[float], forward: Sequence[float],
                    world_pos: Sequence[float], half_h: float,
                    half_v: float) -> Optional[tuple[float, float]]:
    """Shared pinhole-projection core for a camera at cam_pos with the given
    (already-computed) axes. None if world_pos is behind the camera."""
    v = _sub(world_pos, cam_pos)
    depth = _dot(v, forward)
    if depth <= 1e-6:
        return None
    local_right = _dot(v, right)
    local_up = _dot(v, up)
    nx = 0.5 + 0.5 * (local_right / depth) / half_h
    ny = 0.5 - 0.5 * (local_up / depth) / half_v
    return nx, ny


def project_to_screen(cam_pos: Sequence[float], rotation_deg: Sequence[float],
                       lens_mm: float, world_pos: Sequence[float], *,
                       aspect: float = 16 / 9) -> Optional[tuple[float, float]]:
    """Pinhole-project a world point into normalized screen coords
    (0..1, 0..1; (0,0) = top-left, matching ScreenPosition's convention).
    Returns None if the point is behind the camera (depth <= 0)."""
    right, up, forward = camera_axes(rotation_deg)
    half_h = _fov_half_tan(lens_mm, SENSOR_WIDTH_MM)
    half_v = half_h / aspect
    return _project_point(cam_pos, right, up, forward, world_pos, half_h, half_v)


def solve_rotation_for_screen_position(
    cam_pos: Sequence[float], base_rotation_deg: Sequence[float],
    lens_mm: float, subject_pos: Sequence[float],
    target_x: float, target_y: float, *,
    subject_head_pos: Optional[Sequence[float]] = None,
    aspect: float = 16 / 9, max_iters: int = 16, tol: float = 1e-5,
) -> dict:
    """Find rotation_deg = [rx, ry, rz] such that `subject_pos` projects to
    (target_x, target_y) under a camera fixed at `cam_pos` with lens
    `lens_mm`. The camera does not move; only its aim does.

    subject_head_pos: optional second world point, same (x, y) as
    subject_pos with a different z — the top of the subject's visible
    extent (subject_pos then reads as the bottom/foot). When given, the
    solver targets the MIDPOINT of subject_pos's and subject_head_pos's
    independently-projected normalized screen positions, an approximation
    of a standing figure's silhouette centroid, instead of a single fixed
    height. Use it only when the span actually fits the frame: the
    midpoint objective is satisfiable with both endpoints off-screen in
    opposite directions, so at a close-up it will happily aim at the
    subject's midriff with the head out of frame. Omit subject_head_pos
    to keep the single-point behavior.

    Fixed-point iteration: re-project the subject under the current
    rotation guess, convert the normalized screen error to an angular
    pan/tilt delta using the current depth and FOV, apply it, repeat.
    max_iters is 16 rather than the 8 that sufficed for the single-point
    case: when the foot/head span subtends far more than the frame (an
    extreme close-up, where each endpoint sits well off-screen and the
    projection is steeply nonlinear there) the iteration still converges
    but needs roughly twice as many steps — measured at 0.4 m / 50 mm,
    which stopped one step short of tolerance at 8 and converged at 16.
    The extra steps are pure arithmetic, so there is no reason to run the
    tight budget.
    Roll (ry) is held fixed at its base value throughout — only pan (rz)
    and tilt (rx) move, matching how an operator recomposes without
    rolling the frame.

    Returns {"rotation_deg", "iterations", "residual", "converged"};
    residual is the final max(|dx|, |dy|) in normalized screen units.
    converged=False (max_iters exhausted, or either point went behind the
    camera mid-solve) is reported honestly rather than silently accepted
    — callers should treat an unconverged result as a failed composition,
    not a usable one.
    """
    rx, ry, rz = (float(a) for a in base_rotation_deg)
    half_h = _fov_half_tan(lens_mm, SENSOR_WIDTH_MM)
    half_v = half_h / aspect
    err_x = err_y = float("nan")

    for i in range(1, max_iters + 1):
        right, up, forward = camera_axes((rx, ry, rz))
        p_foot = _project_point(cam_pos, right, up, forward, subject_pos, half_h, half_v)
        if p_foot is None:
            return {"rotation_deg": [rx, ry, rz], "iterations": i,
                    "residual": None, "converged": False,
                    "error": "subject behind camera during solve"}
        if subject_head_pos is None:
            cur_x, cur_y = p_foot
        else:
            p_head = _project_point(cam_pos, right, up, forward, subject_head_pos, half_h, half_v)
            if p_head is None:
                return {"rotation_deg": [rx, ry, rz], "iterations": i,
                        "residual": None, "converged": False,
                        "error": "subject behind camera during solve"}
            cur_x = (p_foot[0] + p_head[0]) / 2.0
            cur_y = (p_foot[1] + p_head[1]) / 2.0
        err_x, err_y = target_x - cur_x, target_y - cur_y
        if max(abs(err_x), abs(err_y)) <= tol:
            return {"rotation_deg": [rx, ry, rz], "iterations": i,
                    "residual": max(abs(err_x), abs(err_y)), "converged": True}
        # Normalized screen error -> angular delta at the subject's depth:
        # a normalized-x error err_x corresponds to a lateral offset of
        # err_x * 2*half_h*depth in the image plane at that depth, so the
        # yaw delta needed is atan(err_x * 2*half_h). Signs are calibrated
        # against Blender's actual rotation convention by
        # test_composition_solver.py's round-trip tests, not assumed.
        yaw_delta = math.degrees(math.atan(err_x * 2 * half_h))
        pitch_delta = math.degrees(math.atan(err_y * 2 * half_v))
        rz += yaw_delta
        rx += pitch_delta

    return {"rotation_deg": [rx, ry, rz], "iterations": max_iters,
            "residual": max(abs(err_x), abs(err_y)), "converged": False}
