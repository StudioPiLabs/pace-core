"""Regression test for camera_planner follow-shot composition.

Covers the object-motion + chase-cam helpers that replaced the renderer's
old `--car-speed-mps` hack:

  1. make_linear_motion integrates speed → per-frame world positions.
  2. plan_follow_shot on a straight-line subject degenerates to a plain
     offset add (camera keeps its framing, rotation unchanged).
  3. plan_follow_shot on a rotating subject composes rig offset + rotation
     correctly (matches the manual matrix math).
  4. euler↔matrix helpers round-trip.

Run: uv run python tests/test_follow_shot.py   (or via pytest)
"""
from __future__ import annotations
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent

from pace_core.camera.camera_planner import (  # noqa: E402
    make_linear_motion,
    plan_follow_shot,
    _euler_xyz_to_matrix,
    _matrix_to_euler_xyz,
    _matmul3,
    _matvec3,
)


def _approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def test_linear_motion_integrates_speed(fails: list[str]) -> None:
    # 16 m/s at 16 fps → exactly 1 m per frame along +X.
    track = make_linear_motion(n_frames=4, speed_mps=16.0, fps=16, axis="x")
    _expect(len(track) == 4, f"n_frames: got {len(track)}", fails)
    xs = [kf["position"][0] for kf in track]
    _expect(xs == [0.0, 1.0, 2.0, 3.0], f"x progression: got {xs}", fails)
    _expect(all(kf["position"][1] == 0 and kf["position"][2] == 0 for kf in track),
            "off-axis stays zero", fails)
    _expect([kf["frame_index"] for kf in track] == [0, 1, 2, 3],
            "frame_index sequence", fails)


def test_linear_motion_axis_and_start(fails: list[str]) -> None:
    track = make_linear_motion(n_frames=3, speed_mps=8.0, fps=8, axis="y",
                               start=(1.0, 2.0, 3.0))
    ys = [kf["position"][1] for kf in track]      # 1 m/frame from y=2
    _expect(ys == [2.0, 3.0, 4.0], f"y from start: got {ys}", fails)
    _expect(all(kf["position"][0] == 1.0 for kf in track), "x held at start", fails)


def test_follow_straight_line_is_offset_add(fails: list[str]) -> None:
    """No subject rotation → world cam = subject_pos + rel_pos, rot unchanged."""
    subject = make_linear_motion(n_frames=3, speed_mps=16.0, fps=16, axis="x")
    rel = [{"position": [-4.0, -1.0, 2.0], "rotation_deg": [70.0, 0.0, -50.0],
            "lens_mm": 35}] * 3
    out = plan_follow_shot(subject_track=subject, relative_camera_track=rel)
    cam = out["camera_track"]
    xs = [c["position"][0] for c in cam]
    _expect(xs == [-4.0, -3.0, -2.0], f"cam x follows subject: got {xs}", fails)
    _expect(all(_approx(c["position"][1], -1.0) and _approx(c["position"][2], 2.0)
                for c in cam), "cam y,z constant", fails)
    for c in cam:
        _expect(all(_approx(c["rotation_deg"][k], [70.0, 0.0, -50.0][k]) for k in range(3)),
                f"rotation unchanged: got {c['rotation_deg']}", fails)
    _expect(all(c["lens_mm"] == 35 for c in cam), "lens carried", fails)
    # subject_track echoed back as world-space
    _expect([s["position"][0] for s in out["subject_track"]] == [0.0, 1.0, 2.0],
            "subject echoed world-space", fails)


def test_follow_rotating_subject_composes(fails: list[str]) -> None:
    """Subject yawed 30° → camera offset rotates with it; verify against
    the manual matrix composition world = subj_pos + Rs@rel_pos."""
    subj_rot = [0.0, 0.0, 30.0]
    rel_pos = [-4.0, -1.0, 2.0]
    rel_rot = [70.0, 0.0, -50.0]
    subject = [{"position": [1.0, 2.0, 0.0], "rotation_deg": subj_rot}]
    rel = [{"position": rel_pos, "rotation_deg": rel_rot, "lens_mm": 50}]
    out = plan_follow_shot(subject_track=subject, relative_camera_track=rel)
    c = out["camera_track"][0]

    rs = _euler_xyz_to_matrix(subj_rot)
    exp_pos = [1.0, 2.0, 0.0]
    off = _matvec3(rs, rel_pos)
    exp_pos = [exp_pos[k] + off[k] for k in range(3)]
    _expect(all(_approx(c["position"][k], exp_pos[k]) for k in range(3)),
            f"composed pos: got {c['position']} want {exp_pos}", fails)

    # rotation: compare via matrix (euler representation may vary)
    exp_mat = _matmul3(rs, _euler_xyz_to_matrix(rel_rot))
    got_mat = _euler_xyz_to_matrix(c["rotation_deg"])
    _expect(all(_approx(exp_mat[i][j], got_mat[i][j])
                for i in range(3) for j in range(3)),
            "composed rotation matrix", fails)
    _expect(c["lens_mm"] == 50, f"lens carried: got {c['lens_mm']}", fails)


def test_euler_matrix_roundtrip(fails: list[str]) -> None:
    for rot in ([10.0, 20.0, 30.0], [-45.0, 15.0, 80.0], [0.0, 0.0, 0.0]):
        back = _matrix_to_euler_xyz(_euler_xyz_to_matrix(rot))
        _expect(all(_approx(rot[k], back[k], tol=1e-4) for k in range(3)),
                f"roundtrip {rot}: got {back}", fails)


def test_length_mismatch_raises(fails: list[str]) -> None:
    try:
        plan_follow_shot(
            subject_track=[{"position": [0, 0, 0]}],
            relative_camera_track=[{"position": [0, 0, 0]}, {"position": [1, 0, 0]}],
        )
        fails.append("expected ValueError on length mismatch")
    except ValueError:
        pass

# The hand-rolled runner that used to close this file -- a TESTS list, a main()
# looping it, and in one case a wrapper generator -- is gone. It reimplemented
# what pytest already does: collection (these are plain test_* functions taking
# the `fails` fixture from conftest), fixture injection, and pass/fail
# reporting. Nothing referenced it, no doc or CI invoked the file directly, and
# the wrapper generator was outright dead -- it published 29 module attributes
# named pytest_test_*, which pytest does not collect, so every one of those
# wrappers had never run. Run these with: uv run pytest tests/
