"""Tests for lamp_compile.lamp_to_track — projecting a LAMP DSL trajectory onto
our metric, scene-anchored keyframe track (camera_planner.plan_camera_track
shape). Directional/anchoring sanity vs camera_planner conventions.

Run: uv run python tests/test_lamp_to_track.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from pace_core.camera.lamp_compile import lamp_to_track
from pace_core.camera.lamp_dsl import STATIC_DSL


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def _shot(shot_size="medium", angle="eye_level", lens=50) -> dict:
    return {"scene_ref": "", "camera": {"shot_size": shot_size, "angle": angle, "lens_mm": lens},
            "frame": {"movement": [], "movement_easing": "linear"}}


def _seg(s):  # one segment repeated ×4
    return " ".join([s] * 4)


def test_shape_and_anchor(fails):
    tr = lamp_to_track(STATIC_DSL, _shot(), n_frames=16)
    _expect(len(tr) == 16, f"n_frames not honored: {len(tr)}", fails)
    _expect(len(lamp_to_track(STATIC_DSL, _shot(), n_frames=24)) == 24, "n_frames=24 not honored", fails)
    f0 = tr[0]
    _expect(set(f0) == {"frame_index", "t", "position", "rotation_deg", "lens_mm"},
            f"frame keys mismatch: {set(f0)}", fails)
    # static → anchored at plan_camera start, unmoving
    _expect(f0["position"] == [2.0, 0.0, 1.65], f"start position wrong: {f0['position']}", fails)
    _expect(f0["rotation_deg"] == [90.0, 0.0, 90.0], f"start rotation wrong: {f0['rotation_deg']}", fails)
    _expect(f0["lens_mm"] == 50, f"lens not carried: {f0['lens_mm']}", fails)
    _expect(tr[-1]["position"] == [2.0, 0.0, 1.65], "static should not move", fails)
    _expect(tr[0]["t"] == 0.0 and abs(tr[-1]["t"] - 1.0) < 1e-9, "t must span 0..1", fails)


def test_translation_axes(fails):
    start_x = 2.0
    push = lamp_to_track(_seg("no no far_in 0 0 0"), _shot(), n_frames=16)
    _expect(push[-1]["position"][0] < start_x, f"push_in should reduce x (toward subject): {push[-1]['position']}", fails)
    _expect(push[-1]["position"][0] >= 0.15, "push must not cross subject (min gap)", fails)
    pull = lamp_to_track(_seg("no no far_out 0 0 0"), _shot(), n_frames=16)
    _expect(pull[-1]["position"][0] > start_x, f"pull_out should increase x: {pull[-1]['position']}", fails)
    right = lamp_to_track(_seg("near_right no no 0 0 0"), _shot(), n_frames=16)
    _expect(right[-1]["position"][1] > 0, f"truck right should give +Y: {right[-1]['position']}", fails)
    crane = lamp_to_track(_seg("no far_up no 0 0 0"), _shot(), n_frames=16)
    _expect(crane[-1]["position"][2] > 1.65, f"crane up should raise z: {crane[-1]['position']}", fails)


def test_rotation_signs(fails):
    # pan right: DSL +yaw → rot_z increases (matches camera_planner pan_right).
    pan = lamp_to_track(_seg("no no no 30 0 0"), _shot(), n_frames=16)
    _expect(pan[-1]["rotation_deg"][2] > 90, f"pan right should raise rot_z: {pan[-1]['rotation_deg']}", fails)
    # tilt up: DSL +tilt → rot_x decreases (matches camera_planner tilt_up).
    tilt = lamp_to_track(_seg("no no no 0 10 0"), _shot(), n_frames=16)
    _expect(tilt[-1]["rotation_deg"][0] < 90, f"tilt up should lower rot_x: {tilt[-1]['rotation_deg']}", fails)


def test_clamp_close_up(fails):
    # deep push on the tightest framing must never cross the subject.
    tr = lamp_to_track(_seg("no no far_in 0 0 0"), _shot(shot_size="extreme_close_up"), n_frames=16)
    _expect(all(f["position"][0] >= 0.15 for f in tr), "extreme_close_up push must stay >= 0.15m", fails)


def main() -> int:
    fails: list[str] = []
    for fn in (test_shape_and_anchor, test_translation_axes, test_rotation_signs, test_clamp_close_up):
        fn(fails)
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("OK — all lamp_to_track tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
