"""Golden tests for pace_core/camera/lamp_compile.py — verifies our in-repo port of
LAMP's DSL→trajectory compiler reproduces the *upstream repo's own outputs*
byte-for-byte (cam_traj.txt) and to ≤1e-6 (cam_json 4x4 extrinsics).

Fixtures (tests/fixtures/lamp/) were copied from
LAMP/blender_vis/vis/example_traj/{0000,0001,0005}/.

Run: uv run python tests/test_lamp_compile.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "lamp"

from pace_core.camera.lamp_compile import compile_dsl, cam_json


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def test_cam_traj_golden(fails):
    """compile_dsl(tag).cam_traj_line == upstream cam_traj.txt, exactly."""
    for ex in ("0000", "0001", "0005"):
        tag = (FIX / f"{ex}_cam_tag.txt").read_text().strip()
        want = (FIX / f"{ex}_cam_traj.txt").read_text().strip()
        got = compile_dsl(tag)["cam_traj_line"].strip()
        _expect(got == want, f"[{ex}] cam_traj mismatch\n   want: {want[:90]}...\n    got: {got[:90]}...", fails)


def test_n_frames(fails):
    res = compile_dsl((FIX / "0000_cam_tag.txt").read_text().strip())
    _expect(res["n_frames"] == 21, f"expected 21 frames, got {res['n_frames']}", fails)


def test_cam_json_golden(fails):
    """cam_json 4x4 transform matrices match upstream 0000_cam.json (≤1e-6)."""
    tag = (FIX / "0000_cam_tag.txt").read_text().strip()
    got = cam_json(compile_dsl(tag)["frames"])
    want = json.loads((FIX / "0000_cam.json").read_text())
    gf, wf = got["frames"], want["frames"]
    _expect(len(gf) == len(wf), f"frame count {len(gf)} != {len(wf)}", fails)
    if len(gf) == len(wf):
        max_err = 0.0
        for g, w in zip(gf, wf):
            for grow, wrow in zip(g["transform_matrix"], w["transform_matrix"]):
                for gv, wv in zip(grow, wrow):
                    max_err = max(max_err, abs(float(gv) - float(wv)))
        _expect(max_err <= 1e-6, f"cam_json 4x4 max abs error {max_err:.2e} > 1e-6", fails)


def test_semantics(fails):
    """Sanity on the Phase-3 intermediates (pos_cube / euler_deg)."""
    # 0001 = near_right ×4 → camera trucks right → tx should increase past 256.
    r = compile_dsl((FIX / "0001_cam_tag.txt").read_text().strip())
    _expect(r["frames"][-1]["pos_cube"][0] > 256,
            f"near_right should increase tx, got {r['frames'][-1]['pos_cube'][0]}", fails)
    # 0000 = yaw 30 ×4 → LAMP target_yaw = -rot → final yaw negative.
    r0 = compile_dsl((FIX / "0000_cam_tag.txt").read_text().strip())
    _expect(r0["frames"][-1]["euler_deg"][0] < 0,
            f"yaw30 → final euler yaw should be negative, got {r0['frames'][-1]['euler_deg'][0]}", fails)
    # static hold → no motion at all.
    rs = compile_dsl(" ".join(["no no no 0 0 0"] * 4))
    _expect(all(f["pos_cube"] == [256, 256, 256] for f in rs["frames"]),
            "static DSL should not move the camera", fails)


def main() -> int:
    fails: list[str] = []
    for fn in (test_cam_traj_golden, test_n_frames, test_cam_json_golden, test_semantics):
        fn(fails)
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("OK — all lamp_compile golden tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
