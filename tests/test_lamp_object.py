"""Tests for LAMP object/subject motion (Phase 5):
- lamp_dsl object mode (kind="object": object-framed prompt, same grammar)
- lamp_compile.compile_object_dsl — golden parity vs upstream bbox_to_traj
- lamp_compile.object_to_track — subject-anchored metric path for the viewer

Run: uv run python tests/test_lamp_object.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "lamp"

import json

from pace_core.camera import lamp_dsl
from pace_core.camera.lamp_compile import compile_object_dsl, object_to_track, bbox_json


def _expect(cond, msg, fails):
    if not cond:
        fails.append(msg)


def _seg(s):
    return " ".join([s] * 4)


def _shot(shot_size="medium", angle="eye_level", lens=50):
    return {"scene_ref": "", "camera": {"shot_size": shot_size, "angle": angle, "lens_mm": lens},
            "frame": {"movement": [], "movement_easing": "linear"}}


def test_object_compile_golden(fails):
    # compile_object_dsl(tag).bbox_traj_line == upstream bbox_traj.txt, exactly.
    for ex in ("0000", "0001"):
        tag = (FIX / f"{ex}_bbox_tag.txt").read_text().strip()
        want = (FIX / f"{ex}_bbox_traj.txt").read_text().strip()
        got = compile_object_dsl(tag)["bbox_traj_line"].strip()
        _expect(got == want, f"[{ex}] bbox_traj mismatch:\n  want {want[:80]}\n  got  {got[:80]}", fails)
    r = compile_object_dsl(_seg("right no no 0 0 0"))
    _expect(r["n_frames"] == 21, f"expected 21 frames, got {r['n_frames']}", fails)
    # 'right' → tx (pos_cube[0]) increases from 256
    _expect(r["frames"][-1]["pos_cube"][0] > 256, f"right should raise tx: {r['frames'][-1]['pos_cube']}", fails)
    # static → no motion
    rs = compile_object_dsl(_seg("no no no 0 0 0"))
    _expect(all(f["pos_cube"] == [256, 256, 256] for f in rs["frames"]), "static object should not move", fails)


def test_object_to_track(fails):
    # 'right' subject → +Y in our world; anchored at the subject (≈ origin), not the camera.
    tr = object_to_track(_seg("right no no 0 0 0"), _shot(), n_frames=8)
    _expect(len(tr) == 8, f"n_frames not honored: {len(tr)}", fails)
    _expect(abs(tr[0]["position"][0]) < 0.01 and abs(tr[0]["position"][1]) < 0.01,
            f"object should start at the subject origin, got {tr[0]['position']}", fails)
    _expect(tr[-1]["position"][1] > 0, f"'right' should give +Y: {tr[-1]['position']}", fails)
    # 'far_up' subject → +Z (rises)
    up = object_to_track(_seg("no far_up no 0 0 0"), _shot(), n_frames=8)
    _expect(up[-1]["position"][2] > up[0]["position"][2], f"far_up should raise z: {up[-1]['position']}", fails)


def test_object_dsl_mode(fails):
    # object kind selects the object system prompt + few-shot.
    msgs = lamp_dsl.build_messages("He walks to the right.", kind="object")
    _expect("SUBJECT" in msgs[0]["content"], "object system prompt not selected", fails)
    _expect(any(m["role"] == "assistant" for m in msgs), "object few-shot missing", fails)
    # plan_dsl(kind="object") with an injected fake LLM.
    def fake(_messages):
        return ("right no no 0 0 0 right no no 0 0 0 right no no 0 0 0 right no no 0 0 0", 0.0)
    r = lamp_dsl.plan_dsl("the man crosses screen right", kind="object", call=fake)
    _expect(r["source"] == "llm" and r["dsl"].split()[0] == "right", f"object plan_dsl wrong: {r}", fails)
    # default kind stays camera (no regression)
    cam = lamp_dsl.build_messages("push in", kind="camera")
    _expect("CAMERA" in cam[0]["content"], "camera prompt should remain default", fails)


def test_object_bbox_json_golden(fails):
    # bbox_json(compile_object_dsl(tag)) == upstream object_to_json output, exactly.
    for ex in ("0000", "0001"):
        tag = (FIX / f"{ex}_bbox_tag.txt").read_text().strip()
        want = json.loads((FIX / f"{ex}_bbox.json").read_text())
        got = bbox_json(compile_object_dsl(tag)["frames"])
        _expect(len(got["frames"]) == len(want["frames"]),
                f"[{ex}] bbox_json frame count {len(got['frames'])}!={len(want['frames'])}", fails)
        err = 0.0
        for a, b in zip(want["frames"], got["frames"]):
            for r in range(4):
                for c in range(4):
                    err = max(err, abs(a["transform_matrix"][r][c] - b["transform_matrix"][r][c]))
        _expect(err < 1e-9, f"[{ex}] bbox_json transform mismatch (max err {err:.2e})", fails)
    # static subject → bbox held at the dequantized centre every frame.
    held = bbox_json(compile_object_dsl(_seg("no no no 0 0 0"))["frames"])
    t0 = held["frames"][0]["transform_matrix"]
    _expect(all(f["transform_matrix"] == t0 for f in held["frames"]),
            "static object bbox should not move", fails)


def main() -> int:
    fails: list[str] = []
    for fn in (test_object_compile_golden, test_object_to_track, test_object_dsl_mode,
               test_object_bbox_json_golden):
        fn(fails)
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("OK — all lamp_object tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
