"""LAMP DSL → 3D camera trajectory compiler (Phase 2).

Faithful in-repo port of LAMP's pure-Python compiler chain
(LAMP/src/scripts/{generate_camera_trajectory,traj_to_et,et_to_json}.py) so a
validated 24-token DSL line (see lamp_dsl.py) becomes per-frame camera
extrinsics — with no GPU, no scipy, no SSH. numpy-only.

The port is verified byte-for-byte against the upstream repo's own example
outputs (tests/fixtures/lamp/*_cam_traj.txt + 0000_cam.json) in
tests/test_lamp_compile.py.

Output of `compile_dsl(line)`:
  {
    "n_frames": 21,
    "cam_traj_line": "<rw rx ry rz tx ty tz | ...>",   # == upstream cam_traj.txt
    "frames": [
      {
        "frame_index": int,
        "pos_cube":  [tx, ty, tz],      # 0..512 cube (LAMP's clamped position)
        "euler_deg": [yaw, pitch, roll],# LAMP's per-frame Euler (degrees)
        "quat7":     [rw, rx, ry, rz, tx, ty, tz],  # 0..512 encoded (cam_traj)
        "transform_4x4": [[...],[...],[...],[0,0,0,1]],  # == upstream cam_json
      }, ...
    ],
  }

`pos_cube` + `euler_deg` are the clean intermediates Phase 3 (`lamp_to_track`)
rescales into our metric keyframe track; `transform_4x4` / `cam_traj_line` exist
for parity with LAMP's downstream (projection / cam_json) and the golden tests.
"""

from __future__ import annotations

import math
from collections import Counter

from pace_core.camera.lamp_dsl import parse_dsl, N_SEGMENTS

# Upstream constants (generate_camera_trajectory.py).
SHOT_LENGTH = 21
SEGMENT_COUNT = N_SEGMENTS               # 4
PARAMS_PER_SEGMENT = 6
ORIGINAL_SEGMENT_LENGTH = 5
_BASE_DISTANCE = 64       # camera per-segment travel (generate_camera_trajectory)
_BASE_DISTANCE_OBJ = 54   # object/subject per-segment travel (bbox_to_traj)


# ── generate_camera_trajectory.py port ────────────────────────────────
def _clamp(value: float, lo: int = 0, hi: int = 512) -> int:
    return int(round(max(lo, min(hi, value))))


def _interpolate_sequence(start: float, end: float, num_frames: int) -> list[float]:
    if num_frames < 1:
        return [end]
    return [start + (end - start) * (i / num_frames) for i in range(1, num_frames + 1)]


def _quaternion_from_euler(yaw: float, pitch: float, roll: float) -> dict:
    """Euler°→quaternion, YXZ order (verbatim from upstream)."""
    cy, sy = math.cos(math.radians(yaw) * 0.5), math.sin(math.radians(yaw) * 0.5)
    cp, sp = math.cos(math.radians(pitch) * 0.5), math.sin(math.radians(pitch) * 0.5)
    cr, sr = math.cos(math.radians(roll) * 0.5), math.sin(math.radians(roll) * 0.5)
    return {
        "rw": cr * cp * cy + sr * sp * sy,
        "rx": sr * cp * cy - cr * sp * sy,
        "ry": cr * sp * cy + sr * cp * sy,
        "rz": cr * cp * sy - sr * sp * cy,
    }


def _determine_pattern(segments: list) -> str:
    if not segments or len(segments) != 4:
        return "ABCD"
    full = [tuple(s) for s in segments]
    counts = Counter(full)
    if len(counts) == 1:
        return "AAAA"
    if len(counts) == 4:
        return "ABCD"
    if len(counts) == 3:
        if full[0] == full[1]:
            return "AABC"
        if full[1] == full[2]:
            return "ABBC"
        if full[2] == full[3]:
            return "ABCC"
        return "ABCD"
    if len(counts) == 2:
        if full[0] == full[1] and full[2] == full[3]:
            return "AABB"
        if full[0] == full[1] == full[2]:
            return "AAAB"
        if full[1] == full[2] == full[3]:
            return "ABBB"
        return "ABCD"
    return "ABCD"


def _group_consecutive(pattern: str) -> list[tuple[str, int]]:
    if not pattern:
        return []
    groups, cur, n = [], pattern[0], 1
    for ch in pattern[1:]:
        if ch == cur:
            n += 1
        else:
            groups.append((cur, n))
            cur, n = ch, 1
    groups.append((cur, n))
    return groups


def _move_distance(move_str: str, segment_strength: float = 1.0,
                   base_distance: float = _BASE_DISTANCE) -> float:
    if move_str == "no":
        return 0.0
    max_distance = base_distance * segment_strength
    if "far" in move_str:
        return max_distance
    if "near" in move_str:
        return max_distance / 3.0
    return (max_distance * 2.0) / 3.0


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> list[list[float]]:
    yr, pr, rr = math.radians(yaw), math.radians(pitch), math.radians(roll)
    cy, sy = math.cos(yr), math.sin(yr)
    cp, sp = math.cos(pr), math.sin(pr)
    cr, sr = math.cos(rr), math.sin(rr)
    return [
        [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
        [cp * sr,                 cp * cr,                -sp],
        [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr,  cy * cp],
    ]


def _move_to_world(mx: str, my: str, mz: str, strength: float,
                   yaw: float, pitch: float, roll: float,
                   base_distance: float = _BASE_DISTANCE) -> tuple[float, float, float]:
    dx = _move_distance(mx, strength, base_distance)
    dy = _move_distance(my, strength, base_distance)
    dz = _move_distance(mz, strength, base_distance)
    if dx == 0 and dy == 0 and dz == 0:
        return 0.0, 0.0, 0.0
    right = -dx if "left" in mx else (dx if "right" in mx else 0.0)
    up = dy if "up" in my else (-dy if "down" in my else 0.0)
    fwd = -dz if "in" in mz else (dz if "out" in mz else 0.0)
    R = _rotation_matrix(yaw, pitch, roll)
    return (
        R[0][0] * right + R[0][1] * up + R[0][2] * fwd,
        R[1][0] * right + R[1][1] * up + R[1][2] * fwd,
        R[2][0] * right + R[2][1] * up + R[2][2] * fwd,
    )


def _generate_frames(tag_params: list, base_distance: float = _BASE_DISTANCE) -> list[dict]:
    """Port of generate_trajectory_from_tags — emits per-frame state with BOTH
    the LAMP quat-encoded form (for parity) and raw pos/euler. base_distance
    selects camera (64) vs object/subject (54) per-segment travel."""
    segments = [tag_params[i * 6:(i + 1) * 6] for i in range(SEGMENT_COUNT)]
    groups = _group_consecutive(_determine_pattern(segments))

    frames: list[dict] = []
    tx, ty, tz = 256.0, 256.0, 256.0
    yaw = pitch = roll = 0.0

    def _emit(idx, q):
        frames.append({
            "frame_index": idx,
            "pos_cube": [_clamp(tx), _clamp(ty), _clamp(tz)],
            "euler_deg": [yaw, pitch, roll],
            "quat7": [
                _clamp((q["rw"] + 1) * 256), _clamp((q["rx"] + 1) * 256),
                _clamp((q["ry"] + 1) * 256), _clamp((q["rz"] + 1) * 256),
                _clamp(tx), _clamp(ty), _clamp(tz),
            ],
        })

    _emit(0, _quaternion_from_euler(yaw, pitch, roll))

    seg_idx = 0
    for _char, count in groups:
        mx, my, mz, rot_yaw, rot_pitch, rot_roll = segments[seg_idx]
        s_yaw, s_pitch, s_roll = yaw, pitch, roll
        t_yaw, t_pitch, t_roll = yaw - rot_yaw, pitch + rot_pitch, roll - rot_roll
        gframes = count * ORIGINAL_SEGMENT_LENGTH
        yaw_seq = _interpolate_sequence(s_yaw, t_yaw, gframes)
        pitch_seq = _interpolate_sequence(s_pitch, t_pitch, gframes)
        roll_seq = _interpolate_sequence(s_roll, t_roll, gframes)
        for i in range(gframes):
            yaw, pitch, roll = yaw_seq[i], pitch_seq[i], roll_seq[i]
            strength = count / float(gframes)
            wdx, wdy, wdz = _move_to_world(mx, my, mz, strength, yaw, pitch, roll, base_distance)
            tx, ty, tz = tx + wdx, ty + wdy, tz + wdz
            _emit(len(frames), _quaternion_from_euler(yaw, pitch, roll))
        yaw, pitch, roll = t_yaw, t_pitch, t_roll
        seg_idx += count
    return frames


def cam_traj_line(frames: list[dict]) -> str:
    """Reproduce generate_camera_trajectory.format_traj_line: per-frame
    `rw rx ry rz tx ty tz`, frames joined by ' | '."""
    return " | ".join(" ".join(str(v) for v in f["quat7"]) for f in frames)


def bbox_traj_line(frames: list[dict]) -> str:
    """Reproduce bbox_to_traj.format_traj_line: per-frame `tx ty tz` (the
    translating bbox centre), frames joined by ' | '."""
    return " | ".join(" ".join(str(v) for v in f["pos_cube"]) for f in frames)


# ── traj_to_et.py + et_to_json.py port ────────────────────────────────
def _map_int_to_float(v: int) -> float:
    return (v / 512.0) * 2.0 - 1.0


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> list[list[float]]:
    """scalar-last quaternion → 3x3 rotation matrix (matches scipy from_quat:
    normalizes, active rotation). Pure-Python — core stays stdlib-only."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ]


def _quat7_to_4x4(q7: list[int]) -> list[list[float]]:
    """Port of traj_to_et.process_reverse_line (per frame) + et_to_json's 4x4
    wrap. NOTE the upstream token PERMUTATION: format writes [rw,rx,ry,rz,...]
    but process_reverse_line unpacks as rx,ry,rw,rz,tx,ty,tz — reproduced
    verbatim so our extrinsics match cam_json exactly."""
    rx_int, ry_int, rw_int, rz_int, tx_int, ty_int, tz_int = q7
    rw_f = -_map_int_to_float(rw_int)
    rx_f = _map_int_to_float(rx_int)
    ry_f = -_map_int_to_float(ry_int)
    rz_f = _map_int_to_float(rz_int)
    tx_f = _map_int_to_float(tx_int)
    ty_f = -_map_int_to_float(ty_int)
    tz_f = -_map_int_to_float(tz_int)
    R = _quat_to_matrix(rx_f, ry_f, rz_f, rw_f)
    t = [tx_f, ty_f, tz_f]
    return [R[0] + [t[0]], R[1] + [t[1]], R[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


_INTRINSICS = {"w": 0, "h": 0, "fl_x": 0.0, "fl_y": 0.0, "cx": 0.0, "cy": 0.0}


def track_to_lamp_frames(track: list[dict],
                         units_per_meter: float = 25.0) -> list[dict]:
    """Convert a meter-scale camera_planner track back to LAMP's frame
    format (`{quat7, pos_cube, euler_deg}`) for the wireframe projector.

    Inverse of `camera_planner.plan_camera_track_from_dsl`'s output
    pipeline — round-trips position (meters → LAMP units) and rotation
    (Blender euler → LAMP quat7) so that `cam_json(frames)` produces the
    same shape that `projection.py` consumes.
    """
    out: list[dict] = []
    for i, kf in enumerate(track):
        pos = kf["position"]
        rot = kf["rotation_deg"]   # [pitch_x, roll_y, yaw_z]
        # Inverse axis remap: Blender (X, Y, Z) → LAMP (X, Z, -Y)
        lamp_tx = pos[0] * units_per_meter + 256.0
        lamp_ty = pos[2] * units_per_meter + 256.0
        lamp_tz = -pos[1] * units_per_meter + 256.0
        # LAMP quat is encoded from (yaw, pitch, roll). camera_planner's
        # rotation_deg is [pitch_x, roll_y, yaw_z].
        yaw, pitch, roll = rot[2], rot[0], rot[1]
        q = _quaternion_from_euler(yaw, pitch, roll)
        out.append({
            "frame_index": i,
            "pos_cube": [_clamp(lamp_tx), _clamp(lamp_ty), _clamp(lamp_tz)],
            "euler_deg": [yaw, pitch, roll],
            "quat7": [
                _clamp((q["rw"] + 1) * 256), _clamp((q["rx"] + 1) * 256),
                _clamp((q["ry"] + 1) * 256), _clamp((q["rz"] + 1) * 256),
                _clamp(lamp_tx), _clamp(lamp_ty), _clamp(lamp_tz),
            ],
        })
    return out


def cam_json(frames: list[dict]) -> dict:
    """Reproduce et_to_json output (the cam_json/0000.json dict)."""
    out_frames = []
    for i, f in enumerate(frames, start=1):
        out_frames.append({
            "transform_matrix": _quat7_to_4x4(f["quat7"]),
            "monst3r_im_id": i,
            "color_code": "#000000",
        })
    return {**_INTRINSICS, "frames": out_frames}


# ── object_to_json.py port (bbox centres → projection obj input) ───────
# Dequantize a 0..512 bbox centre to a normalized translation, verbatim from
# object_to_json.dequantize_tx_ty_tz: tx=(i-256)/256; y,z negated; z shifted by
# −100/256 (LAMP pushes the subject box forward so it sits in front of the cam).
_OBJ_TRANS_OFFSET = 256.0
_OBJ_TRANS_SCALE = 1.0 / 256.0
_OBJ_Z_SHIFT = 100.0 / 256.0


def _dequantize_obj(tx_i: int, ty_i: int, tz_i: int) -> tuple[float, float, float]:
    tx = (tx_i - _OBJ_TRANS_OFFSET) * _OBJ_TRANS_SCALE
    ty = (ty_i - _OBJ_TRANS_OFFSET) * _OBJ_TRANS_SCALE
    tz = (tz_i - _OBJ_TRANS_OFFSET) * _OBJ_TRANS_SCALE - _OBJ_Z_SHIFT
    return tx, -ty, -tz


def bbox_json(frames: list[dict]) -> dict:
    """Reproduce object_to_json.build_json_from_triplets: each object frame's
    pos_cube (the translating bbox centre, 0..512) → an identity-rotation
    transform with dequantized translation. This is the `bbox.json` that
    projection.py consumes as the object trajectory. Pass the frames from
    `compile_object_dsl(line)["frames"]`."""
    out_frames = []
    for i, f in enumerate(frames, start=1):
        tx, ty, tz = _dequantize_obj(*f["pos_cube"])
        out_frames.append({
            "transform_matrix": [
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "monst3r_im_id": i,
        })
    return {**_INTRINSICS, "frames": out_frames}


# ── public entry ───────────────────────────────────────────────────────
def compile_dsl(line: str) -> dict:
    """Validated 24-token DSL line → trajectory dict (see module docstring).
    Raises ValueError (via parse_dsl) on an invalid line."""
    segs = parse_dsl(line)
    tag_params: list = []
    for s in segs:
        tag_params += [s.mx, s.my, s.mz, s.yaw, s.tilt, s.roll]
    frames = _generate_frames(tag_params)
    # attach 4x4 to each frame for convenience
    for f in frames:
        f["transform_4x4"] = _quat7_to_4x4(f["quat7"])
    return {
        "n_frames": len(frames),
        "cam_traj_line": cam_traj_line(frames),
        "frames": frames,
    }


# ── Phase 3: LAMP trajectory → our metric keyframe track ───────────────
#
# Reconciles LAMP's normalized/relative 512³ cube (no lens, no scale) with our
# metric, scene-anchored camera (camera_planner). Axis mapping — LAMP world
# axes are camera-aligned at frame 0 (identity start); our camera sits on +X at
# `subject_distance`, looks along −X toward the subject at the origin, up = +Z:
#
#   LAMP +x (truck right)        → our +Y           (camera-right)
#   LAMP +y (boom up)            → our +Z           (camera-up)
#   LAMP +z (dolly OUT/backward) → our +X           (away from subject)
#     (so LAMP "in" lowers cube-z → moves our camera −X, toward the subject)
#
# Rotation signs map LAMP's per-frame Euler onto our rotation_deg so the result
# matches camera_planner's conventions (pan_right → rot_z↑, tilt_up → rot_x↓).
# LAMP encodes a DSL +yaw as Euler −yaw (target = −rot), hence the −1 signs.

CUBE_REACH = 256.0          # cube units from centre (256) to an edge (0/512)
DEFAULT_REACH_FRACTION = 0.6  # a full-reach LAMP move ≈ 0.6 × subject_distance
_MIN_SUBJECT_GAP_M = 0.15     # never let a push-in cross the subject at origin


def _resample_track(metric: list[tuple[list, list]], n_frames: int, lens: int) -> list[dict]:
    """Linear-resample the 21 LAMP frames onto N output keyframes. LAMP already
    bakes its own per-segment timing, so we resample linearly (no extra easing).
    Output shape matches camera_planner.plan_camera_track()."""
    m = len(metric)
    out: list[dict] = []
    for j in range(n_frames):
        t = j / (n_frames - 1) if n_frames > 1 else 0.0
        fpos = t * (m - 1)
        i0 = int(math.floor(fpos))
        i1 = min(i0 + 1, m - 1)
        frac = fpos - i0
        p0, r0 = metric[i0]
        p1, r1 = metric[i1]
        out.append({
            "frame_index": j,
            "t": t,
            "position": [round(p0[k] + (p1[k] - p0[k]) * frac, 4) for k in range(3)],
            "rotation_deg": [round(r0[k] + (r1[k] - r0[k]) * frac, 3) for k in range(3)],
            "lens_mm": lens,
        })
    return out


def lamp_to_track(line: str, shot: dict, *, n_frames: int = 16,
                  bible: dict | None = None,
                  reach_fraction: float = DEFAULT_REACH_FRACTION) -> list[dict]:
    """Compile a DSL line and project it onto our metric keyframe track,
    anchored at `plan_camera(shot)`'s static framing. `shot` is a *planner-input*
    dict (the shape `movement_io.shot_to_planner_dict` emits). Returns N dicts
    `{frame_index, t, position[m], rotation_deg, lens_mm}` — a drop-in for
    Blender's depth-pass + Wan VACE control_video, identical in shape to
    camera_planner.plan_camera_track()."""
    from pace_core.camera.camera_planner import plan_camera  # one-directional, lazy

    start = plan_camera(shot, bible=bible)
    sx, sy, sz = start["position"]
    srx, sry, srz = start["rotation_deg"]
    lens = int(start["lens_mm"])
    dist = float(start.get("subject_distance_m", 2.0))
    scale = (dist * reach_fraction) / CUBE_REACH

    frames = compile_dsl(line)["frames"]
    metric: list[tuple[list, list]] = []
    for f in frames:
        d_right = f["pos_cube"][0] - 256
        d_up = f["pos_cube"][1] - 256
        d_depth = f["pos_cube"][2] - 256          # <0 = moved IN toward subject
        x = max(_MIN_SUBJECT_GAP_M, sx + d_depth * scale)
        y = sy + d_right * scale
        z = sz + d_up * scale
        yaw, pitch, roll = f["euler_deg"]
        rot = [srx - pitch, sry + roll, srz - yaw]  # see axis/sign notes above
        metric.append(([x, y, z], rot))
    return _resample_track(metric, n_frames, lens)


# ── Phase 5: object / subject motion (LAMP bbox path) ──────────────────
def compile_object_dsl(line: str) -> dict:
    """Validated 24-token DSL line → a SUBJECT/OBJECT bbox trajectory — LAMP's
    bbox_to_traj (base_distance 54, output = translating bbox centres). Same
    grammar as the camera DSL; author it with lamp_dsl `kind="object"`.
    Returns {n_frames, bbox_traj_line, frames:[{frame_index, pos_cube, euler_deg, ...}]}."""
    segs = parse_dsl(line)
    tag_params: list = []
    for s in segs:
        tag_params += [s.mx, s.my, s.mz, s.yaw, s.tilt, s.roll]
    frames = _generate_frames(tag_params, base_distance=_BASE_DISTANCE_OBJ)
    return {
        "n_frames": len(frames),
        "bbox_traj_line": bbox_traj_line(frames),
        "frames": frames,
    }


def object_to_track(line: str, shot: dict, *, n_frames: int = 16,
                    bible: dict | None = None,
                    reach_fraction: float = DEFAULT_REACH_FRACTION) -> list[dict]:
    """Project a SUBJECT's bbox trajectory onto our metric frame, anchored at the
    subject (world origin, at the shot's aim height) — the object analogue of
    lamp_to_track. Returns N keyframes {frame_index, t, position[m],
    rotation_deg:[0,0,0], lens_mm:0} so the 3D viewer can draw it as a second
    polyline alongside the camera path. Objects translate, hence zero rotation."""
    from pace_core.camera.camera_planner import plan_camera

    start = plan_camera(shot, bible=bible)
    dist = float(start.get("subject_distance_m", 2.0))
    subj_z = float((start.get("target_pos") or [0.0, 0.0, 1.65])[2])
    scale = (dist * reach_fraction) / CUBE_REACH

    frames = compile_object_dsl(line)["frames"]
    metric: list[tuple[list, list]] = []
    for f in frames:
        d_right = f["pos_cube"][0] - 256
        d_up = f["pos_cube"][1] - 256
        d_depth = f["pos_cube"][2] - 256
        # Same screen→world axis map as the camera: right→+Y, up→+Z, depth→+X,
        # but anchored at the subject (origin) instead of the camera.
        pos = [0.0 + d_depth * scale, 0.0 + d_right * scale, subj_z + d_up * scale]
        metric.append((pos, [0.0, 0.0, 0.0]))
    return _resample_track(metric, n_frames, lens=0)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Compile a LAMP DSL line → trajectory.")
    ap.add_argument("dsl", help="24-token DSL line")
    ap.add_argument("--cam-json", action="store_true", help="emit cam_json instead of summary")
    args = ap.parse_args()
    res = compile_dsl(args.dsl)
    if args.cam_json:
        print(json.dumps(cam_json(res["frames"]), indent=2))
    else:
        print(f"{res['n_frames']} frames")
        for f in (res["frames"][0], res["frames"][len(res['frames']) // 2], res["frames"][-1]):
            print(f"  f{f['frame_index']:>2}  pos_cube={f['pos_cube']}  euler_deg="
                  f"[{f['euler_deg'][0]:.1f},{f['euler_deg'][1]:.1f},{f['euler_deg'][2]:.1f}]")
