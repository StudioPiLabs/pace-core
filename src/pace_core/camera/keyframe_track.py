"""keyframe_track — interpolate authored camera keys into a camera_planner track.

Keys come from a person placing or flying a camera through a 3D stage: a few
hand-set poses, or a dense recorded take. Each key is

    {"time": seconds, "position": [x, y, z], "rotation_deg": [rx, ry, rz], "lens_mm": f}

in the camera_planner convention (Blender Z-up metres, XYZ Euler, M = Rz Ry Rx).
`keyframes_to_track` resamples them onto a uniform frame grid and emits the
same `{frame_index, position, rotation_deg, lens_mm}` entries compile_shot does.

Position uses centripetal Catmull-Rom ("smooth": passes through every key with
no cusps or overshoot loops on uneven spacing) or straight segments ("linear",
right for dense takes). Rotation always slerps between neighbouring keys, so a
camera never spins the long way round; lens interpolates linearly.
"""
from __future__ import annotations

import bisect
import math
from typing import Optional, Sequence

from pace_core.camera.camera_planner import _euler_xyz_to_matrix, _matrix_to_euler_xyz
from pace_core.camera.camera_skills import _ease, lens_value

INTERPOLATIONS = ("smooth", "linear")


def _vector(value, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be three numbers")
    out = [float(v) for v in value]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out


def _normalise(keys: Sequence[dict]) -> list[dict]:
    if not keys:
        raise ValueError("no camera keys supplied")
    out = []
    for n, key in enumerate(keys):
        time = float(key.get("time", math.nan))
        lens = float(key.get("lens_mm", 35.0))
        if not math.isfinite(time) or time < 0:
            raise ValueError(f"key {n}: time must be a non-negative number of seconds")
        if not math.isfinite(lens) or lens <= 0:
            raise ValueError(f"key {n}: lens_mm must be positive")
        out.append({"time": time, "lens_mm": lens,
                    "position": _vector(key.get("position"), f"key {n} position"),
                    "rotation_deg": _vector(key.get("rotation_deg"), f"key {n} rotation_deg")})
    out.sort(key=lambda k: k["time"])
    if any(b["time"] - a["time"] <= 1e-9 for a, b in zip(out, out[1:])):
        raise ValueError("camera keys need distinct times")
    return out


def _quat_from_euler(rot_deg: Sequence[float]) -> list[float]:
    """[w, x, y, z] for the camera_planner Euler (Shepperd's method)."""
    m = _euler_xyz_to_matrix(rot_deg)
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def _euler_from_quat(q: Sequence[float]) -> list[float]:
    w, x, y, z = q
    m = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return _matrix_to_euler_xyz(m)


def _slerp(a: Sequence[float], b: Sequence[float], u: float) -> list[float]:
    dot = sum(p * q for p, q in zip(a, b))
    if dot < 0:                                   # shortest arc
        b, dot = [-c for c in b], -dot
    if dot > 0.9995:
        q = [p + (r - p) * u for p, r in zip(a, b)]
    else:
        theta = math.acos(min(1.0, dot))
        sa, sb = math.sin((1 - u) * theta), math.sin(u * theta)
        q = [(p * sa + r * sb) / math.sin(theta) for p, r in zip(a, b)]
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def _centripetal(p0, p1, p2, p3, u: float) -> list[float]:
    """Point at u in [0, 1] on the centripetal Catmull-Rom span p1 -> p2
    (Barry-Goldman pyramid, knot spacing |dp|^0.5)."""
    if math.dist(p1, p2) < 1e-9:
        return list(p1)
    k0 = 0.0
    k1 = k0 + max(math.dist(p0, p1), 1e-9) ** 0.5
    k2 = k1 + math.dist(p1, p2) ** 0.5
    k3 = k2 + max(math.dist(p2, p3), 1e-9) ** 0.5
    k = k1 + (k2 - k1) * u

    def mix(a, b, ka, kb):
        return [(kb - k) / (kb - ka) * x + (k - ka) / (kb - ka) * y for x, y in zip(a, b)]
    a1, a2, a3 = mix(p0, p1, k0, k1), mix(p1, p2, k1, k2), mix(p2, p3, k2, k3)
    return mix(mix(a1, a2, k0, k2), mix(a2, a3, k1, k3), k1, k2)


def _unwrap(angles: Sequence[float], reference: Sequence[float]) -> list[float]:
    """Each angle shifted by whole turns to lie nearest its reference."""
    return [a + 360.0 * round((r - a) / 360.0) for a, r in zip(angles, reference)]


def keyframes_to_track(keys: Sequence[dict], *, fps: float,
                       n_frames: Optional[int] = None,
                       interpolation: str = "smooth",
                       ease: str = "linear") -> list[dict]:
    """Resample camera keys onto a uniform frame grid.

    Without `n_frames` the track spans the keys' own duration at `fps`
    (round(duration * fps) + 1 frames, both end keys included); with it the
    keys are retimed proportionally to exactly that many frames. `ease`
    reshapes global time (camera_skills easing names), so two keys can start
    and stop gently.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be positive")
    if n_frames is not None and n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    keys = _normalise(keys)
    times = [k["time"] for k in keys]
    start, span = times[0], times[-1] - times[0]
    n = n_frames or round(span * fps) + 1
    quats = [_quat_from_euler(k["rotation_deg"]) for k in keys]
    points = [k["position"] for k in keys]
    last = len(keys) - 1

    track: list[dict] = []
    previous = keys[0]["rotation_deg"]
    for f in range(n):
        u = _ease(f / (n - 1), ease) if n > 1 else 0.0
        t = start + span * u
        i = min(max(bisect.bisect_right(times, t) - 1, 0), max(last - 1, 0))
        if last == 0:
            local, j = 0.0, 0
        else:
            j = i + 1
            local = min(max((t - times[i]) / (times[j] - times[i]), 0.0), 1.0)
        a, b = points[i], points[j]
        if interpolation == "linear" or last == 0:
            position = [x + (y - x) * local for x, y in zip(a, b)]
        else:
            before = points[i - 1] if i > 0 else [2 * x - y for x, y in zip(a, b)]
            after = points[j + 1] if j < last else [2 * y - x for x, y in zip(a, b)]
            position = _centripetal(before, a, b, after, local)
        rotation = _unwrap(_euler_from_quat(_slerp(quats[i], quats[j], local)), previous)
        previous = rotation
        lens = keys[i]["lens_mm"] + (keys[j]["lens_mm"] - keys[i]["lens_mm"]) * local
        track.append({"frame_index": f,
                      "position": [round(c, 4) for c in position],
                      "rotation_deg": [round(c, 3) for c in rotation],
                      "lens_mm": lens_value(lens)})
    return track
