"""frame_rig — author a camera move the way a cinematographer does: compose the
frames, then choose the rig that carries the camera between them.

`frame_subject` composes one frame on a subject from a shot size, an angle and
a screen placement. Shot sizes use the bands the greybox stage itself fits
(`panel_greybox.SHOT_SIZE_FRAMING`), so a "medium close-up" composed here is
the frame the greybox would stage, at whatever lens is chosen.

`rig_track` turns timed frames into a camera_planner track. Each frame after
the first names the rig and ease that brought the camera to it:

    tripod     pans, tilts and zooms from the previous frame's position
    dolly      a straight track between the two positions
    jib        an arc in the move's vertical plane, bowed away from an arm
               base below the lower end
    orbit      around the point both frames look at, holding the composition
    steadicam  a smooth free path through neighbouring free frames
    handheld   the steadicam path with seeded handheld noise

A frame may name the `subject` point it is composed on (frame_subject
returns it). Orbits circle it and lens changes are paced by it, following it
if the two frames name different points; without one, the point where both
optical axes meet stands in. Rotation slerps between frames (orbit re-aims at
the subject instead, carrying each frame's composition). A lens change keeps
the subject's size changing evenly, so a dolly between two equally sized
frames is a true dolly zoom; an unchanged lens is never altered.
"""
from __future__ import annotations

import bisect
import math
from typing import Optional, Sequence

from pace_core.camera.camera_planner import ANGLE_OFFSETS, _euler_xyz_to_matrix, _matrix_to_euler_xyz
from pace_core.camera.camera_skills import (
    _SHAKE_RAMP_S, _ease, _shake_offsets, _shake_phases, _smoothed_levels, lens_value, look_at)
from pace_core.camera.keyframe_track import (
    _centripetal, _euler_from_quat, _normalise, _quat_from_euler, _slerp, _unwrap, _vector)
from pace_core.node.panel_greybox import SHOT_SIZE_FRAMING
from pace_core.setup.composition_solver import SENSOR_WIDTH_MM, camera_axes, solve_rotation_for_screen_position

RIGS = ("tripod", "dolly", "jib", "orbit", "steadicam", "handheld")
EASES = {"feather": "ease_in_out", "feather_in": "ease_in", "feather_out": "ease_out", "linear": "linear"}
SHOT_SIZES = tuple(sorted(SHOT_SIZE_FRAMING, key=lambda s: SHOT_SIZE_FRAMING[s]))
ANGLES = ("eye_level", "low_angle", "high_angle", "low_position", "overhead", "top_down")
_FREE = {"steadicam", "handheld"}
_ASPECT = 16 / 9
_TRIPOD_TOLERANCE_M = 0.005


def _half_vfov_tan(lens_mm: float) -> float:
    return (SENSOR_WIDTH_MM / 2.0 / lens_mm) / _ASPECT


def frame_subject(subject_base: Sequence[float], subject_height: float, *,
                  shot_size: str, bearing_deg: float, angle: str = "eye_level",
                  lens_mm: float = 35.0, screen_x: float = 0.5, screen_y: float = 0.5,
                  roll_deg: float = 0.0) -> dict:
    """One composed pose on a standing subject.

    `subject_base` is the subject's footprint centre on the ground and
    `subject_height` its height. The camera stands at `bearing_deg` (azimuth
    from +X, as seen from the subject), at the distance that makes the shot
    size's band of the body fill its share of the frame at `lens_mm`; `angle`
    raises or lowers it around the band's centre. `screen_x/y` place that
    centre in the frame (0..1, top-left origin) and `roll_deg` is a Dutch
    roll about the lens axis.
    """
    if shot_size not in SHOT_SIZE_FRAMING:
        raise ValueError(f"unknown shot size {shot_size!r}; use one of {SHOT_SIZES}")
    if angle not in ANGLE_OFFSETS:
        raise ValueError(f"unknown angle {angle!r}; use one of {ANGLES}")
    if not (subject_height > 0 and lens_mm > 0):
        raise ValueError("subject_height and lens_mm must be positive")
    band, margin = SHOT_SIZE_FRAMING[shot_size]
    span = band * subject_height
    aim = [float(subject_base[0]), float(subject_base[1]),
           float(subject_base[2]) + subject_height - span / 2.0]
    distance = span * margin / 2.0 / _half_vfov_tan(lens_mm)
    el, az = math.radians(ANGLE_OFFSETS[angle][0]), math.radians(bearing_deg)
    cam = [aim[0] + distance * math.cos(el) * math.cos(az),
           aim[1] + distance * math.cos(el) * math.sin(az),
           aim[2] + distance * math.sin(el)]
    rot = look_at(cam, aim)
    if abs(screen_x - 0.5) > 1e-9 or abs(screen_y - 0.5) > 1e-9:
        solved = solve_rotation_for_screen_position(cam, rot, lens_mm, aim, screen_x, screen_y)
        if not solved["converged"]:
            raise ValueError(f"screen placement ({screen_x}, {screen_y}) is unreachable")
        rot = solved["rotation_deg"]
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        m = _euler_xyz_to_matrix(rot)
        rot = _matrix_to_euler_xyz([[m[i][0] * c + m[i][1] * s, -m[i][0] * s + m[i][1] * c, m[i][2]]
                                    for i in range(3)])
    return {"position": [round(v, 4) for v in cam], "rotation_deg": [round(v, 3) for v in rot],
            "lens_mm": lens_value(lens_mm), "subject": [round(v, 4) for v in aim]}


def _convergence(pa, ra, pb, rb) -> Optional[list[float]]:
    """Midpoint of the closest approach of the two optical axes, if both
    cameras look towards it; None for parallel or diverging axes."""
    fa, fb = camera_axes(ra)[2], camera_axes(rb)[2]
    w = [pa[i] - pb[i] for i in range(3)]
    b = sum(fa[i] * fb[i] for i in range(3))
    d, e = sum(fa[i] * w[i] for i in range(3)), sum(fb[i] * w[i] for i in range(3))
    den = 1.0 - b * b
    if den < 1e-6:
        return None
    s, t = (b * e - d) / den, (e - b * d) / den
    if s <= 1e-6 or t <= 1e-6:
        return None
    return [(pa[i] + s * fa[i] + pb[i] + t * fb[i]) / 2.0 for i in range(3)]


def _qmul(a, b) -> list[float]:
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def _qconj(q) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _jib(pa, pb, u: float) -> list[float]:
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:                                     # straight boom up or down
        return [pa[i] + (pb[i] - pa[i]) * u for i in range(3)]
    zp = min(pa[2], pb[2]) - max(1.0, d)             # arm base below the lower end
    px = (d * d + (pb[2] - zp) ** 2 - (pa[2] - zp) ** 2) / (2 * d)
    arm = math.hypot(px, pa[2] - zp)
    ta, tb = math.atan2(pa[2] - zp, -px), math.atan2(pb[2] - zp, d - px)
    theta = ta + (tb - ta) * u
    along = px + arm * math.cos(theta)
    return [pa[0] + dx / d * along, pa[1] + dy / d * along, zp + arm * math.sin(theta)]


def _orbit(pa, pb, ca, cb, c, u: float) -> list[float]:
    """Polar blend around the subject: ca/cb at the ends, c at this instant."""
    ra, rb = math.hypot(pa[0] - ca[0], pa[1] - ca[1]), math.hypot(pb[0] - cb[0], pb[1] - cb[1])
    aa = math.atan2(pa[1] - ca[1], pa[0] - ca[0])
    ab = math.atan2(pb[1] - cb[1], pb[0] - cb[0])
    sweep = (ab - aa + math.pi) % (2 * math.pi) - math.pi
    r, a = ra + (rb - ra) * u, aa + sweep * u
    za, zb = pa[2] - ca[2], pb[2] - cb[2]
    return [c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2] + za + (zb - za) * u]


def _subjects(a: dict, b: dict) -> tuple[Optional[list[float]], Optional[list[float]]]:
    """The points a segment is composed on: the frames' own, else where their axes meet."""
    if a.get("subject") is not None and b.get("subject") is not None:
        return a["subject"], b["subject"]
    meet = _convergence(a["position"], a["rotation_deg"], b["position"], b["rotation_deg"])
    return meet, meet


def _frames(frames: Sequence[dict]) -> list[dict]:
    out = _normalise(frames, extra=("rig", "ease", "subject"))
    for n, frame in enumerate(out):
        if frame.get("subject") is not None:
            frame["subject"] = _vector(frame["subject"], f"frame {n} subject")
        frame.setdefault("rig", "steadicam")
        frame.setdefault("ease", "feather")
        if frame["rig"] not in RIGS:
            raise ValueError(f"frame {n}: rig must be one of {RIGS}")
        if frame["ease"] not in EASES:
            raise ValueError(f"frame {n}: ease must be one of {tuple(EASES)}")
    for n in range(1, len(out)):
        a, b = out[n - 1], out[n]
        if b["rig"] == "tripod" and math.dist(a["position"], b["position"]) > _TRIPOD_TOLERANCE_M:
            raise ValueError(f"frame {n}: a tripod cannot travel; shoot it from the previous "
                             f"frame's position or choose another rig")
        if b["rig"] == "orbit" and _subjects(a, b)[0] is None:
            raise ValueError(f"frame {n}: an orbit needs both frames to look at the same subject")
    return out


def rig_track(frames: Sequence[dict], *, fps: float, n_frames: Optional[int] = None,
              seed: int = 0) -> list[dict]:
    """Timed frames ({time, position, rotation_deg, lens_mm, rig, ease, subject?}) → track.

    Without `n_frames` the track spans the frames' own duration at `fps`;
    with it the frames are retimed proportionally to exactly that length.
    `seed` fixes the handheld noise.
    """
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be positive")
    if n_frames is not None and n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    keys = _frames(frames)
    times = [k["time"] for k in keys]
    start, span = times[0], times[-1] - times[0]
    n = n_frames or round(span * fps) + 1
    quats = [_quat_from_euler(k["rotation_deg"]) for k in keys]
    last = len(keys) - 1

    raw: list[tuple[list[float], list[float], float, float]] = []   # position, rotation, lens, shake
    previous = keys[0]["rotation_deg"]
    for f in range(n):
        t = start + span * (f / (n - 1) if n > 1 else 0.0)
        if last == 0:
            k = keys[0]
            raw.append((k["position"], k["rotation_deg"], k["lens_mm"], 0.0))
            continue
        i = min(max(bisect.bisect_right(times, t) - 1, 0), last - 1)
        j = i + 1
        a, b = keys[i], keys[j]
        u = _ease(min(max((t - a["time"]) / (b["time"] - a["time"]), 0.0), 1.0), EASES[b["ease"]])
        pa, pb, rig = a["position"], b["position"], b["rig"]
        ca, cb = _subjects(a, b)
        pivot = [ca[k] + (cb[k] - ca[k]) * u for k in range(3)] if ca else None

        if rig == "tripod":
            position = list(pa)
        elif rig == "dolly":
            position = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
        elif rig == "jib":
            position = _jib(pa, pb, u)
        elif rig == "orbit":
            position = _orbit(pa, pb, ca, cb, pivot, u)
        else:
            free_before = i > 0 and a["rig"] in _FREE
            free_after = j < last and keys[j + 1]["rig"] in _FREE
            before = keys[i - 1]["position"] if free_before else [2 * x - y for x, y in zip(pa, pb)]
            after = keys[j + 1]["position"] if free_after else [2 * y - x for x, y in zip(pa, pb)]
            position = _centripetal(before, pa, pb, after, u)

        if rig == "orbit":
            # Re-aim at the subject, carrying each frame's own offset from it.
            def look(p, c):
                return _quat_from_euler(look_at(p, c))
            da = _qmul(_qconj(look(pa, ca)), quats[i])
            db = _qmul(_qconj(look(pb, cb)), quats[j])
            quat = _qmul(look(position, pivot), _slerp(da, db, u))
        else:
            quat = _slerp(quats[i], quats[j], u)
        rotation = _unwrap(_euler_from_quat(quat), previous)
        previous = rotation

        la, lb = a["lens_mm"], b["lens_mm"]
        if la == lb:
            lens = la
        elif pivot:
            sa, sb = math.dist(pa, ca) / la, math.dist(pb, cb) / lb
            lens = math.dist(position, pivot) / (sa + (sb - sa) * u)
        else:
            lens = la + (lb - la) * u
        raw.append((position, rotation, lens, 1.0 if rig == "handheld" else 0.0))

    levels = _smoothed_levels([r[3] for r in raw], max(1, round(_SHAKE_RAMP_S * fps)))
    phases = _shake_phases(seed)
    track = []
    for f, ((position, rotation, lens, _), level) in enumerate(zip(raw, levels)):
        if level > 1e-9:
            dp, dr = _shake_offsets(f / fps, phases)
            position = [position[k] + level * dp[k] for k in range(3)]
            rotation = [rotation[k] + level * dr[k] for k in range(3)]
        track.append({"frame_index": f, "position": [round(v, 4) for v in position],
                      "rotation_deg": [round(v, 3) for v in rotation], "lens_mm": lens_value(lens)})
    return track
