"""Summarise a measured staging path into a declared `Blocking`.

Staging arrives as samples -- from a production scene that recorded it, or from
a measurement of the finished frames -- and a declaration is coarser than that
on purpose: `Blocking` says where someone stands and the legs they walk, not
where they were on every frame. The dense samples stay reachable through
`subject_path`; this derives the readable part beside them.

The rules are the ones a blocking sheet already uses. Someone who barely leaves
their mark holds it, however much their weight shifts. A move is one leg while
it keeps its heading, and a new leg begins where the heading breaks -- which is
what makes a path around a table two legs rather than forty samples.

No Blender and no I/O, so the thresholds can be tested against paths written by
hand.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from ..types_v1 import Blocking, BlockingMove

# A mark is a place to stand, not a point: weight shifts, breathing and a rig's
# own float all move a body a little without anybody crossing the floor.
STILL_M = 0.15
# Heading change that ends a leg. Below this a path is still "the same walk";
# above it, a blocking sheet would draw a second arrow.
TURN_DEG = 25.0


def _xy(sample) -> tuple[float, float]:
    p = sample["xy"] if "xy" in sample else sample["world_xy"]
    return float(p[0]), float(p[1])


def _extent(samples: Sequence[dict]) -> float:
    """The diagonal of the ground the subject covers."""
    xs = [_xy(s)[0] for s in samples]
    ys = [_xy(s)[1] for s in samples]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _heading(a: tuple[float, float], b: tuple[float, float]) -> Optional[float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _angle_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _seconds(samples: Sequence[dict], i: int, j: int, fps: Optional[float]) -> Optional[float]:
    if not fps or "f" not in samples[i] or "f" not in samples[j]:
        return None
    return round((samples[j]["f"] - samples[i]["f"]) / float(fps), 3)


def legs_of(samples: Sequence[dict], *, fps: Optional[float] = None,
            still_m: float = STILL_M, turn_deg: float = TURN_DEG) -> list[BlockingMove]:
    """Cut the path where its heading breaks; each run between cuts is a leg."""
    if len(samples) < 2:
        return []
    legs: list[BlockingMove] = []
    start = 0
    run_heading: Optional[float] = None
    for i in range(1, len(samples)):
        h = _heading(_xy(samples[i - 1]), _xy(samples[i]))
        if h is None:
            continue                       # standing still inside a move; not a turn
        if run_heading is None:
            run_heading = h
            continue
        if _angle_gap(h, run_heading) > turn_deg:
            if math.dist(_xy(samples[start]), _xy(samples[i - 1])) >= still_m:
                legs.append(_leg(samples, start, i - 1, fps))
            start = i - 1
            run_heading = h
    if math.dist(_xy(samples[start]), _xy(samples[-1])) >= still_m:
        legs.append(_leg(samples, start, len(samples) - 1, fps))
    return legs


def _leg(samples: Sequence[dict], i: int, j: int, fps: Optional[float]) -> BlockingMove:
    x, y = _xy(samples[j])
    end = samples[j]
    return BlockingMove(
        to_xy=[round(x, 4), round(y, 4)],
        z=None if end.get("z") is None else round(float(end["z"]), 4),
        facing_deg=None if end.get("facing_deg") is None else round(float(end["facing_deg"]), 3),
        duration_s=_seconds(samples, i, j, fps),
    )


def blocking_from_path(samples: Iterable[dict], *, fps: Optional[float] = None,
                       still_m: float = STILL_M, turn_deg: float = TURN_DEG,
                       subject_path: Optional[str] = None) -> Blocking:
    """The declaration these samples support: the mark, and the legs off it."""
    s = list(samples)
    if not s:
        raise ValueError("no samples to summarise")
    x, y = _xy(s[0])
    mark = [round(x, 4), round(y, 4)]
    z = round(float(s[0].get("z") or 0.0), 4)
    facing = s[0].get("facing_deg")
    facing = None if facing is None else round(float(facing), 3)

    if _extent(s) < still_m:
        return Blocking(world_xy=mark, z=z, facing_deg=facing, static=True,
                        subject_path=subject_path)

    legs = legs_of(s, fps=fps, still_m=still_m, turn_deg=turn_deg)
    if not legs and subject_path is None:
        # The subject covers ground but no run of it survives the leg rules;
        # rather than call that standing still, keep the whole move as one leg.
        legs = [_leg(s, 0, len(s) - 1, fps)]
    return Blocking(world_xy=mark, z=z, facing_deg=facing, static=False,
                    moves=legs, subject_path=subject_path)
