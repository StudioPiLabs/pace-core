"""How long a shot runs, derived from the beats inside it.

`Shot` carries no duration of its own. The only timing in the schema is
`events.actions[].duration_hint_s`, one level down, documented as "rough
seconds -- informs video length, not literal". That is what this reads.

The rule has to answer one modelling question: do a shot's actions run one
after another or at the same time? The schema already says. An action with
`background=True` "happens behind the focal subject", so it is concurrent
and bounds the shot from below rather than adding to it. Everything else is
a beat inside one continuous camera run, so those add up.

Kept as a pure function with no I/O because two callers need the same
answer -- the shot route that reports it and anything that later renders
from it -- and a shot whose length depends on who asked is worse than a
shot with no length at all.
"""
from __future__ import annotations

# Current video backends are trained on clips of a few seconds up to about
# 15 s. A hint outside this band is a typo, not an intent.
MIN_SECONDS = 1.0
MAX_SECONDS = 15.0
DEFAULT_SECONDS = 5.0
DEFAULT_FPS = 16
#: Deprecated alias of DEFAULT_FPS; kept for existing importers.
VACE_FPS = DEFAULT_FPS


def shot_duration_s(shot: dict) -> tuple[float, str]:
    """(seconds, source) for one shot dict.

    source is "actions" when the number came from the shot's own beats and
    "default" when nothing in the shot had a hint -- the caller can then say
    "5s because nobody said" rather than presenting a guess as a fact.
    """
    actions = ((shot or {}).get("events") or {}).get("actions") or []
    seq, concurrent = 0.0, 0.0
    for a in actions:
        hint = (a or {}).get("duration_hint_s")
        if hint is None:
            continue
        try:
            secs = float(hint)
        except (TypeError, ValueError):
            continue                       # a hint that isn't a number isn't a hint
        if secs <= 0:
            continue
        if (a or {}).get("background"):
            concurrent = max(concurrent, secs)
        else:
            seq += secs
    total = max(seq, concurrent)
    if total <= 0:
        return DEFAULT_SECONDS, "default"
    return round(min(max(total, MIN_SECONDS), MAX_SECONDS), 2), "actions"


def frame_count(seconds: float, fps: int = DEFAULT_FPS, quantum: int = 4,
                offset: int = 1) -> int:
    """Seconds → a frame count of the form quantum*n + offset, rounded up.

    Many video models' temporal latent grids want 4n+1 frames (the default).
    The floor is one quantum plus the offset.
    """
    if quantum < 1:
        raise ValueError("quantum must be >= 1")
    n = max(quantum + offset, int(round(seconds * fps)))
    return n + (quantum - (n - offset) % quantum) % quantum


def vace_frames(seconds: float, fps: int = DEFAULT_FPS) -> int:
    """Deprecated alias: `frame_count(seconds, fps)` (4n+1)."""
    return frame_count(seconds, fps)
