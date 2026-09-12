"""How long a shot runs, derived from the beats inside it.

`Shot` carries no duration of its own. The only timing in the schema is
`events.actions[].duration_hint_s`, one level down, documented as "rough
seconds -- informs Wan-I2V length, not literal". That is what this reads.

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

# No backend is trained far outside this band: Wan VACE is a few seconds,
# MiniMax H3's trained range is ~124-362 frames at 24fps (5-15s), LTX 2.5
# tops out around 15s. A hint outside it is a typo, not an intent.
MIN_SECONDS = 1.0
MAX_SECONDS = 15.0
DEFAULT_SECONDS = 5.0
VACE_FPS = 16


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


def vace_frames(seconds: float, fps: int = VACE_FPS) -> int:
    """Seconds → a frame count VACE accepts. Its latent grid wants 4n+1."""
    n = max(5, int(round(seconds * fps)))
    return n + (4 - (n - 1) % 4) % 4
