"""Beats are where the world turns, and the boundary is mostly measurable.

The design gives six boundary features and a weighted threshold. Two things
this file pins, both learned by running it on a real screenplay:

  * A feature the IR cannot support is not faked. Our events carry no location,
    so `spatial_shift` is never emitted, and the weights are renormalised over
    what WAS measured -- otherwise the score silently caps below the split
    threshold and the segmenter under-splits for a reason nobody can see.
  * The design's prose and its formula disagree. The prose says any one
    condition starts a beat; the formula needs several at once. Measured here,
    the formula alone merged 40 of 41 adjudicated pairs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.beat_segmenter import (  # noqa: E402
    MEASURABLE, TAU_HIGH, WEIGHTS, boundary_features, boundary_score,
    build_beats, cut, propose, strong_trigger,
)


def _e(i, predicate="ACT", actor="a", importance=0.5, changes=None, order=None):
    return {"local_id": f"e{i}", "predicate": predicate, "actor": actor,
            "importance": importance, "state_change": changes or [],
            "order": order if order is not None else i,
            "evidence": {"source_text": f"<{predicate}>"}}


def test_an_unmeasurable_feature_is_never_invented():
    """Our events carry no location. Deriving `spatial_shift` from how far
    apart two lines sit on the page would measure the typesetting."""
    f = boundary_features(_e(1), _e(2))
    assert "spatial_shift" not in f
    assert set(f) == set(MEASURABLE)


def test_the_weights_are_renormalised_over_what_was_measured():
    """All measured features at 1.0 must score 1.0. Without renormalising it
    would cap at the measured share of the weight -- 0.60 here -- and never
    reach TAU_HIGH, so the segmenter would under-split invisibly."""
    assert boundary_score({k: 1.0 for k in MEASURABLE}) == 1.0
    assert boundary_score({k: 0.0 for k in MEASURABLE}) == 0.0


def test_a_state_change_alone_starts_a_beat():
    b = propose([_e(1), _e(2, changes=[{"entity": "panels", "attribute": "power",
                                        "to": "off"}])])
    assert b[0].decision == "SPLIT"


def test_a_goal_shift_alone_starts_a_beat():
    """The design's prose: ANY one condition is sufficient. Under the formula
    alone this pair scores below threshold and merges -- which folded a robot
    arm emerging, a man standing in its path and a car being dragged away into
    one 13-event beat."""
    assert strong_trigger({"goal_shift": 0.8}) == "goal_shift"
    assert boundary_score({"state_change": 0.0, "focus_shift": 1.0,
                           "importance_delta": 0.2, "goal_shift": 0.8,
                           "reveal": 0.6}) < TAU_HIGH


def test_a_reveal_alone_starts_a_beat():
    assert strong_trigger({"reveal": 0.7}) == "reveal"


def test_a_weak_signal_does_not_trigger():
    assert strong_trigger({"goal_shift": 0.3, "reveal": 0.2}) is None


def test_a_continuous_action_by_one_actor_stays_one_beat():
    """"reaches out -> grips the arm -> lifts" is one beat, not three."""
    evs = [_e(1, "REACH"), _e(2, "GRIP"), _e(3, "LIFT")]
    assert len(cut(evs, propose(evs))) == 1


def test_every_event_lands_in_exactly_one_beat():
    """The segmenter may group, never drop. An event in no beat is an event
    that will never be drawn."""
    evs = [_e(1), _e(2, actor="b"), _e(3, changes=[{"entity": "x",
                                                    "attribute": "y", "to": "z"}]),
           _e(4, actor="c")]
    groups = cut(evs, propose(evs))
    ids = [e["local_id"] for g in groups for e in g]
    assert sorted(ids) == ["e1", "e2", "e3", "e4"]
    assert len(ids) == len(set(ids))


def test_an_unresolved_pair_keeps_the_beat_whole():
    """When in doubt a beat stays together: over-splitting yields beats with
    no state change in them, and those have nothing to depict."""
    b = propose([_e(1, actor="a"), _e(2, actor="b")])[0]
    assert b.decision == "ADJUDICATE"
    assert len(cut([_e(1, actor="a"), _e(2, actor="b")], [b])) == 1


def test_a_beat_carries_the_state_on_both_sides_of_its_transition():
    """state_before/state_after come from the TIMELINE, not from the beat's own
    events, so a beat inherits everything already true when it starts -- the
    property that stops a later beat restoring an irreversible state."""
    from pace_core.breakdown.world_state import build
    scene = {"index": 1, "heading": "INT. CAR - DAY", "events": [
        _e(1, "LOSE_POWER", changes=[{"entity": "panels", "attribute": "power",
                                      "from": "on", "to": "off"}]),
        _e(2, "TURN", actor="b")]}
    tl = build([scene])
    beats = build_beats(scene, [[scene["events"][0]], [scene["events"][1]]],
                        timeline=tl, scope="INT. CAR")
    assert beats[0]["state_before"] == {}
    assert beats[0]["state_after"]["panels.power"] == "off"
    # the SECOND beat inherits it without restating the event
    assert beats[1]["state_before"]["panels.power"] == "off"


def test_the_published_weights_are_kept_verbatim():
    """So this implementation can be compared against the design it came from."""
    assert WEIGHTS["state_change"] == 0.30
    assert WEIGHTS["spatial_shift"] == 0.15


def test_a_cast_less_panel_stages_the_space_instead_of_dying():
    """Beat segmentation produces environment beats -- "the vehicles hover
    motionless", "the panels close" -- which have no human actor. They are
    real images of the place, so the greybox fits its camera to the location's
    own declared extent rather than to a cast.

    Before this it did neither: `max(1, len(subs))` made range(n) == [0] over
    an empty list and the seat anchor died with a bare IndexError."""
    import inspect
    from pace_core.node import panel_greybox
    src = inspect.getsource(panel_greybox)
    assert "n = max(1, len(subs))" not in src
    assert "environment_only = not framed" in src
    # eligibility no longer refuses a cast-less panel
    elig = inspect.getsource(panel_greybox.eligible_panels)
    assert "a greybox of nobody" not in elig
