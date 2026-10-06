"""The reward orders decisions, which conformance alone was measured not to.

Every weighting of the gate's own clauses turned out non-monotone in how bad
a staging choice was, and several rose as it got worse: a worse declaration is
re-solved into a different self-consistent frame, and a self-consistency check
cannot see that it came from a worse decision. These pin the consequences --
conformance is kept, bounded and droppable; agreement with the human record is
what supplies the order; and neither silently stands in for the other.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.qc.reward import (  # noqa: E402
    CONSTANT, DISCRIMINATIVE, SAME_DISTANCE_PENALTY, SIZE_ORDER, _size_distance,
    agreement, conformance, reward,
)


@dataclass
class C:
    name: str
    ok: bool | None
    value: float | None = None
    threshold: float | None = None


def test_only_discriminative_clauses_reach_conformance():
    """A constant contributes no gradient, so it is not in the denominator."""
    cl = [C("subject_occlusion", True), C("required_entity_visibility", True)]
    score, names = conformance(cl)
    assert names == ("subject_occlusion",)
    assert score == 1.0


def test_the_two_clause_sets_do_not_overlap():
    assert not (DISCRIMINATIVE & CONSTANT)


def test_an_unmeasurable_panel_is_undefined_not_zero():
    """A single cannot be measured for occlusion. Scoring that zero penalises
    the shot's composition, which the policy did not choose, and the group
    baseline would turn it into a negative advantage."""
    score, names = conformance([C("subject_occlusion", None),
                                C("screen_order", None)])
    assert score is None and names == ()
    assert reward([C("subject_occlusion", None)]).reward is None


def test_a_reported_residual_is_graded_against_its_own_tolerance():
    half = conformance([C("screen_order", False, value=0.5, threshold=1.0)])[0]
    none = conformance([C("screen_order", False, value=1.0, threshold=1.0)])[0]
    assert half == pytest.approx(0.5)
    assert none == pytest.approx(0.0)


def test_agreement_grades_shot_size_by_distance():
    """Proposing a medium where a person chose a close-up is wrong; proposing
    an establishing shot is more wrong, and the reward should say so."""
    near = agreement({"shot_size": "medium_close_up"}, {"shot_size": "close_up"})
    far = agreement({"shot_size": "establishing"}, {"shot_size": "close_up"})
    exact = agreement({"shot_size": "close_up"}, {"shot_size": "close_up"})
    assert exact == 1.0
    assert near > far


def test_agreement_scores_cast_by_overlap():
    both = agreement({"cast": ["proog", "emo"]}, {"cast": ["proog", "emo"]})
    half = agreement({"cast": ["proog"]}, {"cast": ["proog", "emo"]})
    extra = agreement({"cast": ["proog", "emo", "harmi"]}, {"cast": ["proog", "emo"]})
    assert both == 1.0
    assert half == pytest.approx(0.5)
    assert extra == pytest.approx(2 / 3)


def test_a_blank_cell_in_the_human_record_neither_rewards_nor_penalises():
    """A breakdown that left shot size out should not make every proposal wrong."""
    assert agreement({"shot_size": "medium"}, {}) is None
    assert agreement({"shot_size": "medium"}, {"framing": "two_shot"}) == 0.0


def test_conformance_alone_is_reported_as_such():
    """With no human record the reward falls back to conformance, and says so,
    because conformance was measured not to order decisions."""
    p = reward([C("subject_occlusion", True)])
    assert p.reward == 1.0 and p.agreement is None
    assert "not to order decisions" in p.note


def test_agreement_moves_the_reward_when_conformance_cannot():
    """The case the measurements found: the frame is self-consistent either
    way, so only the human comparison separates the two declarations."""
    cl = [C("subject_occlusion", True), C("screen_order", True)]
    good = reward(cl, {"shot_size": "close_up"}, {"shot_size": "close_up"})
    bad = reward(cl, {"shot_size": "establishing"}, {"shot_size": "close_up"})
    assert good.conformance == bad.conformance == 1.0
    assert good.reward > bad.reward


def test_beta_zero_reproduces_the_reward_that_does_not_order():
    cl = [C("subject_occlusion", True)]
    a = reward(cl, {"shot_size": "close_up"}, {"shot_size": "close_up"}, beta=0.0)
    b = reward(cl, {"shot_size": "establishing"}, {"shot_size": "close_up"}, beta=0.0)
    assert a.reward == b.reward


def test_a_rejected_declaration_is_penalised_not_scored_low():
    p = reward([C("subject_occlusion", True)], rejected=True)
    assert p.reward == -1.0 and p.rejected


def test_the_reward_stays_in_the_unit_interval():
    cl = [C("subject_occlusion", True), C("screen_order", True)]
    best = reward(cl, {"shot_size": "close_up", "framing": "single",
                       "cast": ["a"]},
                  {"shot_size": "close_up", "framing": "single", "cast": ["a"]})
    # The worst case is not zero and should not be: a size one step from the
    # human choice is wrong, and `master` eight steps away is more wrong, so
    # the grading has to leave room between them. Only a cast with no overlap,
    # the wrong framing and the furthest size reaches the floor.
    worst = reward([C("subject_occlusion", False), C("screen_order", False)],
                   {"shot_size": "establishing", "framing": "crowd", "cast": ["z"]},
                   {"shot_size": "extreme_close_up", "framing": "single",
                    "cast": ["a"]})
    assert best.reward == pytest.approx(1.0)
    assert worst.reward == pytest.approx(0.0)
    assert 0.0 <= worst.reward < best.reward <= 1.0


def test_a_size_one_step_off_scores_above_the_furthest_one():
    """What the floor test above leaves room for."""
    cl = [C("subject_occlusion", False)]
    near = reward(cl, {"shot_size": "close_up"}, {"shot_size": "extreme_close_up"})
    far = reward(cl, {"shot_size": "establishing"}, {"shot_size": "extreme_close_up"})
    assert near.reward > far.reward == pytest.approx(0.0)


def test_the_widest_size_is_establishing_and_not_master():
    """`SIZE_ORDER` lists master last; the camera does not put it there.

    A master is a full-scene coverage shot, which the planner places at the
    same distance as `full`. Scoring by distance rather than by position in
    the name list is what corrects this, so an establishing shot is further
    from a close-up than a master is -- and the two tests above depend on it.
    """
    assert _size_distance("extreme_close_up", "establishing") == pytest.approx(1.0)
    assert _size_distance("extreme_close_up", "master") < 1.0
    assert SIZE_ORDER[-1] == "master"      # the list still says otherwise


def test_two_sizes_at_one_camera_distance_are_close_but_not_equal():
    """`medium` and `medium_full` resolve to the same distance in the planner.

    They are the same framing under two names, so the credit is nearly full;
    scoring it as exactly full would make a wrong answer indistinguishable
    from the right one and put the no-gradient problem back."""
    d = _size_distance("medium", "medium_full")
    assert d == pytest.approx(SAME_DISTANCE_PENALTY)
    assert 0.0 < d < 0.05
    assert _size_distance("medium", "medium") == 0.0


def test_the_size_distance_is_a_ratio_not_a_step_count():
    """Equal steps along the name list are not equal changes of framing."""
    tight = _size_distance("extreme_close_up", "close_up")   # 0.4 m -> 0.7 m
    wide = _size_distance("medium_full", "full")              # 2.0 m -> 3.0 m
    assert tight > wide          # one step each, different cost
    # Halving the distance costs the same wherever it starts.
    assert _size_distance("close_up", "medium_close_up") == pytest.approx(
        _size_distance("full", "wide"), abs=0.02)


def test_the_declared_mark_clause_grades_and_does_not_dilute():
    """It carries a residual, so it must be credited by distance not verdict.

    And a panel with no declared mark must leave conformance exactly where it
    was: the clause is undefined there, and an undefined clause that scored
    zero would teach a policy that an unmeasurable panel is a bad one."""
    from pace_core.qc.reward import DISCRIMINATIVE
    assert "subject_on_declared_mark" in DISCRIMINATIVE

    # Graded at every distance, not just inside the tolerance. The linear
    # credit the other residual clauses use reaches zero AT the tolerance and
    # stays there, which for a distance in metres means half a metre off and
    # thirty metres off pay the same -- the no-gradient condition again, in the
    # one clause added to give reinforcement something to learn from.
    on_mark = conformance([C("subject_on_declared_mark", True,
                             value=0.0, threshold=0.25)])[0]
    at_tol = conformance([C("subject_on_declared_mark", False,
                            value=0.25, threshold=0.25)])[0]
    half_off = conformance([C("subject_on_declared_mark", False,
                              value=0.5, threshold=0.25)])[0]
    way_off = conformance([C("subject_on_declared_mark", False,
                             value=2.5, threshold=0.25)])[0]
    assert on_mark == pytest.approx(1.0)
    assert at_tol == pytest.approx(0.5)
    assert on_mark > at_tol > half_off > way_off > 0.0

    others = [C("screen_order", True), C("subject_occlusion", True)]
    with_mark = conformance(others + [C("subject_on_declared_mark", None)])
    without = conformance(others)
    assert with_mark[0] == without[0]
    assert "subject_on_declared_mark" not in with_mark[1]
