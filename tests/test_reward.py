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
    CONSTANT, DISCRIMINATIVE, agreement, conformance, reward,
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
                   {"shot_size": "master", "framing": "crowd", "cast": ["z"]},
                   {"shot_size": "extreme_close_up", "framing": "single",
                    "cast": ["a"]})
    assert best.reward == pytest.approx(1.0)
    assert worst.reward == pytest.approx(0.0)
    assert 0.0 <= worst.reward < best.reward <= 1.0


def test_a_size_one_step_off_scores_above_the_furthest_one():
    """What the floor test above leaves room for."""
    cl = [C("subject_occlusion", False)]
    near = reward(cl, {"shot_size": "close_up"}, {"shot_size": "extreme_close_up"})
    far = reward(cl, {"shot_size": "master"}, {"shot_size": "extreme_close_up"})
    assert near.reward > far.reward == pytest.approx(0.0)
