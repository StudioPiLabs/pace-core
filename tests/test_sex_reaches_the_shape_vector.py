"""A male character's body proxy is not the neutral mean body.

The gendered SMPL-X models are licence-gated separately from the neutral one
and are frequently absent, so a proxy gets built from `gender="neutral"`. That
was read here as "sex cannot be expressed", and it is wrong: the neutral shape
space is learned over both sexes, so the dimorphism is present as an
unlabelled direction. Measuring it -- sweep each coefficient, watch bust
protrusion and the shoulder-to-hip ratio -- puts it on beta[4].

The numbers in betas_for_subject's comment come from that sweep on the neutral
model; this test pins the behaviour that depends on them, not the sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.pace_regen_packets import betas_for_subject  # noqa: E402

KB = {
    "ryan": {"sex": "male", "anchor": "athletic build"},
    "emily": {"sex": "female", "anchor": "slender"},
    "pat": {"anchor": "average build"},                 # sex not declared
    "sam": {"sex": "Male", "anchor": "average build"},  # declared oddly
}


def test_a_male_body_is_shaped_on_the_axis_that_carries_sex():
    assert betas_for_subject("ryan", KB)[4] < 0, "male shape is the negative direction"


def test_a_female_body_is_left_at_the_neutral_mean():
    """Not pushed positive: the neutral mean already reads female at the chest,
    which is why one shared proxy looked like one person for a whole cast."""
    assert betas_for_subject("emily", KB)[4] == 0.0


def test_an_undeclared_sex_changes_nothing():
    assert betas_for_subject("pat", KB)[4] == 0.0


def test_the_declaration_is_read_case_insensitively():
    assert betas_for_subject("sam", KB)[4] == betas_for_subject("ryan", KB)[4]


def test_sex_does_not_collide_with_age_or_build():
    """betas[0] is age and betas[1] is build; sex must not overwrite either."""
    ryan, emily = betas_for_subject("ryan", KB), betas_for_subject("emily", KB)
    assert ryan[1] != 0.0 and emily[1] != 0.0, "build still reaches the vector"
    assert ryan[1] != emily[1], "and still separates these two"
