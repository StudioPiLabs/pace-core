"""The enrichment says where a subject is; nothing said it as a number.

`screen_position` has four fields and the enrichment fills two: `zone` and
`depth`, because those are what a reader of a screenplay can say. `x` and `y`
are not in the text and nothing else filled them, so all 190 of another production's
subject placements carried `x: null`.

That is silent downstream. `panel_greybox._declared_x` reads a null as 0.5, so
every subject sorts to the same place and the seat assignment spreads them
evenly -- which is how a scene of a hundred people working late staged as
three figures standing in a row.

The zone is a better source than the alternative. The paper's own corpus fills
these by hand and reports the result: three distinct x values across 58
subjects and one y, corpus-wide. A model reading the scene put nine different
zones in another production's.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.derive_screen_position_from_zone import (  # noqa: E402
    ZONE_X, ZONE_Y, split_zone, xy_for,
)


@pytest.mark.parametrize("zone,x,y", [
    ("left", 0.28, 0.52), ("center", 0.50, 0.52), ("right", 0.72, 0.52),
    ("center_left", 0.39, 0.52), ("center_right", 0.61, 0.52),
])
def test_a_horizontal_zone_sets_x_and_leaves_y_at_centre(zone, x, y):
    assert xy_for(zone) == (x, y)


@pytest.mark.parametrize("zone,x,y", [
    ("upper_center", 0.50, 0.38), ("lower_center", 0.50, 0.66),
    ("upper_right", 0.72, 0.38), ("lower_left", 0.28, 0.66),
])
def test_a_zone_naming_both_axes_sets_both(zone, x, y):
    """`upper_right` is one token mixing two axes, so the parse is by which
    half names which axis, not by splitting on the underscore."""
    assert xy_for(zone) == (x, y)


def test_the_horizontal_mapping_is_symmetric():
    """A left subject and a right subject are mirror images; an asymmetric
    table would bias every two-hander in the film."""
    assert ZONE_X["left"] + ZONE_X["right"] == pytest.approx(1.0)
    assert ZONE_X["center_left"] + ZONE_X["center_right"] == pytest.approx(1.0)
    assert ZONE_X["center"] == 0.5


def test_the_inner_pair_matches_what_the_existing_corpus_declares():
    """0.39/0.61 rather than the third lines themselves: a shot composed to a
    derived position and one composed to an authored position should not
    disagree by an accident of rounding."""
    assert ZONE_X["center_left"] == pytest.approx(0.39, abs=0.02)
    assert ZONE_X["center_right"] == pytest.approx(0.61, abs=0.02)
    assert ZONE_Y["center"] == 0.52


@pytest.mark.parametrize("zone", ["", None, "somewhere", "off_screen", "middle"])
def test_an_unreadable_zone_yields_no_number(zone):
    """A wrong position is worse than a missing one: the greybox stages a
    body at it and the composition audit then measures against it."""
    x, _y = xy_for(zone)
    assert x is None


def test_split_zone_reads_both_orders():
    assert split_zone("upper_right") == ("right", "upper")
    assert split_zone("right") == ("right", "center")
    assert split_zone("upper_center") == ("center", "upper")
