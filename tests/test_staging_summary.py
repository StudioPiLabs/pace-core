"""A measured path becomes a declaration a person can read.

`Blocking` is coarser than the samples it comes from on purpose: a blocking
sheet says "crosses to the window, then turns back", not forty positions. These
pin the two judgements that summary rests on -- what counts as holding a mark,
and where one leg ends and the next begins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.setup.staging import blocking_from_path, legs_of  # noqa: E402


def _walk(points, fps=24, step=4):
    return [{"f": 1 + i * step, "xy": list(p), "z": 0.0, "facing_deg": None}
            for i, p in enumerate(points)]


def test_small_wander_is_still_a_held_mark():
    """Weight shifts and rig float are not a move across the floor."""
    b = blocking_from_path(_walk([(0, 0), (0.03, 0.01), (0.0, 0.04), (0.02, 0.0)]))
    assert b.static is True
    assert b.moves == []
    assert b.world_xy == [0.0, 0.0]


def test_a_straight_cross_is_one_leg():
    b = blocking_from_path(_walk([(0, 0), (1, 0), (2, 0), (3, 0)]), fps=24)
    assert b.static is False
    assert len(b.moves) == 1
    assert b.moves[0].to_xy == [3.0, 0.0]


def test_a_corner_is_two_legs():
    """The heading breaks at the corner, which is where a sheet draws the second arrow."""
    b = blocking_from_path(_walk([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]), fps=24)
    assert [m.to_xy for m in b.moves] == [[2.0, 0.0], [2.0, 2.0]]


def test_a_leg_carries_how_long_it_took():
    b = blocking_from_path(_walk([(0, 0), (1, 0), (2, 0)], step=12), fps=24)
    assert b.moves[0].duration_s == pytest.approx(1.0)


def test_the_mark_is_where_they_start():
    b = blocking_from_path(_walk([(1.5, -2.0), (2.5, -2.0), (3.5, -2.0)]))
    assert b.world_xy == [1.5, -2.0]


def test_a_gentle_curve_stays_one_leg():
    """A curve inside the turn threshold is still one walk, not many."""
    pts = [(0, 0), (1, 0.05), (2, 0.15), (3, 0.3)]
    assert len(legs_of(_walk(pts), fps=24)) == 1


def test_facing_rides_along_when_it_was_measured():
    s = _walk([(0, 0), (1, 0)])
    s[0]["facing_deg"] = 90.0
    b = blocking_from_path(s)
    assert b.facing_deg == 90.0


def test_an_empty_path_is_refused():
    with pytest.raises(ValueError, match="no samples"):
        blocking_from_path([])


def test_the_dense_path_can_stand_alone():
    """Samples plus an artifact reference make a legal declaration by themselves."""
    b = blocking_from_path(_walk([(0, 0), (0.01, 0)]),
                           subject_path="assets://staging/x.json")
    assert b.subject_path == "assets://staging/x.json"


def test_a_bobbing_walk_is_still_one_leg():
    """A hopping walk swings the sample-to-sample heading without turning.

    Read between neighbours, this path changes direction on nearly every
    sample; read across a real displacement, it is one crossing."""
    pts = [(i * 0.5, 0.04 if i % 2 else -0.04) for i in range(14)]
    assert len(legs_of(_walk(pts), fps=24)) == 1



def test_milling_about_in_one_spot_is_not_a_dozen_legs():
    """Short back-and-forth steps are not a sequence of crossings.

    Someone shuffling around a spot changes heading constantly and covers real
    ground doing it; a sheet draws that as one move, not one arrow per step."""
    pts = [(0, 0), (0.3, 0.1), (0.1, 0.3), (0.4, 0.2), (0.2, 0.45), (0.5, 0.35)]
    assert len(legs_of(_walk(pts), fps=24)) <= 1
