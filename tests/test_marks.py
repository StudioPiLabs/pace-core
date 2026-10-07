"""Resolving where a subject stands, from a declaration or from a layout.

The reason this module exists is that two places answered the question
differently: the scene builder spaced bodies on a circle and the camera solver
aimed at a table of template slots. Both were defensible alone and neither
read the mark a breakdown actually stated, so these tests pin all three facts
-- the declaration wins, the layout is the fallback, and which one happened is
reported rather than inferred.
"""
from __future__ import annotations

import math

import pytest

from pace_core.setup.marks import (Mark, _xy_of, declared_marks,
                                   mark_error_m, resolve_positions)

LAYOUT = [(0.0, 0.5, 0.0), (0.0, -0.5, 0.0)]


def _doc(*subject_lists):
    return {"shots": [{"setup": {"subjects": list(s)}} for s in subject_lists]}


def test_a_stated_mark_is_read():
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"world_xy": [1.5, -2.0], "z": 0.4,
                                           "facing_deg": 90}}]))
    assert d["a"] == Mark(xy=(1.5, -2.0), z=0.4, facing_deg=90.0, declared=True)


def test_the_camel_case_spelling_is_read_too():
    # The schema's own tier table names the field worldXy; documents in the
    # wild use both.
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"worldXy": [1.0, 2.0]}}]))
    assert d["a"].xy == (1.0, 2.0)


def test_a_subject_with_no_blocking_is_absent_not_defaulted():
    # Absent and "declared at the origin" must not look the same: one is
    # silence and the other is a statement.
    d = declared_marks(_doc([{"character_id": "a"},
                             {"character_id": "b", "blocking": {}}]))
    assert d == {}


def test_the_first_shot_to_state_a_mark_wins():
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"world_xy": [1.0, 0.0]}}],
                            [{"character_id": "a",
                              "blocking": {"world_xy": [9.0, 9.0]}}]))
    assert d["a"].xy == (1.0, 0.0)


def test_a_malformed_mark_is_rejected_not_coerced():
    # Taking the first two numbers of a three-element mark would stage a body
    # somewhere nobody asked for.
    for bad in ([1.0], [1.0, 2.0, 3.0], ["x", "y"], "1,2", None, {}):
        assert _xy_of(bad) is None
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"world_xy": [1.0, 2.0, 3.0]}}]))
    assert d == {}


def test_integers_are_accepted():
    assert _xy_of([1, -2]) == (1.0, -2.0)


def test_resolution_prefers_the_declaration_and_falls_back_per_subject():
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"world_xy": [3.0, 1.0]}}]))
    got = resolve_positions(["a", "b"], d, LAYOUT)
    assert got[0].xy == (3.0, 1.0) and got[0].declared is True
    assert got[1].xy == (0.0, -0.5) and got[1].declared is False


def test_resolution_reports_which_happened():
    # A gate clause has to tell a staged mark from an invented one; if both
    # arrived as bare coordinates it could not.
    got = resolve_positions(["a"], {}, LAYOUT)
    assert got[0].declared is False


def test_more_characters_than_the_layout_covers_land_at_the_origin():
    got = resolve_positions(["a", "b", "c"], {}, LAYOUT)
    assert got[2].as_xyz() == (0.0, 0.0, 0.0)
    assert got[2].declared is False


def test_an_empty_cast_resolves_to_nothing():
    assert resolve_positions([], {}, LAYOUT) == []


def test_fallback_facing_is_used_only_for_undeclared_subjects():
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"world_xy": [0.0, 0.0],
                                           "facing_deg": 10}}]))
    got = resolve_positions(["a", "b"], d, LAYOUT, fallback_facing=[180, 270])
    assert got[0].facing_deg == 10.0        # the declaration, not the layout
    assert got[1].facing_deg == 270


def test_as_xyz_carries_the_declared_height():
    d = declared_marks(_doc([{"character_id": "a",
                              "blocking": {"world_xy": [1.0, 2.0], "z": 0.9}}]))
    assert d["a"].as_xyz() == (1.0, 2.0, 0.9)


def test_the_error_is_a_distance_on_the_floor():
    assert mark_error_m([3.0, 4.0], [0.0, 0.0]) == pytest.approx(5.0)
    assert mark_error_m([1.0, 1.0], [1.0, 1.0]) == 0.0
    # Symmetric, and indifferent to which way round the arguments go.
    assert mark_error_m([0.0, 2.0], [0.0, 0.0]) == mark_error_m([0.0, 0.0],
                                                                [0.0, 2.0])


def test_the_error_ignores_height():
    # Both marks are floor positions; a subject sitting is not off its mark.
    assert mark_error_m([1.0, 0.0], [0.0, 0.0]) == pytest.approx(1.0)
