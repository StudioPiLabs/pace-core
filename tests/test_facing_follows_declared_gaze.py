"""Two subjects declared looking at each other are staged facing each other.

`facing_deg` is in the schema and the assembler applies it, but the compiler
between them wrote 0.0 for every subject in every panel -- so a corpus that
declares `lucas -> abigail` and `abigail -> lucas` staged both of them square
to the windscreen, and the eye-line clause could not fail because no two
bodies ever disagreed. `Gaze`'s own docstring names eyeline-match continuity
across shots as what the field is for.

The turn is derived, not authored: the bearing from one staged seat to the
other, minus the quarter turn between the body's default +Y and atan2's +X.
A gaze at a prop or an off-frame direction leaves the seat alone, because a
glance turns a head and rotating the proxy would move the silhouette the
framing solve is fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import _facing_from_gaze  # noqa: E402


def _at(cid: str) -> dict:
    return {"target_type": "character", "target_ref": cid,
            "direction": None, "note": None}


def test_mutual_gaze_stages_two_bodies_180_apart() -> None:
    subs = [{"character_id": "abigail", "gaze": _at("lucas")},
            {"character_id": "lucas", "gaze": _at("abigail")}]
    places = [(0.22, -0.63), (0.0, 0.22)]
    a = _facing_from_gaze(subs, places, 0)
    b = _facing_from_gaze(subs, places, 1)
    assert abs(abs(a - b) - 180.0) < 1e-6


def test_subject_directly_ahead_needs_no_turn() -> None:
    """The default body already faces +Y, so a target straight ahead is 0."""
    subs = [{"character_id": "a", "gaze": _at("b")}, {"character_id": "b"}]
    assert _facing_from_gaze(subs, [(0.0, 0.0), (0.0, 1.0)], 0) == 0.0


def test_gaze_at_a_prop_leaves_the_seat_alone() -> None:
    subs = [{"character_id": "a",
             "gaze": {"target_type": "object", "target_ref": "car_console"}},
            {"character_id": "b"}]
    assert _facing_from_gaze(subs, [(0.0, 0.0), (1.0, 1.0)], 0) == 0.0


def test_off_frame_direction_leaves_the_seat_alone() -> None:
    subs = [{"character_id": "a",
             "gaze": {"target_type": None, "target_ref": None, "direction": "down"}},
            {"character_id": "b"}]
    assert _facing_from_gaze(subs, [(0.0, 0.0), (1.0, 1.0)], 0) == 0.0


def test_target_outside_this_panel_is_not_a_turn() -> None:
    """A gaze can name someone this panel does not frame; there is no bearing."""
    subs = [{"character_id": "a", "gaze": _at("someone_else")},
            {"character_id": "b"}]
    assert _facing_from_gaze(subs, [(0.0, 0.0), (1.0, 1.0)], 0) == 0.0


def test_no_gaze_is_the_staging_every_panel_already_had() -> None:
    subs = [{"character_id": "a"}, {"character_id": "b"}]
    assert _facing_from_gaze(subs, [(0.0, 0.0), (1.0, 1.0)], 0) == 0.0


def test_a_subject_cannot_turn_toward_its_own_seat() -> None:
    subs = [{"character_id": "a", "gaze": _at("b")},
            {"character_id": "b"}]
    assert _facing_from_gaze(subs, [(0.5, 0.5), (0.5, 0.5)], 0) == 0.0
