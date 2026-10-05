"""Where a subject stands, and where they walk, are declarable.

The camera has carried `trajectory` from the first version of the PACE schema
and lighting has carried `motion`, so a
document could say the lens pushed in and the key light swung round, but not
that anyone crossed the room: a subject carried `screen_position` -- where they
land in the picture -- and nothing about the floor they stand on. Staging could
only be written as prose inside an Action, where it cannot be projected into a
frame, derived from, or compared across a cut.

`Blocking` closes that. These tests pin the two things a reader of the schema
should be able to rely on: a move survives as ordered structured data, and a
document cannot declare a subject both still and moving.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.types_v1 import (  # noqa: E402
    FIELD_TIER, Blocking, BlockingMove, Subject,
)


def test_a_held_mark_is_the_default():
    """Declaring only a mark says the subject stands there for the shot."""
    b = Blocking(world_xy=[1.2, 0.4], facing_deg=90.0)
    assert b.static is True
    assert b.moves == []


def test_a_move_keeps_its_legs_in_order():
    """The legs are the move; reading them back gives the path walked."""
    b = Blocking(
        world_xy=[0.0, 0.0],
        static=False,
        moves=[
            BlockingMove(to_xy=[2.0, 0.0], motion="walk", on_beat=0),
            BlockingMove(to_xy=[2.0, 1.5], motion="turn", on_beat=1),
        ],
    )
    assert [m.to_xy for m in b.moves] == [[2.0, 0.0], [2.0, 1.5]]
    assert [m.motion for m in b.moves] == ["walk", "turn"]


def test_still_and_moving_is_rejected():
    """`static` and `moves` are two statements about one thing; both cannot hold."""
    with pytest.raises(ValueError, match="static=True and also lists moves"):
        Blocking(world_xy=[0.0, 0.0], moves=[BlockingMove(to_xy=[1.0, 0.0])])


def test_moving_nowhere_is_rejected():
    """Saying a subject moves without saying where is not a declaration."""
    with pytest.raises(ValueError, match="names no moves and no"):
        Blocking(world_xy=[0.0, 0.0], static=False)


def test_a_dense_path_stands_in_for_the_legs():
    """Measured staging arrives per-frame, as the camera's does; the artifact
    reference is enough on its own."""
    b = Blocking(world_xy=[0.0, 0.0], static=False,
                 subject_path="assets://staging/subject_path.json")
    assert b.moves == []


def test_the_subject_carries_it():
    s = Subject(character_id="alice", blocking=Blocking(world_xy=[0.5, 2.0]))
    assert s.blocking is not None and s.blocking.world_xy == [0.5, 2.0]
    assert s.screen_position is None, (
        "blocking is a statement about the floor, not about the frame")


def test_the_registry_and_the_tier_table_agree():
    """A new field that only one of the two knows about is the drift this
    repository already guards against elsewhere; check it for these too."""
    reg = {f["path"]: f for f in
           json.loads((ROOT / "src" / "pace_core" / "pace_fields.json").read_text())["fields"]}
    paths = [p for p in reg if p.startswith("setup.subjects[].blocking")]
    assert paths, "blocking is not in the field registry"
    for p in paths:
        assert p in FIELD_TIER, f"{p} is in the registry but not FIELD_TIER"
        assert FIELD_TIER[p] == reg[p]["tier"], f"{p} tier differs"
