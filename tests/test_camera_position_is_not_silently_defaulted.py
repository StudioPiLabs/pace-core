"""An unstated camera position is reported, not silently treated as 'front'.

`BUILDABLE_POSITIONS.get(extr.get("position") or "front")` turned an absent
field into a positive statement. scene_02 declares `three_quarter` on three of
its four shots and left shot_03 empty, so that one shot swung 25 degrees onto
the axis mid-scene: measured off the body mattes, the cast's spread went from
0.255 of frame width on its siblings to 0.400 on shot_03, with the flanking
subjects landing on opposite sides of their declared marks. Nothing reported
it, because nothing had gone wrong as far as the code was concerned — a field
was missing and a default filled in.

The spec carries `position_defaulted` now, the same way it already carried
`coverage_dropped`: a frame staged from a default can say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import BUILDABLE_POSITIONS  # noqa: E402

SRC = (Path(__file__).resolve().parents[1] / "src/pace_core/node/panel_greybox.py").read_text()


def test_the_spec_carries_the_defaulted_flag():
    assert '"position_defaulted": position_defaulted' in SRC


def test_an_unset_position_is_recorded_rather_than_assumed():
    seg = SRC.split("declared_position = extr.get", 1)[1][:400]
    assert "position_defaulted = None if declared_position else" in seg
    assert "unset" in seg and "front" in seg


def test_a_declared_position_records_nothing():
    """The flag must mean 'this frame was staged from a default', not 'this
    frame is front-on' — those are different statements."""
    seg = SRC.split("declared_position = extr.get", 1)[1][:400]
    assert "None if declared_position" in seg


def test_front_remains_the_fallback_so_nothing_stops_rendering():
    """Reporting the gap must not turn it into a hard failure: a shot with no
    position still has to stage."""
    seg = SRC.split("declared_position = extr.get", 1)[1][:400]
    assert 'BUILDABLE_POSITIONS.get(declared_position or "front")' in seg


def test_the_buildable_vocabulary_is_unchanged():
    """three_quarter and front are what the cabin shell can enclose; adding to
    this set is a geometry change, not a data one."""
    assert set(BUILDABLE_POSITIONS) == {"front", "three_quarter"}
