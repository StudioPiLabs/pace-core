"""An over-the-shoulder is placed from what the panel declares, not constants.

`position: "ots"` places the lens from the PAIR -- the near shoulder and the
far face -- which overwrites the fit-to-cast distance solve every other
position runs. Two declared fields fell through that hole:

  * `creative_intent.shot_size` reached the build RESULT and nothing else. A
    panel switched from medium to close_up reported the new size and rendered
    both subjects to the same head height to four decimals (0.2941 / 0.5772),
    with the camera at the same metre position.
  * `extrinsics.angle` was resolved to `elevation_deg` (-8 / 5 / 14) and the
    placement never read it: eye_level, high_angle, low_angle and overhead
    built byte-identical cameras and rotations.

The kernel has read `behind_m` and `rise_m` off `spec["ots"]` as "per-panel
overrides" since the position was built, and nothing ever wrote them. The
size ladder and the elevation table write them now.

Eye level maps to OTS_RISE_M exactly, so a panel that declares no angle keeps
the geometry it already rendered.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import (  # noqa: E402
    OTS_BEHIND_BY_SIZE, OTS_EYE_LEVEL_DEG, OTS_RISE_M, OTS_RISE_RANGE_M,
    _elevation_deg, _ots_offsets,
)


def test_a_tighter_size_stands_the_lens_closer():
    sizes = ["wide", "medium", "medium_close_up", "close_up", "extreme_close_up"]
    behind = [_ots_offsets(s, OTS_EYE_LEVEL_DEG)[0]["behind_m"] for s in sizes]
    assert behind == sorted(behind, reverse=True), behind


def test_every_declared_size_has_a_standoff():
    """A size with no rung would silently fall back to one, which is the
    failure this replaces."""
    from pace_core.types_v1 import ShotSize
    import typing
    assert set(typing.get_args(ShotSize)) <= set(OTS_BEHIND_BY_SIZE)


def test_an_undeclared_size_stages_from_the_middle_rung():
    assert _ots_offsets(None, OTS_EYE_LEVEL_DEG)[0]["behind_m"] == \
        OTS_BEHIND_BY_SIZE["medium"]


def test_eye_level_is_exactly_the_height_already_rendered():
    """The one case that must not move: panels built before this keep their
    camera, so their anchor_version and their frames stand."""
    assert _ots_offsets("medium", OTS_EYE_LEVEL_DEG)[0]["rise_m"] == OTS_RISE_M


def test_a_higher_angle_raises_the_lens_and_a_lower_one_drops_it():
    low = _ots_offsets("medium", _elevation_deg({"angle": "low_angle"}))[0]["rise_m"]
    eye = _ots_offsets("medium", _elevation_deg({"angle": "eye_level"}))[0]["rise_m"]
    high = _ots_offsets("medium", _elevation_deg({"angle": "high_angle"}))[0]["rise_m"]
    assert low < eye < high


def test_the_lens_never_rises_above_the_shoulder():
    """Past the crown the frame is the top of a skull and no shoulder, which
    is not an over-the-shoulder however the angle is declared."""
    offsets, clamped = _ots_offsets("medium", 90.0)
    assert offsets["rise_m"] == OTS_RISE_RANGE_M[1]
    assert clamped and "90" in clamped and "shoulder" in clamped


def test_a_clamp_is_recorded_and_an_unclamped_angle_records_nothing():
    assert _ots_offsets("medium", _elevation_deg({"angle": "high_angle"}))[1] is None
    assert _ots_offsets("medium", -90.0)[1] is not None


def test_the_elevation_table_has_one_reader():
    """The spec's own elevation_deg and this map read the same table; two
    copies is how the corpus got a shot built at a default it never declared."""
    src = (Path(__file__).resolve().parents[1]
           / "src/pace_core/node/panel_greybox.py").read_text()
    assert src.count('"low_angle": -8.0, "high_angle": 14.0') == 1
    assert '"elevation_deg": _elevation_deg(extr)' in src


def test_the_offsets_reach_the_key_the_kernel_reads():
    src = (Path(__file__).resolve().parents[1]
           / "src/pace_core/node/panel_greybox.py").read_text()
    assert "ots_pair.update(offsets)" in src
    assert 'ots.get("behind_m", OTS_BEHIND_M)' in src
    assert 'ots.get("rise_m", OTS_RISE_M)' in src
