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
    OTS_BEHIND_BY_SIZE, OTS_RISE_RANGE_M, _elevation_deg, _ots_offsets,
)

SRC = (Path(__file__).resolve().parents[1]
       / "src/pace_core/node/panel_greybox.py").read_text()


def test_a_tighter_size_stands_the_lens_closer():
    sizes = ["wide", "medium", "medium_close_up", "close_up", "extreme_close_up"]
    behind = [_ots_offsets(s)["behind_m"] for s in sizes]
    assert behind == sorted(behind, reverse=True), behind


def test_every_declared_size_has_a_standoff():
    """A size with no rung would silently fall back to one, which is the
    failure this replaces."""
    from pace_core.types_v1 import ShotSize
    import typing
    assert set(typing.get_args(ShotSize)) <= set(OTS_BEHIND_BY_SIZE)


def test_an_undeclared_size_stages_from_the_middle_rung():
    assert _ots_offsets(None)["behind_m"] == OTS_BEHIND_BY_SIZE["medium"]


def test_the_height_is_solved_against_the_face_the_lens_aims_at():
    """Not approximated. The camera aims at `focus`, so a lens tan(elevation)
    x the horizontal run above that face leaves at exactly the declared
    elevation -- which every other position's solve already delivers."""
    assert "run * math.tan(math.radians(" in SRC
    assert 'float(spec.get("elevation_deg", OTS_EYE_LEVEL_DEG))' in SRC


def test_the_solve_is_closed_form():
    """The horizontal run does not depend on the height, so nothing iterates;
    a loop here would be a sign the placement had been reordered."""
    seg = SRC.split("run = math.hypot", 1)[1][:600]
    assert "while" not in seg and "for " not in seg


def test_the_lens_never_rises_above_the_shoulder_band():
    """Past the crown the frame is the top of a skull and no shoulder, which
    is not an over-the-shoulder however the angle is declared. On this
    corpus's cabin the clamp binds: holding eye level would need the lens
    5 cm over the near crown, where the gate fails the bottom edge at a
    joint."""
    lo, hi = OTS_RISE_RANGE_M
    assert lo < hi <= 0.0
    assert "min(max(want, lo), hi)" in SRC


def test_a_panel_that_states_the_height_outright_is_taken_at_its_word():
    assert 'near_head.z + float(ots["rise_m"])' in SRC
    assert 'if "rise_m" in ots' in SRC


def test_the_build_reports_whether_the_angle_was_held():
    """A clamped panel must be able to say the declared elevation was not
    reproduced, the way the focus conflict says which declaration lost."""
    assert '"elevation_held": abs(cam.location.z - want) < 1e-6' in SRC


def test_the_elevation_classes_stay_ordered():
    """The solve reads this table for the height it aims from, so low below
    eye below high is what keeps the solved heights in the same order."""
    assert (_elevation_deg({"angle": "low_angle"})
            < _elevation_deg({"angle": "eye_level"})
            < _elevation_deg({"angle": "high_angle"}))


def test_the_elevation_table_has_one_reader():
    """The spec's own elevation_deg and this map read the same table; two
    copies is how the corpus got a shot built at a default it never declared."""
    assert SRC.count('"low_angle": -8.0, "high_angle": 14.0') == 1
    assert '"elevation_deg": _elevation_deg(extr)' in SRC


def test_the_standoff_reaches_the_key_the_kernel_reads():
    assert "ots_pair.update(_ots_offsets(shot_size))" in SRC
    assert 'ots.get("behind_m", OTS_BEHIND_M)' in SRC
