"""A panel's own `primary_focus` decides its geometry, not just its packets.

`Panel.primary_focus` is declared ("quick-access override for what dominates
this frame") and `pai_compat.primary_focus_of` resolves it panel-over-shot.
Two consumers call that helper -- the regen packets and the restage pass --
and `build_spec` did not: it read `setup["primary_focus"]` off the resolved
shot, so a panel naming a different subject was staged, aimed and framed on
the shot's subject with nothing reporting the difference.

Found on the corpus's over-the-shoulder, where the panels name the far subject
(the only one that position can frame) and the shot still named the near one:
the build took the shot's, and the conflict clause fired against a declaration
the panel had already corrected.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.pai_compat import primary_focus_of  # noqa: E402

SRC = (Path(__file__).resolve().parents[1]
       / "src/pace_core/node/panel_greybox.py").read_text()


def test_the_geometry_reads_the_panel_first():
    assert "focus = _primary_focus_of(panel, shot)" in SRC
    assert 'focus = (setup.get("primary_focus") or {})' not in SRC


def test_one_helper_resolves_it_everywhere():
    """Three consumers, one precedence rule; a second copy is how the geometry
    and the packets came to disagree about the same panel."""
    assert "primary_focus_of as _primary_focus_of" in SRC


def test_the_panel_wins_over_the_shot():
    shot = {"setup": {"primary_focus": {"type": "character", "ref": "bob"}}}
    panel = {"primary_focus": {"type": "character", "ref": "alice"}}
    assert primary_focus_of(panel, shot)["ref"] == "alice"


def test_a_panel_that_says_nothing_inherits_the_shot():
    shot = {"setup": {"primary_focus": {"type": "character", "ref": "bob"}}}
    assert primary_focus_of({"primary_focus": None}, shot)["ref"] == "bob"
    assert primary_focus_of({}, shot)["ref"] == "bob"


def test_neither_declaring_one_is_not_an_error():
    assert primary_focus_of({}, {}) == {}
