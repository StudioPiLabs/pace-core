"""An OTS panel that names the wrong face says so, instead of dropping it.

`position: "ots"` can only frame the subject seated deeper — the lens sits at
the near subject's shoulder and looks past it — so a panel whose
`primary_focus` names the near subject has declared two things that cannot
both hold. `build_spec` already detected that and built the sentence for it,
then let the local go out of scope: `position_defaulted` and
`coverage_dropped` both reach the spec, and this one reached nothing. Its own
comment said "Recorded rather than silently resolved, the same treatment
coverage and an unstated position already get", which is what the code now
does.

Kept out of `_ANCHOR_FIELDS` on purpose: a note about a contradiction between
two declarations is not geometry, and a panel that acquires one must keep the
anchor_version of the frame it already rendered.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import _ANCHOR_FIELDS, anchor_version  # noqa: E402

SRC = (Path(__file__).resolve().parents[1] / "src/pace_core/node/panel_greybox.py").read_text()


def test_the_conflict_reaches_the_spec():
    assert '"ots_focus_conflict": ots_focus_conflict' in SRC


def test_it_is_present_only_when_there_is_one():
    """A panel that agrees with itself carries no key, so it keeps the
    anchor_version of the build that already ran."""
    assert '**({"ots_focus_conflict": ots_focus_conflict} if ots_focus_conflict else {})' in SRC


def test_the_sentence_names_both_declarations_and_who_wins():
    seg = SRC.split("ots_focus_conflict = (", 1)[1][:400]
    for part in ("primary_focus", "position='ots'", "the seat order decides"):
        assert part in seg


def test_the_note_does_not_move_the_anchor():
    assert "ots_focus_conflict" not in _ANCHOR_FIELDS
    spec = {f: None for f in _ANCHOR_FIELDS}
    before = anchor_version(spec)
    assert anchor_version({**spec, "ots_focus_conflict": "anything"}) == before


def test_ots_still_needs_a_pair():
    """The conflict is about which of two subjects is framed; a panel with one
    subject is refused earlier and never reaches it."""
    seg = SRC.split('if declared_position == "ots":', 2)[2][:600]
    assert "needs two declared subjects" in seg
