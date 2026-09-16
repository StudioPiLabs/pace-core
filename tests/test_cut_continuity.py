"""A cut inside a scene must not change the clothes or the props.

scene_11 of the paper's corpus cut between two angles on the same two people
and the garments changed, the centre screens went from lit to dark, and the
tray table left the cabin. Nothing reported any of it, because a costume was
prose on a character record and every prop's `state` was null: there was no
pair of values for a check to find unequal.

Both are declared fields now, so this clause is arithmetic rather than
judgement -- and where a value is still missing it says so, instead of
passing quietly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.qc.greybox_gate import (                          # noqa: E402
    _continuity_of, cut_continuity_clause,
)


def _decl(costumes=None, states=None):
    return {"costumes": costumes or {}, "prop_states": states or {}}


def test_the_same_garment_across_the_cut_passes():
    d = _decl({"abigail": "abigail_costume"})
    assert cut_continuity_clause(d, d).ok is True


def test_a_changed_garment_fails_and_names_both_sides():
    c = cut_continuity_clause(_decl({"abigail": "lucas_costume"}),
                              _decl({"abigail": "abigail_costume"}))
    assert c.ok is False
    assert "abigail_costume" in c.detail and "lucas_costume" in c.detail


def test_a_prop_that_changes_state_across_the_cut_fails():
    c = cut_continuity_clause(_decl(states={"cabin_panels": "inactive"}),
                              _decl(states={"cabin_panels": "active"}))
    assert c.ok is False and "cabin_panels" in c.detail


def test_the_first_panel_of_a_scene_has_nothing_to_cut_from():
    """A costume change between scenes is legitimate, so the clause abstains
    rather than reporting the previous scene's wardrobe as a break."""
    assert cut_continuity_clause(_decl({"a": "x"}), None).ok is None


def test_an_undeclared_value_is_counted_not_passed_over():
    """This is the state the corpus was in: nothing to compare, and silence
    read as agreement. The clause still passes, and says how much it could
    not check."""
    c = cut_continuity_clause(_decl({"abigail": None}, {"tray_table": None}),
                              _decl({"abigail": "abigail_costume"},
                                    {"tray_table": "deployed"}))
    assert c.ok is True
    assert "2 declared on one side only" in c.detail


def test_a_prop_that_only_appears_after_the_cut_is_not_a_break():
    """Props enter and leave frame; only a prop declared on both sides can
    disagree with itself."""
    c = cut_continuity_clause(_decl(states={"game_device": "active"}), _decl())
    assert c.ok is True


# ── reading the declarations off a staged shot ───────────────────────────
def test_a_staged_shot_yields_its_costumes_and_prop_states():
    setup = {"subjects": [{"character_id": "emily", "costume_id": "emily_costume"},
                          {"character_id": "ryan"}],
             "props": [{"prop_id": "cabin_panels", "state": "active"},
                       {"state": "deployed"}]}
    got = _continuity_of(setup)
    assert got["costumes"] == {"emily": "emily_costume", "ryan": None}
    assert got["prop_states"] == {"cabin_panels": "active"}
