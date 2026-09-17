"""State survives the event that caused it, and a prompt may not deny it.

The corpus's measured failure mode was a compiled prompt that states its own
beat and contradicts it in the same paragraph: `scene_03/shot_01` asserts the
car has lost all power while the prop descriptors it is concatenated with say
"glowing with interface graphics". Finding that cost a render campaign, a
figure, and a section of the paper. It is a dictionary lookup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.world_state import (  # noqa: E402
    Transition, build, contradictions, normalise_entity,
)


def _ir(*scenes):
    return list(scenes)


def _scene(index, heading, events):
    return {"index": index, "heading": heading, "events": events}


def _ev(order, predicate, changes):
    return {"order": order, "predicate": predicate, "state_change": changes,
            "evidence": {"source_text": f"<{predicate}>"}}


def test_articles_and_quantifiers_do_not_split_an_entity():
    """The screenplay writes "all the panels" once and "the panels" later."""
    assert normalise_entity("all_panels") == normalise_entity("the_panels") == "panels"
    assert normalise_entity("main_console") == "main_console"


def test_a_state_persists_until_something_changes_it():
    """Silence means the state holds. If the film never turns the panels back
    on, they are off -- for every later scene in that space."""
    tl = build(_ir(
        _scene(1, "INT. CAR - CONTINUOUS",
               [_ev(17, "LOSE_POWER", [{"entity": "all_panels", "attribute": "power",
                                        "from": "on", "to": "off"}])]),
        _scene(4, "INT. CAR - LATER", [])))
    assert tl.value(4, "panels", "power", scope="INT. CAR") == "off"


def test_a_later_event_overwrites_an_earlier_one():
    tl = build(_ir(_scene(1, "INT. CAR - DAY", [
        _ev(1, "LOSE_POWER", [{"entity": "panels", "attribute": "power",
                               "from": "on", "to": "off"}]),
        _ev(9, "RESTORE", [{"entity": "panels", "attribute": "power",
                            "from": "off", "to": "on"}])])))
    assert tl.value(1, "panels", "power", scope="INT. CAR") == "on"
    # ...and before the restore, it was still off
    assert tl.value(1, "panels", "power", order=5, scope="INT. CAR") == "off"


def test_a_power_cut_in_one_car_does_not_reach_another_car():
    """The corpus has three vehicles. Carrying a state across them would
    invent continuity instead of checking it."""
    tl = build(_ir(
        _scene(1, "INT. CAR - CONTINUOUS",
               [_ev(17, "LOSE_POWER", [{"entity": "panels", "attribute": "power",
                                        "from": "on", "to": "off"}])]),
        _scene(3, "INT. ANOTHER CAR - MOMENTS LATER", [])))
    assert tl.value(3, "panels", "power", scope="INT. CAR") == "off"
    assert tl.value(3, "panels", "power", scope="INT. ANOTHER CAR") is None


def test_the_same_space_under_two_time_qualifiers_is_one_space():
    tl = build(_ir(
        _scene(5, "INT. ANOTHER CAR - CONTINUOUS",
               [_ev(1, "CLOSE", [{"entity": "panels", "attribute": "open",
                                  "from": "open", "to": "closed"}])]),
        _scene(7, "INT. ANOTHER CAR - MOMENTS LATER", [])))
    assert tl.value(7, "panels", "open", scope="INT. ANOTHER CAR") == "closed"


# ───────────────────────── contradiction detection ────────────────────────

def _off():
    return {("panels", "power"): Transition(
        scene=1, order=17, entity="panels", attribute="power", to="off",
        frm="on", predicate="LOSE_POWER",
        evidence={"source_text": "All the panels lose power."})}


def test_a_glowing_descriptor_contradicts_a_dead_panel():
    cs = contradictions(
        "wraparound interior screen panels lining the cabin walls, glowing softly",
        _off(), {"panels": ("screen panel", "panels")})
    assert cs and cs[0]["contradicting_phrase"] == "glowing"
    assert cs[0]["state"] == "off"


def test_a_descriptor_about_a_different_entity_is_not_a_contradiction():
    """`main_console` glowing says nothing about `panels`. Matching on the
    phrase alone would fire on every lit object in the frame."""
    cs = contradictions("the car console glowing with interface graphics",
                        _off(), {"panels": ("screen panel",)})
    assert cs == []


def test_a_prompt_that_respects_the_state_is_clean():
    cs = contradictions("screen panels dark and dead, the cabin in pitch black",
                        _off(), {"panels": ("screen panel", "panels")})
    assert cs == []


# ───────────────── state reaches the prompt by substitution ───────────────

def test_a_declared_state_rewrites_the_registry_appearance():
    """The repair. `prop_phrase` returned the registry description unchanged
    whenever one existed -- which is always -- so `state` was resolved and
    could never take effect."""
    from pace_core.pai_compat import prop_phrase
    kb = {"props": {"wall_panels": {
        "description": "wraparound interior screen panels lining the cabin "
                       "walls, thin bezelless displays, glowing softly"}}}
    lit = prop_phrase({"prop_id": "wall_panels", "count": 1, "state": "active"}, kb)
    assert "glowing softly" in lit
    dead = prop_phrase({"prop_id": "wall_panels", "count": 1,
                        "state": "powered_off"}, kb)
    assert "glowing" not in dead
    assert "dark and unlit" in dead


def test_the_structured_state_from_the_timeline_works_too():
    from pace_core.pai_compat import prop_phrase
    kb = {"props": {"main_console": {
        "description": "a wide curved glass touch panel, glowing with "
                       "interface graphics"}}}
    out = prop_phrase({"prop_id": "main_console", "count": 1,
                       "state": {"power": "off"}}, kb)
    assert "glowing" not in out and "extinguished" in out


def test_a_neutral_state_changes_nothing():
    """`active` is the corpus's value on every prop. The repair must be inert
    for it, or it rewrites the entire film."""
    from pace_core.pai_compat import prop_phrase
    kb = {"props": {"p": {"description": "a glowing screen"}}}
    for st in ("active", "", None, "normal"):
        assert prop_phrase({"prop_id": "p", "count": 1, "state": st}, kb) \
            == prop_phrase({"prop_id": "p", "count": 1}, kb)


def test_substitution_not_negation():
    """cfg=1.0 means the negative prompt is never evaluated, so a denial does
    not remove anything. The word itself has to be gone."""
    from pace_core.breakdown.world_state import rewrite_for_state
    out, ch = rewrite_for_state("panels glowing softly", [("power", "off")])
    assert "glowing" not in out
    assert "not glowing" not in out and "no glow" not in out
    assert ch and ch[0]["was"] == "glowing softly"


def test_the_longest_phrase_wins():
    """Rewriting 'glowing' first would leave 'dark with interface graphics',
    which still asserts a live display."""
    from pace_core.breakdown.world_state import rewrite_for_state
    out, _ = rewrite_for_state("glowing with interface graphics", [("power", "off")])
    assert "interface graphics" not in out or "extinguished" in out
    assert "dark" in out


def test_a_rewritten_clause_no_longer_contradicts_its_own_state():
    """End to end: the contradiction detector and the repair agree."""
    from pace_core.breakdown.world_state import rewrite_for_state
    before = "wraparound screen panels lining the cabin walls, glowing softly"
    assert contradictions(before, _off(), {"panels": ("screen panel",)})
    after, _ = rewrite_for_state(before, [("power", "off")])
    assert contradictions(after, _off(), {"panels": ("screen panel",)}) == []


def test_the_prop_audit_scopes_to_one_prop_and_does_not_cross_entities():
    """Scanning a whole prompt cannot tell which noun a word belongs to: the
    handheld device's "a single glowing screen" read as a panel
    contradiction. Proximity does not rescue it -- a true positive can sit
    further apart than a false one -- so the
    audit is scoped per prop instead of guessing at the seam."""
    from pace_core.breakdown.world_state import audit_props
    kb = {"props": {
        "wall_panels": {"description": "wraparound screen panels, glowing softly"},
        "game_device": {"description": "a handheld slab, its entire face a "
                                       "single glowing screen"}}}
    shot = {"setup": {"props": [
        {"prop_id": "wall_panels", "state": "powered_off"},
        {"prop_id": "game_device", "state": "active"}]}}
    # wall_panels is rewritten by the repair, so nothing contradicts; and the
    # game device's own glow is never attributed to the panels.
    assert [r for r in audit_props(shot, kb) if r["kind"] == "CONTRADICTION"] == []


def test_the_audit_reports_a_prop_that_should_have_a_state_and_does_not():
    """The corpus's actual condition: the world state says these panels are
    dead and the prop declares nothing at all."""
    from pace_core.breakdown.world_state import audit_props
    kb = {"props": {"wall_panels": {"description": "screen panels, glowing softly"}}}
    shot = {"setup": {"props": [{"prop_id": "wall_panels"}]}}
    rows = audit_props(shot, kb, expected={"wall_panels": [("power", "off")]})
    assert [r["kind"] for r in rows] == ["STATE_MISSING"]
    assert rows[0]["expected"] == "off"
