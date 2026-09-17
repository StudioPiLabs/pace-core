"""Who a beat panel shows, and where a prop the beat moves ends up.

The pilot's climax staged a second man on his feet beside the car -- Hal,
who in the screenplay is trapped inside it and only heard -- and parked the
car where the registry keeps it rather than on the man it falls on.
"""
from __future__ import annotations

import json

from pace_core.breakdown import build_beat_panels as bbp
from pace_core.node import panel_greybox as pg

CAST = {"gus", "hal"}
CLIMAX = {"id": "b5", "source_event_ids": [], "evidence": [], "transition": [
    {"predicate": "FALL_ON_TOP_OF", "actor": "the_car", "patient": "gus", "target": None},
    {"predicate": "FADE_SCREAMS", "actor": "hal", "patient": None, "target": None}]}
TEMPLATE = {"setup": {"props": [{"prop_id": "wrecked_car"}, {"prop_id": "robot_arms"}],
                      "subjects": []},
            "camera": {}, "lighting": {}, "_ages": {"gus": "adult_50", "hal": "adult_18"}}
# The project's map from the beat's event actors to registry prop ids.
PROP_MAP = {"event": {"the_car": ["wrecked_car"]}}


def test_an_actor_who_only_makes_a_sound_is_heard_not_shown():
    assert bbp.human_actors(CLIMAX, CAST) == ["gus"]
    assert bbp.heard_actors(CLIMAX, CAST) == ["hal"]


def test_a_sound_aimed_at_someone_keeps_its_actor_in_frame():
    sob = {"transition": [{"predicate": "SOB_OVER", "actor": "gus",
                           "patient": None, "target": "fay"}]}
    assert bbp.human_actors(sob, {"gus", "fay"}) == ["gus", "fay"]


def test_an_actor_heard_in_one_event_and_seen_in_another_is_shown():
    b = {"transition": [
        {"predicate": "SCREAM", "actor": "gus", "patient": None, "target": None},
        {"predicate": "PUSH_AGAINST", "actor": "gus", "patient": "the_car", "target": None}]}
    assert bbp.human_actors(b, CAST) == ["gus"]
    assert bbp.heard_actors(b, CAST) == []


def _panel(plan):
    return bbp.to_panel(CLIMAX, plan, TEMPLATE, 5, "scene_20", PROP_MAP)


def test_the_panel_drops_a_subject_the_beat_does_not_show():
    sh = _panel({"subjects": [{"character_id": "gus", "pose": "lying"},
                              {"character_id": "hal", "pose": "standing"}]})
    assert [s["character_id"] for s in sh["setup"]["subjects"]] == ["gus"]


def test_a_prop_that_falls_on_someone_rests_on_them():
    sh = _panel({"subjects": [{"character_id": "gus", "pose": "lying"}]})
    props = {p["prop_id"]: p for p in sh["setup"]["props"]}
    assert props["wrecked_car"]["rests_on"] == "gus"
    assert "rests_on" not in props["robot_arms"]
    assert "rests_on" not in TEMPLATE["setup"]["props"][0]      # template untouched


def test_without_a_map_no_prop_is_guessed_for_an_event_actor():
    sh = bbp.to_panel(CLIMAX, {"subjects": [{"character_id": "gus"}]}, TEMPLATE, 5, "scene_20")
    assert all("rests_on" not in p for p in sh["setup"]["props"])


def test_an_entity_with_no_map_entry_is_taken_as_a_prop_id():
    beat = {"state_after": {"main_screen.power": "off"}}
    assert bbp.prop_states(beat) == {"main_screen": "powered_off"}
    assert bbp.prop_states(beat, {"main_screen": ["a", "b"]}) == {"a": "powered_off", "b": "powered_off"}


def test_the_greybox_carries_the_relation_onto_the_fixture(tmp_path):
    kb = {"wrecked_car": {"placement": {"anchor": "ground", "span": [4.3, 1.85, 1.45],
                                        "offset": [0.35, -1.7, 0.0]}}}
    (tmp_path / "props.json").write_text(json.dumps(kb))

    class P:
        props_file = tmp_path / "props.json"
        storage = tmp_path

    on = pg._panel_fixtures(P, {"props": [{"prop_id": "wrecked_car", "rests_on": "gus"}]})
    assert on[0]["on_subject"] == "gus"
    # Absent unless set: `fixtures` is an anchor field, so a key on every
    # fixture would restamp every panel's anchor for geometry that did not move.
    plain = pg._panel_fixtures(P, {"props": [{"prop_id": "wrecked_car"}]})
    assert "on_subject" not in plain[0]


def test_the_panel_asks_only_for_a_camera_the_greybox_can_build():
    one = [{"character_id": "gus"}]
    assert _panel({"camera_position": "side", "subjects": one}
                  )["camera"]["extrinsics"]["position"] in pg.BUILDABLE_POSITIONS
    assert _panel({"camera_position": "front", "subjects": one}
                  )["camera"]["extrinsics"]["position"] == "front"


def test_the_framing_pattern_follows_the_cast_it_has():
    tpl = {**TEMPLATE, "camera": {"creative_intent": {"framing": "two_shot"}}}
    sh = bbp.to_panel(CLIMAX, {"subjects": [{"character_id": "gus"}]}, tpl, 5, "scene_20")
    assert sh["camera"]["creative_intent"]["framing"] == "single"
    empty = {"id": "b6", "transition": [{"predicate": "CLOSE", "actor": "panels"}]}
    sh = bbp.to_panel(empty, {"subjects": []}, tpl, 6, "scene_20")
    assert "framing" not in sh["camera"]["creative_intent"]


def test_a_beat_that_changes_a_state_stages_what_it_starts_from():
    """The panel depicting a change shows the state being lost, not the one
    being arrived at, so the change falls between two panels rather than
    before both of them."""
    from pace_core.breakdown.build_beat_panels import prop_states
    smap = {"panels": ("wall_panels", "main_screen")}
    turns_off = {"state_before": {"panels.power": "on"},
                 "state_after": {"panels.power": "off"}}
    already_off = {"state_before": {"panels.power": "off"},
                   "state_after": {"panels.power": "off"}}
    assert prop_states(turns_off, smap) == {}
    assert prop_states(already_off, smap) == {"wall_panels": "powered_off",
                                              "main_screen": "powered_off"}


def test_a_state_with_no_recorded_before_is_staged_as_it_stands():
    from pace_core.breakdown.build_beat_panels import prop_states
    assert prop_states({"state_after": {"door.open": "closed"}}) == {"door": "closed"}
