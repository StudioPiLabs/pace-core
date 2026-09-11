"""Who a beat panel shows, and where a prop the beat moves ends up.

The pilot's climax staged a second man on his feet beside the car -- Ethan,
who in the screenplay is trapped inside it and only heard -- and parked the
car where the registry keeps it rather than on the man it falls on.
"""
from __future__ import annotations

import json

from pace_core.breakdown import build_beat_panels as bbp
from pace_core.node import panel_greybox as pg

CAST = {"ryan", "ethan"}
CLIMAX = {"id": "b5", "source_event_ids": [], "evidence": [], "transition": [
    {"predicate": "FALL_ON_TOP_OF", "actor": "the_car", "patient": "ryan", "target": None},
    {"predicate": "FADE_SCREAMS", "actor": "ethan", "patient": None, "target": None}]}
TEMPLATE = {"setup": {"props": [{"prop_id": "wrecked_car"}, {"prop_id": "robot_arms"}],
                      "subjects": []},
            "camera": {}, "lighting": {}, "_ages": {"ryan": "adult_50", "ethan": "adult_18"}}


def test_an_actor_who_only_makes_a_sound_is_heard_not_shown():
    assert bbp.human_actors(CLIMAX, CAST) == ["ryan"]
    assert bbp.heard_actors(CLIMAX, CAST) == ["ethan"]


def test_a_sound_aimed_at_someone_keeps_its_actor_in_frame():
    sob = {"transition": [{"predicate": "SOB_OVER", "actor": "ryan",
                           "patient": None, "target": "emily"}]}
    assert bbp.human_actors(sob, {"ryan", "emily"}) == ["ryan", "emily"]


def test_an_actor_heard_in_one_event_and_seen_in_another_is_shown():
    b = {"transition": [
        {"predicate": "SCREAM", "actor": "ryan", "patient": None, "target": None},
        {"predicate": "PUSH_AGAINST", "actor": "ryan", "patient": "the_car", "target": None}]}
    assert bbp.human_actors(b, CAST) == ["ryan"]
    assert bbp.heard_actors(b, CAST) == []


def _panel(plan):
    return bbp.to_panel(CLIMAX, plan, TEMPLATE, 5, "scene_20")


def test_the_panel_drops_a_subject_the_beat_does_not_show():
    sh = _panel({"subjects": [{"character_id": "ryan", "pose": "lying"},
                              {"character_id": "ethan", "pose": "standing"}]})
    assert [s["character_id"] for s in sh["setup"]["subjects"]] == ["ryan"]


def test_a_prop_that_falls_on_someone_rests_on_them():
    sh = _panel({"subjects": [{"character_id": "ryan", "pose": "lying"}]})
    props = {p["prop_id"]: p for p in sh["setup"]["props"]}
    assert props["wrecked_car"]["rests_on"] == "ryan"
    assert "rests_on" not in props["robot_arms"]
    assert "rests_on" not in TEMPLATE["setup"]["props"][0]      # template untouched


def test_the_greybox_carries_the_relation_onto_the_fixture(tmp_path):
    kb = {"wrecked_car": {"placement": {"anchor": "ground", "span": [4.3, 1.85, 1.45],
                                        "offset": [0.35, -1.7, 0.0]}}}
    (tmp_path / "props.json").write_text(json.dumps(kb))

    class P:
        props_file = tmp_path / "props.json"
        storage = tmp_path

    on = pg._panel_fixtures(P, {"props": [{"prop_id": "wrecked_car", "rests_on": "ryan"}]})
    assert on[0]["on_subject"] == "ryan"
    # Absent unless set: `fixtures` is an anchor field, so a key on every
    # fixture would restamp every panel's anchor for geometry that did not move.
    plain = pg._panel_fixtures(P, {"props": [{"prop_id": "wrecked_car"}]})
    assert "on_subject" not in plain[0]


def test_the_panel_asks_only_for_a_camera_the_greybox_can_build():
    one = [{"character_id": "ryan"}]
    assert _panel({"camera_position": "side", "subjects": one}
                  )["camera"]["extrinsics"]["position"] in pg.BUILDABLE_POSITIONS
    assert _panel({"camera_position": "front", "subjects": one}
                  )["camera"]["extrinsics"]["position"] == "front"


def test_the_framing_pattern_follows_the_cast_it_has():
    tpl = {**TEMPLATE, "camera": {"creative_intent": {"framing": "two_shot"}}}
    sh = bbp.to_panel(CLIMAX, {"subjects": [{"character_id": "ryan"}]}, tpl, 5, "scene_20")
    assert sh["camera"]["creative_intent"]["framing"] == "single"
    empty = {"id": "b6", "transition": [{"predicate": "CLOSE", "actor": "panels"}]}
    sh = bbp.to_panel(empty, {"subjects": []}, tpl, 6, "scene_20")
    assert "framing" not in sh["camera"]["creative_intent"]
