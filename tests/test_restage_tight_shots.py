"""Restaging a tight shot the camera cannot deliver, or splitting it into singles.

The probe -- a greybox build in production -- is replaced by a scripted one,
so these pin the decisions and the document edits, not Blender.
"""
from __future__ import annotations

import copy

from pace_core.breakdown.restage_tight_shots import (
    reaction_action, restage_scene, restaged_subjects, tight_multi_panels)


def _scene(size="close_up", framing="two_shot", position="front"):
    subs = [
        {"character_id": "a", "screen_position": {"x": 0.39, "y": 0.52},
         "pose": "standing", "gaze": {"target_type": "character", "target_ref": "b"}},
        {"character_id": "b", "screen_position": {"x": 0.61, "y": 0.52},
         "pose": "standing, head tilted", "gaze": {"target_type": "character", "target_ref": "a"}},
    ]
    return {
        "scene_id": "scene_09", "_max_panel_n": 5,
        "compile_hints": [{"panel_id": "scene_09_shot_02_panel_0005"}],
        "shots": [
            {"shot_id": "shot_01",
             "camera": {"creative_intent": {"shot_size": "wide"}, "extrinsics": {}},
             "setup": {"subjects": copy.deepcopy(subs)},
             "panels": [{"id": "scene_09_shot_01_panel_0004", "panel_number": 1}]},
            {"shot_id": "shot_02",
             "camera": {"creative_intent": {"shot_size": size, "framing": framing},
                        "extrinsics": {"position": position}},
             "setup": {"subjects": subs,
                       "primary_focus": {"type": "character", "ref": "a"}},
             "events": {"actions": [{"description_en": "a speaks"}],
                        "emotions": [{"implicit": "trembling hands"}]},
             "panels": [{"id": "scene_09_shot_02_panel_0005", "panel_number": 1,
                         "primary_focus": {"type": "character", "ref": "a",
                                           "coverage_pct": 50}}]},
        ],
    }


def _probe(*answers):
    calls = []

    def probe(scene, scene_id, panel_id):
        calls.append(copy.deepcopy(scene))
        return answers[min(len(calls), len(answers)) - 1]
    probe.calls = calls
    return probe


def test_only_tight_multi_subject_panels_are_candidates():
    assert [p["id"] for _, p in tight_multi_panels(_scene())] == ["scene_09_shot_02_panel_0005"]
    assert tight_multi_panels(_scene(size="medium")) == []
    assert tight_multi_panels(_scene(framing="ots")) == []
    assert tight_multi_panels(_scene(position="ots")) == []


def test_a_panel_the_size_already_binds_on_is_kept():
    scene = _scene()
    before = copy.deepcopy(scene)
    rows = restage_scene(scene, "scene_09", _probe({"bound_by": "height"}), {}, 0.35)
    assert rows == [{"panel": "scene_09_shot_02_panel_0005", "action": "kept",
                     "bound_by": "height"}]
    assert scene == before


def test_restaged_subjects_draw_in_and_stagger_behind_the_focus():
    subs = _scene()["shots"][1]["setup"]["subjects"]
    out = {s["character_id"]: s["screen_position"] for s in restaged_subjects(subs, "a")}
    assert out["a"] == {"x": 0.39, "y": 0.52, "depth": "foreground"}
    assert out["b"]["x"] == 0.51 and out["b"]["depth"] == "midground"
    # The original document is not touched.
    assert "depth" not in subs[0]["screen_position"]


def test_a_restage_that_binds_is_written_as_a_panel_override():
    scene = _scene()
    probe = _probe({"bound_by": "width"}, {"bound_by": "height", "overlap": 0.12})
    rows = restage_scene(scene, "scene_09", probe, {}, 0.35)
    assert rows[0]["action"] == "restaged"
    panel = scene["shots"][1]["panels"][0]
    xs = {s["character_id"]: s["screen_position"]["x"]
          for s in panel["setup_override"]["subjects"]}
    assert xs == {"a": 0.39, "b": 0.51}
    # The shot keeps its declared blocking; only this panel is restaged.
    assert scene["shots"][1]["setup"]["subjects"][1]["screen_position"]["x"] == 0.61
    assert len(scene["shots"][1]["panels"]) == 1


def test_a_restage_that_hides_someone_is_not_kept():
    scene = _scene()
    probe = _probe({"bound_by": "width"}, {"bound_by": "height", "overlap": 0.6})
    assert restage_scene(scene, "scene_09", probe, {}, 0.35)[0]["action"] == "split"


def test_split_gives_each_subject_a_single_numbered_like_the_studio():
    scene = _scene()
    chars = {"subject_b": {"name": "b", "name_en": "Bee"}, "a": {"name_en": "Ay"}}
    rows = restage_scene(scene, "scene_09", _probe({"bound_by": "width"}), chars, 0.35)
    assert rows[0]["action"] == "split"
    assert rows[0]["new_panels"] == ["scene_09_shot_02_panel_0006"]
    first, second = scene["shots"][1]["panels"]
    assert [p["panel_number"] for p in (first, second)] == [1, 2]
    assert [s["character_id"] for s in first["setup_override"]["subjects"]] == ["a"]
    assert [s["character_id"] for s in second["setup_override"]["subjects"]] == ["b"]
    for p in (first, second):
        assert p["camera_override"]["creative_intent"]["framing"] == "single"
    assert second["primary_focus"] == {"type": "character", "ref": "b", "coverage_pct": 50}
    # The shot's action and emotion belong to whoever acts; the reaction
    # single gets its own beat and no borrowed cue.
    assert "events_override" not in first
    assert second["events_override"]["emotions"] == []
    assert second["events_override"]["actions"][0]["description_en"] == \
        "Bee watches Ay, standing, head tilted."
    assert scene["_max_panel_n"] == 6
    assert scene["compile_hints"][-1]["panel_id"] == "scene_09_shot_02_panel_0006"


def test_reaction_without_a_gaze_target_says_so():
    act = reaction_action({"character_id": "gus", "pose": None}, {})
    assert act["description_en"] == "Gus reacts." or act["description_en"] == "gus reacts."
