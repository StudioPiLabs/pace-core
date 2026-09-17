"""A dutch angle: a canted horizon about the optical axis.

The vocabulary has had "dutch" as an angle since PAI 1.1 and nothing built
it: the builder read it as eye level and the compiler said nothing. A roll is
its own field, so a high or low camera can be canted and keep its elevation.
"""
from __future__ import annotations

from pace_core.compilers.compile_common import CompileContext
from pace_core.compilers.compile_flux2 import compile_flux2
from pace_core.node.panel_greybox import DUTCH_ROLL_DEG, _roll_of, anchor_version


def test_dutch_alone_implies_the_conventional_cant():
    assert _roll_of({"angle": "dutch"}) == DUTCH_ROLL_DEG


def test_a_declared_roll_wins_and_keeps_the_elevation_class():
    assert _roll_of({"angle": "high", "roll_deg": -10}) == -10.0
    assert _roll_of({"angle": "dutch", "roll_deg": 8}) == 8.0


def test_an_upright_camera_has_no_roll():
    assert _roll_of({"angle": "eye_level"}) == 0.0
    assert _roll_of({}) == 0.0
    assert _roll_of({"roll_deg": True}) == 0.0


def test_roll_changes_the_anchor_only_when_there_is_one():
    upright = {"lens_mm": 35, "subjects": []}
    assert anchor_version(upright) == anchor_version({**upright, "roll_deg": 0.0})
    assert anchor_version(upright) != anchor_version({**upright, "roll_deg": 15.0})


def _scene(extrinsics):
    shot = {"shot_id": "shot_01",
            "setup": {"backdrop": {"setting": "ext", "location": "a roadside"},
                      "subjects": [{"character_id": "gus"}]},
            "camera": {"extrinsics": extrinsics,
                       "creative_intent": {"shot_size": "full"}},
            "events": {"actions": [{"description_en": "gus pushes against the car"}]},
            "panels": [{"id": "scene_01_shot_01_panel_0001", "panel_number": 1}]}
    return {"scene_id": "scene_01", "shots": [shot]}, shot


def test_the_prompt_says_dutch():
    scene, shot = _scene({"angle": "dutch"})
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], CompileContext())
    assert "dutch angle" in pos


def test_a_canted_low_camera_keeps_its_low_angle_words():
    scene, shot = _scene({"angle": "low", "roll_deg": 15})
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], CompileContext())
    assert "low angle" in pos and "dutch angle" in pos


def test_an_upright_camera_is_not_called_dutch():
    scene, shot = _scene({"angle": "low"})
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], CompileContext())
    assert "dutch" not in pos


def test_the_vocabulary_spellings_of_high_and_low_reach_the_elevation_table():
    """`high`/`low` are the type vocabulary's words; the table was keyed only
    by the corpus's `high_angle`/`low_angle`, so a shot declared `high` was
    built at eye level."""
    from pace_core.node.panel_greybox import _angle_class
    assert _angle_class({"angle": "high"}) == "high_angle"
    assert _angle_class({"angle": "low"}) == "low_angle"
    assert _angle_class({"angle": "high_angle"}) == "high_angle"
    assert _angle_class({"angle": "eye_level"}) == "eye_level"
    assert _angle_class({}) is None
