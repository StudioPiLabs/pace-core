"""A prop declared away from the cast is written with its place.

On one scene's second beat the only prop text was "mechanical robotic arms
emerging from the wreckage", declared in the background and staged nowhere,
and the sampler attached the arms to the one man in frame: his hands came back
mechanical. Written with its declared place, the prop has somewhere to be.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.pai_compat import (  # noqa: E402
    prop_left_unstaged_away_from_cast, prop_phrase, prop_placement,
)

KB = {
    "robot_arms": {"id": "robot_arms", "physical_attributes": {"size_m": [1.5, 0.5, 0.5]}},
    "wrecked_car": {"placement": {"anchor": "ground", "span": [4.3, 1.85, 1.45]}},
    "game_device": {"id": "game_device", "physical_attributes": {"size_m": [0.2, 0.1, 0.01]}},
}


def test_a_large_unstaged_prop_away_from_the_cast_is_not_named():
    arms = {"prop_id": "robot_arms", "screen_position": {"zone": "background_midground"}}
    assert prop_left_unstaged_away_from_cast(arms, KB)


def test_a_prop_the_greybox_stages_is_named():
    car = {"prop_id": "wrecked_car", "screen_position": {"zone": "background_midground"}}
    assert not prop_left_unstaged_away_from_cast(car, KB)


def test_a_small_prop_is_named_wherever_it_is():
    device = {"prop_id": "game_device", "screen_position": {"zone": "background"}}
    assert not prop_left_unstaged_away_from_cast(device, KB)


def test_a_prop_on_the_cast_is_named_however_large():
    held = {"prop_id": "robot_arms", "screen_position": {"zone": "hands", "depth": "foreground"}}
    assert not prop_left_unstaged_away_from_cast(held, KB)
    assert not prop_left_unstaged_away_from_cast({"prop_id": "robot_arms"}, KB)


def test_declared_vehicle_scale_counts_as_large_without_a_registry():
    arms = {"prop_id": "robot_arms", "size": "vehicle_scale",
            "screen_position": {"zone": "background_midground"}}
    assert prop_left_unstaged_away_from_cast(arms, None)


def _prop(zone: str | None = None, depth: str | None = None) -> dict:
    p = {"prop_id": "robot_arms", "cls": "mechanical_system"}
    if zone or depth:
        p["screen_position"] = {k: v for k, v in (("zone", zone), ("depth", depth)) if v}
    return p


def test_a_background_prop_is_placed_in_the_background():
    assert prop_placement(_prop("background_midground", "midground")) == " in the background"
    assert prop_placement(_prop("background")) == " in the background"


def test_a_midground_prop_is_placed_in_the_middle_distance():
    assert prop_placement(_prop("midground")) == " in the middle distance"


def test_a_prop_in_the_hands_or_foreground_is_left_as_written():
    """Its place is the cast's; saying so would add nothing the cast clause
    does not already carry."""
    assert prop_placement(_prop("hands", "foreground")) == ""
    assert prop_placement(_prop("foreground")) == ""


def test_a_prop_with_no_declared_place_is_left_as_written():
    assert prop_placement(_prop()) == ""
    assert prop_placement({"prop_id": "robot_arms", "screen_position": None}) == ""


def test_the_place_follows_the_prop_it_belongs_to():
    phrase = prop_phrase(_prop("background_midground")) + prop_placement(_prop("background_midground"))
    assert phrase.endswith("robot arms in the background")


# ── a prop staged out of the camera's view ──
#
# The cabin camera stands at the front looking back, so the console anchored
# at the front of the cabin is built behind the lens. Named anyway, it has
# nowhere in the control image to be drawn: the same seed put it on the back
# wall with the console lit and across the cast as a pane of glass with it
# dark. The build records what its camera sees; the compiler reads it.

import json  # noqa: E402

from pace_core.compilers.compile_common import CompileContext  # noqa: E402
from pace_core.compilers.compile_flux2 import compile_flux2  # noqa: E402
from pace_core.pai_compat import gaze_clauses_of, props_out_of_frame  # noqa: E402


def _cabin_scene():
    shot = {
        "shot_id": "shot_01",
        "setup": {
            "backdrop": {"setting": "int", "location": "inside a small car", "time_of_day": "day"},
            "subjects": [{"character_id": "nina",
                          "gaze": {"target_type": "object", "target_ref": "car_console"}}],
            "primary_focus": {"type": "character", "ref": "nina"},
            "props": [{"prop_id": "car_console"}, {"prop_id": "game_device"}],
        },
        "camera": {"extrinsics": {"angle": "eye_level"},
                   "creative_intent": {"shot_size": "medium"}},
        "events": {"actions": [{"description_en": "nina stares ahead"}]},
        "panels": [{"id": "scene_01_shot_01_panel_0001", "panel_number": 1}],
    }
    return {"scene_id": "scene_01", "shots": [shot]}, shot


def _record(tmp_path, rec):
    (tmp_path / "scene_01_shot_01_panel_0001_90001_.png.props.json").write_text(json.dumps(rec))


def test_the_build_record_says_which_staged_props_are_out_of_frame(tmp_path):
    _record(tmp_path, {"car_console": {"in_frame": False, "frame_share": 0.0},
                       "cabin_panels": {"in_frame": True, "frame_share": 0.41}})
    assert props_out_of_frame(tmp_path, "scene_01_shot_01_panel_0001") == {"car_console"}


def test_with_no_build_record_nothing_is_left_out(tmp_path):
    assert props_out_of_frame(tmp_path, "scene_01_shot_01_panel_0001") == frozenset()
    assert props_out_of_frame(None, "scene_01_shot_01_panel_0001") == frozenset()


def test_an_eyeline_to_a_prop_out_of_frame_does_not_name_it():
    _scene, shot = _cabin_scene()
    assert gaze_clauses_of(shot) == ["nina gazing at the car console"]
    assert gaze_clauses_of(shot, {"car_console"}) == ["nina gazing at something out of frame"]


def test_a_prop_staged_behind_the_lens_is_not_named(tmp_path):
    scene, shot = _cabin_scene()
    _record(tmp_path, {"car_console": {"in_frame": False, "frame_share": 0.0},
                       "game_device": {"in_frame": True, "frame_share": 0.02}})
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], CompileContext(greyboxes_dir=tmp_path))
    assert "console" not in pos
    assert "game device" in pos


def test_the_same_prop_is_named_when_the_build_has_not_said(tmp_path):
    scene, shot = _cabin_scene()
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], CompileContext(greyboxes_dir=tmp_path))
    assert "car console" in pos


# ── what the panel declares in frame outranks the build's record ──

def _compile(tmp_path, shot, scene):
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], CompileContext(greyboxes_dir=tmp_path))
    return pos


def test_a_declaration_in_frame_outranks_the_build_record(tmp_path):
    scene, shot = _cabin_scene()
    _record(tmp_path, {"car_console": {"in_frame": False, "frame_share": 0.0}})
    shot["setup"]["props"][0]["in_frame"] = "yes"
    pos = _compile(tmp_path, shot, scene)
    assert "car console" in pos and "gazing at the car console" in pos


def test_a_prop_declared_out_of_frame_is_not_named_without_a_build(tmp_path):
    scene, shot = _cabin_scene()
    shot["setup"]["props"][0]["in_frame"] = "no"
    pos = _compile(tmp_path, shot, scene)
    assert "console" not in pos and "gazing at something out of frame" in pos


def test_a_partly_framed_prop_says_how_much_of_it_is_seen(tmp_path):
    scene, shot = _cabin_scene()
    shot["setup"]["props"][0].update(in_frame="partial",
                                     in_frame_extent="its right end at the bottom edge")
    assert "car console (only its right end at the bottom edge visible)" in _compile(tmp_path, shot, scene)


def test_to_be_confirmed_defers_to_the_build_record(tmp_path):
    scene, shot = _cabin_scene()
    _record(tmp_path, {"car_console": {"in_frame": False, "frame_share": 0.0}})
    shot["setup"]["props"][0]["in_frame"] = "tbd"
    assert "console" not in _compile(tmp_path, shot, scene)
