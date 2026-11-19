"""KB keys are opaque: text names an entry by its name and matches it by its names."""
from __future__ import annotations

import json
import re

from pace_core.breakdown.extract_characters import merge_into_characters_json
from pace_core.breakdown.split_script import _match_location
from pace_core.compilers.compile_common import CompileContext
from pace_core.compilers.compile_flux2 import compile_flux_for_base

ALICE, BOB = "3f1c0a7e9b2d4c6e8a0b1c2d3e4f5a6b", "7d2e4f6a8b0c4d1e9f3a5b7c9d1e3f5a"
CONSOLE, LAMP = "a4b6c8d0e2f44a6b8c0d2e4f6a8b0c2d", "c1d3e5f7a9b14c3d9e7f9a1b3c5d7e9f"
ATTIC = "0cafe0bed0d34f6e8a0b1c2d3e4f5a6b"         # hex words "cafe" and "bed" inside
KEYS = (ALICE, BOB, CONSOLE, LAMP, ATTIC)
UUID4_HEX = re.compile(r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}")


def test_a_flux_prompt_names_the_cast_and_location_and_prints_no_key():
    scene = {"scene_id": "scene_01", "narrative_meta": {"time_of_day": "night"}, "shots": [{
        "shot_id": "shot_01",
        "setup": {
            "backdrop": {"setting": "int", "location": ATTIC},
            "subjects": [
                {"character_id": ALICE, "gaze": {"target_type": "character", "target_ref": BOB}},
                {"character_id": BOB, "gaze": {"target_type": "object", "target_ref": CONSOLE}},
            ],
            "props": [{"prop_id": CONSOLE}],
            "primary_focus": {"type": "character", "ref": ALICE},
            "excluded": [LAMP],
        },
        "camera": {"extrinsics": {"angle": "eye_level"}, "creative_intent": {"shot_size": "medium"}},
        "events": {"actions": [{"description_en": "they talk quietly"}]},
        "panels": [{"id": "scene_01_shot_01_panel_0001", "panel_number": 1}],
    }]}
    ctx = CompileContext(
        characters_kb={ALICE: {"name": "Alice"}, BOB: {"name": "Bob"}},
        props_kb={"props": {CONSOLE: {"name": "main console"}, LAMP: {"name": "desk lamp"}}},
        location_stubs={ATTIC: {"name": "lighthouse attic"}})
    shot = scene["shots"][0]
    positive, negative = compile_flux_for_base(None, scene, shot, shot["panels"][0], ctx)
    for name in ("Alice", "Bob", "main console", "lighthouse attic"):
        assert name in positive
    assert "desk lamp" in negative
    assert not any(k in positive or k in negative for k in KEYS)


def test_extraction_merges_a_known_name_and_mints_a_uuid4_key_for_a_new_one(tmp_path):
    chars = tmp_path / "characters.json"
    chars.write_text(json.dumps({ALICE: {"name": "Alice", "anchor": "artist anchor"}}))
    created, updated, keys = merge_into_characters_json(
        [{"name": "Alice", "anchor": "model anchor", "scenes": ["scene_02"]},
         {"name": "Carol", "anchor": "a tall woman"}], chars)
    registry = json.loads(chars.read_text())
    assert keys[0] == ALICE and registry[ALICE]["anchor"] == "artist anchor"
    assert (created, updated) == (1, 1)
    assert UUID4_HEX.fullmatch(keys[1])
    assert registry[keys[1]]["name"] == registry[keys[1]]["trigger"] == "Carol"


def test_a_script_location_matches_a_bible_by_name_never_by_its_key():
    bibles = {ATTIC: {"name": "Lighthouse Attic"}}
    assert _match_location("INT. LIGHTHOUSE ATTIC - NIGHT", bibles) == ATTIC
    assert _match_location("cafe", bibles) is None
    assert _match_location("bed", bibles) is None
