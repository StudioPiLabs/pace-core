"""The action names the cast it acts on, and the prompt does not contradict it with a default posture."""
from pace_core.compilers.compile_common import CompileContext
from pace_core.compilers.compile_flux2 import compile_flux2

KB = {"lena": {"name": "Lena", "sex": "female", "generic_anchor": "slender, dark curly hair"},
      "ryan": {"name": "Ryan", "sex": "male", "generic_anchor": "slim build, short dark hair"},
      "ann": {"name": "Ann", "sex": "female", "generic_anchor": "tall, red hair"}}


def prompt(action, subjects):
    shot = {"shot_id": "shot_01",
            "setup": {"backdrop": {"setting": "ext", "location": "a village gate"},
                      "subjects": subjects, "primary_focus": {"type": "character", "ref": subjects[0]["character_id"]}},
            "camera": {"creative_intent": {"shot_size": "medium"}},
            "events": {"actions": [{"description_en": action}]},
            "panels": [{"id": "p1"}]}
    scene = {"scene_id": "s1", "shots": [shot]}
    return compile_flux2(scene, shot, shot["panels"][0], CompileContext(characters_kb=KB))[0]


def test_a_named_actor_is_named_in_its_description():
    p = prompt("Lena turns to Ryan and asks a question.",
               [{"character_id": "lena", "screen_position": {"x": 0.39}, "pose": "standing"},
                {"character_id": "ryan", "screen_position": {"x": 0.61}, "pose": "standing"}])
    assert "Lena, " in p.split("Lena turns")[0] and "Ryan, " in p.split("Lena turns")[0]


def test_a_name_inside_another_word_binds_nothing():
    p = prompt("Anna waves.", [{"character_id": "ann", "pose": "standing"}])
    assert "Ann, " not in p


def test_standing_is_not_written_but_a_marked_posture_is():
    standing = prompt("Lena kneels by the gate.", [{"character_id": "lena", "pose": "standing"}])
    assert "standing" not in standing
    seated = prompt("Lena waits.", [{"character_id": "lena", "pose": "seated"}])
    assert "seated" in seated
