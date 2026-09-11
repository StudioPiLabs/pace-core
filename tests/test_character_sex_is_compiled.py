"""A character's declared sex reaches the prompt.

`sex` is authored on every character in both corpora and was read by no
compiler. The descriptors it left behind are true of either sex — "athletic
build, Caucasian, with short greying dark hair, weathered features and a
determined expression" — and the geometry does not settle it: every subject is
staged from the same gender="neutral" SMPL-X proxy, whose chest measures 1.03x
its waist.

With no LoRA trained for this cast, that left exactly one sex-bearing token in
the whole prompt: the inert trigger word `emily_female`, emitted by
lora_trigger_for for the FIRST-listed character only and attached to no screen
position. scene_02_shot_02_panel_0002 rendered its left-hand male character as
a woman — the sampler answering a question nothing in the specification had
answered.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.compilers.compile_common import _character_hint  # noqa: E402

KB = {
    "ryan":  {"sex": "male",   "generic_anchor": "athletic build, weathered features"},
    "emily": {"sex": "female", "generic_anchor": "slender build, lively eyes"},
    "kid":   {"sex": "male",   "generic_anchors": {"child_7": "lean build, restless"}},
    "teen":  {"sex": "female", "generic_anchors": {"adult_16": "tall, quiet"}},
    "nosex": {"generic_anchor": "average build, unremarkable"},
    "said":  {"sex": "male",   "generic_anchor": "an older man, stooped"},
}


def test_a_male_character_is_named_a_man():
    assert "man," in _character_hint("ryan@adult_50", KB)


def test_a_female_character_is_named_a_woman():
    out = _character_hint("emily@adult_50", KB)
    assert "woman," in out and "man," not in out.replace("woman,", "")


def test_a_child_is_named_by_age_not_as_an_adult():
    assert "boy," in _character_hint("kid@child_7", KB)


def test_a_teenager_is_not_called_a_child_or_an_adult():
    assert "teenage girl," in _character_hint("teen@adult_16", KB)


def test_a_descriptor_that_already_says_who_it_is_is_left_alone():
    """Doubling it would read 'man, an older man, stooped'."""
    out = _character_hint("said@adult_60", KB)
    assert out.count("man") == 1


def test_a_character_with_no_declared_sex_is_not_invented():
    out = _character_hint("nosex@adult_40", KB)
    for w in ("man,", "woman,", "boy,", "girl,"):
        assert w not in out


def test_the_age_prefix_still_comes_first():
    """'50-year-old man, athletic build' — not 'man, 50-year-old ...'."""
    out = _character_hint("ryan@adult_50", KB)
    assert out.startswith("50-year-old man,")
