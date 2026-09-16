"""The breakdown's criteria come from the skills package, not from constants.

Three decomposition criteria used to live here as prompt strings and
module-level tuples: where a scene ends, how a film's theme decides the way
each stretch is shot, and where a beat begins. A criterion written in two
places is a criterion that will be changed in one of them, and the copy a
person reads was never going to be the copy the program sends.

These tests pin the direction of the dependency. They do not check the values
themselves -- `pace-scene-skills` owns those and tests them there.
"""
from __future__ import annotations

import pathlib

import pytest

from pace_core.breakdown import beat_segmenter, derive_shot_design, split_script

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/pace_core/breakdown"


def test_the_scene_criterion_is_the_skills_text():
    s = pytest.importorskip("pace_scene_skills").load("split-into-scenes")
    assert split_script.SYSTEM_PROMPT == s.instructions


def test_the_shot_design_brief_is_the_skills_text():
    s = pytest.importorskip("pace_scene_skills").load("derive-shot-design")
    assert derive_shot_design.SYSTEM_PROMPT == s.instructions


def test_the_vocabularies_are_the_skills_vocabularies():
    v = pytest.importorskip("pace_scene_skills").load(
        "derive-shot-design").reference("vocabulary.yaml")
    assert derive_shot_design.SHOT_SIZES == tuple(v["shot_size"])
    assert derive_shot_design.ANGLES == tuple(v["angle"])
    assert derive_shot_design.MOVEMENTS == tuple(v["camera_movement"])


def test_the_segmenter_parameters_are_the_skills_parameters():
    p = pytest.importorskip("pace_scene_skills").load(
        "segment-on-state-change").reference("parameters.yaml")
    assert beat_segmenter.WEIGHTS == p["weights"]
    assert beat_segmenter.MEASURABLE == tuple(p["measurable_from_ir"])
    assert beat_segmenter.TAU_HIGH == p["thresholds"]["tau_high"]
    assert beat_segmenter.TAU_LOW == p["thresholds"]["tau_low"]
    assert beat_segmenter.STRONG_TRIGGERS == p["sufficient_alone"]


@pytest.mark.parametrize("module,marker", [
    ("split_script.py", 'SYSTEM_PROMPT = """You are a script supervisor'),
    ("derive_shot_design.py", 'SYSTEM_PROMPT = f"""You are a director'),
])
def test_the_prompt_is_not_written_here_again(module, marker):
    """A literal prompt reappearing in these files is the regression: the
    skill would still load, and nothing would be sending it."""
    assert marker not in (SRC / module).read_text()


def test_the_numbers_are_not_written_here_again():
    t = (SRC / "beat_segmenter.py").read_text()
    for literal in ("0.55, 0.20", '"state_change":     0.30', '"goal_shift": 0.7'):
        assert literal not in t, f"{literal} is back as a constant"


def test_the_segmenter_still_scores_the_way_it_did():
    """Reading the weights from a file must not change what they mean. A
    boundary carrying every measurable signal splits; one carrying none
    merges."""
    hot = {k: 1.0 for k in beat_segmenter.MEASURABLE}
    cold = {k: 0.0 for k in beat_segmenter.MEASURABLE}
    assert beat_segmenter.boundary_score(hot) == pytest.approx(1.0)
    assert beat_segmenter.boundary_score(cold) == pytest.approx(0.0)
    assert beat_segmenter.strong_trigger({"state_change": 1.0}) == "state_change"
    assert beat_segmenter.strong_trigger({"focus_shift": 1.0}) is None
