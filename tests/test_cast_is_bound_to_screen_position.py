"""A multi-person prompt names its cast in the order they stand on screen.

The greybox stages every body from `screen_position`. The compiled prompt used
to name the same people in registry order with no spatial anchor at all, so
geometry said one thing and the text said nothing about place: across repeated
renders of one panel the subject count and the positions came out right while
the *people* moved between seats, because nothing tied a description to a
side. Both halves now read the same declared `screen_position`.

The single-subject case is deliberately left alone: the composition solver has
already aimed the camera at that subject, and a side named in the text would
compete with the solve rather than agree with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.compilers.compile_common import (  # noqa: E402
    screen_x_phrase, subjects_left_to_right,
)


def _shot(*subjects: dict) -> dict:
    return {"setup": {"subjects": list(subjects)}}


def test_subjects_come_back_in_screen_order_not_registry_order():
    shot = _shot(
        {"character_id": "fay", "screen_position": {"x": 0.50}},
        {"character_id": "gus",  "screen_position": {"x": 0.38}},
        {"character_id": "hal", "screen_position": {"x": 0.62}},
    )
    assert [s["character_id"] for s in subjects_left_to_right(shot)] == \
        ["gus", "fay", "hal"]


def test_each_subject_carries_the_phrase_for_where_it_stands():
    shot = _shot(
        {"character_id": "gus",  "screen_position": {"x": 0.38}},
        {"character_id": "fay", "screen_position": {"x": 0.50}},
        {"character_id": "hal", "screen_position": {"x": 0.62}},
    )
    got = {s["character_id"]: s["phrase"] for s in subjects_left_to_right(shot)}
    assert got == {"gus": "on the left", "fay": "in the centre",
                   "hal": "on the right"}


def test_a_zone_and_the_x_it_means_get_the_same_words():
    """A subject declared `center_left` and one declared x=0.38 are the same
    staging said two ways, and must not read differently to the encoder."""
    by_zone = subjects_left_to_right(
        _shot({"character_id": "a", "screen_position": {"zone": "center_left"}}))
    by_x = subjects_left_to_right(
        _shot({"character_id": "a", "screen_position": {"x": 0.38}}))
    assert by_zone[0]["phrase"] == by_x[0]["phrase"]


def test_an_unplaced_subject_sorts_last_and_is_not_given_a_side():
    """An unplaced subject is a KB gap. Inventing a side for it would put the
    text and the geometry back into disagreement in the one case where the
    disagreement is real."""
    shot = _shot(
        {"character_id": "placed",   "screen_position": {"x": 0.30}},
        {"character_id": "unplaced"},
    )
    out = subjects_left_to_right(shot)
    assert [s["character_id"] for s in out] == ["placed", "unplaced"]
    assert out[-1]["phrase"] is None


def test_declaration_order_breaks_ties_between_two_unplaced_subjects():
    shot = _shot({"character_id": "first"}, {"character_id": "second"})
    assert [s["character_id"] for s in subjects_left_to_right(shot)] == \
        ["first", "second"]


def test_the_phrase_bands_run_left_to_right():
    xs = [0.05, 0.30, 0.50, 0.70, 0.95]
    phrases = [screen_x_phrase(x) for x in xs]
    assert phrases == ["at the far left of the frame", "on the left",
                       "in the centre", "on the right",
                       "at the far right of the frame"]


def test_a_subject_with_no_character_id_is_dropped():
    shot = _shot({"screen_position": {"x": 0.5}},
                 {"character_id": "real", "screen_position": {"x": 0.4}})
    assert [s["character_id"] for s in subjects_left_to_right(shot)] == ["real"]


def test_a_group_prompt_anchors_every_description(monkeypatch):
    from pace_core.compilers import compile_flux2

    scene = {"scene_id": "s", "shots": []}
    shot = {
        "shot_id": "shot_01",
        "camera": {"creative_intent": {"shot_size": "medium"}},
        "setup": {"subjects": [
            {"character_id": "fay", "screen_position": {"x": 0.50}},
            {"character_id": "gus",  "screen_position": {"x": 0.38}},
            {"character_id": "hal", "screen_position": {"x": 0.62}},
        ]},
    }
    panel = {"id": "p", "panel_number": 1}
    kb = {c: {"generic_anchors": {"default": f"{c}-looks"}, "anchor": f"{c}-looks"}
          for c in ("fay", "gus", "hal")}
    ctx = compile_flux2.CompileContext(characters_kb=kb)
    pos, _ = compile_flux2.compile_flux_for_base(
        "flux2_dev_fp8mixed.safetensors", scene, shot, panel, ctx)

    # left to right, and each anchored
    assert pos.index("on the left") < pos.index("in the centre") < pos.index("on the right")
    assert pos.index("gus-looks") < pos.index("fay-looks") < pos.index("hal-looks")


def test_a_single_subject_panel_is_not_given_a_side(monkeypatch):
    from pace_core.compilers import compile_flux2

    scene = {"scene_id": "s", "shots": []}
    shot = {
        "shot_id": "shot_01",
        "camera": {"creative_intent": {"shot_size": "medium"}},
        "setup": {"subjects": [
            {"character_id": "fay", "screen_position": {"x": 0.38}},
        ]},
    }
    panel = {"id": "p", "panel_number": 1}
    ctx = compile_flux2.CompileContext(
        characters_kb={"fay": {"generic_anchors": {"default": "fay-looks"},
                                 "anchor": "fay-looks"}})
    pos, _ = compile_flux2.compile_flux_for_base(
        "flux2_dev_fp8mixed.safetensors", scene, shot, panel, ctx)

    assert "fay-looks" in pos
    assert "on the left" not in pos
