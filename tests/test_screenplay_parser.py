"""Pass 0 segments a screenplay and keeps the offsets.

The offsets are the point. Without them no field can cite the words that
support it, and a value produced by a lookup table reads exactly like a value
read off the page -- which is how every `screen_position.x` in the corpus came
to be a constant chosen by list index. An evidence span that cannot be sliced
back out of the source is not evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.screenplay_parser import (  # noqa: E402
    parse, scenes, verify_spans,
)

SAMPLE = """FADE IN:

EXT. CITY WALL - NIGHT

SUPER: 2035

Enemy soldiers raise ladders against the wall.

OLD SOLDIER
(shouting)
Hold the line!

He pulls the young soldier upright.

INT. GUARD HOUSE - CONTINUOUS

A lantern burns low.

YOUNG SOLDIER (CONT'D)
I can still fight.
"""


def _kinds(els):
    return [(e.kind, e.text) for e in els]


def test_every_element_slices_back_out_of_the_source():
    els = parse(SAMPLE)
    assert verify_spans(SAMPLE, els) == []


def test_scene_headings_are_found_and_numbered():
    sc = scenes(parse(SAMPLE))
    assert [s["heading"] for s in sc] == [
        "EXT. CITY WALL - NIGHT", "INT. GUARD HOUSE - CONTINUOUS"]
    assert [s["index"] for s in sc] == [0, 1]


def test_a_transition_is_not_an_action():
    els = parse(SAMPLE)
    assert ("transition", "FADE IN:") in _kinds(els)


def test_a_super_is_not_a_character_cue():
    """`SUPER: 2035` is all caps and short, so the cue heuristic would take it
    for a speaker if the super rule did not run first."""
    els = parse(SAMPLE)
    assert ("super", "SUPER: 2035") in _kinds(els)
    assert not any(k == "character" and t.startswith("SUPER") for k, t in _kinds(els))


def test_a_cue_carries_its_speech_and_its_parenthetical():
    els = parse(SAMPLE)
    speech = [e for e in els if e.kind == "dialogue"]
    assert ("Hold the line!", "OLD SOLDIER") in [(e.text, e.speaker) for e in speech]
    paren = [e for e in els if e.kind == "parenthetical"]
    assert ("(shouting)", "OLD SOLDIER") in [(e.text, e.speaker) for e in paren]


def test_a_contd_suffix_does_not_become_part_of_the_name():
    els = parse(SAMPLE)
    assert any(e.speaker == "YOUNG SOLDIER" for e in els if e.kind == "dialogue")


def test_speech_ends_at_a_blank_line():
    """`He pulls the young soldier upright.` follows dialogue but is action.
    Treating it as speech would attribute a stage direction to a character."""
    els = parse(SAMPLE)
    action = [e.text for e in els if e.kind == "action"]
    assert "He pulls the young soldier upright." in action


def test_a_caps_line_with_no_speech_under_it_is_action():
    """A shouted action line is all caps and short, so only the line AFTER it
    can decide whether it was a cue."""
    els = parse("EXT. ROAD - DAY\n\nSILENCE.\n\nA car passes.\n")
    kinds = dict((t, k) for k, t in _kinds(els))
    assert kinds["SILENCE."] == "action"


def test_a_page_number_left_by_the_pdf_is_dropped():
    els = parse("EXT. ROAD - DAY\n\n2.\n\nA car passes.\n")
    assert all(e.text != "2." for e in els)


def test_offsets_are_into_the_text_as_given_not_a_normalised_copy():
    """If the parser normalised whitespace before recording offsets, every
    span would point a few characters off -- silently, and worse the further
    into the script you read."""
    els = parse(SAMPLE)
    heading = next(e for e in els if e.kind == "scene_heading")
    assert SAMPLE[heading.start:heading.end] == "EXT. CITY WALL - NIGHT"
