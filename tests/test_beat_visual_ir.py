"""What a panel must SHOW, written down so a verifier can check it.

Between a beat and its geometry there was nothing: the panel builder went
straight from "these events happened" to staged geometry, so the only claim on
record was geometric and the image verifier could check nothing else.

Both defects pinned below were found by running this over the real corpus, not
by imagining failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.beat_visual_ir import (  # noqa: E402
    INFERRED, SCRIPT, VISUALIZATION, facts_from_beat, moment_for, plan,
)

CAST = {"gus", "fay", "hal"}


def _beat(**over):
    b = {"id": "s_beat_01", "scene_id": "scene_07",
         "source_event_ids": ["e11", "e12"],
         "transition": [
             {"predicate": "FALL_ON_TOP_OF", "actor": "the_car", "patient": "gus"},
             {"predicate": "FADE_SCREAMS", "actor": "hal", "patient": None}],
         "state_before": {"fay.alive": "false"},
         "state_after": {"fay.alive": "false", "gus.injured": "crushed"},
         "state_changes": [{"entity": "gus", "attribute": "injured",
                            "to": "crushed"}]}
    b.update(over)
    return b


# ── the two defects the corpus exposed ────────────────────────────────────

def test_inherited_state_does_not_put_an_absent_character_in_frame():
    """The climax beat carries `fay.alive = false` from a wreck two scenes
    earlier. She is not in that frame, and asking a VQA model whether she is
    visible gets a confident wrong answer either way."""
    ir = plan(_beat(), CAST)
    assert "fay" not in ir.subjects
    assert not any(q["about"] == "fay" for q in ir.verification_questions)
    # ...but the fact is still RECORDED, just not required
    fay = [f for f in ir.visible_facts if f["subject"] == "fay"]
    assert fay and fay[0]["provenance"] == INFERRED
    assert fay[0]["required"] is False


def test_a_predicate_ending_in_a_preposition_gets_its_object():
    """"Is the car falling on top of?" is not answerable, and a model asked an
    unanswerable question still answers."""
    ir = plan(_beat(), CAST)
    asks = [q["ask"] for q in ir.verification_questions]
    assert "Is the car falling on top of gus?" in asks
    for a in asks:
        assert a.rstrip("?").split()[-1] not in {"of", "on", "at", "to", "with"}, a


# ── dependency structure ──────────────────────────────────────────────────

def test_every_question_about_a_subject_is_gated_on_its_presence():
    """Asking whether a man is kneeling in a frame containing no man produces
    an answer, and the answer is noise."""
    ir = plan(_beat(), CAST)
    by_id = {q["id"]: q for q in ir.verification_questions}
    for q in ir.verification_questions:
        if q["ask"].endswith("visible in the image?"):
            assert q["depends_on"] == []
        else:
            assert q["depends_on"], q
            gate = by_id[q["depends_on"][0]]
            assert gate["about"] == q["about"]
            assert "visible" in gate["ask"]


def test_a_non_cast_actor_still_gets_a_presence_gate():
    """A car is not a cast member, but "is the car falling on him" still needs
    a frame with a car in it."""
    ir = plan(_beat(), CAST)
    assert any(q["ask"] == "Is the car visible in the image?"
               for q in ir.verification_questions)


# ── required vs optional, and provenance ──────────────────────────────────

def test_only_this_beat_s_own_facts_are_required():
    """A verifier that hard-fails every unstated detail fails a panel for not
    drawing a ladder in the background."""
    ir = plan(_beat(), CAST)
    assert all(f["provenance"] == SCRIPT for f in ir.required())
    assert any(f["provenance"] == INFERRED and not f["required"]
               for f in ir.visible_facts)


def test_the_state_the_beat_causes_is_required():
    ir = plan(_beat(), CAST)
    crushed = [f for f in ir.required()
               if f["subject"] == "gus" and f["predicate"] == "injured"]
    assert crushed and crushed[0]["value"] == "crushed"


def test_a_visualization_is_never_script_truth():
    """The vocabulary must keep the three apart: the corpus already had one
    field whose table constant was read as direction."""
    assert SCRIPT != INFERRED != VISUALIZATION
    facts = facts_from_beat(_beat(), CAST)
    assert all(f.provenance in (SCRIPT, INFERRED) for f in facts)


# ── visual moment ─────────────────────────────────────────────────────────

def test_a_beat_is_a_span_and_the_panel_picks_an_instant():
    m, why = moment_for(_beat())
    assert m == "ACTION_PEAK" and "FALL_ON_TOP_OF" in why


def test_a_held_state_reads_after_the_action():
    m, _ = moment_for(_beat(transition=[{"predicate": "LOSE_POWER",
                                         "actor": "panels"}], state_changes=[]))
    assert m == "POST_ACTION"


def test_a_beat_with_no_action_is_not_mid_way_through_one():
    m, why = moment_for(_beat(transition=[]))
    assert m == "POST_ACTION" and "no action" in why


def test_an_unrecognised_predicate_still_gets_a_moment():
    m, _ = moment_for(_beat(transition=[{"predicate": "BLORP", "actor": "gus"}],
                            state_changes=[]))
    assert m in ("MID_ACTION", "ACTION_PEAK", "PRE_ACTION",
                 "ACTION_START", "POST_ACTION")


# ── the question has to be answerable ─────────────────────────────────────

def test_gerunds_are_formed_correctly():
    """`word + "ing"` gave "pauseing" and "changeing", and because
    endswith("ing") is true of SING it gave "Is fay sing?". A VQA model
    asked a malformed question still answers, so this is not cosmetic."""
    from pace_core.breakdown.beat_visual_ir import _gerund
    assert _gerund("pause") == "pausing"
    assert _gerund("change") == "changing"
    assert _gerund("sing") == "singing"
    assert _gerund("lie") == "lying"
    assert _gerund("sit") == "sitting"
    assert _gerund("press") == "pressing"


def test_an_existing_gerund_is_left_alone():
    from pace_core.breakdown.beat_visual_ir import _gerund
    assert _gerund("traveling") == "traveling"


def test_a_plural_subject_gets_a_plural_verb():
    """"Is vehicles traveling fast?" is not a question."""
    ir = plan(_beat(transition=[{"predicate": "TRAVEL_FAST",
                                 "actor": "vehicles", "patient": None}],
                    source_event_ids=["e1"], state_changes=[],
                    state_before={}), CAST)
    asks = [q["ask"] for q in ir.verification_questions]
    assert any(a.startswith("Are vehicles") for a in asks), asks
    assert not any(a.startswith("Is vehicles") for a in asks), asks


def test_a_collective_noun_is_singular():
    from pace_core.breakdown.beat_visual_ir import _be
    assert _be("family") == "Is"
    assert _be("panels") == "Are"
