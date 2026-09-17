"""Measuring the verifier: does an injected error actually reach it?

Two halves. The first checks the mutations do what they claim to the text --
a benchmark built on a mutation that silently failed to apply reports a
verifier blind spot that is really a harness bug.

The second is the part that needs no API call and is the stronger result:
seven of the design's twelve mutations cannot be detected by this verifier at
all, and rather than assert that, `test_invisible_mutations_do_not_change_the_prompt`
compiles the estimator's messages before and after each one and requires them
to be identical. A mutation the model is never shown cannot be a model failure.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.inject_errors import (  # noqa: E402
    INVISIBLE_TO_PROMPT, NO_OUTPUT_FIELD, WITH_CHANNEL, Mutation, apply, plan,
    score_detection, signals, swap_names,
)
from pace_core.breakdown.verify_breakdown import build_messages  # noqa: E402


def _doc(sid: str, actions: list[str]) -> dict:
    return {"scene_id": sid, "shots": [
        {"shot_id": f"shot_{i:02d}",
         "events": {"actions": [{"description_en": a}]}}
        for i, a in enumerate(actions, start=1)]}


SCRIPT_IR = [{
    "index": 0, "heading": "EXT. HIGHWAY - DAY", "start": 0, "end": 100,
    "entities": [
        {"local_id": "gus", "type": "CHARACTER", "canonical_name": "GUS",
         "aliases": ["he"]},
        {"local_id": "hal", "type": "CHARACTER", "canonical_name": "HAL",
         "aliases": []},
    ],
    "events": [
        {"local_id": "e1", "predicate": "PULL_FREE", "actor": "gus",
         "patient": "hal", "order": 1, "importance": 0.9,
         "evidence": {"source_text": "Gus pulls Hal free"}},
        {"local_id": "e2", "predicate": "STAND", "actor": "hal",
         "patient": None, "order": 2, "importance": 0.4,
         "evidence": {"source_text": "Hal stands"}},
    ],
    "states": [],
}]

BASELINE = {"alignments": [{
    "script_scene": 0,
    "matches": [
        {"script_event": "e1", "breakdown_actions": ["scene_01/shot_01/a0"],
         "match": "EXACT", "role_ok": True},
        {"script_event": "e2", "breakdown_actions": ["scene_01/shot_02/a0"],
         "match": "SEMANTIC", "role_ok": True},
    ],
    "invented_events": [], "over_specified": [],
}]}

ACTIONS = {"scene_01/shot_01/a0": "Gus drags Hal out of the wreck.",
           "scene_01/shot_02/a0": "Hal gets to his feet."}
MAPPING = {"0": ["scene_01"]}


# ── the mutations do what they say ────────────────────────────────────────

def test_swap_names_keeps_the_surface_form():
    """The IR spells names in screenplay caps; the text may not."""
    out = swap_names("Gus drags Hal out.", "GUS", "HAL")
    assert out == "Hal drags Gus out."


def test_swap_names_is_one_pass():
    """A naive two-step replace puts both names back as the same one."""
    assert swap_names("a then b", "a", "b") == "b then a"


def test_swap_names_leaves_text_alone_when_a_name_is_absent():
    assert swap_names("Gus alone.", "GUS", "HAL") == "Gus alone."


def _apply(kind: str, action_id: str, event: str | None, expect: str):
    docs = {"scene_01": _doc("scene_01", [ACTIONS["scene_01/shot_01/a0"],
                                          ACTIONS["scene_01/shot_02/a0"]])}
    mu = Mutation(kind, 0, "scene_01", action_id, event, expect,
                  "GUS <-> HAL" if kind.startswith("Swap") else "")
    landed = apply([mu], docs, SCRIPT_IR)
    return docs, landed


def test_delete_event_removes_the_action():
    docs, landed = _apply("DeleteEvent", "scene_01/shot_01/a0", "e1", "MISSING")
    assert len(landed) == 1
    assert docs["scene_01"]["shots"][0]["events"]["actions"] == []
    # and leaves the other shot alone
    assert docs["scene_01"]["shots"][1]["events"]["actions"]


def test_insert_event_appends_and_renumbers_its_own_id():
    docs, landed = _apply("InsertEvent", "scene_01/shot_01/a0", None, "INVENTED")
    acts = docs["scene_01"]["shots"][0]["events"]["actions"]
    assert len(acts) == 2
    # The manifest must name the INSERTED action, not the one it was seeded from.
    assert landed[0].action_id == "scene_01/shot_01/a1"


def test_specialize_keeps_the_event_and_adds_only_craft():
    docs, landed = _apply("SpecializePredicate", "scene_01/shot_01/a0", "e1",
                          "OVERSPEC")
    t = docs["scene_01"]["shots"][0]["events"]["actions"][0]["description_en"]
    assert t.startswith("Gus drags Hal out of the wreck")
    assert len(t) > len(ACTIONS["scene_01/shot_01/a0"])


def test_craft_detail_introduces_no_new_participant():
    """The one test that separates over-specification from fabrication is
    void if the appended clause itself invents someone."""
    from pace_core.breakdown.inject_errors import CRAFT_DETAIL
    for c in CRAFT_DETAIL:
        assert not any(w in c.lower().split()
                       for w in ("his", "her", "their", "he", "she", "they"))


def test_swap_actor_reverses_the_roles():
    docs, _ = _apply("SwapActor", "scene_01/shot_01/a0", "e1", "ROLE")
    assert (docs["scene_01"]["shots"][0]["events"]["actions"][0]["description_en"]
            == "Hal drags Gus out of the wreck.")


def test_generalize_keeps_the_actor_and_drops_the_predicate():
    docs, _ = _apply("GeneralizePredicate", "scene_01/shot_01/a0", "e1", "OVERGEN")
    t = docs["scene_01"]["shots"][0]["events"]["actions"][0]["description_en"]
    assert "gus" in t.lower() and "drags" not in t.lower()


# ── planning ──────────────────────────────────────────────────────────────

def test_plan_never_puts_two_mutations_in_one_shot():
    """A second edit fights the first, and a delete renumbers the rest."""
    got = plan(BASELINE, SCRIPT_IR, ACTIONS, MAPPING, per_scene=99)
    shots = ["/".join(m.action_id.split("/")[:2]) for m in got]
    assert len(shots) == len(set(shots))


def test_plan_only_mutates_what_the_baseline_says_is_covered():
    """An error injected where the verifier was already failing proves
    nothing about whether it can see the injection."""
    blind = {"alignments": [{**BASELINE["alignments"][0], "matches": [
        {"script_event": "e1", "breakdown_actions": ["scene_01/shot_01/a0"],
         "match": "MISSING", "role_ok": True}]}]}
    got = plan(blind, SCRIPT_IR, ACTIONS, MAPPING, per_scene=99)
    assert all(m.kind == "InsertEvent" for m in got)


def test_plan_only_emits_kinds_with_a_channel():
    got = plan(BASELINE, SCRIPT_IR, ACTIONS, MAPPING, per_scene=99)
    assert {m.kind for m in got} <= set(WITH_CHANNEL)


# ── scoring ───────────────────────────────────────────────────────────────

def _run(missing=(), invented=(), overspec=(), overgen=(), role=()):
    return {"alignments": [{
        "script_scene": 0,
        "matches": ([{"script_event": e, "match": "MISSING", "role_ok": True}
                     for e in missing]
                    + [{"script_event": e, "match": "OVER_GENERALIZED",
                        "role_ok": True} for e in overgen]
                    + [{"script_event": e, "match": "EXACT", "role_ok": False}
                       for e in role]),
        "invented_events": [{"action": a} for a in invented],
        "over_specified": [{"action": a} for a in overspec],
    }]}


def test_role_ok_on_a_missing_event_is_not_a_detection():
    """Mirrors verify_breakdown.score: nothing matched, so nothing was
    swapped. Counting those once turned 2 role errors into 30."""
    r = {"alignments": [{"script_scene": 0, "matches": [
        {"script_event": "e1", "match": "MISSING", "role_ok": False}]}]}
    assert ("ROLE", 0, "e1") not in signals(r)


def test_recall_counts_only_signals_that_appeared():
    base = _run(missing=["e9"])
    mut = _run(missing=["e9", "e1"])
    mus = [Mutation("DeleteEvent", 0, "scene_01", "scene_01/shot_01/a0",
                    "e1", "MISSING")]
    r = score_detection(base, mut, mus)
    assert (r["recall"], r["precision"], r["true_detected"]) == (1.0, 1.0, 1)


def test_an_error_already_in_the_baseline_is_not_credited():
    """The corpus breakdown has 48 genuinely missing events. Scoring the
    mutated run absolutely would credit the verifier for all of them."""
    base = _run(missing=["e1"])
    mus = [Mutation("DeleteEvent", 0, "scene_01", "scene_01/shot_01/a0",
                    "e1", "MISSING")]
    r = score_detection(base, _run(missing=["e1"]), mus)
    assert r["recall"] == 0.0 and r["true_detected"] == 0


def test_collateral_detections_cost_precision():
    base = _run()
    mut = _run(missing=["e1", "e2"])          # e2 was not injected
    mus = [Mutation("DeleteEvent", 0, "scene_01", "scene_01/shot_01/a0",
                    "e1", "MISSING")]
    r = score_detection(base, mut, mus)
    assert r["recall"] == 1.0 and r["precision"] == 0.5
    assert r["spurious"] == ["('MISSING', 0, 'e2')"]


def test_craft_detail_read_as_invention_is_named():
    aid = "scene_01/shot_01/a0"
    mus = [Mutation("SpecializePredicate", 0, "scene_01", aid, "e1", "OVERSPEC")]
    r = score_detection(_run(), _run(overspec=[aid], invented=[aid]), mus)
    assert r["craft_detail_misread_as_invention"] == [aid]


def test_a_control_run_reports_drift_as_spurious():
    """No mutations: every signal that moved is model noise, and recall is
    undefined rather than zero."""
    r = score_detection(_run(), _run(missing=["e7"]), [])
    assert r["recall"] is None and r["spurious"] == ["('MISSING', 0, 'e7')"]


def test_per_kind_recall_is_reported():
    mus = [Mutation("DeleteEvent", 0, "s", "a", "e1", "MISSING"),
           Mutation("DeleteEvent", 0, "s", "b", "e2", "MISSING")]
    r = score_detection(_run(), _run(missing=["e1"]), mus)
    assert r["per_kind"]["DeleteEvent"] == {
        "injected": 2, "detected": 1, "detected_any_signal": 1,
        "unmeasurable": 0, "recall": 0.5}


# ── the seven that cannot be measured ─────────────────────────────────────

@pytest.mark.parametrize("kind", sorted(INVISIBLE_TO_PROMPT))
def test_invisible_mutations_do_not_change_the_prompt(kind):
    """Proof, not assertion: the estimator is handed identical messages.

    Each of these edits a field somewhere in a scene document that
    `load_breakdown_actions` never reads. Rather than trust that reading, the
    field is actually mutated and the compiled messages compared.
    """
    from pace_core.breakdown.verify_breakdown import load_breakdown_actions

    doc = _doc("scene_01", [ACTIONS["scene_01/shot_01/a0"]])
    doc["narrative_meta"] = {"location_ref": "highway", "characters_present": ["gus"]}
    doc["shots"][0]["setup"] = {
        "backdrop": {"location": "highway", "time_of_day": "day"},
        "subjects": [{"character_id": "gus", "costume": "jacket",
                      "state": "injured"}],
        "props": [{"prop_id": "car", "state": "intact"}],
        "relations": [],
    }
    mutate = {
        "ChangeLocation": lambda d: d["shots"][0]["setup"]["backdrop"]
                                     .__setitem__("location", "tunnel"),
        "RemoveState": lambda d: d["shots"][0]["setup"]["subjects"][0]
                                  .pop("state"),
        "ContradictState": lambda d: d["shots"][0]["setup"]["props"][0]
                                      .__setitem__("state", "destroyed"),
        "DropRelation": lambda d: d["shots"][0]["setup"].pop("relations"),
        "InvertRelation": lambda d: d["shots"][0]["setup"]["relations"]
                                     .append({"a": "gus", "rel": "LEFT_OF",
                                              "b": "hal"}),
        "ReplaceCostume": lambda d: d["shots"][0]["setup"]["subjects"][0]
                                     .__setitem__("costume", "spacesuit"),
    }
    after = copy.deepcopy(doc)
    mutate[kind](after)
    assert after != doc, f"{kind} did not change the document at all"

    def messages(d, tmp):
        (tmp / "scene_01.json").write_text(json.dumps(d))
        acts = load_breakdown_actions("_", ["scene_01"], str(tmp))
        return build_messages(SCRIPT_IR[0], acts)

    import tempfile
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        assert messages(doc, Path(a)) == messages(after, Path(b))


def test_reordering_is_visible_but_unreportable():
    """SwapOrder is the other kind of blind spot: the estimator DOES see the
    change and still has nowhere to report it, so closing it needs a new
    output field rather than a new input."""
    assert "SwapOrder" in NO_OUTPUT_FIELD
    schema = build_messages(SCRIPT_IR[0], [])[0]["content"]
    for word in ("order", "sequence", "before", "after"):
        assert f'"{word}"' not in schema


def test_a_scene_the_mutated_run_lost_is_excluded_not_counted_as_missed():
    """A gateway failure is an outage, not a verifier blind spot.

    Scoring a run that lost scene 1 without excluding it would report the
    error injected there as undetected, and depress recall for a reason that
    has nothing to do with the estimator.
    """
    base = {"alignments": [{"script_scene": 0, "matches": [],
                            "invented_events": [], "over_specified": []},
                           {"script_scene": 1, "matches": [],
                            "invented_events": [], "over_specified": []}]}
    mut = {"alignments": [{"script_scene": 0,
                           "matches": [{"script_event": "e1",
                                        "match": "MISSING", "role_ok": True}],
                           "invented_events": [], "over_specified": []}]}
    mus = [Mutation("DeleteEvent", 0, "s", "a", "e1", "MISSING"),
           Mutation("DeleteEvent", 1, "s", "b", "e5", "MISSING")]
    r = score_detection(base, mut, mus)
    assert r["scenes_dropped"] == [1]
    assert r["injected"] == 1 and r["recall"] == 1.0
    assert "DeleteEvent" in r["per_kind"]
    assert r["per_kind"]["DeleteEvent"]["injected"] == 1


def test_a_signal_already_in_the_baseline_is_unmeasurable_not_missed():
    """The first real run scored SpecializePredicate 0.00 and the estimator
    had done nothing wrong: all four targets were ALREADY flagged
    over-specified, so no new signal could appear. That is an error in the
    experiment, and reporting it as a verifier failure would be a lie."""
    aid = "scene_01/shot_01/a0"
    mus = [Mutation("SpecializePredicate", 0, "s", aid, "e1", "OVERSPEC")]
    r = score_detection(_run(overspec=[aid]), _run(overspec=[aid]), mus)
    assert r["unmeasurable_already_flagged"] == [aid]
    assert r["measurable"] == 0
    assert r["recall_measurable"] is None
    assert r["recall"] == 0.0          # the strict figure still says 0


def test_plan_does_not_inject_over_specification_where_it_is_already_found():
    base = copy.deepcopy(BASELINE)
    base["alignments"][0]["over_specified"] = [
        {"action": "scene_01/shot_01/a0"}, {"action": "scene_01/shot_02/a0"}]
    got = plan(base, SCRIPT_IR, ACTIONS, MAPPING, per_scene=99)
    assert not any(m.kind == "SpecializePredicate" for m in got)


def test_a_detection_under_a_different_code_still_counts_as_noticing():
    """Generalising a predicate often comes back MISSING rather than
    OVER_GENERALIZED. The estimator noticed; it graded differently."""
    mus = [Mutation("GeneralizePredicate", 0, "s", "a", "e1", "OVERGEN")]
    r = score_detection(_run(), _run(missing=["e1"]), mus)
    assert r["recall_measurable"] == 0.0
    assert r["recall_any_signal"] == 1.0
    assert r["undetected_entirely"] == []


def test_a_mutation_nothing_responded_to_is_named():
    mus = [Mutation("DeleteEvent", 0, "s", "a", "e2", "MISSING")]
    r = score_detection(_run(), _run(missing=["e9"]), mus)
    assert r["recall_any_signal"] == 0.0
    assert r["undetected_entirely"] == ["DeleteEvent 0 e2"]
