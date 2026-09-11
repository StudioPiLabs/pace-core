"""The estimator proposes; the scorer counts. Two bugs this pins.

Both were live, and both inflated the verdict against the breakdown -- which
is the direction a verifier must never be wrong in, because an auditor that
overstates is discarded and takes its true findings with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.verify_breakdown import score  # noqa: E402


def _scenes():
    return [
        {"index": 6, "events": [
            {"local_id": "e11", "predicate": "FALL_ON_TOP_OF", "importance": 1.0,
             "evidence": {"source_text": "The car falls on top of him."}}]},
        {"index": 7, "events": [
            {"local_id": "e11", "predicate": "LOSE_POWER", "importance": 1.0,
             "evidence": {"source_text": "All the panels lose power."}}]},
    ]


def test_a_missing_event_cannot_also_have_its_roles_wrong():
    """The estimator returns role_ok=false on MISSING too, which is vacuous:
    nothing matched, so nothing swapped actor for patient. Counting those
    turned 2 real role errors into 30."""
    al = [{"script_scene": 6, "matches": [
        {"script_event": "e11", "match": "MISSING", "role_ok": False}]}]
    assert score(al, _scenes())["counts"]["wrong_role"] == 0


def test_a_role_error_on_a_matched_event_still_counts():
    al = [{"script_scene": 6, "matches": [
        {"script_event": "e11", "match": "SEMANTIC", "role_ok": False}]}]
    assert score(al, _scenes())["counts"]["wrong_role"] == 1


def test_two_scenes_reusing_an_event_id_are_not_the_same_event():
    """Scenes restart local ids at e1, so keying the lookup on local_id alone
    made every scene's e11 one event -- and reported a single dropped climax
    four times over, crowding the real list out of the top of the report."""
    al = [{"script_scene": 6, "matches": [
             {"script_event": "e11", "match": "MISSING", "role_ok": True}]},
          {"script_scene": 7, "matches": [
             {"script_event": "e11", "match": "MISSING", "role_ok": True}]}]
    miss = score(al, _scenes())["missing_events"]
    assert len(miss) == 2
    assert {m["predicate"] for m in miss} == {"FALL_ON_TOP_OF", "LOSE_POWER"}
    assert {m["scene"] for m in miss} == {6, 7}


def test_over_generalisation_is_partial_credit_not_a_miss():
    al = [{"script_scene": 6, "matches": [
        {"script_event": "e11", "match": "OVER_GENERALIZED", "role_ok": True}]}]
    r = score(al, _scenes())
    assert r["counts"]["missing"] == 0
    assert 0.0 < r["coverage"] < 1.0
