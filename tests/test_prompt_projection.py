"""The projection is a gate, not a description of one.

The failure this file exists to catch is specific and already happened once:
the table said two fields were quarantined and the payload went on carrying
both, because `project_scene` built its dict by hand and never consulted
`FIELDS`. Two implementations of one rule drift, silently, and the drift is
invisible from either side — the table reads correct, the payload looks
plausible, and a model is being handed fields nobody admitted.
"""
import json

import pytest

from pace_core.compilers.prompt_projection import (
    FIELDS, Mode, Rule, project_scene, visible, writable,
)


def _scene():
    return {
        "scene_id": "scene_x",
        "shots": [{
            "shot_id": "shot_01",
            "camera": {
                "intrinsics": {"lens_mm": 28, "sensor_width_mm": 36},
                "extrinsics": {"angle": "low", "position": "left"},
                "creative_intent": {"shot_size": "medium", "framing": "two_shot"},
                "_design": {"intent": "a director's note"},
            },
            "setup": {
                "backdrop": {"location": "family_car", "setting": "INT car", "era": "2035"},
                "environment": {"lighting": "hard sun", "density": "crowded_crash_site",
                                "style": "photoreal"},
                "space": {"scale_meters": [2.0, 3.2, 1.5]},
                "subjects": [{
                    "character_id": "nina", "age_state": "middle_aged",
                    "continuity_anchor": "young adult, slender build",
                    "costume": "a cream t-shirt", "hair": "long brown hair",
                    "pose": "seated",
                    "screen_position": {"x": 0.5, "zone": "center"},
                }],
                "props": [{"prop_id": "p1", "name": "tray table", "state": "active",
                           "screen_position": {"zone": "center"}}],
                "primary_focus": {"ref": "nina", "type": "character", "coverage_pct": 55},
            },
            "events": {"actions": [{"description_en": "she looks up",
                                    "intensity": "dramatic", "foreground": "focal"}]},
            "panels": [{"id": "scene_x_shot_01_panel_0001", "panel_number": 1}],
        }],
    }


def test_geometry_never_reaches_the_model():
    """The load-bearing constraint. The greybox decides where things are; if
    prose re-decides it, that is two sources for one decision, which measured
    3/3 -> 2/3 on occlusion in the system this borrows from. A regression here
    would not raise anything — it would quietly make composition worse."""
    blob = json.dumps(project_scene(_scene()))
    for leaked in ("screen_position", "lens_mm", "sensor_width_mm",
                   "scale_meters", "0.5", "28"):
        assert leaked not in blob, f"{leaked!r} reached the model"


def test_quarantined_fields_are_absent():
    shot = project_scene(_scene())["shots"][0]
    assert "density" not in shot["write"]
    subj = shot["write"]["subjects"][0]
    assert "anchor" not in subj
    # setup.subjects[].costume is quarantined too — found to contradict
    # characters.json's own costume for the same character. The subject
    # survives with the fields that ARE admitted, rather than being dropped
    # along with the bad one.
    assert "costume" not in subj
    assert subj["hair"] == "long brown hair"


def test_character_registry_supplies_the_identity_the_scene_lacks(monkeypatch, tmp_path):
    """Found by diffing a written prompt against the compiled one: every
    written prompt had been missing physical identity entirely, because it
    lives in characters.json — a file this projection did not read. No LoRA
    is trained for this cast (lora.path is null), so this prose is the only
    thing telling the model who is on screen."""
    import json as _json
    from pace_core import paths as paths_mod
    chars_file = tmp_path / "characters.json"
    chars_file.write_text(_json.dumps({
        "nina": {"trigger": "nina_female",
                  "anchor": "slender build, with long brown hair",
                  "costumes": {"default": "a plain cream cotton t-shirt"},
                  "lora": {"path": None}},
    }))
    fake_paths = type("P", (), {"chars_file": chars_file})()
    monkeypatch.setattr(paths_mod, "paths_for", lambda proj: fake_paths)
    import pace_core.compilers.prompt_projection as ppmod
    monkeypatch.setattr(ppmod, "paths_for", lambda proj: fake_paths)

    scene = _scene()
    scene["shots"][0]["setup"]["subjects"][0]["character_id"] = "nina"
    scene["shots"][0]["setup"]["subjects"][0]["age_state"] = "adult"
    shot = project_scene(scene, project="proj")["shots"][0]

    subj = shot["write"]["subjects"][0]
    assert "cream cotton t-shirt" in subj["descriptor"]
    assert "long brown hair" in subj["descriptor"]
    # the trigger is context, never prose
    assert shot["input_only"]["character_trigger_words"] == ["nina_female"]
    assert "nina_female" not in _json.dumps(shot["write"])


def test_input_only_is_visible_but_never_writable():
    shot = project_scene(_scene())["shots"][0]
    assert shot["input_only"]["design_intent"] == "a director's note"
    assert "design_intent" not in shot["write"]
    assert visible("camera._design.intent") and not writable("camera._design.intent")


def test_the_table_is_the_gate(monkeypatch):
    """Flip a rule and the payload must follow with no other edit. This is
    the exact property that was missing when the module was first written."""
    assert "lighting" in project_scene(_scene())["shots"][0]["write"]
    monkeypatch.setitem(FIELDS, "setup.environment.lighting",
                        Rule(Mode.QUARANTINED, "test"))
    assert "lighting" not in project_scene(_scene())["shots"][0]["write"]


def test_unlisted_fields_are_withheld_by_construction():
    """An additive gate's whole point: a field the KB grows later reaches
    nobody until someone weighs it."""
    assert not visible("setup.some_field_invented_next_quarter")
    assert not writable("setup.some_field_invented_next_quarter")


@pytest.mark.parametrize("path,rule", sorted(FIELDS.items()))
def test_every_rule_carries_a_reason(path, rule):
    """A verdict with no reason gets re-litigated, and usually reversed by
    whoever is annoyed by it next."""
    assert rule.reason.strip(), path
    assert rule.evidence in ("reasoned", "measured", "untested"), path


def test_write_rules_admitted_without_checking_values_are_visible():
    """Not a failure — a standing report. Fill rate is not truth: the two
    fields now quarantined both passed a fill audit (27/27 and 58/58) while
    holding wrong values. This keeps the unverified admissions countable
    rather than forgotten."""
    unverified = [p for p, r in FIELDS.items()
                  if r.mode is Mode.WRITE and not r.values_checked]
    # Nothing to assert about the count; assert the flag exists and is honest.
    assert all(isinstance(FIELDS[p].values_checked, bool) for p in unverified)
