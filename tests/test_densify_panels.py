"""A moving shot needs two panels; a held frame needs one.

The rule is structural, not stylistic: a panel carries the composition target,
so a camera that translates has two framings and one panel can only state one
of them. The distinction it turns on -- movement versus gear -- is exactly what
the corpus got wrong, so it is tested rather than trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.densify_panels import densify_scene, framing_moves  # noqa: E402


def _scene(movement_3d, gear="tripod", static=False, n_panels=1):
    return {
        "scene_id": "scene_t", "_max_panel_n": n_panels,
        "shots": [{
            "shot_id": "shot_01",
            "camera": {"trajectory": {"movement_2d": [], "movement_3d": movement_3d,
                                      "gear": gear, "static": static}},
            "panels": [{"id": f"scene_t_shot_01_panel_{i+1:04d}", "panel_number": i + 1,
                        "scene_id": "scene_t", "shot_id": "shot_01"}
                       for i in range(n_panels)],
        }],
    }


def test_gear_is_not_movement():
    """handheld is rig character. It sat in movement_3d across 16 shots of the
    corpus, where movement_of drops it -- and where a naive reader counts it as
    a camera move and densifies a frame that never travels."""
    assert framing_moves(_scene(["handheld"])["shots"][0]) == []
    assert framing_moves(_scene(["static"])["shots"][0]) == []


def test_translation_is_movement_in_both_spellings():
    """The schema literal is `crane`; movement_of emits `crane_up`. Both must
    be recognised, since scene documents on disk carry each."""
    assert framing_moves(_scene(["crane"])["shots"][0]) == ["crane_up"]
    assert framing_moves(_scene(["push_in"])["shots"][0]) == ["push_in"]


def test_moving_shot_gains_an_end_panel():
    doc = _scene(["crane"])
    r = densify_scene(doc, apply=True)
    panels = doc["shots"][0]["panels"]
    assert len(r["added"]) == 1 and len(panels) == 2
    assert panels[1]["panel_number"] == 2
    assert "end framing" in panels[1]["notes"]
    # Reframing: the subject holds its screen position across the move.
    assert panels[1].get("screen_position") == panels[0].get("screen_position")
    # Every panel needs a compile-hints slot or its prompt overrides are unreachable.
    assert any(h["panel_id"] == panels[1]["id"] for h in doc["compile_hints"])


def test_held_frame_stays_at_one_panel():
    doc = _scene(["handheld"])
    r = densify_scene(doc, apply=True)
    assert not r["added"] and len(doc["shots"][0]["panels"]) == 1


def test_is_idempotent():
    doc = _scene(["crane"])
    densify_scene(doc, apply=True)
    again = densify_scene(doc, apply=True)
    assert not again["added"] and len(doc["shots"][0]["panels"]) == 2


def test_panel_ids_do_not_collide_across_shots():
    """Panel numbering is scene-wide. Deriving the counter per shot handed both
    shots of a scene the same id on a dry run."""
    doc = _scene(["crane"])
    doc["shots"].append({
        "shot_id": "shot_02",
        "camera": {"trajectory": {"movement_3d": ["crane"], "gear": "tripod"}},
        "panels": [{"id": "scene_t_shot_02_panel_0002", "panel_number": 1,
                    "scene_id": "scene_t", "shot_id": "shot_02"}],
    })
    r = densify_scene(doc, apply=False)
    ids = [a["end_panel"] for a in r["added"]]
    assert len(ids) == 2 and len(set(ids)) == 2, ids
