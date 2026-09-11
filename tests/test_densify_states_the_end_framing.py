"""A moving shot's second panel says where the frame ends up.

Densification exists because a moving shot has two framings and one panel, so
"the end framing is simply unstated". Cloning the start panel stated the START
framing twice, which left the end framing exactly as unstated as before — the
pair differed only by panel_number and a prose note.

On the video path that was invisible: plan_camera_track interpolates between
the two panels, so the pair works. On the still path each panel renders alone,
and scene_10/shot_01's two panels came back pixel-identical.

The end panel now carries the size the move arrives at as a camera_override,
one step along the ShotSize ladder. That is a composition target — where the
frame ends up — not a camera transform, which stays plan_camera_track's job.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.breakdown.densify_panels import (  # noqa: E402
    densify_scene, end_shot_size,
)
from pace_core.pai_compat import resolve_shot, shot_size_of  # noqa: E402


def _scene(shot_size="medium", movement_3d=("push_in",), movement_2d=()):
    return {
        "scene_id": "scene_10",
        "shots": [{
            "shot_id": "shot_01",
            "camera": {"creative_intent": {"shot_size": shot_size},
                       "trajectory": {"movement_3d": list(movement_3d),
                                      "movement_2d": list(movement_2d),
                                      "static": False}},
            "panels": [{"id": "scene_10_shot_01_panel_0001", "panel_number": 1}],
        }],
    }


def test_a_push_in_ends_one_step_tighter():
    assert end_shot_size("medium", ["push_in"]) == "medium_close_up"
    assert end_shot_size("wide", ["dolly_in"]) == "full"


def test_a_pull_out_ends_one_step_wider():
    assert end_shot_size("medium", ["pull_out"]) == "medium_full"


def test_the_ladder_does_not_run_off_either_end():
    """A push-in on an extreme close-up ends on an extreme close-up. Inventing
    a tighter size than the schema declares would be worse than saying so."""
    assert end_shot_size("extreme_close_up", ["push_in"]) is None
    assert end_shot_size("establishing", ["pull_out"]) is None


def test_a_reframing_move_changes_no_size():
    """A pan or a crane moves the frame without changing how much of the
    subject is in it, and for those the clone is already correct."""
    assert end_shot_size("medium", ["pan_left"]) is None
    assert end_shot_size("medium", ["crane_up"]) is None


def test_one_step_per_shot_not_one_per_move():
    """push_in and dolly_in in one shot are the same movement said twice.
    Stepping per move would turn a medium into a close-up on a vocabulary
    artefact."""
    assert end_shot_size("medium", ["push_in", "dolly_in"]) == "medium_close_up"


def test_contradictory_moves_get_no_step():
    assert end_shot_size("medium", ["push_in", "pull_out"]) is None


def test_an_undeclared_shot_size_is_left_alone():
    assert end_shot_size(None, ["push_in"]) is None
    assert end_shot_size("not_a_size", ["push_in"]) is None


def test_the_end_panel_carries_the_override():
    doc = _scene()
    out = densify_scene(doc, apply=True)
    assert len(out["added"]) == 1
    panels = doc["shots"][0]["panels"]
    assert len(panels) == 2
    assert panels[1]["camera_override"] == {"creative_intent": {"shot_size": "medium_close_up"}}
    assert panels[0].get("camera_override") is None


def test_the_pair_no_longer_resolves_to_the_same_camera():
    """The regression this exists to prevent: two panels that resolve to the
    same shot_size render the same image."""
    doc = _scene()
    densify_scene(doc, apply=True)
    shot = doc["shots"][0]
    start, end = shot["panels"]
    assert shot_size_of(resolve_shot(doc, shot, start)) == "medium"
    assert shot_size_of(resolve_shot(doc, shot, end)) == "medium_close_up"


def test_a_reframe_only_shot_still_gets_its_panel_but_no_override():
    """The second panel is still structurally needed — it carries the end
    composition target — it just does not restate the size.

    A pan is authored in movement_2d as pan_left; movement_of() canonicalises
    it to the planner's pan_rl, which a FRAMING_MOVES built from the schema
    literals alone never matched — so a panning shot got no second panel."""
    doc = _scene(movement_3d=(), movement_2d=("pan_left",))
    densify_scene(doc, apply=True)
    panels = doc["shots"][0]["panels"]
    assert len(panels) == 2
    assert "camera_override" not in panels[1] or panels[1].get("camera_override") is None


def test_the_dry_run_writes_nothing():
    doc = _scene()
    before = len(doc["shots"][0]["panels"])
    out = densify_scene(doc, apply=False)
    assert out["added"] and len(doc["shots"][0]["panels"]) == before


def test_a_crane_ends_at_a_higher_angle_not_a_tighter_size():
    """A crane changes the camera's height, not how much of the subject is in
    frame, so its end framing differs in angle."""
    from pace_core.breakdown.densify_panels import end_angle
    assert end_angle("eye_level", ["crane_up"]) == "high"
    assert end_angle("eye_level", ["crane_down"]) == "low"
    assert end_shot_size("medium", ["crane_up"]) is None


def test_the_angle_ladder_reads_the_spelling_on_disk():
    """The documents say `low_angle` and `high_angle`; the Angle Literal says
    `low` and `high`. Failing to recognise the angle a shot plainly declares
    would silently skip its end framing."""
    from pace_core.breakdown.densify_panels import end_angle
    assert end_angle("high_angle", ["crane_up"]) == "overhead"
    assert end_angle("low_angle", ["crane_up"]) == "eye_level"


def test_a_shot_that_declares_static_is_not_densified():
    """`static: true` says the frame is held. A shot that also lists a move is
    contradicting itself, and which half is wrong is an authoring decision.

    Every 2-panel shot in the evaluation corpus came from this pair of statements —
    4 declaring static alongside a crane, 0 genuinely moving — and the pairs
    rendered identically because there was no second framing to state.
    """
    doc = _scene()
    doc["shots"][0]["camera"]["trajectory"]["static"] = True
    out = densify_scene(doc, apply=True)
    assert not out["added"]
    assert len(doc["shots"][0]["panels"]) == 1
    why = out["skipped"][0]["why"]
    assert "contradiction" in why and "static" in why
