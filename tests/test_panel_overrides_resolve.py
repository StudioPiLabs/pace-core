"""A panel's camera_override actually changes the camera.

The schema documents an inheritance ladder — scene defaults → shot → panel —
and only the first two rungs were built. `Panel.camera_override` is declared in
types_v1 as a "sparse Camera-shaped diff vs the parent shot's camera", is
initialised to None by the splitter, and was read by nothing: a panel could not
move the camera however the field was filled in. `setup_override` was
half-wired, with `excluded_of` reading one leaf and the rest ignored.

This is the failure the corpus showed as two identical panels. Densification
adds a moving shot's end framing as a second panel, and the only place "the
camera has finished the move" can be written is a camera override — so the
second panel had nowhere to differ, and start and end rendered the same frame.

The greybox is included here because it read `shot["camera"]` raw, so it
skipped scene defaults as well as panel overrides: the prompt and the staged
geometry have to agree about which camera this panel is shot on, or the
override would move one and not the other.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.pai_compat import (  # noqa: E402
    resolve_shot, shot_size_of, angle_of, excluded_of,
)

SCENE = {"shot_defaults": {"camera": {"creative_intent": {"aspect_ratio": "2.35:1"},
                                      "extrinsics": {"angle": "eye_level"}}}}
SHOT = {"camera": {"creative_intent": {"shot_size": "wide"},
                   "intrinsics": {"lens_mm": 35}},
        "setup": {"subjects": [{"character_id": "nina"}]}}


def test_a_panel_without_an_override_gets_the_shot_camera():
    assert shot_size_of(resolve_shot(SCENE, SHOT, {"id": "p1"})) == "wide"


def test_a_camera_override_changes_the_shot_size():
    end = {"id": "p2", "camera_override": {"creative_intent": {"shot_size": "close_up"}}}
    assert shot_size_of(resolve_shot(SCENE, SHOT, end)) == "close_up"


def test_an_override_is_a_sparse_diff_not_a_replacement():
    """The schema calls it a sparse diff. If it replaced the block, overriding
    the shot size would silently drop the lens the shot declared."""
    end = {"id": "p2", "camera_override": {"creative_intent": {"shot_size": "close_up"}}}
    cam = resolve_shot(SCENE, SHOT, end)["camera"]
    assert cam["intrinsics"]["lens_mm"] == 35
    assert cam["creative_intent"]["aspect_ratio"] == "2.35:1"


def test_an_override_still_sits_on_top_of_scene_defaults():
    end = {"id": "p2", "camera_override": {"extrinsics": {"angle": "low"}}}
    assert angle_of(resolve_shot(SCENE, SHOT, end)) == "low"
    assert angle_of(resolve_shot(SCENE, SHOT, {"id": "p1"})) == "eye_level"


def test_setup_override_resolves_too_not_just_its_excluded_leaf():
    """`excluded_of` read setup_override.excluded directly, so that one leaf
    worked while everything else in the block was ignored."""
    end = {"id": "p2", "setup_override": {"environment": {"style": "sketch_bw"}}}
    out = resolve_shot(SCENE, SHOT, end)
    assert out["setup"]["environment"]["style"] == "sketch_bw"
    # …and the shot's own setup survives alongside it
    assert out["setup"]["subjects"][0]["character_id"] == "nina"


def test_the_excluded_leaf_keeps_working():
    end = {"id": "p2", "setup_override": {"excluded": ["seat belts"]}}
    assert excluded_of(end, SHOT) == ["seat belts"]


def test_resolve_shot_without_a_panel_is_unchanged():
    """Every existing caller passes two arguments; the panel is optional."""
    assert shot_size_of(resolve_shot(SCENE, SHOT)) == "wide"
    assert resolve_shot({}, SHOT) is SHOT          # no defaults, no overrides


def test_an_empty_override_does_not_count_as_an_override():
    """The splitter writes `camera_override: None` on every panel, and an
    empty dict is the same statement. Neither may perturb the merge."""
    for empty in (None, {}):
        p = {"id": "p", "camera_override": empty}
        assert shot_size_of(resolve_shot(SCENE, SHOT, p)) == "wide"


def test_the_greybox_resolves_the_same_ladder_as_the_compilers():
    """The staged camera and the compiled prompt must agree about which camera
    this panel is on. build_spec read shot["camera"] raw."""
    import inspect
    from pace_core.node import panel_greybox
    src = inspect.getsource(panel_greybox.build_spec)
    assert "resolve_shot(scene, shot, panel)" in src, (
        "build_spec stopped resolving the panel's camera — the prompt and the "
        "greybox can now disagree about the camera for the same panel")
