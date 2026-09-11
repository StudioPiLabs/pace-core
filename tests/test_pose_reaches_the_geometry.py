"""A subject's authored pose picks the proxy it is staged from.

`_mesh_for` has taken a `pose` argument since it was written, and the caller
passed one value for the whole shot: `"sitting" if shape == "subway" else
"standing"`. So the pose came from the LOCATION, and a subject's own `pose`
field -- "kneeling or crouched grief pose", "lying motionless" -- reached the
prompt as prose and never reached the geometry.

The vocabulary was then three buckets, and kneeling lived inside the SITTING
one: the corpus's authored kneels staged as somebody sitting down. It is six
now -- standing, sitting, kneeling, walking, reaching, lying -- because the
joint angles for the missing three already existed in the breakdown's own
preset table and had simply never been turned into meshes.

`lying` still resolves to a STANDING mesh, which is not an omission: SMPL-X
body_pose is relative to the pelvis, so lying is orientation, and the
assembler lays the body down itself. Giving it a recumbent mesh as well would
tip it twice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import (  # noqa: E402
    _POSE_FALLBACK, _mesh_for, pose_key_for,
)


def test_kneeling_is_its_own_pose_now():
    """It used to answer "sitting", so every kneel in the corpus staged as a
    seated body -- in scene_06 and scene_07 that is the grief beat."""
    for text in ("kneeling or crouched grief pose", "crouching by the door",
                 "kneels on the road", "kneeling, hands on her shoulder"):
        assert pose_key_for(text) == "kneeling", text


def test_sitting_is_still_sitting():
    for text in ("seated at the console", "sits in the front seat",
                 "seated upper-body performance"):
        assert pose_key_for(text) == "sitting", text


def test_a_beat_that_puts_someone_on_the_ground_reads_as_lying():
    for text in ("lying motionless", "the woman lies on the road",
                 "collapsed, unconscious", "sprawled on the ground"):
        assert pose_key_for(text) == "lying", text


def test_travelling_on_foot_is_walking_not_standing():
    """"limps to the other side of the car" is a stride, and a standing proxy
    gives it the silhouette of someone waiting."""
    for text in ("walking toward the wreckage", "he limps to the car",
                 "runs across the road", "approaches the panel"):
        assert pose_key_for(text) == "walking", text


def test_an_upper_body_action_on_your_feet_is_reaching():
    for text in ("pushes against it with all his strength",
                 "braces both hands against the flank", "bangs on the panel"):
        assert pose_key_for(text) == "reaching", text


def test_plain_upright_is_still_standing():
    for text in ("standing performance pose", "stands by the door"):
        assert pose_key_for(text) == "standing", text


def test_an_unauthored_pose_falls_back_to_the_location_default():
    """A cabin seats its occupants; open ground does not."""
    assert pose_key_for(None, default="sitting") == "sitting"
    assert pose_key_for("", default="standing") == "standing"
    assert pose_key_for("gesturing vaguely", default="sitting") == "sitting"


# A library path the pose picks a file name under; it need not exist.
LIB = Path("meshes/characters")


def test_lying_resolves_to_a_mesh_that_exists():
    assert _mesh_for("adult_40", LIB, "lying").endswith("smplx_standing_175.obj")
    assert _mesh_for("child_8", LIB, "lying").endswith("smplx_standing_125.obj")


def test_a_project_without_the_new_meshes_still_stages():
    """A project generated before the vocabulary grew has four files. It must
    degrade to the nearest silhouette rather than fail."""
    empty = Path("/nonexistent-meshes")
    assert Path(_mesh_for("adult_40", empty, "kneeling")).name == "smplx_sitting_175.obj"
    assert Path(_mesh_for("adult_40", empty, "walking")).name == "smplx_standing_175.obj"


def test_the_fallback_chain_terminates():
    """A cycle would hang the staging loop."""
    for start in _POSE_FALLBACK:
        seen, cur = set(), start
        while cur in _POSE_FALLBACK:
            assert cur not in seen, f"cycle at {cur}"
            seen.add(cur); cur = _POSE_FALLBACK[cur]


def test_motionless_is_stillness_not_posture():
    """It was in the `lying` bucket at highest precedence, so "standing tall,
    motionless, watching building" staged a body on the floor -- 10 of another production's
    190 subjects. It only ever looked right because the corpus that taught the
    table wrote "lying motionless", where "lying" already matches."""
    from pace_core.node.panel_greybox import pose_key_for
    assert pose_key_for("standing tall, motionless, watching building") == "standing"
    assert pose_key_for("seated motionless at the desk") == "sitting"
    # The phrase that taught the table still resolves, on its real word.
    assert pose_key_for("lying motionless on the road") == "lying"
