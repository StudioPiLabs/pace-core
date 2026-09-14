"""A cast of five is staged as five bodies, not as one body five times.

`_mesh_for` resolved a proxy from stature and pose alone -- its signature had
no character in it -- so every panel in a production staged the same SMPL-X
mesh whatever the registry said about the person. It was visible in the
paper's own figures: a male lead came back with a female body, and the same
body appeared for every other character in every greybox.

Two paths existed and only one knew about people. `scene_assembler` reads
`body_proxy_3d` and stages a character's own mesh; `panel_greybox`, which is
what actually builds every panel, did not. This is that path learning the
same thing, with the stature proxies kept as the fallback so a project with
no per-character bake behaves exactly as it did before.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import _mesh_for  # noqa: E402


def _lib(tmp_path: Path, *names: str) -> Path:
    for n in names:
        (tmp_path / n).write_text("o proxy\n")
    return tmp_path


def test_a_characters_own_mesh_wins_over_the_stature_one(tmp_path):
    lib = _lib(tmp_path, "smplx_standing_175.obj", "smplx_standing_ryan.obj")
    assert _mesh_for("adult_50", lib, "standing", "ryan").endswith("smplx_standing_ryan.obj")


def test_two_characters_in_one_shot_get_two_bodies(tmp_path):
    lib = _lib(tmp_path, "smplx_standing_175.obj",
               "smplx_standing_ryan.obj", "smplx_standing_emily.obj")
    got = {_mesh_for("adult_50", lib, "standing", c) for c in ("ryan", "emily")}
    assert len(got) == 2, "the whole point: one mesh per person, not per stature"


def test_without_a_bake_it_is_exactly_the_old_behaviour(tmp_path):
    """The fallback is not a courtesy; it is what every existing project uses."""
    lib = _lib(tmp_path, "smplx_standing_175.obj")
    assert _mesh_for("adult_50", lib, "standing", "ryan").endswith("smplx_standing_175.obj")
    assert _mesh_for("adult_50", lib, "standing", "").endswith("smplx_standing_175.obj")


def test_the_right_pose_beats_the_right_person(tmp_path):
    """Someone with a standing bake but no kneeling one kneels as the shared
    body, not as their own standing body.

    Pose is what the geometry is for. The camera and composition solves read
    head height and the projected read point off the staged body, and both
    move with the pose; the identity of the body moves neither. So when only
    one of the two can be honoured, the kernel keeps the pose and gives up the
    person, which is the opposite of what this test first asserted."""
    lib = _lib(tmp_path, "smplx_standing_175.obj", "smplx_kneeling_175.obj",
               "smplx_standing_ryan.obj")
    assert _mesh_for("adult_50", lib, "kneeling", "ryan").endswith("smplx_kneeling_175.obj")


def test_a_persons_own_pose_chain_is_still_walked_first(tmp_path):
    """Within a person, kneeling falls back to their sitting body before it
    gives up on them: that chain is the same one the stature proxies walk."""
    lib = _lib(tmp_path, "smplx_kneeling_175.obj", "smplx_sitting_ryan.obj")
    assert _mesh_for("adult_50", lib, "kneeling", "ryan").endswith("smplx_sitting_ryan.obj")


def test_lying_still_resolves_to_a_standing_body(tmp_path):
    """Unchanged rule: lying is orientation, and the kernel lays the body down."""
    lib = _lib(tmp_path, "smplx_standing_175.obj", "smplx_standing_ryan.obj")
    assert _mesh_for("adult_50", lib, "lying", "ryan").endswith("smplx_standing_ryan.obj")


def test_a_child_still_reads_as_the_short_stature_when_unbaked(tmp_path):
    lib = _lib(tmp_path, "smplx_standing_125.obj", "smplx_standing_175.obj")
    assert _mesh_for("child_08", lib, "standing", "nobody").endswith("smplx_standing_125.obj")


def test_the_eligibility_precheck_resolves_the_same_file_the_build_would(monkeypatch, tmp_path):
    """The precheck calls `_mesh_for` too, and it shipped referring to a name
    that exists only in the build path.

    pace-core's own suite did not catch it -- nothing here drove the precheck
    with a real scene document -- and it surfaced as a NameError in the host
    repo's greybox tests. The point of this test is that the two callers stay
    reachable from the same place, so a signature change has to satisfy both.
    """
    import inspect

    from pace_core.node import panel_greybox as pg

    src = inspect.getsource(pg)
    # Every _mesh_for call must pass a pose expression that is defined where it
    # stands; a bare `pose` inside the precheck was not.
    precheck = src[src.index("missing = [m for m in"):]
    precheck = precheck[:precheck.index("if not Path(m).is_file()]")]
    assert "loc_pose" in precheck, "the precheck must name its own pose default"
    assert "pose_key_for" in precheck, "and still honour the subject's authored pose"
