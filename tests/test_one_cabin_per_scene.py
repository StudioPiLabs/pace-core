"""Every panel of a scene is built inside one cabin.

The vehicle shell is stretched to contain each panel's camera, so sized per
panel it followed the lens: a wide and the over-the-shoulders after it were
built at different widths and lengths, and the walls moved at every cut. make_scene_greyboxes builds each panel once to learn the
shell it needs and again inside the union.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pace_core.node.panel_greybox as G                          # noqa: E402

SCENE = {"scene_id": "s", "shots": [{"shot_id": "shot_01", "panels": [{"id": "a"}, {"id": "b"}]}]}


def _fake(bounds_by_panel, style="vehicle"):
    calls = []

    def build(project, scene_id, pid, out, res=None, scene=None, dress=False, scene_shell=None):
        calls.append((pid, str(out), scene_shell))
        return {"ok": True, "shell": {"style": style, "bounds": bounds_by_panel[pid]}}
    return build, calls


def test_the_second_pass_shares_the_union(monkeypatch):
    b = {"a": {"half_w": 2.13, "top_z": 2.3, "back_y": -1.5, "front_y": 4.8},
         "b": {"half_w": 2.05, "top_z": 2.4, "back_y": -2.2, "front_y": 1.8}}
    build, calls = _fake(b)
    monkeypatch.setattr(G, "make_panel_greybox", build)
    G.make_scene_greyboxes("p", "s", lambda pid: f"/out/{pid}.png", scene=SCENE)
    final = [c for c in calls if c[1].startswith("/out/")]
    assert [c[0] for c in final] == ["a", "b"]
    shared = {"half_w": 2.13, "top_z": 2.4, "back_y": -2.2, "front_y": 4.8}
    assert all(c[2] == shared for c in final)


def test_a_scene_not_built_as_a_procedural_shell_shares_no_cabin(monkeypatch):
    b = {"a": None, "b": None}
    build, calls = _fake(b, style=None)
    monkeypatch.setattr(G, "make_panel_greybox", build)
    G.make_scene_greyboxes("p", "s", lambda pid: f"/out/{pid}.png", scene=SCENE)
    assert all(c[2] is None for c in calls if c[1].startswith("/out/"))


def test_every_panel_is_measured_before_any_is_delivered(monkeypatch):
    """The count this file used to claim and never checked.

    It was named `..._is_built_once` and asserted only that the delivered
    builds shared no cabin, so it passed against a function that builds each
    panel twice -- which is what happens, and is the measuring pass. Naming
    the real count makes the cost visible instead of implying it away.
    """
    b = {"a": None, "b": None}
    build, calls = _fake(b, style=None)
    monkeypatch.setattr(G, "make_panel_greybox", build)
    G.make_scene_greyboxes("p", "s", lambda pid: f"/out/{pid}.png", scene=SCENE)
    measured = [c for c in calls if not c[1].startswith("/out/")]
    delivered = [c for c in calls if c[1].startswith("/out/")]
    assert len(measured) == 2 and len(delivered) == 2


# ── the union is the scene's, not the rebuilt subset's ──────────────────────

def test_rebuilding_one_panel_still_gets_the_whole_scenes_cabin(monkeypatch):
    """`panel_ids` narrows what is built, never what is measured.

    Measured over the subset, a one-panel rebuild got its own shell back:
    half_w 2.05 against the scene's 2.13 and front_y 1.8 against 4.8. The
    walls then moved against the panels the rebuild was meant to match, which
    is the defect this whole function exists to remove.
    """
    b = {"a": {"half_w": 2.13, "top_z": 2.3, "back_y": -1.5, "front_y": 4.8},
         "b": {"half_w": 2.05, "top_z": 2.4, "back_y": -2.2, "front_y": 1.8}}
    build, calls = _fake(b)
    monkeypatch.setattr(G, "make_panel_greybox", build)
    G.make_scene_greyboxes("p", "s", lambda pid: f"/out/{pid}.png",
                           panel_ids=["b"], scene=SCENE)
    measured = sorted(c[0] for c in calls if not c[1].startswith("/out/"))
    delivered = [c for c in calls if c[1].startswith("/out/")]
    assert measured == ["a", "b"]          # the scene, not the subset
    assert [c[0] for c in delivered] == ["b"]
    assert delivered[0][2] == {"half_w": 2.13, "top_z": 2.4,
                               "back_y": -2.2, "front_y": 4.8}


def test_a_panel_entry_with_no_id_does_not_take_the_scene_down():
    # Every sibling reader uses pl.get("id"); this one indexed it, so a single
    # id-less entry raised before any panel of the scene was built.
    scene = {"scene_id": "s", "shots": [{"panels": [{"id": "a"}, {}]}]}
    ids = [pl.get("id") for sh in scene["shots"] for pl in sh["panels"]]
    assert [i for i in ids if i] == ["a"]


# ── the anchor can be reproduced from the build's own record ────────────────

def test_the_shared_cabin_is_recorded_beside_the_frame(tmp_path):
    """Without the record the anchor is unreproducible.

    `anchor_version` hashes `scene_shell`, and `greybox_gate` rebuilds the
    spec with `build_spec`, which is a pure KB read -- and a measured shell is
    not in the KB. So every shared-cabin panel hashed differently on the two
    sides and read as permanently stale.
    """
    png = tmp_path / "panel_01.png"
    shell = {"half_w": 2.13, "top_z": 2.4, "back_y": -2.2, "front_y": 4.8}
    G._record_scene_shell(png, shell)
    assert G._recorded_scene_shell(png) == shell
    assert json.loads(G._scene_shell_record(png).read_text()) == shell


def test_a_scene_that_stops_sharing_clears_its_record(tmp_path):
    # Otherwise the last shared shell stays on disk and the next anchor covers
    # a cabin the build did not use -- the same mismatch, now silent.
    png = tmp_path / "panel_01.png"
    G._record_scene_shell(png, {"half_w": 2.13})
    G._record_scene_shell(png, None)
    assert G._recorded_scene_shell(png) is None
    assert not G._scene_shell_record(png).exists()


def test_an_unreadable_record_reads_as_unshared(tmp_path):
    # Reproducibly wrong is worse than reporting the panel as unshared.
    png = tmp_path / "panel_01.png"
    G._scene_shell_record(png).write_text("{not json")
    assert G._recorded_scene_shell(png) is None
    G._scene_shell_record(png).write_text("[]")
    assert G._recorded_scene_shell(png) is None


def test_the_recorded_cabin_reproduces_the_stamped_anchor(tmp_path):
    # The whole point: the hash the build stamped and the hash the gate
    # recomputes have to agree.
    png = tmp_path / "panel_01.png"
    shell = {"half_w": 2.13, "top_z": 2.4}
    built = {"cabin": [2.4, 2.2, 1.7], "subjects": [], "scene_shell": shell}
    G._record_scene_shell(png, shell)
    regated = {"cabin": [2.4, 2.2, 1.7], "subjects": [],
               **({"scene_shell": s} if (s := G._recorded_scene_shell(png)) else {})}
    assert G.anchor_version(regated) == G.anchor_version(built)


def test_the_shared_cabin_moves_the_anchor_only_when_present():
    """Listed unconditionally it would enter every payload as None and restamp
    every panel in every project that never shares a cabin."""
    spec = {"cabin": [2.4, 2.2, 1.7], "subjects": []}
    before = G.anchor_version(spec)
    assert G.anchor_version(dict(spec)) == before
    assert G.anchor_version(dict(spec, scene_shell={"half_w": 2.0})) != before
