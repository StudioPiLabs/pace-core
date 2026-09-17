"""Every panel of a scene is built inside one cabin.

The vehicle shell is stretched to contain each panel's camera, so sized per
panel it followed the lens: a wide and the over-the-shoulders after it were
built at different widths and lengths, and the walls moved at every cut. make_scene_greyboxes builds each panel once to learn the
shell it needs and again inside the union.
"""
from __future__ import annotations

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


def test_a_scene_not_built_as_a_procedural_shell_is_built_once(monkeypatch):
    b = {"a": None, "b": None}
    build, calls = _fake(b, style=None)
    monkeypatch.setattr(G, "make_panel_greybox", build)
    G.make_scene_greyboxes("p", "s", lambda pid: f"/out/{pid}.png", scene=SCENE)
    assert all(c[2] is None for c in calls if c[1].startswith("/out/"))


def test_the_shared_cabin_moves_the_anchor_only_when_present():
    """Listed unconditionally it would enter every payload as None and restamp
    every panel in every project that never shares a cabin."""
    spec = {"cabin": [2.4, 2.2, 1.7], "subjects": []}
    before = G.anchor_version(spec)
    assert G.anchor_version(dict(spec)) == before
    assert G.anchor_version(dict(spec, scene_shell={"half_w": 2.0})) != before
