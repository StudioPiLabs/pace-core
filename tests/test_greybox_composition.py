"""Composition clauses of the greybox gate, and the kernel's pure helpers.

Mattes are drawn by hand so each clause is pinned on the fault it names and
on the frame that does not have it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from pace_core.qc import greybox_gate as gg
from pace_core.node.panel_greybox import (
    DEFAULT_ELEVATION_RANGE_DEG, camera_search_order, inside_shell, joint_points)

H, W = 272, 640


def _matte(path, box):
    a = np.zeros((H, W), dtype=np.uint8)
    y0, y1, x0, x1 = box
    a[y0:y1, x0:x1] = 255
    Image.fromarray(np.dstack([a, a, a, a]), "RGBA").save(path)


def _frame(tmp_path):
    out = tmp_path / "p_90001_.png"
    Image.new("RGB", (W, H), (128, 128, 128)).save(out)
    return out


# ── head_clear_of_background_lines ──

def _set_pass(out, stripe_x=None):
    g = np.full((H, W), 128, dtype=np.uint8)
    if stripe_x is not None:
        g[:, stripe_x:stripe_x + 3] = 40
    Image.fromarray(g, "L").save(out.parent / (out.name + ".set.png"))


def test_a_vertical_line_into_a_head_fails(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".head_a.png"), (80, 130, 300, 340))
    _set_pass(out, stripe_x=318)
    c = gg.background_line_clause(out, {"a"})
    assert c.ok is False and "a (" in c.detail


def test_a_line_beside_the_head_passes(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".head_a.png"), (80, 130, 300, 340))
    _set_pass(out, stripe_x=420)
    assert gg.background_line_clause(out, {"a"}).ok is True


def test_no_set_pass_is_unmeasured(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".head_a.png"), (80, 130, 300, 340))
    assert gg.background_line_clause(out, {"a"}).ok is None


# ── frame_not_cut_at_joint ──

def _joints(out, js):
    (out.parent / (out.name + ".joints.json")).write_text(json.dumps(js))


def test_a_frame_edge_through_the_knee_fails(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (20, H, 280, 360))
    _joints(out, {"a": {"knee_l": [0.47, 0.995, 4.0], "neck": [0.5, 0.2, 4.0]}})
    c = gg.joint_cut_clause(out, {"a"})
    assert c.ok is False and "bottom edge cuts a at the knee" in c.detail


def test_a_cut_between_joints_passes(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (20, H, 280, 360))
    _joints(out, {"a": {"knee_l": [0.47, 1.12, 4.0], "hip_l": [0.47, 0.9, 4.0]}})
    assert gg.joint_cut_clause(out, {"a"}).ok is True


def test_a_joint_at_an_edge_the_body_does_not_touch_passes(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (20, 200, 280, 360))
    _joints(out, {"a": {"knee_l": [0.47, 1.0, 4.0]}})
    assert gg.joint_cut_clause(out, {"a"}).ok is True


# ── silhouettes_not_tangent ──

def test_two_outlines_that_just_touch_fail(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (60, H, 200, 300))
    _matte(out.parent / (out.name + ".body_b.png"), (60, H, 302, 400))
    c = gg.tangency_clause(out, {"a", "b"})
    assert c.ok is False and "a and b meet" in c.detail


@pytest.mark.parametrize("b_box", [(60, H, 250, 350),      # overlap: depth reads
                                   (60, H, 360, 460)])     # clearly apart
def test_overlapping_or_separated_outlines_pass(tmp_path, b_box):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (60, H, 200, 300))
    _matte(out.parent / (out.name + ".body_b.png"), b_box)
    assert gg.tangency_clause(out, {"a", "b"}).ok is True


def test_a_crown_grazing_the_top_edge_fails(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (2, H, 280, 360))
    c = gg.tangency_clause(out, {"a"})
    assert c.ok is False and "grazes the top edge" in c.detail


# ── cast_not_in_a_row ──

def _heads(out, boxes):
    for cid, box in boxes.items():
        _matte(out.parent / (out.name + f".head_{cid}.png"), box)


def test_three_heads_at_one_height_and_size_fail(tmp_path):
    out = _frame(tmp_path)
    _heads(out, {"a": (60, 100, 100, 140), "b": (61, 101, 300, 340), "c": (60, 100, 500, 540)})
    assert gg.lineup_clause(out, {"a", "b", "c"}).ok is False


def test_a_staggered_group_passes(tmp_path):
    out = _frame(tmp_path)
    _heads(out, {"a": (60, 100, 100, 140), "b": (100, 150, 300, 350), "c": (60, 100, 500, 540)})
    assert gg.lineup_clause(out, {"a", "b", "c"}).ok is True


def test_two_heads_cannot_form_a_row(tmp_path):
    out = _frame(tmp_path)
    _heads(out, {"a": (60, 100, 100, 140), "b": (60, 100, 300, 340)})
    assert gg.lineup_clause(out, {"a", "b"}).ok is None


# ── reverse_shot_eye_lines_match ──

def test_reverse_shot_eye_lines(tmp_path):
    out = _frame(tmp_path)
    _heads(out, {"a": (40, 140, 280, 360)})
    mine = gg.eye_line(out, "a")
    assert mine == pytest.approx((40 + 0.42 * 100) / H)
    near = [{"panel_id": "p2", "eye_y": mine + 0.01}]
    far = [{"panel_id": "p2", "eye_y": mine + 0.12}]
    assert gg.eye_line_clause(out, {"a"}, near).ok is True
    bad = gg.eye_line_clause(out, {"a"}, far)
    assert bad.ok is False and "p2" in bad.detail
    assert gg.eye_line_clause(out, {"a"}, None).ok is None


def test_evaluate_reports_the_composition_clauses(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (20, 200, 280, 360))
    rep = gg.evaluate({"panel_id": "p", "subjects": [{"character_id": "a"}]}, out)
    names = [c.name for c in rep.clauses]
    for n in ("head_clear_of_background_lines", "frame_not_cut_at_joint",
              "silhouettes_not_tangent", "cast_not_in_a_row",
              "reverse_shot_eye_lines_match"):
        assert n in names
    # Clauses with nothing to measure stay unmeasured and do not fail the panel.
    assert "cast_not_in_a_row" in rep.unmeasured


# ── kernel helpers ──

def test_camera_search_starts_at_the_solved_pose_and_stays_in_class():
    order = camera_search_order(5.0, 25.0, DEFAULT_ELEVATION_RANGE_DEG, 12.0)
    assert order[0] == (5.0, 25.0)
    lo, hi = DEFAULT_ELEVATION_RANGE_DEG
    assert all(lo <= e <= hi and abs(y - 25.0) <= 12.0 for e, y in order[1:])
    change = [abs(e - 5.0) + abs(y - 25.0) for e, y in order[1:]]
    assert change == sorted(change)


def test_inside_shell():
    b = {"half_w": 2.0, "top_z": 2.4, "front_y": 5.0}
    assert inside_shell(0.5, 3.0, 1.5, b)
    assert not inside_shell(1.95, 3.0, 1.5, b)         # in the wall
    assert not inside_shell(0.5, 3.0, 2.35, b)         # through the roof
    assert not inside_shell(0.5, 4.95, 1.5, b)         # past the open front
    assert inside_shell(0.0, 99.0, 1.0, None) and not inside_shell(0.0, 0.0, 0.1, None)


def test_joint_points_from_part_groups():
    line = lambda a, b, n=200: [tuple(a[i] + (b[i] - a[i]) * t / (n - 1) for i in range(3))
                                for t in range(n)]
    parts = {
        "torso": line((0.0, 0.0, 0.95), (0.0, 0.0, 1.45)),
        "head": line((0.0, 0.0, 1.50), (0.0, 0.0, 1.75)),
        "arm_l": line((0.20, 0.0, 1.40), (0.90, 0.0, 1.40)),
        "leg_l": line((0.10, 0.0, 0.90), (0.10, 0.0, 0.00)),
    }
    j = joint_points(parts)
    assert j["neck"][2] == pytest.approx(1.51, abs=0.02)
    assert j["elbow_l"][0] == pytest.approx(0.20 + 0.42 * 0.70, abs=0.03)
    assert j["knee_l"][2] == pytest.approx(0.90 - 0.46 * 0.90, abs=0.03)
    assert j["ankle_l"][2] == pytest.approx(0.90 - 0.915 * 0.90, abs=0.03)
    assert joint_points({"head": parts["head"]}) == {}


# ── declared_in_frame_matches_staging ──
#
# A blocking sheet lists what enters the frame apart from where things stand;
# the clause holds that declaration against the build's own record.

def _props_record(out, rec):
    (out.parent / (out.name + ".props.json")).write_text(json.dumps(rec))


def test_a_prop_declared_in_frame_but_staged_out_of_it_fails(tmp_path):
    out = _frame(tmp_path)
    _props_record(out, {"main_console": {"in_frame": False, "frame_share": 0.04, "shown": 0.007}})
    c = gg.declared_in_frame_clause(out, {("prop", "main_console"): "yes"})
    assert c.ok is False and "main_console declared yes, staged no" in c.detail


def test_a_partial_declaration_matches_a_prop_the_frame_cuts(tmp_path):
    out = _frame(tmp_path)
    _props_record(out, {"main_console": {"in_frame": True, "frame_share": 0.2, "shown": 0.4}})
    assert gg.declared_in_frame_clause(out, {("prop", "main_console"): "partial"}).ok is True
    assert gg.declared_in_frame_clause(out, {("prop", "main_console"): "yes"}).ok is True
    assert gg.declared_in_frame_clause(out, {("prop", "main_console"): "no"}).ok is False


def test_a_subject_declared_out_of_frame_but_staged_fails(tmp_path):
    out = _frame(tmp_path)
    _matte(out.parent / (out.name + ".body_a.png"), (80, 200, 300, 360))
    assert gg.declared_in_frame_clause(out, {("subject", "a"): "no"}).ok is False
    assert gg.declared_in_frame_clause(out, {("subject", "a"): "yes"}).ok is True


def test_to_be_confirmed_is_reported_not_passed(tmp_path):
    c = gg.declared_in_frame_clause(_frame(tmp_path), {("prop", "main_console"): "tbd"})
    assert c.ok is None and "main_console" in c.detail


def test_a_panel_that_declares_nothing_is_unmeasured(tmp_path):
    assert gg.declared_in_frame_clause(_frame(tmp_path), {}).ok is None
