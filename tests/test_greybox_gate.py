"""The gate between the 3D anchor and the generator.

A gate that cannot fail is decoration, so the load-bearing tests here are the
ones that stage a specific defect and require the gate to name it: a body that
did not render, a matte that lost its alpha, a cast delivered in the wrong
left-to-right order, and a control pass left over from an earlier build.

The screen-order case is not hypothetical. The evaluation corpus's 2026-09-02 build
of scene_01_shot_03_panel_0003 rendered the three-person cabin reversed --
omar, declared leftmost at x=0.38, came out on the right -- and nothing in the
pipeline noticed. `test_reversed_cast_fails` is that regression in miniature.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.qc.greybox_gate import (  # noqa: E402
    MIN_SUBJECT_AREA, STALE_TOLERANCE_S, control_provenance, evaluate,
    matte_stats, screen_order_agreement,
)

W, H = 200, 100


def _matte(path: Path, x0: int, x1: int, *, alpha: bool = True,
           y0: int = 20, y1: int = 80) -> None:
    """Write one body matte: a filled block between two x positions."""
    a = np.zeros((H, W, 4), dtype=np.uint8)
    a[y0:y1, x0:x1, :3] = 200
    a[y0:y1, x0:x1, 3] = 255
    im = Image.fromarray(a, "RGBA")
    if not alpha:
        im = im.convert("RGB")
    im.save(path)


def _spec(tmp: Path, declared: list[tuple[str, float]]) -> dict:
    return {"panel_id": "p", "subjects": [
        {"character_id": c, "declared_x": x, "mesh": "m.glb", "pose": "sitting"}
        for c, x in declared]}


def _build(tmp: Path, bodies: dict[str, tuple[int, int]], *,
           no_alpha: set[str] = frozenset()) -> Path:
    """A beauty frame plus one matte per body, all written now."""
    out = tmp / "gb.png"
    Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(out)
    for cid, (x0, x1) in bodies.items():
        _matte(out.parent / (out.name + f".body_{cid}.png"), x0, x1,
               alpha=cid not in no_alpha)
    return out


# ── the pure pieces ───────────────────────────────────────────────────────

def test_matte_stats_reads_the_alpha_not_the_colour(tmp_path):
    f = tmp_path / "m.png"
    _matte(f, 0, W // 2)                       # left half, full height band
    s = matte_stats(f)
    assert s["no_alpha"] is False
    assert s["area"] == pytest.approx(0.5 * 0.6, abs=0.01)
    assert s["centroid_x"] == pytest.approx(0.25, abs=0.01)


def test_a_matte_without_alpha_is_not_read_as_a_full_body(tmp_path):
    """Losing film_transparent makes every body opaque everywhere.

    Read as colour that is a body filling the frame, which would pass both
    visibility and area for a subject that may not be in shot at all.
    """
    f = tmp_path / "m.png"
    _matte(f, 0, W // 2, alpha=False)
    s = matte_stats(f)
    assert s["no_alpha"] is True
    assert s["area"] is None


def test_an_empty_matte_measures_zero_not_missing(tmp_path):
    f = tmp_path / "m.png"
    _matte(f, 0, 0)
    assert matte_stats(f) == {"no_alpha": False, "area": 0.0, "centroid_x": None}


def test_missing_matte_is_none(tmp_path):
    assert matte_stats(tmp_path / "nope.png") is None


def test_screen_order_agreement_counts_pairs():
    declared = [("a", 0.2), ("b", 0.5), ("c", 0.8)]
    frac, bad = screen_order_agreement(declared, {"a": 0.1, "b": 0.5, "c": 0.9})
    assert (frac, bad) == (1.0, [])
    # c and a swapped: (a,c) and (b,c) break, (a,b) holds.
    frac, bad = screen_order_agreement(declared, {"a": 0.5, "b": 0.7, "c": 0.1})
    assert frac == pytest.approx(1 / 3)
    assert len(bad) == 2


def test_equal_declared_x_asserts_no_order():
    """Two subjects declared at the same x cannot be rendered in the wrong
    order, because they did not ask for one."""
    frac, bad = screen_order_agreement([("a", 0.5), ("b", 0.5)],
                                       {"a": 0.9, "b": 0.1})
    assert (frac, bad) == (1.0, [])


# ── provenance ────────────────────────────────────────────────────────────

def test_a_pass_older_than_its_frame_fails(tmp_path):
    out = _build(tmp_path, {"a": (10, 40)})
    depth = tmp_path / (out.name + ".depth.png")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(depth)
    old = out.stat().st_mtime - 3600
    os.utime(depth, (old, old))
    c = control_provenance(out)
    assert c.ok is False and "depth.png" in c.detail


def test_a_pass_written_after_its_frame_is_normal(tmp_path):
    """The kernel writes the frame first and the derived passes after, so
    positive skew is the healthy case and must not be flagged."""
    out = _build(tmp_path, {"a": (10, 40)})
    depth = tmp_path / (out.name + ".depth.png")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(depth)
    later = out.stat().st_mtime + 120
    os.utime(depth, (later, later))
    assert control_provenance(out).ok is True


def test_timestamp_noise_is_absorbed(tmp_path):
    out = _build(tmp_path, {"a": (10, 40)})
    depth = tmp_path / (out.name + ".depth.png")
    Image.fromarray(np.zeros((H, W), np.uint8)).save(depth)
    t = out.stat().st_mtime - (STALE_TOLERANCE_S / 2)
    os.utime(depth, (t, t))
    assert control_provenance(out).ok is True


# ── the gate ──────────────────────────────────────────────────────────────

def _clause(report, name):
    return next(c for c in report.clauses if c.name == name)


def test_a_correctly_staged_panel_passes(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50), "nina": (80, 120),
                            "theo": (150, 190)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5),
                                  ("theo", 0.62)]), out)
    assert r.ok
    assert r.failures == []


def test_reversed_cast_fails(tmp_path):
    """The 2026-09-02 regression: declared order, rendered mirrored."""
    out = _build(tmp_path, {"omar": (150, 190), "nina": (80, 120),
                            "theo": (10, 50)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5),
                                  ("theo", 0.62)]), out)
    assert not r.ok
    order = _clause(r, "screen_order")
    assert order.ok is False and order.value == 0.0
    assert "omar declared left of nina" in order.detail


def test_a_body_that_did_not_render_fails_visibility(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50), "nina": (80, 120)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5),
                                  ("theo", 0.62)]), out)
    vis = _clause(r, "required_entity_visibility")
    assert vis.ok is False and vis.value == pytest.approx(2 / 3, abs=1e-4)
    assert "no matte: theo" in vis.detail


def test_a_body_staged_outside_the_frustum_fails_visibility(tmp_path):
    """The matte exists and is empty -- a different defect from no matte."""
    out = _build(tmp_path, {"omar": (10, 50), "nina": (0, 0)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5)]), out)
    vis = _clause(r, "required_entity_visibility")
    assert vis.ok is False and "matte empty" in vis.detail


def test_an_alphaless_matte_fails_rather_than_passing_as_full_frame(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50), "nina": (80, 120)},
                 no_alpha={"nina"})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5)]), out)
    vis = _clause(r, "required_entity_visibility")
    assert vis.ok is False and "no alpha" in vis.detail


def test_a_subject_below_the_area_floor_fails(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50), "nina": (100, 101)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5)]), out)
    area = _clause(r, "projected_subject_area")
    assert area.ok is False
    assert area.value < MIN_SUBJECT_AREA and "nina" in area.detail


def test_required_can_be_narrowed_to_the_beats_own_facts(tmp_path):
    """A subject the geometry staged but the beat does not require is not a
    reason to refuse the panel."""
    out = _build(tmp_path, {"omar": (10, 50)})
    spec = _spec(tmp_path, [("omar", 0.38), ("nina", 0.5)])
    assert evaluate(spec, out).ok is False
    assert evaluate(spec, out, required={"omar"}).ok is True


# ── what the gate does NOT claim ──────────────────────────────────────────

def test_unmeasurable_clauses_are_reported_not_silently_passed(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50), "nina": (80, 120)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5)]), out)
    assert "focal_action_readability" in r.unmeasured
    assert _clause(r, "focal_action_readability").ok is None
    assert r.ok is True          # unmeasurable does not block


def test_a_single_subject_declares_no_screen_order(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38)]), out)
    assert "screen_order" in r.unmeasured
    assert r.ok is True


# ── occlusion ─────────────────────────────────────────────────────────────

def test_two_subjects_on_one_sight_line_fail(tmp_path):
    """Scene 2's defect, in miniature. Its three-quarter azimuth stacked two
    of three subjects, leaving 67% of one under the other, and every other
    clause passed -- so a panel that rendered a declared subject as a sliver
    of forehead was called an anchor."""
    out = _build(tmp_path, {"nina": (60, 120), "theo": (75, 135)})
    r = evaluate(_spec(tmp_path, [("nina", 0.5), ("theo", 0.62)]), out)
    occ = _clause(r, "subject_occlusion")
    assert occ.ok is False
    assert occ.value > 0.35
    assert "under" in occ.detail
    assert not r.ok


def test_side_by_side_subjects_pass(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50), "nina": (80, 120)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38), ("nina", 0.5)]), out)
    assert _clause(r, "subject_occlusion").ok is True
    assert r.ok


def test_some_overlap_is_how_depth_reads(tmp_path):
    """A gate demanding zero overlap would forbid a two-shot, so the floor is
    against a subject being effectively absent, not against overlap itself."""
    out = _build(tmp_path, {"a": (10, 60), "b": (52, 102)})
    r = evaluate(_spec(tmp_path, [("a", 0.4), ("b", 0.6)]), out)
    occ = _clause(r, "subject_occlusion")
    assert 0 < occ.value < 0.35 and occ.ok is True


def test_a_single_subject_cannot_occlude_itself(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50)})
    r = evaluate(_spec(tmp_path, [("omar", 0.38)]), out)
    assert _clause(r, "subject_occlusion").ok is None
    assert "subject_occlusion" in r.unmeasured


# ── read point behind the set ─────────────────────────────────────────────

def _visible(path: Path, x0: int, x1: int, y0: int = 20, y1: int = 80) -> None:
    """The flat two-tone pass: white where the body is seen past the set."""
    a = np.zeros((H, W), dtype=np.uint8)
    a[y0:y1, x0:x1] = 255
    Image.fromarray(a, "L").save(path)


def _head(path: Path, x0: int, x1: int) -> None:
    _matte(path, x0, x1, y0=20, y1=40)


def test_a_head_behind_the_set_fails(tmp_path):
    """Scene 2 again, one level out. omar's body was 51% visible and his head
    0%: the set stood in front of exactly the part a frame is composed
    around, and body-vs-body occlusion could not see it because those mattes
    are rendered with the set hidden."""
    out = _build(tmp_path, {"omar": (10, 50)})
    _head(tmp_path / (out.name + ".head_omar.png"), 10, 50)
    _visible(tmp_path / (out.name + ".visible_omar.png"), 10, 50, y0=45, y1=80)
    r = evaluate(_spec(tmp_path, [("omar", 0.38)]), out)
    rp = _clause(r, "read_point_visible")
    assert rp.ok is False and "omar 0%" in rp.detail
    assert not r.ok


def test_a_head_the_camera_can_see_passes(tmp_path):
    out = _build(tmp_path, {"omar": (10, 50)})
    _head(tmp_path / (out.name + ".head_omar.png"), 10, 50)
    _visible(tmp_path / (out.name + ".visible_omar.png"), 10, 50)
    assert _clause(evaluate(_spec(tmp_path, [("omar", 0.38)]), out),
                   "read_point_visible").ok is True


def test_a_build_without_the_visible_pass_says_so(tmp_path):
    """An older greybox has no visible-matte beside it. That is unmeasured,
    not passing: reporting it as a pass would let a stale build clear a clause
    it was never rendered for."""
    out = _build(tmp_path, {"omar": (10, 50)})
    _head(tmp_path / (out.name + ".head_omar.png"), 10, 50)
    r = evaluate(_spec(tmp_path, [("omar", 0.38)]), out)
    assert _clause(r, "read_point_visible").ok is None
    assert "read_point_visible" in r.unmeasured


def test_action_readability_still_says_it_cannot_measure_an_action(tmp_path):
    """Visibility is not readability: seeing a head does not say whether the
    action reads, and the clause must keep admitting that."""
    out = _build(tmp_path, {"omar": (10, 50)})
    _head(tmp_path / (out.name + ".head_omar.png"), 10, 50)
    _visible(tmp_path / (out.name + ".visible_omar.png"), 10, 50)
    far = _clause(evaluate(_spec(tmp_path, [("omar", 0.38)]), out),
                  "focal_action_readability")
    assert far.ok is None and "read_point_visible" in far.detail
