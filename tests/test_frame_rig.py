"""frame_rig: composed frames and the rigs that move the camera between them."""
from __future__ import annotations

import math

import pytest

from pace_core.camera.frame_rig import SHOT_SIZES, frame_subject, rig_track
from pace_core.node.panel_modeling import SHOT_SIZE_FRAMING
from pace_core.setup.composition_solver import camera_axes, project_to_screen

BASE, HEIGHT = [1.0, 2.0, 0.0], 1.8
TOP = [1.0, 2.0, 1.8]


def frame(shot_size, bearing, **kw):
    return frame_subject(BASE, HEIGHT, shot_size=shot_size, bearing_deg=bearing, **kw)


def screen(pose, point):
    return project_to_screen(pose["position"], pose["rotation_deg"], pose["lens_mm"], point)


def timed(time, pose, rig="steadicam", ease="feather"):
    return {"time": time, **pose, "rig": rig, "ease": ease}


@pytest.mark.parametrize("size", ["close_up", "medium_close_up", "medium", "full"])
@pytest.mark.parametrize("lens", [24, 50, 85])
def test_shot_size_band_fills_the_modeling_share_of_the_frame(size, lens):
    band, margin = SHOT_SIZE_FRAMING[size]
    pose = frame(size, 30, lens_mm=lens)
    cut = [1.0, 2.0, HEIGHT * (1 - band)]
    top_y, cut_y = screen(pose, TOP)[1], screen(pose, cut)[1]
    assert cut_y - top_y == pytest.approx(1 / margin, abs=2e-3)
    assert (top_y + cut_y) / 2 == pytest.approx(0.5, abs=2e-3)


def test_sizes_are_ordered_tightest_first():
    assert SHOT_SIZES[0] == "extreme_close_up" and SHOT_SIZES[-1] == "establishing"


def test_bearing_angle_placement_and_dutch():
    side = frame("medium", 90)
    assert side["position"][0] == pytest.approx(BASE[0], abs=1e-3) and side["position"][1] > BASE[1]
    high, low = frame("medium", 0, angle="high_angle"), frame("medium", 0, angle="low_angle")
    assert high["position"][2] > side["position"][2] > low["position"][2]
    thirds = frame("medium", 0, screen_x=1 / 3)
    band_mid = [1.0, 2.0, HEIGHT * (1 - SHOT_SIZE_FRAMING["medium"][0] / 2)]
    assert screen(thirds, band_mid)[0] == pytest.approx(1 / 3, abs=2e-3)
    level, dutch = camera_axes(frame("medium", 0)["rotation_deg"]), camera_axes(frame("medium", 0, roll_deg=12)["rotation_deg"])
    assert dutch[2] == pytest.approx(level[2], abs=1e-4)                       # same lens axis
    assert math.degrees(math.acos(sum(x * y for x, y in zip(dutch[1], level[1])))) == pytest.approx(12, abs=1e-2)


def test_unknown_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="shot size"):
        frame("tight", 0)
    with pytest.raises(ValueError, match="angle"):
        frame("medium", 0, angle="worm")


def test_tripod_pans_from_one_place_and_refuses_to_travel():
    a, b = frame("medium", 0), frame("medium", 0)
    b = {**b, "rotation_deg": [b["rotation_deg"][0], 0, b["rotation_deg"][2] + 40], "lens_mm": 50}
    track = rig_track([timed(0, a), timed(1, b, "tripod")], fps=12)
    assert all(p["position"] == a["position"] for p in track)
    assert track[-1]["lens_mm"] == 50
    with pytest.raises(ValueError, match="tripod cannot travel"):
        rig_track([timed(0, a), timed(1, frame("full", 0), "tripod")], fps=12)


def test_dolly_runs_on_a_straight_track():
    a, b = frame("full", 0), frame("close_up", 20)
    track = rig_track([timed(0, a), timed(2, b, "dolly")], fps=12)
    pa, pb = a["position"], b["position"]
    for p in track:
        cross = [(p["position"][i] - pa[i]) for i in range(3)]
        along = [(pb[i] - pa[i]) for i in range(3)]
        n = math.dist(pa, pb)
        proj = sum(c * l for c, l in zip(cross, along)) / n
        assert math.dist(p["position"], [pa[i] + along[i] * proj / n for i in range(3)]) < 1e-3


def test_jib_bows_up_in_the_vertical_plane_of_the_move():
    a = {"position": [0.0, -6.0, 1.0], "rotation_deg": [80, 0, 0], "lens_mm": 35}
    b = {"position": [0.0, -3.0, 3.0], "rotation_deg": [60, 0, 0], "lens_mm": 35}
    track = rig_track([timed(0, a), timed(2, b, "jib", "linear")], fps=10)
    assert track[0]["position"] == a["position"] and track[-1]["position"] == pytest.approx(b["position"], abs=1e-4)
    mid = track[len(track) // 2]["position"]
    assert all(abs(p["position"][0]) < 1e-6 for p in track)
    assert mid[2] > (a["position"][2] + b["position"][2]) / 2


def test_orbit_holds_the_composition_around_the_subject():
    a, b = frame("medium", -60, screen_x=1 / 3), frame("medium", 60, screen_x=1 / 3)
    band_mid = [1.0, 2.0, HEIGHT * (1 - SHOT_SIZE_FRAMING["medium"][0] / 2)]
    track = rig_track([timed(0, a), timed(3, b, "orbit")], fps=8)
    radii = [math.hypot(p["position"][0] - 1.0, p["position"][1] - 2.0) for p in track]
    assert max(radii) - min(radii) < 2e-3
    for p in track:
        assert screen(p, band_mid)[0] == pytest.approx(1 / 3, abs=3e-3)


def test_orbit_follows_a_moving_subject():
    a = frame("medium", -45)
    b = frame_subject([4.0, 2.0, 0.0], HEIGHT, shot_size="medium", bearing_deg=45)
    track = rig_track([timed(0, a), timed(2, b, "orbit", "linear")], fps=8)
    assert track[-1]["position"] == pytest.approx(b["position"], abs=1e-3)
    mid, subject_mid = track[8], [2.5, 2.0, a["subject"][2]]
    assert math.hypot(mid["position"][0] - 2.5, mid["position"][1] - 2.0) == pytest.approx(
        math.hypot(a["position"][0] - 1.0, a["position"][1] - 2.0), abs=1e-3)
    assert screen(mid, subject_mid) == pytest.approx((0.5, 0.5), abs=2e-3)


def test_orbit_needs_a_shared_subject():
    a = {"position": [0, 0, 1], "rotation_deg": [90, 0, 0], "lens_mm": 35}
    b = {"position": [3, 0, 1], "rotation_deg": [90, 0, 0], "lens_mm": 35}
    with pytest.raises(ValueError, match="same subject"):
        rig_track([timed(0, a), timed(1, b, "orbit")], fps=8)


def test_steadicam_flows_through_frames_and_handheld_shakes_it():
    poses = [frame("full", -40), frame("medium", 0), frame("medium_close_up", 40)]
    steady = rig_track([timed(0, poses[0]), timed(1, poses[1]), timed(2, poses[2])], fps=12)
    for f, pose in ((0, poses[0]), (12, poses[1]), (24, poses[2])):
        assert steady[f]["position"] == pytest.approx(pose["position"], abs=1e-4)
    hand = rig_track([timed(0, poses[0]), timed(1, poses[1], "handheld"), timed(2, poses[2], "handheld")], fps=12)
    assert hand == rig_track([timed(0, poses[0]), timed(1, poses[1], "handheld"),
                              timed(2, poses[2], "handheld")], fps=12)
    offsets = [math.dist(p["position"], q["position"]) for p, q in zip(hand, steady)]
    assert 0 < max(offsets) < 0.05


def test_lens_change_between_equal_sizes_is_a_dolly_zoom():
    wide, tele = frame("medium", 0, lens_mm=24), frame("medium", 0, lens_mm=85)
    band_mid = [1.0, 2.0, HEIGHT * (1 - SHOT_SIZE_FRAMING["medium"][0] / 2)]
    track = rig_track([timed(0, tele), timed(2, wide, "dolly", "linear")], fps=10)
    sizes = [math.dist(p["position"], band_mid) / p["lens_mm"] for p in track]
    assert max(sizes) - min(sizes) < 1e-3 * sizes[0]
    same = rig_track([timed(0, frame("full", 0)), timed(2, frame("close_up", 0), "dolly")], fps=10)
    assert {p["lens_mm"] for p in same} == {35}


def test_retime_and_vocabulary_errors():
    a, b = frame("full", 0), frame("medium", 0)
    assert len(rig_track([timed(0, a), timed(5, b, "dolly")], fps=24, n_frames=33)) == 33
    with pytest.raises(ValueError, match="rig must be"):
        rig_track([timed(0, a), timed(1, b, "crane")], fps=24)
    with pytest.raises(ValueError, match="ease must be"):
        rig_track([timed(0, a), timed(1, b, "dolly", "bounce")], fps=24)
