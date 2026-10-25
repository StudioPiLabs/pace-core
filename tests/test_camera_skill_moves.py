"""compile_shot's camera-relative moves: dolly_zoom, truck, frame, handheld."""
from __future__ import annotations

import math

import pytest

from pace_core.camera.camera_movement_kb import KB
from pace_core.camera.camera_skills import KNOWN_SKILLS, compile_shot
from pace_core.setup.composition_solver import camera_axes, project_to_screen

AIM = [0.0, 0.0, 0.6]


def test_new_skills_are_known_and_kb_still_validates():
    assert {"dolly_zoom", "truck", "frame", "handheld"} <= KNOWN_SKILLS
    assert all(s["skill"] in KNOWN_SKILLS for e in KB.values() for s in e.skills)


def test_zoom_keeps_whole_millimetre_lenses():
    track = compile_shot([{"skill": "zoom", "from_mm": 35, "to_mm": 75, "frames": 9}])
    assert all(isinstance(p["lens_mm"], int) for p in track)


def test_dolly_zoom_holds_the_subject_size():
    track = compile_shot([{"skill": "dolly_zoom", "to": 2.0, "frames": 17}], lens_mm=50)
    ratios = [math.dist(p["position"], AIM) / p["lens_mm"] for p in track]
    assert math.dist(track[-1]["position"], AIM) < math.dist(track[0]["position"], AIM)
    assert track[-1]["lens_mm"] < track[0]["lens_mm"]
    assert max(ratios) - min(ratios) < 1e-3 * ratios[0]


def test_parallel_truck_keeps_orientation_and_moves_right():
    track = compile_shot([{"skill": "truck", "distance": 3.0, "frames": 9}])
    assert all(p["rotation_deg"] == track[0]["rotation_deg"] for p in track)
    right, _, _ = camera_axes(track[0]["rotation_deg"])
    move = [b - a for a, b in zip(track[0]["position"], track[-1]["position"])]
    assert sum(m * r for m, r in zip(move, right)) == pytest.approx(3.0, abs=1e-3)


def test_truck_without_follow_reaims_at_the_subject():
    track = compile_shot([{"skill": "truck", "distance": 3.0, "follow": 0, "frames": 9}])
    end = project_to_screen(track[-1]["position"], track[-1]["rotation_deg"], 35, AIM)
    assert end == pytest.approx((0.5, 0.5), abs=1e-3)


def test_frame_places_the_subject_and_persists():
    track = compile_shot([{"skill": "frame", "x": 1 / 3, "y": 0.4, "frames": 9},
                          {"skill": "orbit", "degrees": 60, "frames": 9}])
    for pose in (track[8], track[-1]):
        xy = project_to_screen(pose["position"], pose["rotation_deg"], pose["lens_mm"], AIM)
        assert xy == pytest.approx((1 / 3, 0.4), abs=2e-3)


def test_frame_outside_the_picture_is_rejected():
    with pytest.raises(ValueError, match="inside the frame"):
        compile_shot([{"skill": "frame", "x": 1.2, "y": 0.5}])


def test_handheld_is_seeded_bounded_and_ramped():
    skills = [{"skill": "hold", "frames": 32}, {"skill": "handheld", "frames": 32},
              {"skill": "hold", "frames": 32}]
    a, b = compile_shot(skills, seed=1), compile_shot(skills, seed=1)
    assert a == b
    assert a != compile_shot(skills, seed=2)
    still = compile_shot([{"skill": "hold", "frames": 96}])
    assert a[:24] == still[:24] and a[-24:] == still[-24:]
    offsets = [math.dist(p["position"], q["position"]) for p, q in zip(a, still)]
    assert 0 < max(offsets) < 0.05
    steps = [math.dist(p["position"], q["position"]) for p, q in zip(a, a[1:])]
    assert max(steps) < 0.01


def test_shake_overlays_a_move_without_changing_where_it_ends():
    plain = compile_shot([{"skill": "orbit", "degrees": 90, "frames": 24},
                          {"skill": "hold", "frames": 24}])
    shaken = compile_shot([{"skill": "orbit", "degrees": 90, "frames": 24, "shake": 1.0},
                           {"skill": "hold", "frames": 24}])
    assert shaken[-1] == plain[-1]
