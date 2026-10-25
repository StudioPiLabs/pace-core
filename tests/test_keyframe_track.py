"""keyframes_to_track: authored camera keys resampled onto a frame grid."""
from __future__ import annotations

import math

import pytest

from pace_core.camera.keyframe_track import keyframes_to_track


def key(time, position, rotation=(80.0, 0.0, 0.0), lens=35):
    return {"time": time, "position": list(position), "rotation_deg": list(rotation), "lens_mm": lens}


def test_track_passes_through_every_key():
    keys = [key(0, (0, -6, 2)), key(1, (3, -5, 2), (80, 0, 30), 50), key(2, (4, -1, 3), (70, 0, 75))]
    track = keyframes_to_track(keys, fps=8)
    assert len(track) == 17
    for frame, k in ((0, keys[0]), (8, keys[1]), (16, keys[2])):
        assert track[frame]["position"] == pytest.approx(k["position"], abs=1e-4)
        assert track[frame]["rotation_deg"] == pytest.approx(k["rotation_deg"], abs=1e-3)
        assert track[frame]["lens_mm"] == k["lens_mm"]


def test_single_key_holds_for_the_requested_length():
    track = keyframes_to_track([key(0, (1, 2, 3))], fps=24, n_frames=5)
    assert [p["position"] for p in track] == [[1.0, 2.0, 3.0]] * 5


def test_n_frames_retimes_and_ease_starts_gently():
    keys = [key(0, (0, 0, 0)), key(3, (6, 0, 0))]
    linear = keyframes_to_track(keys, fps=24, n_frames=11, interpolation="linear")
    eased = keyframes_to_track(keys, fps=24, n_frames=11, interpolation="linear", ease="ease_in_out")
    assert len(linear) == 11 and linear[-1]["position"][0] == pytest.approx(6.0)
    assert eased[1]["position"][0] < linear[1]["position"][0]
    assert eased[-1]["position"] == linear[-1]["position"]


def test_rotation_takes_the_short_way_round():
    track = keyframes_to_track([key(0, (0, 0, 0), (80, 0, 170)), key(1, (0, 0, 0), (80, 0, -170))],
                               fps=10)
    yaws = [p["rotation_deg"][2] for p in track]
    assert all(170 - 1e-6 <= y <= 190 + 1e-6 for y in yaws)
    assert all(b >= a - 1e-6 for a, b in zip(yaws, yaws[1:]))


def test_smooth_stays_on_a_straight_path():
    keys = [key(0, (0, 0, 0)), key(1, (1, 0, 0)), key(3, (5, 0, 0)), key(4, (6, 0, 0))]
    track = keyframes_to_track(keys, fps=10)
    xs = [p["position"][0] for p in track]
    assert all(abs(p["position"][1]) < 1e-9 and abs(p["position"][2]) < 1e-9 for p in track)
    assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))


def test_smooth_curve_is_continuous_through_a_corner():
    keys = [key(0, (0, 0, 0)), key(1, (2, 0, 0)), key(2, (2, 2, 0))]
    track = keyframes_to_track(keys, fps=30)
    steps = [math.dist(p["position"], q["position"]) for p, q in zip(track, track[1:])]
    assert max(steps) < 0.12


def test_dense_take_resamples_linearly():
    take = [key(t / 60, (t / 60, 0, 1)) for t in range(61)]
    track = keyframes_to_track(take, fps=24, interpolation="linear")
    assert len(track) == 25
    assert [p["position"][0] for p in track] == pytest.approx([f / 24 for f in range(25)], abs=1e-4)


@pytest.mark.parametrize("keys, message", [
    ([], "no camera keys"),
    ([key(1, (0, 0, 0)), key(1, (1, 0, 0))], "distinct times"),
    ([{"time": 0, "position": [0, 0], "rotation_deg": [0, 0, 0]}], "three numbers"),
    ([key(0, (0, 0, 0), lens=0)], "lens_mm"),
])
def test_invalid_keys_are_rejected(keys, message):
    with pytest.raises(ValueError, match=message):
        keyframes_to_track(keys, fps=24)


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError, match="interpolation"):
        keyframes_to_track([key(0, (0, 0, 0))], fps=24, interpolation="cubic")
