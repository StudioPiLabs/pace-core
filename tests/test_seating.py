"""Seat positions taken from the standard cars are measured by.

The point of this module is independence, so that is what these pin. A mark
derived from the declared box would be the greybox builder's own number, and
a clause comparing staged against declared would read zero on every panel by
construction -- the no-gradient condition arriving inside the one clause added
to escape it. These assert the lateral coordinate comes from W5 instead, that
it does not track the box, and that the one coordinate with no published value
is the caller's rather than invented here.
"""
from __future__ import annotations

import pytest

from pace_core.setup.seating import (COUPLE_DISTANCE_L50_M, HIP_ROOM_W5_M,
                                     SEATED_Z_M, cabin_rows, cabin_seat_marks,
                                     hip_room_for, row_seat_xs)


def test_the_spread_does_not_follow_the_declared_box():
    # The whole reason for this module: two cabins of different declared width
    # seat people at the same hip room, because hips are the same width.
    a = cabin_seat_marks(rows=(2,), cabin_width_m=2.05, front_row_y_m=0.0)
    b = cabin_seat_marks(rows=(2,), cabin_width_m=1.80, front_row_y_m=0.0)
    assert [m.xy[0] for m in a] == [m.xy[0] for m in b]


def test_the_spread_is_the_standards_hip_room():
    marks = cabin_seat_marks(rows=(2,), cabin_width_m=2.05, front_row_y_m=0.0)
    assert marks[0].xy[0] - marks[1].xy[0] == pytest.approx(
        HIP_ROOM_W5_M["compact"])


def test_a_wider_cabin_is_held_to_a_wider_interior_not_to_its_own_box():
    assert hip_room_for(2.5) == HIP_ROOM_W5_M["wide"]
    assert hip_room_for(2.05) == HIP_ROOM_W5_M["compact"]
    # A proxy mesh narrower than any published interior does not narrow people.
    assert hip_room_for(0.9) == HIP_ROOM_W5_M["narrow"]


def test_rows_step_back_by_the_couple_distance():
    marks = cabin_seat_marks(rows=(2, 1), cabin_width_m=2.05,
                             front_row_y_m=0.21)
    assert marks[0].xy[1] == pytest.approx(0.21)
    assert marks[-1].xy[1] == pytest.approx(0.21 - COUPLE_DISTANCE_L50_M)


def test_the_longitudinal_offset_is_the_callers():
    # No published figure places a row along the cabin, so this module must not
    # supply one; passing a different offset must move the rows and nothing
    # else.
    a = cabin_seat_marks(rows=(2, 1), cabin_width_m=2.05, front_row_y_m=0.0)
    b = cabin_seat_marks(rows=(2, 1), cabin_width_m=2.05, front_row_y_m=1.0)
    assert [m.xy[0] for m in a] == [m.xy[0] for m in b]
    assert [m.xy[1] + 1.0 for m in a] == pytest.approx([m.xy[1] for m in b])


def test_a_row_of_one_seat_is_centred():
    # A 2+1 cabin's rear seat. This is a row with one SEAT, not a row with one
    # occupant: a lone occupant of a two-seat row is in one of its two seats,
    # and which one is an assignment rather than a fact about the car.
    assert row_seat_xs(1, 0.69) == [0.0]


def test_a_rows_seats_do_not_depend_on_who_is_in_them():
    # The argument counts seats. Conflating it with occupancy is what put a
    # lone front-row occupant at a centre the row has no seat at.
    assert row_seat_xs(2, 0.69) == [0.69, -0.69]


def test_three_across_seats_the_middle_one_in_the_middle():
    assert row_seat_xs(3, 0.69) == [0.69, 0.0, -0.69]


def test_seats_are_screen_left_first():
    # Screen-left is world +X, so the list descends; a consumer zips it against
    # subjects already sorted by declared screen x.
    xs = [m.xy[0] for m in cabin_seat_marks(rows=(2,), cabin_width_m=2.05,
                                            front_row_y_m=0.0)]
    assert xs == sorted(xs, reverse=True)


def test_a_declared_mark_says_it_is_declared():
    for m in cabin_seat_marks(rows=(2, 1), cabin_width_m=2.05,
                              front_row_y_m=0.0):
        assert m.declared is True
        assert m.z == SEATED_Z_M


def test_facing_is_carried_through_rather_than_assumed():
    # Which way the seats face is the location's fact, not the standard's.
    marks = cabin_seat_marks(rows=(2,), cabin_width_m=2.05, front_row_y_m=0.0,
                             facing_deg=90)
    assert all(m.facing_deg == 90 for m in marks)
    assert cabin_seat_marks(rows=(2,), cabin_width_m=2.05,
                            front_row_y_m=0.0)[0].facing_deg is None


# ── the cabin's own row structure ───────────────────────────────────────────

def test_a_cabins_rows_are_read_off_its_seat_count():
    assert cabin_rows(2) == (2,)
    assert cabin_rows(3) == (2, 1)
    assert cabin_rows(5) == (2, 3)


def test_the_rows_describe_the_car_and_not_the_cast():
    # Same cabin, same marks, whoever turns up. A row structure that shrank
    # with the cast would make the declaration a function of the staging.
    a = cabin_seat_marks(rows=cabin_rows(3), cabin_width_m=2.05,
                         front_row_y_m=0.21)
    assert len(a) == 3


def test_a_cabin_with_no_seats_has_no_rows():
    assert cabin_rows(0) == ()
