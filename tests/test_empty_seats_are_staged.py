"""A vehicle's unoccupied seats are in the control image.

Seats were staged one per SUBJECT, so a two-person shot in a four-seat car put
two chairs on the floor and left the rest of it bare. Bare floor in a control
image is an invitation rather than a constraint: measured on
one panel, the greybox staged two seats and the delivered
panel came back with a third chair and a wraparound console invented into the
empty half of the cabin — while the two SUBJECTS landed inside their staged
mattes, so the bodies followed the greybox and only the set did not.

How many seats a car has is a fact about the set, and the stubs had nowhere to
state it. `seat_count` on the location says it; the occupied placement is
untouched, because it is tuned and measured against declared screen_position
and nothing here should move it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import _cabin_slots, _unoccupied_slots  # noqa: E402

CABIN = [2.0, 3.2, 1.5]


def test_a_car_with_more_seats_than_passengers_stages_the_rest():
    occ = [(0.48, 0.32), (-0.48, 0.32)]
    assert len(_unoccupied_slots(4, occ, CABIN)) == 2


def test_a_full_car_stages_no_extras():
    occ = [(0.48, 0.32), (-0.48, 0.32)]
    assert _unoccupied_slots(2, occ, CABIN) == []


def test_an_undeclared_seat_count_changes_nothing():
    """Every location without the field must behave exactly as before."""
    occ = [(0.48, 0.32), (-0.48, 0.32)]
    assert _unoccupied_slots(0, occ, CABIN) == []
    assert _unoccupied_slots(None, occ, CABIN) == []


def test_more_passengers_than_seats_does_not_go_negative():
    occ = [(0.48, 0.32), (-0.48, 0.32), (0.0, -0.7)]
    assert _unoccupied_slots(2, occ, CABIN) == []


def test_each_passenger_claims_the_seat_nearest_them():
    """An occupant sitting in the rear must not free the rear slot and leave a
    chair standing inside them."""
    slots = _cabin_slots(4, CABIN)
    rear = slots[2]
    free = _unoccupied_slots(4, [rear], CABIN)
    assert rear not in free
    assert len(free) == 3


def test_the_layout_puts_the_front_row_first():
    """Front-first is how a car fills and how these stubs describe the
    vehicles ('front seats')."""
    slots = _cabin_slots(4, CABIN)
    assert slots[0][1] > slots[2][1], "first row should sit ahead of the second"


def test_the_occupied_placement_is_not_derived_from_this():
    """The tuned two/three-subject placement stays the source of truth for
    where people sit; these slots only fill in around it."""
    src = (Path(__file__).resolve().parents[1] / "src/pace_core/node/panel_greybox.py").read_text()
    spec_line = [l for l in src.splitlines() if '"empty_seats"' in l]
    assert spec_line, "spec must carry empty_seats"
    assert "_unoccupied_slots" in "\n".join(spec_line)
