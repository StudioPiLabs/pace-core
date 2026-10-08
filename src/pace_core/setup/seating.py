"""Where a vehicle's seats are, from the dimensions vehicles are measured by.

A declared mark is only worth scoring against if it came from somewhere the
staging did not. The greybox builder derives its seat positions from the box
the location declares -- half the cabin width times a spread fraction -- so a
mark copied back out of it would be the builder's own number, and a clause
comparing the two would read zero on every panel by construction. That is the
no-gradient condition arriving inside the one clause added to escape it.

So the numbers here come from SAE J1100, the interior-dimension standard car
interiors are actually specified in. Two of its codes place a row:

  W5  hip room -- trimmed wall to trimmed wall across one row, which bounds
      how far apart two occupants' hips can be. Published figures run from
      about 1321 mm in a two-seat sports car to 1452 mm in a wide coupe, with
      a mainstream compact sedan near 1379 mm.
  L50 couple distance -- front SgRP to second-row SgRP, the distance between
      rows. A longer cabin gets a boot rather than more legroom, so this is a
      metre figure and not a fraction of the declared box.

Only the lateral coordinate is independent. A row's absolute position along
the cabin is a packaging choice with no single published value, so the caller
passes the longitudinal offset it already uses and this module does not invent
one. A consumer therefore reads the resulting clause as a measure of seat
*spread*, not of where the row sits -- which is the discrepancy worth
measuring anyway, because a spread wider than W5 puts two occupants further
apart than the cabin they are sitting in is wide.

Pure functions, no I/O: the builder, a mark-declaring pass and a gate clause
can all call this and provably agree on what a seat is.
"""
from __future__ import annotations

from typing import Optional, Sequence

from pace_core.setup.marks import Mark

# SAE J1100 W5, hip room across one row, in metres. Keyed by how wide the
# cabin is rather than by model, so a caller with only a declared box can pick
# one: the standard measures the trimmed interior, which is what a declared
# cabin width is trying to describe.
HIP_ROOM_W5_M: dict[str, float] = {
    "narrow": 1.321,    # two-seat sports car
    "compact": 1.379,   # mainstream compact sedan -- the default
    "wide": 1.452,      # wide coupe
}

# SAE J1100 L50, front row to second row. The builder already uses this value
# and arrived at it from cars rather than from a cabin fraction, so a
# declaration that changed it would be asserting a difference that is not
# there.
COUPLE_DISTANCE_L50_M = 0.85

# A seated occupant's mark sits below the floor datum. The schema's own value
# for sitting, so a declaration states what the schema means by it.
SEATED_Z_M = -0.5


def hip_room_for(cabin_width_m: float) -> float:
    """The W5 figure to hold a cabin of this width to.

    A declared box narrower than the narrowest published interior is still
    held to that interior: the standard describes people, who do not get
    narrower because a proxy mesh did.
    """
    if cabin_width_m <= 1.6:
        return HIP_ROOM_W5_M["narrow"]
    if cabin_width_m >= 2.4:
        return HIP_ROOM_W5_M["wide"]
    return HIP_ROOM_W5_M["compact"]


def row_seat_xs(n_seats: int, half_spread_m: float) -> list[float]:
    """Where a row's seats are, screen-left first.

    `n_seats` is how many seats the row HAS, not how many people are in it.
    The two were one argument once, which made a lone occupant of a two-seat
    row come back at the centre -- a position that row has no seat at. Who
    takes which seat of a partly filled row is an assignment, and assignment
    needs something stated about the occupants rather than about the car.

    Screen-left is world +X, so this descends. A single-seat row is centred,
    which is what a 2+1 cabin's rear seat is; the outer seats of a wider row
    sit at the hip-room bound and the rest are spread evenly between them.
    """
    if n_seats <= 0:
        return []
    if n_seats == 1:
        return [0.0]
    if n_seats == 2:
        return [half_spread_m, -half_spread_m]
    if n_seats == 3:
        return [half_spread_m, 0.0, -half_spread_m]
    step = (2 * half_spread_m) / (n_seats - 1)
    return [half_spread_m - step * i for i in range(n_seats)]


def cabin_seat_marks(*, rows: Sequence[int], cabin_width_m: float,
                     front_row_y_m: float,
                     hip_room_m: Optional[float] = None,
                     facing_deg: Optional[float] = None,
                     ) -> list[Mark]:
    """One declared Mark per seat, front row first and screen-left within a row.

    `rows` is how many seats each row has, front to back -- `(2, 1)` is a
    cabin with two in front and one behind them. It describes the car and not
    its occupants, so the same cabin returns the same marks however many
    people are in it. `front_row_y_m` is the caller's own longitudinal offset
    for the front row, for the reason in the module docstring; the rows behind
    it step back by L50.

    Every mark comes back `declared=True`: these are positions a standard
    states, which is what makes them worth comparing a render against.
    """
    half = (hip_room_m if hip_room_m is not None
            else hip_room_for(cabin_width_m)) / 2.0
    out: list[Mark] = []
    for r, n in enumerate(rows):
        y = front_row_y_m - r * COUPLE_DISTANCE_L50_M
        for x in row_seat_xs(n, half):
            out.append(Mark(xy=(x, y), z=SEATED_Z_M,
                            facing_deg=facing_deg, declared=True))
    return out


def cabin_rows(seat_count: int, front_seats: int = 2) -> tuple[int, ...]:
    """How a cabin's seats divide into rows, front to back.

    A property of the car, read off its seat count: three seats are two in
    front and one behind, five are two and three. The front row fills first
    because that is the row a car has before it has any other.
    """
    seats = max(0, int(seat_count))
    if seats <= front_seats:
        return (seats,) if seats else ()
    return (front_seats, seats - front_seats)
