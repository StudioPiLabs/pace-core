"""A mesh fixture's span says where it goes; the mesh says what shape it is.

`fitted_mesh` scaled every axis independently to fill the declared span, which
means a prop whose natural proportions differ from its slot is anisotropically
distorted -- and nothing said so. Measured in-build on the evaluation corpus's cabin,
each mesh yawed as it is actually placed:

    car_console   1.75 x 0.56 x 0.38  ->  1.89 x 0.35 x 0.18    2.29x
    cabin_panels  1.85 x 1.96 x 0.45  ->  0.07 x 0.95 x 0.26   14.60x

cabin_panels is a wall lining squeezed to 3.9% of its own width to reach a
7 cm door-card slot. Whatever that GLB was modelled as, that is not what
renders: dropping the mesh and letting the box fallback take the slot changes
0.70% of the frame, so at that compression the mesh contributes nothing a
primitive was not already contributing.

The fix is not a better universal rule -- uniform fit was tried and shrank the
console to a third of the cabin, which is why per-axis was there in the first
place, and re-tried here under `fit="x"` it inflates the console until it
buries two of the cast to the waist (8.27% of the frame, worse). The choice
belongs to whoever authored the span. What was missing is the reading.

The numbers above are taken inside Blender on purpose. Its glTF importer
applies the Y-up to Z-up conversion, so measuring the same GLB's vertices
outside Blender returns a permuted extent and a wrong anisotropy -- 3.90x for
cabin_panels rather than 14.60x.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node.panel_greybox import (  # noqa: E402
    _FIT_DEFAULT,
    _FIT_POLICIES,
    fit_scale,
)

# The two fixtures above, exactly as the kernel measured them.
CONSOLE_EXT = [1.7523, 0.5589, 0.3823]
CONSOLE_BOX = [1.8860, 0.3507, 0.1800]
PANELS_EXT = [1.8516, 1.9556, 0.4524]
PANELS_BOX = [0.0717, 0.9450, 0.2560]


def test_the_default_is_still_stretch():
    """Every span in the KB was authored against per-axis fit. Changing the
    default would resize every fixture in every project at once, on the
    strength of a policy nobody has looked at the render for."""
    assert _FIT_DEFAULT == "stretch"
    k, _ = fit_scale(CONSOLE_EXT, CONSOLE_BOX)
    assert k == fit_scale(CONSOLE_EXT, CONSOLE_BOX, "stretch")[0]
    for i in range(3):
        assert abs(CONSOLE_EXT[i] * k[i] - CONSOLE_BOX[i]) < 1e-9


def test_anisotropy_is_reported_whatever_policy_is_chosen():
    """It measures the mesh against the box, not the fit against the box. A
    fixture reported at 14x was not modelled for the slot it stands in, and
    that stays true however we decide to seat it."""
    seen = {p: fit_scale(PANELS_EXT, PANELS_BOX, p)[1] for p in _FIT_POLICIES}
    assert len(set(round(v, 6) for v in seen.values())) == 1
    assert 14.5 < seen["stretch"] < 14.7


def test_a_uniform_policy_keeps_the_mesh_proportions():
    for policy in ("uniform", "x", "y", "z"):
        k, _ = fit_scale(PANELS_EXT, PANELS_BOX, policy)
        assert k[0] == k[1] == k[2], f"{policy} must be a single scale"


def test_uniform_fits_inside_the_slot_and_an_axis_policy_need_not():
    """The two non-stretch policies answer different questions. `uniform` is
    'the biggest this mesh can be without leaving its box'. An axis policy is
    'this one declared dimension was the real claim' -- and it will overflow
    the other two, which is the point: the console IS 0.56 m deep."""
    ku, _ = fit_scale(CONSOLE_EXT, CONSOLE_BOX, "uniform")
    for i in range(3):
        assert CONSOLE_EXT[i] * ku[i] <= CONSOLE_BOX[i] + 1e-9

    kx, _ = fit_scale(CONSOLE_EXT, CONSOLE_BOX, "x")
    assert abs(CONSOLE_EXT[0] * kx[0] - CONSOLE_BOX[0]) < 1e-9
    assert CONSOLE_EXT[2] * kx[2] > CONSOLE_BOX[2]


def test_a_zero_extent_axis_does_not_divide_by_zero():
    """A flat mesh -- a plane, a degenerate import -- has one extent of 0."""
    k, anis = fit_scale([1.0, 0.0, 1.0], [2.0, 2.0, 2.0])
    assert k[1] == 1.0
    assert anis == 2.0


def test_an_unknown_policy_resolves_the_same_way_in_both_places():
    """A typo in the KB costs the fixture its policy, not the panel its
    control geometry -- the same rule `fitted_mesh` applies to a mesh that
    will not import. `_panel_fixtures` validates and substitutes the default;
    `fit_scale` must substitute the SAME default, or a fixture is reported
    under one policy and built under another."""
    assert (fit_scale(CONSOLE_EXT, CONSOLE_BOX, "widthwise")
            == fit_scale(CONSOLE_EXT, CONSOLE_BOX, _FIT_DEFAULT))
