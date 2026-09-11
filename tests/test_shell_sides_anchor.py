"""A fixture that lines the side walls is placed once per wall.

The anchor vocabulary had no side. `shell_center`'s own comment says it is for
"a fixture that surrounds the space rather than standing at one end of it --
screen panels lining the walls around the seats are not at the front or the
back, they are around", but it places ONE object at the middle of the cabin.
For a wraparound sized like a wall that is not a wraparound, it is an occluder:
staged that way, `cabin_panels` sat between the lens and the cast and buried
both heads — the failure `_fixtures_for`'s docstring warns about.

`shell_sides` places two instances instead, mirrored in x and each turned to
face inward, the same shape as `per_seat` placing one per seat.

The span convention is the part worth pinning. `fitted_mesh` applies the yaw
BEFORE it measures the bounding box and scales it to `span`, so a span is read
in world axes after rotation — a side panel is written thin in x, long in y,
tall in z, and the yaw only decides which face points into the cabin. Written
in the mesh's own axes instead, every side fixture would come out rotated.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node import panel_greybox  # noqa: E402
from pace_core.node.panel_greybox import _FIXTURE_ANCHORS  # noqa: E402


def test_the_side_anchor_is_in_the_accepted_vocabulary():
    """_fixtures_for drops any placement whose anchor it does not know, so an
    unregistered anchor means the prop is silently never staged."""
    assert "shell_sides" in _FIXTURE_ANCHORS


def test_a_side_fixture_is_placed_once_per_wall():
    src = inspect.getsource(panel_greybox)
    block = src.split('if anchor == "shell_sides":', 1)[1].split("continue", 1)[0]
    # two instances, mirrored, each yawed to face inward
    assert "(-1.0, 1.0)" in block
    assert "90.0 * sign" in block
    assert "W / 2 - size[0] / 2" in block, "a side panel sits against the wall"


def test_the_span_contract_is_world_axes_after_rotation():
    """If fitted_mesh ever scales before it rotates, every shell_sides span in
    the KB silently means something else."""
    src = inspect.getsource(panel_greybox)
    body = src.split("def fitted_mesh", 1)[1].split("def fixture", 1)[0]
    yaw_at = body.index("rotation_euler")
    scale_at = body.index("o.scale = (")
    assert yaw_at < scale_at, (
        "fitted_mesh must yaw before it scales — shell_sides spans are "
        "authored in world axes on that basis")
