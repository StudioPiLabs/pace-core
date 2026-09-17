"""The USD export must agree with the render, and must not overclaim.

Two failure modes are worth a test. The first is drift: an exported camera that
is *nearly* the rendered one is worse than no export, because it looks
authoritative. The first version of the exporter solved from the raw shot,
dropped the panel's composition target, and was wrong by about five degrees
while looking entirely plausible.

The second is overclaiming: writing PACE's own vocabulary into USD as though a
USD consumer could act on it. Only fields the shared table marks native or
derived may become real USD attributes; everything else belongs under `pace:`,
where its opacity is explicit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pxr", reason="usd-core not installed: uv sync --extra usd")

from pace_core.usd_export import export_scene, UNITS_PER_METRE  # noqa: E402
from pace_core.usd_map import verdict_for  # noqa: E402

SCENE = {
    "scene_id": "scene_t",
    "narrative_meta": {"characters_present": ["fay"]},
    "shots": [{
        "shot_id": "shot_01",
        "camera": {
            "intrinsics": {"lens_mm": 35, "sensor_width_mm": 36, "fov_class": "wide"},
            "extrinsics": {"angle": "eye_level", "position": "three_quarter"},
            "creative_intent": {"shot_size": "wide", "framing": "crowd"},
        },
        "setup": {
            "space": {"scale_meters": [3.2, 2.1, 1.6]},
            "subjects": [{"character_id": "fay", "pose": "seated",
                          "screen_position": {"zone": "center", "x": 0.5, "y": 0.52}}],
            "props": [{"prop_id": "game_device"}],
        },
        "panels": [{"panel_id": "p1"}],
    }],
}


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    from pxr import Usd
    out = tmp_path_factory.mktemp("usd") / "scene_t.usda"
    export_scene(SCENE, out, scene_id="scene_t")
    return Usd.Stage.Open(str(out))


def test_stage_conventions_are_declared(stage):
    """A stage without upAxis/metersPerUnit is a stage every consumer guesses
    about, and the guesses differ."""
    from pxr import UsdGeom
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(0.01)
    assert stage.GetDefaultPrim().GetName() == "World"


def test_declared_lens_reaches_usd_in_millimetres(stage):
    """USD defines focalLength in tenths of a world unit, so the centimetre
    world is what makes 35 mm read as 35."""
    from pxr import UsdGeom
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/shot_01"))
    assert cam.GetFocalLengthAttr().Get() == pytest.approx(35.0)
    assert cam.GetHorizontalApertureAttr().Get() == pytest.approx(36.0)


def test_exported_camera_equals_the_planner(stage):
    """The whole point of the export: this file must describe the camera that
    actually renders."""
    from pxr import UsdGeom
    from pace_core.camera.camera_planner import plan_camera
    from pace_core.node.blender_box import _flatten_panels, _synthesize_shot

    entry = _flatten_panels(SCENE)[0]
    plan = plan_camera(_synthesize_shot(entry, "scene_t", SCENE["narrative_meta"]))
    ops = {o.GetOpName(): o.Get() for o in UsdGeom.Xformable(
        UsdGeom.Camera(stage.GetPrimAtPath("/World/Cameras/shot_01"))).GetOrderedXformOps()}
    for i in range(3):
        assert ops["xformOp:translate"][i] == pytest.approx(
            plan["position"][i] * UNITS_PER_METRE)
        assert ops["xformOp:rotateXYZ"][i] == pytest.approx(
            plan["rotation_deg"][i], abs=1e-4)


def test_non_native_fields_stay_in_the_pace_namespace(stage):
    """Intent is preserved but never disguised as interchange."""
    prim = stage.GetPrimAtPath("/World/Cameras/shot_01")
    assert prim.GetAttribute("pace:CameraCreativeIntent:shot_size").Get() == "wide"
    assert prim.GetAttribute("pace:CameraExtrinsics:angle").Get() == "eye_level"
    # The invariant, stated directly: everything we add that USD assigns no
    # meaning to is a custom attribute, and every custom attribute lives under
    # `pace:`. Schema attributes (exposure:fStop, shutter:open) are USD's own
    # and are not custom -- an earlier version of this test parsed their
    # namespace as a PACE schema name and failed on them.
    for attr in prim.GetAttributes():
        if attr.IsCustom():
            assert attr.GetName().startswith("pace:"), (
                f"{attr.GetName()} is custom but outside the pace: namespace")
    # And the fields the table calls native really did become schema attributes.
    assert verdict_for("CameraIntrinsics", "focal_length_mm")[0] == "native"
    assert not prim.GetAttribute("focalLength").IsCustom()


def test_subjects_carry_declaration_but_no_invented_transform(stage):
    """PACE states a *screen* position. Writing a world transform here would put
    a number into an interchange file that no PACE field authorises."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath("/World/Subjects/shot_01_fay")
    assert prim.IsValid()
    assert prim.GetAttribute("pace:Subject:pose").Get() == "seated"
    assert json.loads(prim.GetAttribute("pace:ScreenPosition:target").Get())["x"] == 0.5
    assert not UsdGeom.Xformable(prim).GetOrderedXformOps()
