"""Lower a PACE scene document into an OpenUSD stage.

PACE and USD are not rivals and the paper should stop reading as though they
were. PACE is an authoring layer: it states what a shot must achieve
(`screen_position`, `shot_size`, `angle`) and leaves the camera transform to be
solved. USD is an interchange layer: it carries the transform once solved.
They sit on opposite sides of the solve, which is why USD natively covers 27 of
252 PACE fields and none of the 29 Events fields -- it describes a stage, not a
performance.

So this module *compiles*: it runs the production camera planner and writes the
solved result as real USD, alongside the declaration that produced it.

Two tiers, and the split is the point:

  * Fields the shared table (`usd_map.MAPPING`) marks native or derived become
    real USD attributes -- `focalLength`, `xformOp:translate`, `framesPerSecond`
    -- with meaning any USD consumer already understands.
  * Everything else is written under a `pace:` namespace. Nothing is lost, and
    nothing masquerades as interchange: a DCC opening this file gets exactly the
    27 fields it can act on, and can see the rest is payload it must know PACE
    to read.

Units. `metersPerUnit = 0.01`, so the world unit is the centimetre and
translations are scaled by 100. This is not decoration. USD defines
`focalLength` and the apertures in *tenths of a world unit*, so a centimetre
world makes them millimetres -- the unit every lens is actually quoted in. In a
metre world a 35 mm lens would have to be written `0.35`, which is correct and
which somebody downstream will eventually read as 0.35 mm. `upAxis = "Z"`,
matching the Blender scenes these coordinates come from.

`pxr` is imported lazily so this module can be imported (and the mapping
inspected) without usd-core installed:  `uv sync --extra usd`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from pace_core.usd_map import verdict_for

# USD's camera attributes are in tenths of a world unit; a centimetre world
# therefore expresses them in millimetres. Everything positional is metres in
# PACE, so it is multiplied by this on the way out.
UNITS_PER_METRE = 100.0
METERS_PER_UNIT = 0.01


def _san(name: str) -> str:
    """A USD prim name must be a valid identifier."""
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))
    return out if out and not out[0].isdigit() else f"_{out}"


def _set_custom(prim, schema: str, field: str, value: Any) -> bool:
    """Write a PACE field USD assigns no meaning to, under the `pace:`
    namespace. Returns False for values USD has no scalar type for -- those are
    serialised as their JSON text rather than silently dropped, because a field
    that vanishes at export is indistinguishable from one that was never
    declared."""
    from pxr import Sdf
    import json as _json

    attr_name = f"pace:{schema}:{field}"
    if isinstance(value, bool):
        t = Sdf.ValueTypeNames.Bool
    elif isinstance(value, int):
        t = Sdf.ValueTypeNames.Int
    elif isinstance(value, float):
        t = Sdf.ValueTypeNames.Double
    elif isinstance(value, str):
        t = Sdf.ValueTypeNames.String
    else:
        value, t = _json.dumps(value, ensure_ascii=False), Sdf.ValueTypeNames.String
    prim.CreateAttribute(attr_name, t, custom=True).Set(value)
    return True


def _write_declared(prim, schema: str, payload: dict) -> dict:
    """Write every declared field of one PACE schema onto a prim, splitting on
    the shared table's verdict. Returns a per-verdict count so the caller can
    report what actually crossed the boundary."""
    counts = {"native": 0, "derived": 0, "custom": 0}
    for field, value in (payload or {}).items():
        if value is None:
            continue
        verdict, _target, _note = verdict_for(schema, field)
        # native/derived fields are written by the caller as real USD
        # attributes; recording them here would duplicate the value under two
        # names and let the copies disagree.
        if verdict == "custom":
            _set_custom(prim, schema, field, value)
        counts[verdict] += 1
    return counts


def export_scene(scene_doc: dict, out_path: Path, *,
                 scene_id: Optional[str] = None,
                 plan: bool = True) -> dict:
    """Write `scene_doc` as a USD stage at `out_path`. Returns a summary."""
    from pxr import Usd, UsdGeom, Sdf, Gf

    scene_id = scene_id or scene_doc.get("scene_id") or "scene"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Built in memory and exported at the end, so a failure part-way through
    # leaves no half-written stage behind for someone to open and trust.
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)

    # FrameRate is rational in PACE (num/denom) and a double in USD, so
    # 24000/1001 cannot round-trip exactly. Write the float and keep the exact
    # pair as payload rather than pretend the conversion was lossless.
    fr = scene_doc.get("frame_rate") or (scene_doc.get("shot_defaults") or {}).get("frame_rate")
    num, den = ((fr or {}).get("num"), (fr or {}).get("denom")) if isinstance(fr, dict) else (None, None)
    fps = float(num) / float(den) if num and den else 24.0
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    root = world.GetPrim()
    root.CreateAttribute("pace:scene_id", Sdf.ValueTypeNames.String, custom=True).Set(str(scene_id))
    if num and den:
        root.CreateAttribute("pace:FrameRate:num", Sdf.ValueTypeNames.Int, custom=True).Set(int(num))
        root.CreateAttribute("pace:FrameRate:denom", Sdf.ValueTypeNames.Int, custom=True).Set(int(den))

    totals = {"native": 0, "derived": 0, "custom": 0}
    n_cam = n_sub = n_prop = 0
    # The exported camera must be the camera that renders, or this file is
    # decoration. That means using the same panel synthesis the renderer uses:
    # solving from the raw shot alone drops the panel's composition target and
    # yields a default level aim, which looked plausible and was wrong by ~5
    # degrees on the first shot tried.
    #
    # _flatten_panels / _synthesize_shot are pure dict logic that happen to live
    # in the Blender module; they import without bpy. They belong in core beside
    # the planner, and moving them is a separate change.
    planner = None
    if plan:
        try:
            from pace_core.camera.camera_planner import plan_camera
            from pace_core.node.blender_box import _flatten_panels, _synthesize_shot
            planner = (plan_camera, _flatten_panels, _synthesize_shot)
        except Exception:
            planner = None

    # panel entries keyed by shot, so each camera is solved for the panel that
    # actually states a composition target
    panels_by_shot: dict[str, list] = {}
    if planner:
        _pc, _flatten, _synth = planner
        for entry in _flatten(scene_doc):
            panels_by_shot.setdefault(str(entry.get("shot_id")), []).append(entry)

    for shot in scene_doc.get("shots") or []:
        sid = _san(shot.get("shot_id") or f"shot_{n_cam+1}")
        setup = shot.get("setup") or {}
        cam_doc = shot.get("camera") or {}
        intr = cam_doc.get("intrinsics") or {}

        cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{sid}")
        p = cam.GetPrim()

        # ── native: the physical camera ──────────────────────────────────
        lens = intr.get("focal_length_mm") or intr.get("lens_mm")
        if lens:
            cam.CreateFocalLengthAttr(float(lens))
            totals["native"] += 1
        sensor = intr.get("sensor_mm") or intr.get("sensor_width_mm")
        if sensor:
            cam.CreateHorizontalApertureAttr(float(sensor))
            totals["native"] += 1
            ar = (cam_doc.get("creative_intent") or {}).get("aspect_ratio")
            if isinstance(ar, (int, float)) and ar:
                cam.CreateVerticalApertureAttr(float(sensor) / float(ar))
                totals["derived"] += 1
        for key, setter, tier in (("aperture_f", cam.CreateFStopAttr, "native"),
                                  ("focus_distance_m", cam.CreateFocusDistanceAttr, "native"),
                                  ("iso_value", None, "native")):
            v = intr.get(key)
            if v is None:
                continue
            if setter is not None:
                setter(float(v) * (UNITS_PER_METRE if key == "focus_distance_m" else 1.0))
            else:
                p.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float, custom=False).Set(float(v))
            totals[tier] += 1
        # Shutter angle is degrees in PACE and open/close *times* in USD.
        ang = intr.get("shutter_angle_deg")
        if ang:
            half = (float(ang) / 360.0) / fps / 2.0
            p.CreateAttribute("shutter:open", Sdf.ValueTypeNames.Double, custom=False).Set(-half)
            p.CreateAttribute("shutter:close", Sdf.ValueTypeNames.Double, custom=False).Set(half)
            totals["derived"] += 1

        # ── derived: the solved transform ────────────────────────────────
        if planner:
            try:
                pc, _fl, synth = planner
                entries = panels_by_shot.get(str(shot.get("shot_id"))) or []
                if not entries:
                    raise ValueError("no panel carries a composition target for this shot")
                plan_out = pc(synth(entries[0], str(scene_id),
                                    scene_doc.get("narrative_meta") or {}))
                pos = [float(v) * UNITS_PER_METRE for v in plan_out["position"]]
                rot = [float(v) for v in plan_out["rotation_deg"]]
                xf = UsdGeom.Xformable(cam)
                xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
                if not lens and plan_out.get("lens_mm"):
                    cam.CreateFocalLengthAttr(float(plan_out["lens_mm"]))
                totals["derived"] += 1
                p.CreateAttribute("pace:solved_by", Sdf.ValueTypeNames.String,
                                  custom=True).Set("camera_planner.plan_camera via _synthesize_shot(panel)")
            except Exception as exc:   # a planner failure must not lose the shot
                p.CreateAttribute("pace:solve_error", Sdf.ValueTypeNames.String,
                                  custom=True).Set(str(exc)[:400])

        # ── custom: the intent that produced it ──────────────────────────
        for schema, payload in (("CameraIntrinsics", intr),
                                ("CameraExtrinsics", cam_doc.get("extrinsics")),
                                ("CameraTrajectory", cam_doc.get("trajectory")),
                                ("CameraCreativeIntent", cam_doc.get("creative_intent"))):
            c = _write_declared(p, schema, payload or {})
            totals["custom"] += c["custom"]
        n_cam += 1

        # ── subjects: declaration only ───────────────────────────────────
        # No world transform is written. PACE states a subject's *screen*
        # position; its world placement is produced downstream by the
        # assembler, and inventing one here would put a number in an
        # interchange file that no PACE field authorises.
        for sub in setup.get("subjects") or []:
            cid = _san(sub.get("character_id") or f"subject_{n_sub+1}")
            sp = UsdGeom.Xform.Define(stage, f"/World/Subjects/{sid}_{cid}").GetPrim()
            c = _write_declared(sp, "Subject", sub)
            totals["custom"] += c["custom"]
            if isinstance(sub.get("screen_position"), dict):
                _set_custom(sp, "ScreenPosition", "target", sub["screen_position"])
            n_sub += 1

        for prop in setup.get("props") or []:
            pid = _san(prop.get("prop_id") or f"prop_{n_prop+1}")
            pp = UsdGeom.Xform.Define(stage, f"/World/Props/{sid}_{pid}").GetPrim()
            totals["custom"] += _write_declared(pp, "Prop", prop)["custom"]
            n_prop += 1

        # Declared interior extent. camera_planner does not read this today,
        # which is why it places cameras outside closed sets; exporting it puts
        # the constraint where a DCC can at least see it.
        space = setup.get("space") or {}
        if isinstance(space.get("scale_meters"), list) and len(space["scale_meters"]) == 3:
            ext = [float(v) * UNITS_PER_METRE for v in space["scale_meters"]]
            p.CreateAttribute("pace:SceneSpace:scale_units", Sdf.ValueTypeNames.Double3,
                              custom=True).Set(Gf.Vec3d(*ext))

    stage.GetRootLayer().Export(str(out_path))
    return {"out": str(out_path), "cameras": n_cam, "subjects": n_sub,
            "props": n_prop, "fps": fps, "attributes": totals,
            "meters_per_unit": METERS_PER_UNIT, "up_axis": "Z"}
