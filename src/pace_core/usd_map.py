"""Where each PACE field lands in OpenUSD — the single source of truth.

Two consumers read this table and they must not drift:

  * the correspondence was measured by a script since removed, which printed the coverage the
    paper reports;
  * `pace_core.usd_export` writes it, deciding for every field whether it
    becomes a real USD attribute with real semantics or a namespaced `pace:`
    custom attribute that only a PACE-aware consumer can read.

Keeping them in one place means a field that gains a USD target starts being
exported and starts counting toward the published number in the same commit.
A table that claims coverage the exporter does not deliver is exactly the kind
of unverifiable claim this project keeps finding in its own pipeline.

Three verdicts, and the burden of proof runs toward "custom":

  native   a typed USD attribute carries the same quantity with the same
           meaning, so any USD consumer understands the value
  derived  a USD attribute carries it after a documented conversion
  custom   USD can store the value but assigns it no meaning; every consumer
           must already know PACE to read it
"""
from __future__ import annotations

# PACE field -> (verdict, USD target, note). Keyed "Schema.field".
# Anything absent defaults to custom, so silence never reads as coverage.
MAPPING: dict[str, tuple[str, str, str]] = {
    # ── Camera: the one pillar USD genuinely models ──────────────────────
    "CameraIntrinsics.focal_length_mm":  ("native",  "Camera.focalLength", ""),
    "CameraIntrinsics.aperture_f":       ("native",  "Camera.fStop", ""),
    "CameraIntrinsics.sensor_mm":        ("native",  "Camera.horizontalAperture", ""),
    "CameraIntrinsics.focus_distance_m": ("native",  "Camera.focusDistance", ""),
    "CameraIntrinsics.iso_value":        ("native",  "Camera.exposure:iso", ""),
    "CameraIntrinsics.shutter_angle_deg": ("derived", "Camera.shutter:open",
                                           "angle converts to open/close times given fps"),
    "CameraExtrinsics.position":         ("native",  "Camera.xformOpOrder",
                                           "authored as xformOp:translate"),
    "CameraTrajectory.camera_path":      ("native",  "Camera.xformOpOrder",
                                           "6-DoF keyframes are time-sampled xformOps"),
    "CameraCreativeIntent.aspect_ratio": ("derived", "Camera.verticalAperture",
                                           "ratio of horizontal to vertical aperture"),
    "FrameRate.num":                     ("derived", "stage.framesPerSecond",
                                           "LOSSY: USD stores a float, so 24000/1001 cannot be exact"),
    "FrameRate.denom":                   ("derived", "stage.framesPerSecond",
                                           "LOSSY: same float"),
    # ── Lighting: partial ────────────────────────────────────────────────
    "Lighting.color_temp_k":             ("native",  "RectLight.inputs:colorTemperature", ""),
    "Lighting.color_gels":               ("derived", "RectLight.inputs:color",
                                           "a gel is a colour multiplier"),
    # ── Placement: the metric layout is real geometry ────────────────────
    "WorldEntity.world_xy":              ("native",  "Xformable.xformOpOrder", "xformOp:translate"),
    "WorldEntity.z":                     ("native",  "Xformable.xformOpOrder", "xformOp:translate"),
    "WorldEntity.facing_deg":            ("native",  "Xformable.xformOpOrder", "xformOp:rotateXYZ"),
    "WorldEntity.scale":                 ("native",  "Xformable.xformOpOrder", "xformOp:scale"),
    "CameraSetup.world_xy":              ("native",  "Camera.xformOpOrder", ""),
    "CameraSetup.z":                     ("native",  "Camera.xformOpOrder", ""),
    "CameraSetup.looking_at_xy":         ("derived", "Camera.xformOpOrder",
                                           "an aim point becomes a rotation"),
    "CameraSetup.lens_mm":               ("native",  "Camera.focalLength", ""),
    "PhysicalLayout.frame_of_reference": ("derived", "stage.upAxis",
                                           "stage metadata: upAxis + metersPerUnit"),
    # ── Props / subjects: identity and material, not intent ──────────────
    "Prop.prop_id":                      ("native",  "prim.path", "a prim path is an identifier"),
    "Prop.material":                     ("native",  "Material.outputs:surface",
                                           "UsdShade material binding"),
    "Prop.color":                        ("derived", "Material.outputs:surface", "a shader input"),
    "Prop.size":                         ("derived", "Xformable.xformOpOrder", "xformOp:scale"),
    "Subject.character_id":              ("native",  "prim.path", "a prim path is an identifier"),
}

# Which USD prim type to check a target against.
SCHEMA_FOR = {"Camera": "Camera", "RectLight": "RectLight", "DistantLight": "DistantLight",
              "Xformable": "Xform", "Material": "Material"}
# Targets that are not prim attributes and so cannot be introspected this way.
NON_ATTRIBUTE = {"prim.path", "stage.framesPerSecond", "stage.upAxis"}

def verdict_for(schema: str, field: str) -> tuple[str, str, str]:
    """(verdict, usd_target, note). Absent defaults to custom, so silence
    never reads as coverage."""
    return MAPPING.get(f"{schema}.{field}", ("custom", "", ""))
