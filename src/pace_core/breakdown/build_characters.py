"""
build_characters.py — Procedural humanoid generator for Blender greybox.

Builds anatomically-correct stand-in humans from primitives (head/torso/limbs/hair).
Replaces the box mannequins in build_scenes.py — gives the Layout Solver real human
proportions for ControlNet depth maps, OpenPose-style skeletons, and screen-space
projection in the alignment engine.

Usage from another Blender python script:
    from pailang.core.breakdown.build_characters import build_humanoid
    build_humanoid("Subject_A", character_id="jia",  origin=(0, 0.5, 0))
    build_humanoid("Subject_B", character_id="wei",  origin=(0, -0.5, 0))

What it produces (per character):
    A parent Empty named <subject_name> with these child meshes:
        head, hair, neck, torso, pelvis,
        arm_upper_l/r, arm_lower_l/r, hand_l/r,
        leg_upper_l/r, leg_lower_l/r, foot_l/r
    All children share a common pass_index so segmentation passes can isolate
    the whole character with a single ID Mask node.

What it does NOT do (left for later):
    - rigging / armature (only static mesh)
    - facial features (head is a smooth sphere)
    - cloth simulation (clothing is just a torso color)
    - hair physics (hair is a static cap mesh)
"""

import bpy, math

# ── Per-character proportions (meters) ───────────────────────────────────────
#
# Defaults derived from kb/on_scene/characters.json anchor strings:
#   jia: 28yo slim asian woman, shoulder-length black hair
#   wei: 30yo asian man, short side-parted black hair
# Adjust here when adding new characters; values are intentionally simple.

CHARACTER_PROPORTIONS = {
    "jia": {
        "height_m":         1.62,   # 5'4"
        "shoulder_width_m": 0.36,
        "hip_width_m":      0.34,
        "head_radius_m":    0.10,
        "torso_color":      (0.72, 0.65, 0.55, 1.0),   # beige sweater
        "hair_color":       (0.05, 0.04, 0.03, 1.0),   # near-black
        "skin_color":       (0.95, 0.83, 0.72, 1.0),
        "hair_style":       "shoulder_length",
    },
    "wei": {
        "height_m":         1.78,   # 5'10"
        "shoulder_width_m": 0.46,
        "hip_width_m":      0.36,
        "head_radius_m":    0.11,
        "torso_color":      (0.18, 0.18, 0.20, 1.0),   # charcoal trench
        "hair_color":       (0.05, 0.04, 0.03, 1.0),
        "skin_color":       (0.92, 0.78, 0.65, 1.0),
        "hair_style":       "short_side_part",
    },
    "_default_": {
        "height_m":         1.70,
        "shoulder_width_m": 0.40,
        "hip_width_m":      0.34,
        "head_radius_m":    0.10,
        "torso_color":      (0.40, 0.40, 0.45, 1.0),
        "hair_color":       (0.10, 0.08, 0.06, 1.0),
        "skin_color":       (0.90, 0.80, 0.70, 1.0),
        "hair_style":       "neutral",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_material(name: str, rgba: tuple) -> bpy.types.Material:
    """Get or create a material with a flat diffuse color. Reused across calls
    so we don't bloat the scene with duplicate materials per body part."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
    return mat


def _add_mesh(prim: str, name: str, location, scale, parent=None,
              rotation=(0, 0, 0), material=None, pass_index=0):
    """Add a primitive mesh (sphere/cube/cylinder), set transform, attach material."""
    if prim == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(location=location)
    elif prim == "cube":
        bpy.ops.mesh.primitive_cube_add(location=location)
    elif prim == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(location=location, vertices=16)
    else:
        raise ValueError(f"unknown primitive {prim}")
    obj = bpy.context.active_object
    obj.name           = name
    obj.scale          = scale
    obj.rotation_euler = rotation
    obj.display_type   = "SOLID"
    obj.pass_index     = pass_index
    if material:
        obj.data.materials.append(material)
    if parent:
        # IMPORTANT: setting obj.parent without also clearing
        # matrix_parent_inverse causes Blender to apply the parent transform
        # ON TOP of the child's existing world coords — so when the parent
        # later moves, children "double-shift". Setting parent_inverse to the
        # inverse of the parent's current world matrix keeps the child where
        # it currently is in world space and makes its local transform exactly
        # what we just set, so subsequent parent.location changes work as
        # expected: parent moves N → child moves N (no doubling).
        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


# ── Public entry ──────────────────────────────────────────────────────────────

def build_humanoid(subject_name: str, character_id: str, origin=(0.0, 0.0, 0.0),
                   facing_camera: bool = True, pass_index: int = 1) -> bpy.types.Object:
    """Build a parameterized humanoid at `origin` (world coords, feet on ground).

    Returns the parent Empty so callers can re-position / hide / rotate the
    whole character with one transform.

    `pass_index` is set on every body-part mesh — the renderer then uses an
    ID Mask compositor node to isolate this character in the segmentation pass
    (different characters get different pass indices).
    """
    p = CHARACTER_PROPORTIONS.get(character_id, CHARACTER_PROPORTIONS["_default_"])

    h            = p["height_m"]
    head_r       = p["head_radius_m"]
    shoulder_w   = p["shoulder_width_m"]
    hip_w        = p["hip_width_m"]
    torso_h      = h * 0.32                       # ~ chest+belly height
    pelvis_h     = h * 0.10
    leg_h        = h * 0.50                       # full leg incl thigh+shin
    upper_leg_h  = leg_h * 0.50
    lower_leg_h  = leg_h * 0.45
    arm_h        = h * 0.42
    upper_arm_h  = arm_h * 0.48
    lower_arm_h  = arm_h * 0.42

    # Vertical anchors: feet at z=0, ground them
    z_foot       = 0.0
    z_lower_leg  = z_foot + lower_leg_h * 0.5
    z_upper_leg  = z_foot + lower_leg_h + upper_leg_h * 0.5
    z_pelvis     = z_foot + leg_h + pelvis_h * 0.5
    z_torso      = z_pelvis + pelvis_h * 0.5 + torso_h * 0.5
    z_neck       = z_torso + torso_h * 0.5 + 0.04
    z_head       = z_neck + 0.04 + head_r

    # Materials (one per channel — body parts share)
    mat_skin   = _make_material(f"{character_id}_skin",   p["skin_color"])
    mat_torso  = _make_material(f"{character_id}_torso",  p["torso_color"])
    mat_hair   = _make_material(f"{character_id}_hair",   p["hair_color"])
    mat_pants  = _make_material(f"{character_id}_pants",
                                (p["torso_color"][0]*0.4,
                                 p["torso_color"][1]*0.4,
                                 p["torso_color"][2]*0.5, 1.0))
    mat_shoes  = _make_material(f"{character_id}_shoes", (0.05, 0.05, 0.05, 1.0))

    # Parent empty — the single transform handle for the whole character
    bpy.ops.object.empty_add(type="PLAIN_AXES",
                              location=(origin[0], origin[1], origin[2]))
    parent             = bpy.context.active_object
    parent.name        = subject_name
    if facing_camera:
        # Camera looks down -X by convention, so character should face +X
        parent.rotation_euler = (0, 0, math.radians(0))
    else:
        parent.rotation_euler = (0, 0, math.radians(180))   # back to camera

    common = dict(parent=parent, pass_index=pass_index)

    # ── Head + hair ──────────────────────────────────────────────────────────
    _add_mesh("sphere", f"{subject_name}_head",
              location=(origin[0], origin[1], z_head),
              scale=(head_r, head_r * 0.85, head_r * 1.05),
              material=mat_skin, **common)

    # Hair as a slightly larger half-sphere on top of the head
    hair_z_off = 0.02 if p["hair_style"] == "short_side_part" else 0.0
    hair_scale_z = {
        "short_side_part":   head_r * 0.6,
        "shoulder_length":   head_r * 1.6,
        "neutral":           head_r * 0.9,
    }.get(p["hair_style"], head_r * 0.9)
    _add_mesh("sphere", f"{subject_name}_hair",
              location=(origin[0] - 0.01, origin[1], z_head + hair_z_off),
              scale=(head_r * 1.05, head_r * 0.95, hair_scale_z),
              material=mat_hair, **common)

    # ── Neck ─────────────────────────────────────────────────────────────────
    _add_mesh("cylinder", f"{subject_name}_neck",
              location=(origin[0], origin[1], z_neck),
              scale=(0.045, 0.045, 0.04),
              material=mat_skin, **common)

    # ── Torso ────────────────────────────────────────────────────────────────
    _add_mesh("cube", f"{subject_name}_torso",
              location=(origin[0], origin[1], z_torso),
              scale=(0.10, shoulder_w * 0.5, torso_h * 0.5),
              material=mat_torso, **common)

    # ── Pelvis ───────────────────────────────────────────────────────────────
    _add_mesh("cube", f"{subject_name}_pelvis",
              location=(origin[0], origin[1], z_pelvis),
              scale=(0.10, hip_w * 0.5, pelvis_h * 0.5),
              material=mat_pants, **common)

    # ── Arms (left = +Y, right = -Y, both hanging straight) ─────────────────
    arm_y_off = shoulder_w * 0.5 + 0.02
    z_upper_arm = z_torso + torso_h * 0.25
    z_lower_arm = z_upper_arm - upper_arm_h * 0.5 - lower_arm_h * 0.5
    z_hand      = z_lower_arm - lower_arm_h * 0.5 - 0.04
    for side, sign in (("l", 1), ("r", -1)):
        _add_mesh("cylinder", f"{subject_name}_arm_upper_{side}",
                  location=(origin[0], origin[1] + sign * arm_y_off, z_upper_arm),
                  scale=(0.045, 0.045, upper_arm_h * 0.5),
                  material=mat_torso, **common)
        _add_mesh("cylinder", f"{subject_name}_arm_lower_{side}",
                  location=(origin[0], origin[1] + sign * arm_y_off, z_lower_arm),
                  scale=(0.040, 0.040, lower_arm_h * 0.5),
                  material=mat_skin, **common)
        _add_mesh("sphere", f"{subject_name}_hand_{side}",
                  location=(origin[0], origin[1] + sign * arm_y_off, z_hand),
                  scale=(0.045, 0.035, 0.05),
                  material=mat_skin, **common)

    # ── Legs ─────────────────────────────────────────────────────────────────
    leg_y_off = hip_w * 0.5 - 0.05
    for side, sign in (("l", 1), ("r", -1)):
        _add_mesh("cylinder", f"{subject_name}_leg_upper_{side}",
                  location=(origin[0], origin[1] + sign * leg_y_off, z_upper_leg),
                  scale=(0.06, 0.06, upper_leg_h * 0.5),
                  material=mat_pants, **common)
        _add_mesh("cylinder", f"{subject_name}_leg_lower_{side}",
                  location=(origin[0], origin[1] + sign * leg_y_off, z_lower_leg),
                  scale=(0.05, 0.05, lower_leg_h * 0.5),
                  material=mat_pants, **common)
        _add_mesh("cube", f"{subject_name}_foot_{side}",
                  location=(origin[0] + 0.05, origin[1] + sign * leg_y_off, z_foot + 0.025),
                  scale=(0.10, 0.04, 0.025),
                  material=mat_shoes, **common)

    return parent


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run from inside Blender:
    #   blender --background --python build_characters.py
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Floor for context
    bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.scale = (3, 3, 1)
    floor.name  = "Floor"

    # Two characters side-by-side
    build_humanoid("Subject_A", "jia", origin=(0,  0.5, 0), pass_index=1)
    build_humanoid("Subject_B", "wei", origin=(0, -0.5, 0), pass_index=2)

    # Camera in front
    bpy.ops.object.camera_add(location=(3, 0, 1.4))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.rotation_euler = (math.radians(85), 0, math.radians(90))
    bpy.context.scene.camera = cam

    bpy.ops.wm.save_as_mainfile(filepath="/tmp/characters_demo.blend")
    print("[build_characters] saved /tmp/characters_demo.blend")
