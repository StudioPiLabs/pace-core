"""
import_3d_assets.py — Import .glb / .gltf / .obj 3D meshes into Blender scenes.

Purpose: bridge between AI-generated 3D meshes (from Hunyuan3D-2, TripoSR,
Trellis, Meshy, etc., all of which output .glb) and our Blender greybox
pipeline. Called from build_scenes.py to drop AI-generated props into a
location instead of the current cube/cylinder primitives.

Why a helper module:
  - Hunyuan3D outputs in arbitrary scale/orientation; we have to normalize
    (height, rotation, pivot) before placing in a scene.
  - Imported .glb meshes come with their own materials and pass_index=0,
    breaking our segmentation passes. We have to re-tag pass_index per
    asset and optionally swap materials.
  - Want a single signature `import_3d_asset()` that build_scenes.py can
    call exactly like add_cube/add_cylinder.

Usage from inside Blender python:
    from pailang.core.breakdown.import_3d_assets import import_3d_asset
    obj = import_3d_asset(
        name="LampPost",
        glb_path="kb/3d_models/lamp_post.glb",
        location=(0, 0, 0),
        scale_to_height_m=5.0,
        pass_index=0,
    )

If the .glb file is missing: returns None and logs a warning. Caller decides
whether to fall back to a primitive. This keeps build_scenes.py's logic
unchanged while letting us incrementally replace primitives with AI-generated
meshes as the kb/3d_models/ library grows.
"""

import bpy
import mathutils
from pathlib import Path


def import_3d_asset(
    name: str,
    glb_path: str | Path,
    location: tuple = (0, 0, 0),
    scale_to_height_m: float | None = None,
    rotation: tuple = (0, 0, 0),
    pass_index: int = 0,
    parent: bpy.types.Object | None = None,
    fallback_factory=None,
) -> bpy.types.Object | None:
    """Import a .glb file as a single Blender object.

    Args:
        name:                The name to assign to the imported object.
        glb_path:            Path to .glb (relative to cwd or absolute).
        location:            World-space (x, y, z) in meters where the
                             object's pivot lands.
        scale_to_height_m:   Normalize the imported mesh's Z extent (top-to-
                             bottom in world space) to this many meters.
                             None = no scaling (use whatever scale .glb has).
        rotation:            Euler rotation (rx, ry, rz) in radians applied
                             after import.
        pass_index:          Object index for the segmentation render pass.
                             Use 0 for non-character assets, 1 for jia, 2 for
                             wei (matching build_characters.py convention).
        parent:              Optional Empty to parent the import to (lets
                             callers re-position via parent.location later
                             without affecting the child's local transform —
                             see build_characters.py's matrix_parent_inverse
                             pattern).
        fallback_factory:    Optional callable(name, location, ...) that
                             builds a primitive if glb_path is missing. If
                             None and file is missing, returns None.

    Returns:
        The imported Blender object (or fallback if file missing, or None).
    """
    glb_path = Path(glb_path).expanduser()
    if not glb_path.is_file():
        msg = f"[import_3d_assets] .glb not found: {glb_path}"
        if fallback_factory is not None:
            print(msg + " — using fallback primitive")
            return fallback_factory(name=name, location=location)
        print(msg + " — skipping (no fallback)")
        return None

    # Track existing objects so we can find the newly-imported ones
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    new_objs = [o for o in bpy.data.objects if o not in before]

    if not new_objs:
        print(f"[import_3d_assets] gltf import produced no objects: {glb_path}")
        return None

    # If the import created multiple meshes, join them into one for downstream
    # transforms. This keeps the "one prop = one Blender object" invariant.
    mesh_objs = [o for o in new_objs if o.type == "MESH"]
    if len(mesh_objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
        # Drop now-empty parents that came in with the gltf
        for o in new_objs:
            if o.type != "MESH" and o not in bpy.data.objects.values():
                continue
            if o.type != "MESH" and o not in mesh_objs:
                bpy.data.objects.remove(o, do_unlink=True)
        obj = mesh_objs[0]
    elif len(mesh_objs) == 1:
        obj = mesh_objs[0]
        # Drop accompanying empties / lights / cameras imported with the gltf
        for o in new_objs:
            if o is obj:
                continue
            bpy.data.objects.remove(o, do_unlink=True)
    else:
        # No mesh objects — strange .glb, return whatever we got
        obj = new_objs[0]

    obj.name = name
    obj.location       = mathutils.Vector(location)
    obj.rotation_euler = mathutils.Euler(rotation, "XYZ")
    obj.pass_index     = pass_index
    obj.display_type   = "SOLID"

    # Normalize height — Hunyuan3D and similar generators output meshes at
    # arbitrary unit scale (often 1.0 = bounding-box diagonal, not meters).
    if scale_to_height_m is not None and scale_to_height_m > 0:
        # Apply current transform first so dimensions reflect actual extent
        bpy.context.view_layer.update()
        cur_h = obj.dimensions.z
        if cur_h > 1e-6:
            s = scale_to_height_m / cur_h
            obj.scale = (obj.scale[0] * s, obj.scale[1] * s, obj.scale[2] * s)

    # Parent (with matrix_parent_inverse so child world position is preserved
    # — same fix as in build_characters.py)
    if parent is not None:
        bpy.context.view_layer.update()
        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()

    print(f"[import_3d_assets] imported {name} from {glb_path.name} "
          f"(dims={tuple(round(d, 2) for d in obj.dimensions)}, "
          f"pass_index={pass_index})")
    return obj


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run from inside Blender:
    #   blender --background --python import_3d_assets.py -- /path/to/test.glb
    import sys
    glb = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else None
    if not glb:
        print("usage: blender --background --python import_3d_assets.py -- <path.glb>")
        sys.exit(2)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    import_3d_asset(name="TestImport", glb_path=glb,
                    location=(0, 0, 0), scale_to_height_m=1.0)
    bpy.ops.wm.save_as_mainfile(filepath="/tmp/import_test.blend")
    print("[import_3d_assets] saved /tmp/import_test.blend")
