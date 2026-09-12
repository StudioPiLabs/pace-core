"""Blender greybox primitive library.

A small, fixed vocabulary of geometric building blocks for ControlNet depth
passes. The LLM in `build_locations.py` does NOT generate Blender code —
it produces a JSON `primitives_spec` whose entries are dispatched to the
functions defined here.

Design rule: keep the vocabulary tiny (7 kinds). Anything more complex than
a wall, post, or box becomes an `imported_mesh` that delegates to
`import_3d_assets.py` (the same path used by Hunyuan3D-generated props).

This module is importable without `bpy` (the schemas are pure data); the
Blender-side functions only run when bpy is available, which is the case
when the module is imported from inside `blender --background --python`.

Public API:
  PRIMITIVE_SCHEMAS  — dict, schema for each primitive kind (LLM constraint)
  build_from_spec(spec, glb_root)  — run inside Blender; consumes the spec
                                     and creates the geometry
"""

from pathlib import Path
import json

# ──────────────────────────────────────────────────────────────────────
#  Schemas — pure data, importable without bpy
# ──────────────────────────────────────────────────────────────────────

PRIMITIVE_SCHEMAS = {
    "ground": {
        "_description": "Horizontal floor plane. Default normal = +Z.",
        "name":     "(str) unique mesh name",
        "size":     "[w_m, d_m]   width and depth in meters",
        "position": "[x, y, z]    default [0, 0, 0]",
    },
    "ceiling": {
        "_description": "Horizontal ceiling plane. Default normal = -Z.",
        "name":     "(str)",
        "size":     "[w_m, d_m]",
        "position": "[x, y, z]    typically [0, 0, room_height]",
    },
    "wall": {
        "_description": "Vertical wall plane. rotation_z controls which way it faces.",
        "name":     "(str)",
        "size":     "[w_m, h_m]   width along its surface, height up",
        "position": "[x, y, z]    centroid in world coords",
        "rotation": "[0, 0, rz_deg]  Z-axis rotation in degrees, default 0",
    },
    "backdrop": {
        "_description": "Distant background plane (e.g. sky, neon haze, painted backdrop). Same geometry as wall but semantically distant.",
        "name":     "(str)",
        "size":     "[w_m, h_m]",
        "position": "[x, y, z]",
        "rotation": "[0, 0, rz_deg]   default 0",
    },
    "box_prop": {
        "_description": "Solid cuboid for furniture, equipment, set pieces.",
        "name":     "(str)",
        "size":     "[w_m, d_m, h_m]",
        "position": "[x, y, z]   centroid",
        "rotation": "[rx_deg, ry_deg, rz_deg]   default [0,0,0]",
    },
    "post": {
        "_description": "Vertical cylinder: lamp posts, columns, tree trunks, signposts.",
        "name":     "(str)",
        "radius_m": "(float)",
        "height_m": "(float)",
        "position": "[x, y, z]   bottom-center; cylinder extends upward",
    },
    "imported_mesh": {
        "_description": "Delegate to import_3d_assets.import_3d_asset() for complex objects (vehicles, furniture, hero props, characters).",
        "name":         "(str)",
        "glb_path":     "(str) path relative to repo root or absolute",
        "position":     "[x, y, z]   default [0, 0, 0]",
        "scale_to_height_m": "(float, optional) auto-uniform-scale so the bounding box height matches this; null = use mesh native scale",
        "rotation":     "[rx_deg, ry_deg, rz_deg]   default [0,0,0]",
    },
}


# ──────────────────────────────────────────────────────────────────────
#  Blender-side dispatch (only runs inside `blender --python`)
# ──────────────────────────────────────────────────────────────────────


def _bpy():
    """Lazy-import bpy so the module is importable outside Blender."""
    import bpy  # type: ignore
    return bpy


def _set_pass_index(obj, idx: int):
    if idx is not None:
        obj.pass_index = int(idx)


def _add_ground(spec: dict):
    bpy = _bpy()
    w, d = spec["size"]
    pos = spec.get("position", [0, 0, 0])
    bpy.ops.mesh.primitive_plane_add(size=1, location=pos)
    obj = bpy.context.active_object
    obj.scale = (w, d, 1)
    obj.name = spec["name"]
    _set_pass_index(obj, spec.get("pass_index"))
    return obj


def _add_ceiling(spec: dict):
    obj = _add_ground(spec)
    obj.rotation_euler = (3.14159265, 0, 0)  # flip normal -Z
    return obj


def _add_wall(spec: dict):
    bpy = _bpy()
    w, h = spec["size"]
    pos = spec.get("position", [0, 0, 0])
    rz = (spec.get("rotation") or [0, 0, 0])[2]
    import math
    bpy.ops.mesh.primitive_plane_add(size=1, location=pos)
    obj = bpy.context.active_object
    # Plane lies in XY (flat). Scale Y by h so that after 90° X rotation,
    # the Y extent becomes the world Z extent (vertical height).
    obj.scale = (w, h, 1)
    obj.rotation_euler = (math.radians(90), 0, math.radians(rz))
    obj.name = spec["name"]
    _set_pass_index(obj, spec.get("pass_index"))
    return obj


def _add_backdrop(spec: dict):
    return _add_wall(spec)


def _add_box_prop(spec: dict):
    bpy = _bpy()
    w, d, h = spec["size"]
    pos = spec.get("position", [0, 0, 0])
    rot = spec.get("rotation") or [0, 0, 0]
    import math
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    obj = bpy.context.active_object
    # primitive_cube_add(size=1) produces a 1x1x1 unit cube (verts at ±0.5).
    obj.scale = (w, d, h)
    obj.rotation_euler = tuple(math.radians(r) for r in rot)
    obj.name = spec["name"]
    _set_pass_index(obj, spec.get("pass_index"))
    return obj


def _add_post(spec: dict):
    bpy = _bpy()
    r = spec["radius_m"]
    h = spec["height_m"]
    px, py, pz = spec.get("position", [0, 0, 0])
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=(px, py, pz + h / 2))
    obj = bpy.context.active_object
    obj.name = spec["name"]
    _set_pass_index(obj, spec.get("pass_index"))
    return obj


def _add_imported_mesh(spec: dict, glb_root: Path):
    """Import a .glb / .gltf via import_3d_assets.import_3d_asset()."""
    from pace_core.breakdown.import_3d_assets import import_3d_asset  # type: ignore

    glb_path = Path(spec["glb_path"])
    if not glb_path.is_absolute():
        glb_path = (glb_root / glb_path).resolve()
    if not glb_path.exists():
        print(f"  [primitives] WARN imported_mesh missing: {glb_path}; skipping")
        return None
    obj = import_3d_asset(
        name=spec["name"],
        glb_path=str(glb_path),
        location=tuple(spec.get("position", [0, 0, 0])),
        scale_to_height_m=spec.get("scale_to_height_m"),
        rotation=tuple(spec.get("rotation", [0, 0, 0])),
        pass_index=spec.get("pass_index", 0),
    )
    return obj


_DISPATCH = {
    "ground":         _add_ground,
    "ceiling":        _add_ceiling,
    "wall":           _add_wall,
    "backdrop":       _add_backdrop,
    "box_prop":       _add_box_prop,
    "post":           _add_post,
}


def build_from_spec(spec: dict, glb_root: Path | None = None) -> list:
    """Iterate the spec's primitives and create them in the active Blender scene.

    Returns the list of created Blender objects (those that succeeded).
    """
    if glb_root is None:
        glb_root = Path(__file__).resolve().parent.parent
    items = spec.get("primitives", spec) if isinstance(spec, dict) else spec
    created = []
    for i, p in enumerate(items):
        kind = p.get("kind")
        try:
            if kind == "imported_mesh":
                obj = _add_imported_mesh(p, glb_root)
            elif kind in _DISPATCH:
                obj = _DISPATCH[kind](p)
            else:
                print(f"  [primitives] WARN unknown kind {kind!r} at index {i}; skipping")
                continue
            if obj is not None:
                created.append(obj)
        except Exception as e:
            print(f"  [primitives] FAIL {kind} ({p.get('name')}): {e}")
    print(f"  [primitives] built {len(created)}/{len(items)} primitives")
    return created


# ──────────────────────────────────────────────────────────────────────
#  CLI: blender --background --python primitives.py -- spec.json out.blend
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--" not in sys.argv:
        print("usage: blender --background --python primitives.py -- spec.json out.blend")
        sys.exit(1)
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 2:
        print("usage: blender --background --python primitives.py -- spec.json out.blend")
        sys.exit(1)
    spec_path, out_blend = args

    bpy = _bpy()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    spec = json.loads(Path(spec_path).read_text())
    build_from_spec(spec)

    bpy.ops.wm.save_as_mainfile(filepath=str(Path(out_blend).resolve()))
    print(f"saved {out_blend}")
