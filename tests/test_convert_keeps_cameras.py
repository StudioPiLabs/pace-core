"""BlenderBox.convert keeps cameras and reads USD (runs only where Blender is installed)."""
from __future__ import annotations

import json
import os
import struct
import subprocess

import pytest

from pace_core.node.blender_box import BlenderBox

BLENDER = os.environ.get("BLENDER_BIN")

MAKE = '''
import bpy, sys
out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
bpy.ops.object.camera_add(location=(0, -6, 1.6))
cam = bpy.context.object; cam.keyframe_insert("location", frame=1)
cam.location = (3, -4, 2); cam.keyframe_insert("location", frame=25)
if out.endswith(".fbx"):
    bpy.ops.export_scene.fbx(filepath=out, bake_anim=True)
else:
    bpy.ops.wm.usd_export(filepath=out, export_animation=True)
'''


def glb_json(path) -> dict:
    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length])


@pytest.mark.skipif(not BLENDER, reason="set BLENDER_BIN to run Blender kernels")
@pytest.mark.parametrize("source", ["shot.fbx", "shot.usdc"])
def test_converted_glb_keeps_the_camera(tmp_path, source):
    script = tmp_path / "make.py"; script.write_text(MAKE)
    subprocess.run([BLENDER, "--background", "--python", script, "--", tmp_path / source],
                   check=True, capture_output=True)
    out = tmp_path / "out.glb"
    assert BlenderBox(bin=BLENDER).convert(tmp_path / source, out)["ok"]
    gltf = glb_json(out)
    assert gltf.get("cameras") and any("camera" in node for node in gltf["nodes"])
