"""Host-facing configuration: late configure(), host kernels, model roles, aliases."""
from __future__ import annotations

import pytest

from pace_core import llm_client, paths
from pace_core.events.shot_duration import frame_count, vace_frames
from pace_core.node import blender_box


def test_frame_count_is_quantised_and_the_old_name_agrees():
    assert frame_count(5.0) == 81                      # 16 fps, 4n+1
    assert (frame_count(1.0) - 1) % 4 == 0
    assert frame_count(0.0) == 5                       # floor: one quantum + offset
    assert frame_count(5.0, fps=24, quantum=8, offset=1) % 8 == 1
    for s in (0.3, 1.0, 2.7, 5.0, 15.0):
        assert vace_frames(s) == frame_count(s)


def test_a_missing_kernel_is_an_error_not_a_dead_path(monkeypatch, tmp_path):
    monkeypatch.setattr(blender_box, "KERNEL_DIRS", [tmp_path])
    with pytest.raises(blender_box.KernelNotFound, match="register_kernel_dir"):
        blender_box.BlenderBox(bin="blender")._kernel("render_depth")


def test_a_registered_kernel_dir_is_searched(monkeypatch, tmp_path):
    monkeypatch.setattr(blender_box, "KERNEL_DIRS", list(blender_box.KERNEL_DIRS))
    (tmp_path / blender_box.KERNELS["render_depth"]).write_text("# kernel\n")
    blender_box.register_kernel_dir(tmp_path)
    assert blender_box.BlenderBox(bin="blender")._kernel("render_depth").parent == tmp_path.resolve()


def test_configure_after_import_reaches_blender_box(monkeypatch):
    monkeypatch.setattr(paths.BOX, "blender_bin", paths.BOX.blender_bin)
    paths.configure(blender_bin="/opt/blender-test/blender")
    assert blender_box.BlenderBox().bin == "/opt/blender-test/blender"


def test_model_roles_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("PACE_MODEL_GENERATOR", "host-generator-key")
    assert llm_client.default_model("generator") == "host-generator-key"
    monkeypatch.delenv("PACE_MODEL_GENERATOR")
    assert llm_client.default_model("generator") == llm_client.DEFAULT_MODELS["generator"]


def test_renamed_motion_compiler_keeps_its_old_name():
    from pace_core.camera import movement_io as m
    assert m.compile_wan_motion is m.compile_motion_phrase
    assert m.compile_motion_phrase(["push_in"]) == m.compile_wan_motion(["push_in"])
