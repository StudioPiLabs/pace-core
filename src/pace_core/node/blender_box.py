"""blender_box — the single host-side caller for ALL Blender work.

`BlenderBox` is to Blender what `GpuBox` is to ComfyUI: the one module
that spawns the `blender` binary. Everything else (pai CLI, studio
routers, the VACE pipeline) goes through a `BlenderBox` method and never
shells out to Blender itself.

    box = BlenderBox()
    box.convert(src, out_glb)        # export  — FBX/OBJ/glTF → GLB
    box.assemble(SceneAssemblySpec)  # build   — PAI scene → .blend
    box.render_preview(ShotPreviewSpec)   # render — single Workbench frame
    box.render_depth(DepthSeqSpec)   # motion+render — N-frame depth seq
    box.render_passes(...)           # render  — depth/normal/seg passes
    box.export_glb(GlbExportSpec)    # export  — .blend → GLB (web viewer)
    box.health()                     # blender --version probe

Two layers, kept distinct on purpose:

  * BlenderBox (this file) is host-side Python — it runs in the venv and
    only ever *launches* Blender. It never imports `bpy`.
  * The per-op work runs inside Blender as a **kernel** script (one of
    `KERNELS` below). Kernels `import bpy` and CAN'T be imported here;
    BlenderBox passes each a spec/flags and parses the `RESULT_JSON={…}`
    line it prints. BlenderBox passes this package's
    location in $PACE_CORE_PATH; a host registers the directory of any
    kernel it supplies with `register_kernel_dir`.

The caller computes all motion (camera + object tracks) up front via
`camera_planner`; the depth kernel is a dumb playback engine.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Dual-mode module. Imported in the venv it is the host-side BlenderBox
# caller. Run by Blender —
#     blender --background [scene.blend] --python blender_box.py -- \
#         --kernel {convert|passes} …
# — it is a bpy KERNEL (the code at the bottom, formerly convert_to_glb.py
# and blender_render.py). `import bpy` succeeds only inside Blender.
try:
    import bpy
    import mathutils  # noqa: F401 — used by the passes kernel
except ImportError:                       # venv host — no Blender
    bpy = None
    mathutils = None
else:
    # Blender's Python ignores PYTHONPATH; BlenderBox passes this package's location.
    sys.path.insert(0, os.environ.get("PACE_CORE_PATH") or str(Path(__file__).resolve().parents[2]))

from pace_core import paths as _paths
from pace_core.paths import paths_for, BOX
from pace_core import timeouts as _to

# Bound lazily inside _kernel_passes (Blender only) so the venv host import
# stays free of camera_planner. The passes-kernel helpers reference these.
_plan_camera = _plan_cameras = _plan_camera_track = None


BLENDER_BIN = BOX.blender_bin

# Kernel directories, searched in order. A host application registers the
# directory holding the kernels this package does not ship.
KERNEL_DIRS: list[Path] = [Path(__file__).resolve().parent]


def register_kernel_dir(d: str | Path) -> None:
    """Search `d` for kernels after this package's own."""
    d = Path(d).resolve()
    if d not in KERNEL_DIRS:
        KERNEL_DIRS.append(d)


def _package_path() -> Path:
    """A directory holding only pace_core, for Blender's sys.path.

    Adding the whole site-packages would put the venv's numpy ahead of the one
    built for Blender's Python.
    """
    pkg = Path(__file__).resolve().parents[1]
    d = Path(tempfile.gettempdir()) / f"pace_core_path_{hashlib.sha1(str(pkg).encode()).hexdigest()[:10]}"
    link = d / "pace_core"
    if not link.exists():
        d.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(pkg, target_is_directory=True)
        except FileExistsError:
            pass
    return d


def _blender_env() -> dict:
    """Environment for a Blender child: this package importable, same project roots."""
    return {**os.environ,
            "PACE_CORE_PATH": str(_package_path()),
            "FILM_REPO_ROOT": str(_paths.FILM_REPO_ROOT),
            "PAI_PROJECTS_ROOT": str(_paths.PAI_PROJECTS_ROOT),
            "PAI_BRANCH": _paths.PAI_BRANCH,
            "MODELS_FILE": str(_paths.MODELS_FILE)}

# Logical op → bpy kernel file. These run INSIDE Blender, not here. Single
# source of truth for "which file carries each kernel". Some kernels are
# embedded (dual-mode) in a host module and selected with `--kernel <name>`:
#   convert / render_passes → this file (blender_box.py)
#   render_depth            → vace_pipeline.py
# The rest are standalone bpy scripts; assemble, render_preview, render_depth
# and export_glb are supplied by the host application.
KERNELS = {
    "convert":        "blender_box.py",           # embedded — --kernel convert
    "assemble":       "scene_assembler.py",       # PAI scene → .blend
    "render_preview": "render_shot_preview.py",   # single Workbench frame
    "render_depth":   "vace_pipeline.py",         # embedded — --kernel depth
    "render_passes":  "blender_box.py",           # embedded — --kernel passes
    "export_glb":     "export_blend_glb.py",       # .blend → GLB (Phase 2)
    "list_parts":     "blender_box.py",            # embedded — --kernel list_parts
    "panel_greybox":  "panel_greybox.py",          # embedded — --kernel greybox
}


# ─── spec dataclasses ─────────────────────────────────────────────────────

@dataclass
class SceneAssemblySpec:
    """Inputs for BlenderBox.assemble."""
    project:            str
    scene_id:           str
    out_blend:          str                      # absolute path
    sim_frames:         int  = 60
    include_props:      Optional[list[str]] = None
    include_characters: bool = True
    include_location:   bool = True
    keep_rigid_body:    bool = False
    typing_characters:  list[str] = field(default_factory=list)


@dataclass
class ShotPreviewSpec:
    """Inputs for BlenderBox.render_preview."""
    project:    str
    scene_id:   str
    shot_id:    str
    in_blend:   str         # absolute path
    out_png:    str         # absolute path
    mode:       str  = "layout"           # "layout" | "shot"
    width:      int  = 1280
    height:     int  = 720


@dataclass
class PanelGreyboxSpec:
    """Inputs for BlenderBox.render_panel_greybox.

    `spec` is the whole control-geometry description -- cabin, seats, subject
    meshes with their measured seat contact, lens, solved azimuth/elevation --
    built host-side by `panel_greybox.build_spec` off the panel and its
    location stub. It arrives as one nested dict rather than as flags because
    it is a scene description, not a handful of knobs; the kernel reads it from
    a file in the job dir, the way render_depth passes its track.
    """
    spec:    dict
    out_png: str                                 # absolute path
    # Optional: also export the SET (everything but the cast) as a GLB. The
    # assembly is otherwise rebuilt and discarded per panel.
    export_glb: str | None = None
    # Include the seated cast proxies in that GLB too, instead of just the
    # set. For a video job's mask_from_subject path, which needs a subject
    # mesh to render a matte from — the set-only export has nothing to mask.
    export_glb_include_bodies: bool = False


@dataclass
class DepthSeqSpec:
    """Inputs for BlenderBox.render_depth.

    Supply exactly one of `in_blend` (assembled .blend) or `glb_file`
    (raw GLB/glTF — imported into a fresh Blender scene at render time).
    """
    track:           list[dict]           # camera_planner N-keyframe list
    out_dir:         str
    in_blend:        Optional[str] = None
    glb_file:        Optional[str] = None
    width:           int   = 832
    height:          int   = 480
    render_keyframe: bool  = False        # also render frame 0 in colour → keyframe.png
    keyframe_textured: bool = False       # keyframe uses GLB image textures (else flat clay)
    add_ground_plane: bool = False        # add 400×400m flat plane at Z=0
    add_road_markers: bool = False        # line the +X path with posts (depth parallax)
    ground_contour:   bool = False        # rolling terrain (parallax, no posts in output)
    # Extra control passes (same camera/animation, re-rendered per modality) for
    # the multi-channel VACE path. normal/seg → extra control channels; matte →
    # binary subject mask for an in-graph ColorToMask → control_masks.
    emit_normal:      bool = False        # also render normal_NNNN.png
    emit_seg:         bool = False        # also render seg_NNNN.png (random colours)
    emit_matte:       bool = False        # also render matte_NNNN.png (white subj/black bg)
    # Animation directives — the single motion source. List of {target, kind, …}:
    #   kind="clip" retimes the imported asset's EMBEDDED glTF animation onto the
    #               shot's frames (retime: fit|loop).
    #   kind="spin" rotates a target about an axle; target="wheels" auto-detects
    #               the car wheels + adds rolling spoked proxies (roll=true → Δθ=Δs/r).
    # Realized in the depth kernel.
    animations:       Optional[list] = None
    # Per-frame object motion. Each entry: {target, track} where target is
    # "subject" (whole imported mesh) or an object name, and track is a list
    # of {position[m], rotation_deg}. Built by camera_planner — see below.
    object_tracks:   Optional[list] = None
    # Back-compat convenience: animate the imported mesh + chase cam forward
    # at this speed. When >0 (and object_tracks unset) the shim below uses
    # camera_planner.make_linear_motion + plan_follow_shot to synthesize
    # object_tracks and rewrite `track`. Prefer passing object_tracks directly.
    car_speed_mps:   float = 0.0
    fps:             int   = 16
    # Generic heading for the imported subject (deg about world up / Z) — the
    # "which way the car faces" knob. Baked at import for a still subject;
    # folded into the drive track (so it faces its travel) when car_speed_mps>0.
    subject_yaw_deg: float = 0.0


@dataclass
class GlbExportSpec:
    """Inputs for BlenderBox.export_glb (Phase 2)."""
    in_blend:        str
    out_glb:         str
    include_hidden:  bool = False


# ─── the caller ───────────────────────────────────────────────────────────

class BlenderBox:
    """Single facade over the `blender` binary. Stateless apart from which
    binary to launch (`bin`), so it's cheap to construct per call."""

    def __init__(self, bin: str | None = None):
        self.bin = bin or BLENDER_BIN

    # ---- kernel + invoke plumbing ----------------------------------------

    def _kernel(self, op: str) -> Path:
        """Resolve a logical op to its bpy kernel path."""
        name = KERNELS[op]
        for d in KERNEL_DIRS:
            if (d / name).is_file():
                return d / name
        return KERNEL_DIRS[0] / name

    @staticmethod
    def _job_dir() -> Path:
        """Tempdir under /tmp/blender_jobs/<uuid> for spec + log."""
        d = Path(tempfile.gettempdir()) / "blender_jobs" / uuid.uuid4().hex[:8]
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _parse_result(stdout: str) -> dict:
        """Pick the last RESULT_JSON line from a kernel's stdout."""
        line = next((l for l in reversed(stdout.splitlines())
                     if l.startswith("RESULT_JSON=")), None)
        if line is None:
            return {"ok": False, "_no_result_json": True,
                    "stdout_tail": stdout[-500:]}
        return json.loads(line[len("RESULT_JSON="):])

    def _invoke(self, cmd: list[str], timeout: int = 600, full: bool = False) -> dict:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=_blender_env())
        if r.returncode != 0:
            d = {"ok": False, "returncode": r.returncode,
                 "stderr_tail": (r.stderr or "")[-500:],
                 "stdout_tail": (r.stdout or "")[-500:]}
            if full:                       # untruncated, for server-side logging
                d["stderr_full"] = r.stderr or ""
                d["stdout_full"] = r.stdout or ""
            return d
        return self._parse_result(r.stdout)

    # ---- health ----------------------------------------------------------

    def health(self) -> dict:
        """Probe the binary (mirrors GpuBox.health). Returns
        {ok, version, bin} or {ok: False, error}."""
        try:
            r = subprocess.run([self.bin, "--version"],
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError) as e:
            return {"ok": False, "bin": self.bin, "error": str(e)}
        if r.returncode != 0:
            return {"ok": False, "bin": self.bin,
                    "error": (r.stderr or r.stdout or "")[-200:]}
        version = next((l.strip() for l in r.stdout.splitlines()
                        if l.lower().startswith("blender")), r.stdout[:80].strip())
        return {"ok": True, "bin": self.bin, "version": version}

    # ---- op: export — FBX/OBJ/glTF → GLB ---------------------------------

    def list_glb_parts(self, glb: str | Path) -> dict:
        """Import a GLB exactly as the depth kernel does and report the object
        names Blender assigns — the ground-truth animation targets (the depth
        kernel matches `o.name == target`). Returns {ok, parts:[{name, type,
        is_mesh}]}. Lets the UI list names that are guaranteed to animate,
        instead of three.js-side names that can drift from Blender's import."""
        glb = Path(glb)
        if not glb.is_file():
            return {"ok": False, "_missing": str(glb)}
        cmd = [self.bin, "--background",
               "--python", str(self._kernel("list_parts")), "--",
               "--kernel", "list_parts", "--in", str(glb)]
        return self._invoke(cmd, timeout=120)

    def convert(self, src: str | Path, out_glb: str | Path, full: bool = False) -> dict:
        """Convert a Mixamo/mocap FBX (or OBJ/glTF/.blend) to a GLB with its
        embedded animation, so the depth kernel's `clip` source can play it.
        `full=True` returns untruncated stdout/stderr for server-side logging."""
        src = Path(src); out_glb = Path(out_glb)
        if not src.is_file():
            return {"ok": False, "_missing": str(src)}
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.bin, "--background",
               "--python", str(self._kernel("convert")), "--",
               "--kernel", "convert",
               "--in", str(src), "--out", str(out_glb)]
        return self._invoke(cmd, timeout=_to.BLENDER_CONVERT, full=full)

    # ---- op: build — PAI scene → .blend ----------------------------------

    def assemble(self, spec: SceneAssemblySpec | dict) -> dict:
        """Build a .blend for `spec.scene_id` via the assemble kernel."""
        if isinstance(spec, dict):
            spec = SceneAssemblySpec(**spec)
        cmd = [self.bin, "--background", "--python", str(self._kernel("assemble")),
               "--",
               "--project", spec.project,
               "--scene",   spec.scene_id,
               "--out",     spec.out_blend,
               "--sim-frames", str(spec.sim_frames)]
        if spec.include_props:
            cmd += ["--props", *spec.include_props]
        if not spec.include_characters:
            cmd += ["--no-characters"]
        if not spec.include_location:
            cmd += ["--no-location"]
        if spec.keep_rigid_body:
            cmd += ["--keep-rigid-body"]
        for cid in spec.typing_characters:
            cmd += ["--typing-character", cid]
        return self._invoke(cmd, timeout=300)

    # ---- op: render — single Workbench preview ---------------------------

    def render_preview(self, spec: ShotPreviewSpec | dict) -> dict:
        """Single-frame Workbench preview of one shot."""
        if isinstance(spec, dict):
            spec = ShotPreviewSpec(**spec)
        if not Path(spec.in_blend).is_file():
            return {"ok": False, "_missing_blend": spec.in_blend}
        cmd = [self.bin, "--background", spec.in_blend,
               "--python", str(self._kernel("render_preview")), "--",
               "--project", spec.project,
               "--scene",   spec.scene_id,
               "--shot",    spec.shot_id,
               "--out",     spec.out_png,
               "--width",   str(spec.width),
               "--height",  str(spec.height),
               "--mode",    spec.mode]
        return self._invoke(cmd, timeout=180)

    # ---- op: render — one panel's control geometry ------------------------

    def render_panel_greybox(self, spec: PanelGreyboxSpec | dict) -> dict:
        """Flat Workbench frame of a panel's declared cast, in its own camera.

        The frame feeds `structure_flux2` as an init image, where it carries
        the subject count and staging that a prompt cannot hold on its own.
        """
        if isinstance(spec, dict):
            spec = PanelGreyboxSpec(**spec)
        out = Path(spec.out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        job = self._job_dir()
        spec_json = job / "greybox_spec.json"
        spec_json.write_text(json.dumps({
            **spec.spec, "out": str(out),
            **({"export_glb": spec.export_glb,
                "export_glb_include_bodies": spec.export_glb_include_bodies}
               if spec.export_glb else {})}))
        cmd = [self.bin, "--background",
               "--python", str(self._kernel("panel_greybox")), "--",
               "--kernel", "greybox", "--spec", str(spec_json)]
        return self._invoke(cmd, timeout=300)

    # ---- op: motion + render — N-frame depth sequence --------------------

    def render_depth(self, spec: DepthSeqSpec | dict) -> dict:
        """N-frame Workbench depth render driven by a pre-computed camera
        track + object tracks + animation directives.

        Scene source — one of:
          spec.in_blend  : .blend loaded as Blender's positional arg
          spec.glb_file  : GLB imported inside the kernel (fresh scene)
        """
        if isinstance(spec, dict):
            spec = DepthSeqSpec(**spec)
        if not spec.in_blend and not spec.glb_file and not (spec.ground_contour or spec.add_ground_plane):
            return {"ok": False, "_error": "DepthSeqSpec needs in_blend or glb_file "
                                           "(or ground_contour/add_ground_plane for a no-mesh scene render)"}
        if spec.in_blend and not Path(spec.in_blend).is_file():
            return {"ok": False, "_missing_blend": spec.in_blend}
        if spec.glb_file and not Path(spec.glb_file).is_file():
            return {"ok": False, "_missing_glb": spec.glb_file}
        out_dir = Path(spec.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Camera + object tracks. The kernel is dumb playback; all motion is
        # computed here / in camera_planner before we shell out.
        track = spec.track
        object_tracks = spec.object_tracks

        # Back-compat shim: car_speed_mps>0 (and no explicit object_tracks)
        # means "drive the imported mesh + chase cam forward at this speed".
        # Synthesize world-space subject + follow-camera tracks via the
        # planner — keeps the physics in the planner, not the Blender kernel.
        if spec.car_speed_mps > 0 and not object_tracks:
            from pace_core.camera.camera_planner import make_linear_motion, plan_follow_shot
            subject = make_linear_motion(
                n_frames=len(track), speed_mps=spec.car_speed_mps,
                fps=spec.fps, axis="x",
                rotation_deg=[0.0, 0.0, spec.subject_yaw_deg])  # face its heading while driving
            follow = plan_follow_shot(subject_track=subject,
                                      relative_camera_track=track)
            track = follow["camera_track"]
            object_tracks = [{"target": "subject", "track": follow["subject_track"]}]

        track_json = out_dir / "track.json"
        track_json.write_text(json.dumps(track))
        kernel = self._kernel("render_depth")
        kf_flag      = ["--render-keyframe"] if spec.render_keyframe else []
        kf_tex_flag  = ["--keyframe-textured"] if spec.keyframe_textured else []
        ground_flag  = ["--add-ground-plane"] if spec.add_ground_plane else []
        markers_flag = ["--add-road-markers"] if spec.add_road_markers else []
        contour_flag = ["--ground-contour"] if spec.ground_contour else []
        normal_flag  = ["--emit-normal"] if spec.emit_normal else []
        seg_flag     = ["--emit-seg"] if spec.emit_seg else []
        matte_flag   = ["--emit-matte"] if spec.emit_matte else []
        obj_args: list[str] = []
        if object_tracks:
            obj_json = out_dir / "object_tracks.json"
            obj_json.write_text(json.dumps(object_tracks))
            obj_args = ["--object-tracks", str(obj_json)]
        anim_args: list[str] = []
        if spec.animations:
            anim_json = out_dir / "animations.json"
            anim_json.write_text(json.dumps(spec.animations))
            anim_args = ["--animations", str(anim_json)]
        yaw_args: list[str] = (["--subject-yaw-deg", str(spec.subject_yaw_deg)]
                               if spec.subject_yaw_deg else [])
        aids = [*obj_args, *anim_args, *yaw_args, *ground_flag, *markers_flag, *contour_flag,
                *normal_flag, *seg_flag, *matte_flag, *kf_flag, *kf_tex_flag]
        if spec.glb_file:
            # GLB path: no positional .blend — kernel imports the mesh itself.
            cmd = [self.bin, "--background",
                   "--python", str(kernel), "--", "--kernel", "depth",
                   "--glb-file",   spec.glb_file,
                   "--track-json", str(track_json),
                   "--out-dir",    str(out_dir),
                   "--width",      str(spec.width),
                   "--height",     str(spec.height),
                   *aids]
        elif spec.in_blend:
            cmd = [self.bin, "--background", spec.in_blend,
                   "--python", str(kernel), "--", "--kernel", "depth",
                   "--track-json", str(track_json),
                   "--out-dir",    str(out_dir),
                   "--width",      str(spec.width),
                   "--height",     str(spec.height),
                   *aids]
        else:
            # Scene render: neither a subject mesh nor a .blend — the kernel renders
            # the camera move over ground geometry (ground_contour/plane in `aids`)
            # for parallax; appearance comes from the VACE keyframe + prompt.
            cmd = [self.bin, "--background",
                   "--python", str(kernel), "--", "--kernel", "depth",
                   "--track-json", str(track_json),
                   "--out-dir",    str(out_dir),
                   "--width",      str(spec.width),
                   "--height",     str(spec.height),
                   *aids]
        # Scale the timeout with frame count — 240-frame cinematic tracks of a
        # heavy GLB (+ road markers) run well past the 81-frame budget.
        n_frames = len(track) if isinstance(track, list) else 81
        timeout = max(_to.DEPTH_RENDER_FLOOR, int(n_frames * 12))
        return self._invoke(cmd, timeout=timeout)

    # ---- op: cam_track — extract an embedded camera animation from a GLB -----

    def extract_camera_track(self, glb_file: str | Path, *, cam_name: str | None = None,
                             timeout: int = 120) -> dict:
        """Extract a GLB's embedded animated camera as a trajectory track.
        Returns {has_camera, frames, track, camera, cameras, fps}; `cameras` lists
        every animated camera so the UI can offer a picker, and `cam_name` selects
        which one to extract (default: the first). has_camera=False when the GLB
        has no animated camera. Lets the Playground skip the trajectory upload when
        the camera move ships inside the GLB."""
        kernel = self._kernel("render_depth")          # vace_pipeline.py (dual-mode)
        out = self._job_dir() / "cam_track.json"
        cmd = [self.bin, "--background", "--python", str(kernel), "--",
               "--kernel", "cam_track", "--glb-file", str(glb_file), "--out", str(out)]
        if cam_name:
            cmd += ["--cam", cam_name]
        res = self._invoke(cmd, timeout=timeout)
        try:
            return json.loads(out.read_text())
        except Exception:
            return {"has_camera": False, "frames": 0, "track": [],
                    "error": res.get("error") or "extraction failed"}

    # ---- op: render — depth/normal/seg passes (single frame, ControlNet) -

    def render_passes(self, shot_id: str, scene_ref: str,
                      scene_path: str | Path, project: str,
                      out_dir: str | Path | None = None,
                      timeout: int = 600) -> dict:
        """Render depth/normal/segmentation passes for one shot via the
        embedded multi-pass compositor kernel (--kernel passes). Returns a
        dict of pass-name → PNG path for the passes that were produced.

        Scene geometry is the assembled .blend at
        <project>/blender_scenes/<scene_ref>.blend; camera + lighting + panel
        list are read from the PAI scene JSON at `scene_path`.
        """
        scene_path = Path(scene_path)
        if not scene_path.exists():
            return {"ok": False, "_missing_scene_json": str(scene_path)}
        p = paths_for(project)
        out = Path(out_dir) if out_dir else p.depth_dir / shot_id
        out.mkdir(parents=True, exist_ok=True)
        blend_file = p.blender_scenes_dir / f"{scene_ref}.blend"
        if not blend_file.exists():
            return {"ok": False, "_missing_blend": str(blend_file)}

        cmd = [self.bin, "--background", str(blend_file),
               "--python", str(self._kernel("render_passes")), "--",
               "--kernel", "passes",
               "--pai", str(scene_path),
               "--out", str(out),
               "--ids", shot_id]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=_blender_env())
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-800:]
            return {"ok": False, "returncode": r.returncode, "stderr_tail": tail}

        # blender_render's FileOutput node appends "0001" to each slot prefix.
        cand = {
            "depth":      out / f"depth_{shot_id}_0001.png",
            "normal":     out / f"normal_{shot_id}_0001.png",
            "seg_a":      out / f"seg_a_{shot_id}_0001.png",
            "seg_b":      out / f"seg_b_{shot_id}_0001.png",
            "depth_end":  out / f"depth_{shot_id}_end_0001.png",
            "normal_end": out / f"normal_{shot_id}_end_0001.png",
        }
        if not cand["depth"].exists():
            return {"ok": False, "_no_depth_output": str(cand["depth"])}
        passes = {k: str(v) for k, v in cand.items() if v.exists()}
        return {"ok": True, "passes": passes}

    # ---- op: export — .blend → GLB (Phase 2 web viewer) ------------------

    def export_glb(self, spec: GlbExportSpec | dict) -> dict:
        """Export the .blend as a single glTF binary for in-browser viewing."""
        if isinstance(spec, dict):
            spec = GlbExportSpec(**spec)
        if not Path(spec.in_blend).is_file():
            return {"ok": False, "_missing_blend": spec.in_blend}
        kernel = self._kernel("export_glb")
        if not kernel.is_file():
            return {"ok": False, "_pending": f"{kernel.name} not implemented yet"}
        Path(spec.out_glb).parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.bin, "--background", spec.in_blend,
               "--python", str(kernel), "--",
               "--out", spec.out_glb,
               "--include-hidden", str(spec.include_hidden).lower()]
        return self._invoke(cmd, timeout=180)


# ─── convenience: spec from PAI project ───────────────────────────────────

def default_blend_path(project: str, scene_id: str) -> Path:
    """Convention: <project>/blender_scenes/<scene_id>.blend"""
    return paths_for(project).storage / "blender_scenes" / f"{scene_id}.blend"


def default_preview_path(project: str, scene_id: str, shot_id: str) -> Path:
    """Convention: <project>/blender_scenes/previews/<scene_id>_<shot_id>.png"""
    return (paths_for(project).storage / "blender_scenes" / "previews"
            / f"{scene_id}_{shot_id}.png")


# ════════════════════════════════════════════════════════════════════════
#  Blender KERNELS (run INSIDE Blender; bpy present). Reached via
#  `--kernel <name>`. None of this runs in the venv.
#    convert  — FBX/OBJ/glTF → GLB           (formerly convert_to_glb.py)
#    passes   — depth/normal/seg pass render (formerly blender_render.py)
# ════════════════════════════════════════════════════════════════════════


# ── kernel: convert (FBX/OBJ/glTF → GLB, keeps embedded animation) ────────

def _kernel_convert(argv: list) -> None:
    """Import a mesh and re-export as one GLB with its embedded animation.
    Prints RESULT_JSON={ok, out, actions} for BlenderBox.convert."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="source FBX/OBJ/glTF")
    ap.add_argument("--out", required=True, help="destination .glb")
    args = ap.parse_args(argv)

    ext = Path(args.inp).suffix.lower()
    if ext == ".blend":
        # A .blend is OPENED (it IS a Blender scene) — keeps real named
        # objects + materials, then we export the whole scene to GLB.
        bpy.ops.wm.open_mainfile(filepath=args.inp)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if ext == ".fbx":
            # automatic_bone_orientation keeps Mixamo rigs sane on re-export.
            bpy.ops.import_scene.fbx(filepath=args.inp,
                                     automatic_bone_orientation=True)
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=args.inp)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=args.inp)
        else:
            raise SystemExit(f"unsupported input extension: {ext!r}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB",
                              export_animations=True, use_selection=True)
    n_actions = len(bpy.data.actions)
    print("RESULT_JSON=" + json.dumps(
        {"ok": True, "out": args.out, "actions": n_actions}))


def _kernel_list_parts(argv: list) -> None:
    """Import a GLB the SAME way the depth kernel does (factory-empty scene +
    import_scene.gltf) and report the object names Blender assigns. These are
    the exact `subject_objs` names the animation matchers compare against, so
    the UI can list ground-truth targets. RESULT_JSON={ok, parts:[…]}."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="source GLB/glTF")
    args = ap.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.inp)
    parts = [{"name": o.name, "type": o.type, "is_mesh": o.type == "MESH"}
             for o in bpy.context.scene.objects]
    print("RESULT_JSON=" + json.dumps({"ok": True, "parts": parts}))


# ── kernel: passes (PAI-driven depth/normal/seg + VACE flow sequence) ─────
# Helpers below are module-level (used by _kernel_passes). They reference
# the _plan_* globals, bound from camera_planner inside _kernel_passes.

_BIBLE_CACHE: dict = {}


def _load_bible(scene_id: str):
    if scene_id in _BIBLE_CACHE:
        return _BIBLE_CACHE[scene_id]
    bible_path = (Path(__file__).resolve().parent
                  / "kb" / "on_scene" / "scenes" / f"{scene_id}.json")
    bible = json.loads(bible_path.read_text()) if bible_path.exists() else None
    _BIBLE_CACHE[scene_id] = bible
    return bible


def _apply_camera_plan(cam_obj, plan: dict):
    """Set Blender camera location, rotation, and lens from a camera_plan dict.

    For movement plans, applies the START frame. Use _apply_camera_xform
    directly for the END frame.
    """
    mv = plan.get("movement")
    if mv:
        _apply_camera_xform(cam_obj, mv["start_position"],
                            mv["start_rotation_deg"], mv["start_lens_mm"])
    else:
        _apply_camera_xform(cam_obj, plan["position"],
                            plan["rotation_deg"], plan["lens_mm"])


def _apply_camera_xform(cam_obj, position, rotation_deg, lens_mm):
    cam_obj.location = mathutils.Vector(position)
    cam_obj.rotation_euler = mathutils.Euler(
        [math.radians(d) for d in rotation_deg], "XYZ"
    )
    cam_obj.data.lens = float(lens_mm)


def _setup_depth_only_compositor(scene, out_dir: Path, panel_id: str,
                                  emit_flow: bool = False):
    """Compositor used by the VACE sequence renderer and the clean re-pass.

    Outputs depth_<panel_id>_<frame>.png always; when emit_flow=True also
    emits flow EXR + flow_rgb PNG + normal + seg (the four VACE ControlNet
    channels, produced together in one Cycles render call).
    """
    scene.use_nodes = True
    tree  = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rl = nodes.new("CompositorNodeRLayers")

    # ── Depth: Normalize → Invert → 16-bit BW PNG ──
    norm   = nodes.new("CompositorNodeNormalize")
    invert = nodes.new("CompositorNodeInvert")
    depth  = nodes.new("CompositorNodeOutputFile")
    depth.base_path           = str(out_dir)
    depth.file_slots[0].path  = f"depth_{panel_id}_"
    depth.format.file_format  = "PNG"
    depth.format.color_mode   = "BW"
    depth.format.color_depth  = "16"
    links.new(rl.outputs["Depth"], norm.inputs[0])
    links.new(norm.outputs[0],     invert.inputs["Color"])
    links.new(invert.outputs[0],   depth.inputs[0])

    # ── Flow: raw 4-channel Vector pass → OpenEXR (lossless float) ──
    if emit_flow and "Vector" in rl.outputs:
        # (a) Archival float EXR — full fidelity, signed pixel deltas
        flow_exr = nodes.new("CompositorNodeOutputFile")
        flow_exr.base_path           = str(out_dir)
        flow_exr.file_slots[0].path  = f"flow_{panel_id}_"
        flow_exr.format.file_format  = "OPEN_EXR"
        flow_exr.format.color_mode   = "RGBA"
        flow_exr.format.color_depth  = "32"
        links.new(rl.outputs["Vector"], flow_exr.inputs[0])

        # (b) Browser-viewable PNG. Encoding (clamp ±16 px):
        #   R = (out_dx + 16) / 32, G = (out_dy + 16) / 32, B = 0
        sep = nodes.new("CompositorNodeSepRGBA")
        links.new(rl.outputs["Vector"], sep.inputs["Image"])
        m_x = nodes.new("CompositorNodeMapRange")
        m_x.inputs["From Min"].default_value = -16.0
        m_x.inputs["From Max"].default_value =  16.0
        m_x.inputs["To Min"].default_value   =  0.0
        m_x.inputs["To Max"].default_value   =  1.0
        m_x.use_clamp = True
        links.new(sep.outputs["B"], m_x.inputs["Value"])

        m_y = nodes.new("CompositorNodeMapRange")
        m_y.inputs["From Min"].default_value = -16.0
        m_y.inputs["From Max"].default_value =  16.0
        m_y.inputs["To Min"].default_value   =  0.0
        m_y.inputs["To Max"].default_value   =  1.0
        m_y.use_clamp = True
        links.new(sep.outputs["A"], m_y.inputs["Value"])

        comb = nodes.new("CompositorNodeCombRGBA")
        links.new(m_x.outputs["Value"], comb.inputs["R"])
        links.new(m_y.outputs["Value"], comb.inputs["G"])

        flow_png = nodes.new("CompositorNodeOutputFile")
        flow_png.base_path           = str(out_dir)
        flow_png.file_slots[0].path  = f"flow_rgb_{panel_id}_"
        flow_png.format.file_format  = "PNG"
        flow_png.format.color_mode   = "RGB"
        flow_png.format.color_depth  = "8"
        links.new(comb.outputs["Image"], flow_png.inputs[0])

    # ── Normal pass: 8-bit RGB PNG (surface normals in camera space) ──
    if emit_flow and "Normal" in rl.outputs:
        normal = nodes.new("CompositorNodeOutputFile")
        normal.base_path           = str(out_dir)
        normal.file_slots[0].path  = f"normal_{panel_id}_"
        normal.format.file_format  = "PNG"
        normal.format.color_mode   = "RGB"
        normal.format.color_depth  = "8"
        links.new(rl.outputs["Normal"], normal.inputs[0])

    # ── Segmentation pass: IndexOB mask, 8-bit BW PNG ──
    if emit_flow and "IndexOB" in rl.outputs:
        idmask = nodes.new("CompositorNodeIDMask")
        idmask.index = 1
        seg = nodes.new("CompositorNodeOutputFile")
        seg.base_path           = str(out_dir)
        seg.file_slots[0].path  = f"seg_{panel_id}_"
        seg.format.file_format  = "PNG"
        seg.format.color_mode   = "BW"
        seg.format.color_depth  = "8"
        links.new(rl.outputs["IndexOB"], idmask.inputs[0])
        links.new(idmask.outputs[0],     seg.inputs[0])


def _setup_compositor(scene, out_dir: Path, panel_id: str):
    """Multi-pass compositor: depth + normal + seg (IndexOB)."""
    scene.use_nodes = True
    tree  = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rl = nodes.new("CompositorNodeRLayers")

    # ── Depth pass: Normalize → Invert → 16-bit BW PNG ──
    norm   = nodes.new("CompositorNodeNormalize")
    invert = nodes.new("CompositorNodeInvert")
    depth  = nodes.new("CompositorNodeOutputFile")
    depth.base_path           = str(out_dir)
    depth.file_slots[0].path  = f"depth_{panel_id}_"
    depth.format.file_format  = "PNG"
    depth.format.color_mode   = "BW"
    depth.format.color_depth  = "16"
    links.new(rl.outputs["Depth"], norm.inputs[0])
    links.new(norm.outputs[0],     invert.inputs["Color"])
    links.new(invert.outputs[0],   depth.inputs[0])

    # ── Normal pass: 8-bit RGB ──
    normal = nodes.new("CompositorNodeOutputFile")
    normal.base_path           = str(out_dir)
    normal.file_slots[0].path  = f"normal_{panel_id}_"
    normal.format.file_format  = "PNG"
    normal.format.color_mode   = "RGB"
    normal.format.color_depth  = "8"
    links.new(rl.outputs["Normal"], normal.inputs[0])

    # ── Seg pass: IndexOB → 8-bit BW ──
    idmask = nodes.new("CompositorNodeIDMask")
    idmask.index = 1   # Subject_A.pass_index=1; build_characters.py sets these
    seg = nodes.new("CompositorNodeOutputFile")
    seg.base_path           = str(out_dir)
    seg.file_slots[0].path  = f"seg_{panel_id}_"
    seg.format.file_format  = "PNG"
    seg.format.color_mode   = "BW"
    seg.format.color_depth  = "8"
    links.new(rl.outputs["IndexOB"], idmask.inputs[0])
    links.new(idmask.outputs[0],     seg.inputs[0])


def _render_vace_sequence(cam_obj, scene, planner_shot: dict, out_dir: Path,
                          sid_with_cam: str, num_frames: int) -> None:
    """Render N depth frames along the planner's eased trajectory.

    Output: out_dir/vace_<sid>/depth_<sid>_<frame:04d>.png (one per frame)
            + camera_track_<sid>.json. Uses camera_planner.plan_camera_track
            — picks up composite movement + frame.movement_easing."""
    sub_out = out_dir / f"vace_{sid_with_cam}"
    sub_out.mkdir(parents=True, exist_ok=True)

    _setup_depth_only_compositor(scene, sub_out, sid_with_cam, emit_flow=True)

    saved_samples = scene.cycles.samples
    scene.cycles.samples = 1

    track = _plan_camera_track(planner_shot, n_frames=num_frames)

    for i, frame in enumerate(track):
        _apply_camera_xform(cam_obj,
                            frame["position"],
                            frame["rotation_deg"],
                            frame["lens_mm"])
        scene.frame_set(i + 1)        # so OutputFile suffix increments
        scene.render.filepath = str(sub_out / f"_vace_frame_{i+1:04d}")
        bpy.ops.render.render(write_still=False)

    (sub_out / f"camera_track_{sid_with_cam}.json").write_text(
        json.dumps({
            "sid":       sid_with_cam,
            "n_frames":  num_frames,
            "track":     track,
            "passes": [
                {"kind": "depth", "format": "png",
                 "pattern": f"depth_{sid_with_cam}_####.png"},
                {"kind": "optical_flow", "format": "exr",
                 "pattern": f"flow_{sid_with_cam}_####.exr",
                 "encoding": "RGBA32F; RG=incoming motion (px), BA=outgoing motion (px)"},
                {"kind": "optical_flow_rgb", "format": "png",
                 "pattern": f"flow_rgb_{sid_with_cam}_####.png",
                 "encoding": "RGB8 with ±16px clamp; R=(out_dx+16)/32, G=(out_dy+16)/32, B=0"},
                {"kind": "normal", "format": "png",
                 "pattern": f"normal_{sid_with_cam}_####.png",
                 "encoding": "RGB8; surface normals in camera space"},
                {"kind": "segmentation", "format": "png",
                 "pattern": f"seg_{sid_with_cam}_####.png",
                 "encoding": "BW8; IndexOB mask for pass_index=1 (Subject_A)"},
            ],
        }, indent=2))

    scene.cycles.samples = saved_samples
    scene.frame_set(1)


def _set_visible(parent_obj, visible: bool):
    """Show/hide a humanoid (Empty parent + child meshes)."""
    if parent_obj is None:
        return
    parent_obj.hide_render   = not visible
    parent_obj.hide_viewport = not visible
    for child in parent_obj.children:
        child.hide_render   = not visible
        child.hide_viewport = not visible


def _mannequin_slot_positions(n: int, scene_name: str) -> list[tuple[float, float, float]]:
    """World (x, y, z) for each mannequin slot (index 0 = Subject_A, 1 =
    Subject_B), for a scene with `n` characters present. Single source of
    truth for both _position_mannequins (which moves the actual bpy
    objects) and the composition-solver wiring in _synthesize_shot (which
    needs the same positions without a live bpy scene to read them from).

    Camera looks along -X (camera_planner.plan_camera); local +X (right,
    verified against Blender's own mathutils.Euler.to_matrix — see
    composition_solver.camera_axes) maps to world +Y, so +Y is
    image-RIGHT, not image-left as an earlier version of this comment
    claimed.
    """
    z = 0.0 if scene_name in ("street", "apartment") else 0.40
    sep = {"street": 0.55, "cafe_interior": 0.4,
           "cafe_flashback": 0.4, "apartment": 0.35}.get(scene_name, 0.5)
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0, z)]
    return [(0.0, sep, z), (0.0, -sep, z)][:n]


def _descendant_meshes(root) -> list:
    """Every MESH at or under `root`, including the root when it is one."""
    out, stack = [], [root]
    while stack:
        o = stack.pop()
        if getattr(o, "type", None) == "MESH":
            out.append(o)
        stack.extend(getattr(o, "children", ()) or ())
    return out


def _subject_object_for(cid: str):
    """The staged body for one character, under either naming convention.

    Two conventions exist and never met. This module places bodies into the
    positional slots Subject_A / Subject_B / Subject_Ref that a mannequin
    template blend provides; scene_assembler.assemble() instead names each
    proxy after the character it is (add_body_proxy(cid, ...)), because a
    segmentation index has to be addressable by *who*, not by slot order.
    Scenes built by the assembler therefore contained no Subject_* object, so
    every mannequin lookup missed, nothing was made visible, and the IndexOB
    pass came back empty -- a composition measurement that reported ABSENT on
    every shot without anything being wrong with the camera or the solver.

    Resolution order is semantic first: a proxy named for the character is
    preferred over a positional slot, because that is the name a per-character
    mask can be attributed to.
    """
    if not cid:
        return None
    stem = str(cid).partition("@")[0].strip()
    for name in (stem, f"{stem}_proxy", f"proxy_{stem}"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            return obj
    # Case-insensitive sweep: the assembler lower-cases ids, a hand-built
    # template may not.
    low = stem.lower()
    for obj in bpy.data.objects:
        if obj.name.lower() == low or obj.name.lower().startswith(low + "_"):
            return obj
    return None


def _position_mannequins(characters: list, scene_name: str):
    """Show/hide and position the staged bodies for this shot's characters.

    Prefers a proxy named for the character (assembler convention); falls back
    to the Subject_A / Subject_B template slots.
    """
    named = [_subject_object_for(c if isinstance(c, str) else
                                 (c or {}).get("character_id", "")) for c in (characters or [])]
    if any(named):
        # Assembler-built scene: the bodies are already placed at their staged
        # world positions, so only visibility and identity are ours to set.
        #
        # pass_index is what makes a body *addressable*. A proxy imported from
        # OBJ carries index 0, which every IndexOB mask reads as background, so
        # a staged body that is plainly visible in the render contributes
        # nothing to a segmentation pass. Bodies reached previs geometry and
        # were still unmeasurable -- the specific shape of this project's
        # missing geometry channel. Slots 1 and 2 match the mannequin rig's own
        # convention so a consumer written for either sees the same indices.
        for slot, obj in enumerate(named, start=1):
            if obj is None:
                continue
            _set_visible(obj, True)
            # add_body_proxy() names the *parent Empty* after the character and
            # parents the imported mesh under it. An Empty renders nothing, so
            # setting pass_index on it leaves every mesh at index 0 and the
            # IndexOB mask empty -- visible bodies, blank segmentation. The
            # index has to reach the meshes.
            for mesh in _descendant_meshes(obj):
                _set_visible(mesh, True)
                mesh.pass_index = slot
        return

    obj_a = bpy.data.objects.get("Subject_A") or bpy.data.objects.get("Subject_Ref")
    obj_b = bpy.data.objects.get("Subject_B")

    _set_visible(obj_a, False)
    _set_visible(obj_b, False)

    n = len(characters or [])
    if n == 0:
        return

    slots = _mannequin_slot_positions(n, scene_name)
    for obj, pos in zip((obj_a, obj_b), slots):
        if obj is None:
            continue
        obj.location = mathutils.Vector(pos)
        _set_visible(obj, True)


def _flatten_panels(scine_doc: dict) -> list[dict]:
    """Flatten scine_doc.shots[].panels[] into one entry per real panel.

    The current (v1.1) KB schema nests panels under shots — there is no
    top-level "panels" list, and panels carry no "kind" field. Both were
    true of an earlier schema version this replaces; scine_doc.get(
    "panels", []) silently returns [] against every real production scene
    file today. Each entry pairs a panel with its parent shot so camera/
    setup override resolution and movement_io.shot_to_planner_dict (the
    adapter the trajectory API and LaMP endpoints already use) have what
    they need.
    """
    out: list[dict] = []
    for shot in scine_doc.get("shots", []):
        for panel in shot.get("panels", []):
            out.append({"panel": panel, "shot": shot, "shot_id": shot.get("shot_id")})
    return out


def _effective_shot(shot: dict, panel: dict) -> dict:
    """shot with any panel-level *_override applied (whole-block override,
    matching how the *_override fields are typed — same shape as the base
    field, Optional). Every real panel checked so far has null overrides
    (an authoring feature not yet exercised in production), so this is a
    no-op today but resolved correctly rather than ignored."""
    eff = dict(shot)
    for key in ("camera", "setup", "lighting", "events"):
        override = panel.get(f"{key}_override")
        if override is not None:
            eff[key] = override
    return eff


def _composition_for_panel(setup: dict, characters: list[str],
                           scene_id: str) -> dict | None:
    """Build shot.composition (subject_xy/target_x/target_y) for
    camera_planner's screen_position solver from a resolved setup's
    primary_focus + subjects[].screen_position.

    Grounds the primary subject at whichever mannequin slot (Subject_A/B)
    _position_mannequins will actually place it at — the solver needs the
    subject's TRUE position, not an assumed one, or the composed frame
    would not match what's actually rendered. Returns None (default
    centered aim, unchanged from before this mechanism existed) when
    there's no resolvable primary subject, it isn't in a mannequin slot
    (only 2 exist today), or its screen_position has neither a zone nor
    explicit (x, y).
    """
    from pace_core.setup.composition_solver import target_xy

    subjects = setup.get("subjects") or []
    if not subjects:
        return None

    primary_focus = setup.get("primary_focus") or {}
    of_character = (primary_focus.get("of_character") or "").split("@")[0]

    idx = next((i for i, s in enumerate(subjects)
               if s.get("character_id") == of_character), None) if of_character else None
    if idx is None:
        idx = 0   # no resolvable primary_focus -> compose around the first subject

    if idx >= min(2, len(characters)):
        return None   # only Subject_A/Subject_B exist; can't ground beyond slot 1

    target = target_xy(subjects[idx].get("screen_position"))
    if target is None:
        return None

    slots = _mannequin_slot_positions(min(2, len(characters)), scene_id)
    if idx >= len(slots):
        return None
    x, y, _z = slots[idx]
    return {"subject_xy": [x, y], "target_x": target[0], "target_y": target[1]}


def _synthesize_shot(entry: dict, scene_id: str, narrative_meta: dict) -> dict:
    """Build a shot dict that camera_planner understands from one flattened
    (panel, parent shot) entry (see _flatten_panels). Camera/movement
    resolution goes through movement_io.shot_to_planner_dict — the same
    canonical adapter the studio's trajectory API and LaMP endpoints
    already use — rather than a separate, drift-prone reimplementation.
    """
    from pace_core.camera.movement_io import read_movement, shot_to_planner_dict

    panel, shot = entry["panel"], entry["shot"]
    eff_shot = _effective_shot(shot, panel)

    scene_stub = {"narrative_meta": narrative_meta}
    planner_shot = shot_to_planner_dict(eff_shot, scene=scene_stub, scene_id=scene_id)
    movements, easing = read_movement(eff_shot)

    setup = eff_shot.get("setup") or {}
    characters = [s.get("character_id") for s in (setup.get("subjects") or [])
                 if s.get("character_id")]

    result = {
        "id":         panel.get("id") or f"{scene_id}__{entry.get('shot_id')}__P{panel.get('panel_number', 0):02d}",
        "scene_ref":  planner_shot.get("scene_ref") or scene_id,
        "characters": characters,
        "camera":     planner_shot["camera"],
        "frame": {
            "movement":        movements,
            "movement_easing": easing,
        },
    }
    composition = _composition_for_panel(setup, characters, scene_id)
    if composition is not None:
        result["composition"] = composition
    return result


def _kernel_passes(argv: list) -> None:
    """The depth/normal/seg pass-render kernel (formerly blender_render.py).
    Renders into --out and writes cameras.json. No RESULT_JSON — BlenderBox
    .render_passes verifies the depth PNG on disk."""
    global _plan_camera, _plan_cameras, _plan_camera_track
    from pace_core.camera.camera_planner import (
        plan_camera as _pc, plan_cameras as _pcs, plan_camera_track as _pct)
    _plan_camera, _plan_cameras, _plan_camera_track = _pc, _pcs, _pct

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pai",  required=True,
                        help="path to a PAI doc (kb/on_scene/scenes/scene_NN.json)")
    parser.add_argument("--out",    required=True,
                        help="output directory for depth/normal/seg PNGs")
    parser.add_argument("--ids",    nargs="*",
                        help="render only these panel IDs; default: all visual panels")
    parser.add_argument("--vace-frames", type=int, default=0,
                        help="when >0, render an N-frame depth video sequence per panel")
    parser.add_argument("--no-clean-depth", action="store_true",
                        help="skip the mannequin-hidden re-render")
    args = parser.parse_args(argv)

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    with open(args.pai) as f:
        scine_doc = json.load(f)

    scene_id       = scine_doc.get("scene_id", "")
    narrative_meta = scine_doc.get("narrative_meta") or {}
    panels = _flatten_panels(scine_doc)
    if args.ids:
        # Accepts either panel ids (scenes_panels.py's router) or a shot id
        # (BlenderBox.render_passes()'s --ids <shot_id>) — both call sites
        # exist today and expect their own id shape to work.
        panels = [e for e in panels
                 if e["panel"].get("id") in args.ids or e["shot_id"] in args.ids]

    print(f"[blender_render] scene={scene_id}  panels={[e['panel'].get('id') for e in panels]}"
          f"  vace_frames={args.vace_frames}")

    scene = bpy.context.scene
    scene.render.resolution_x          = 1280
    scene.render.resolution_y          = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format  = "PNG"
    scene.render.image_settings.color_mode  = "RGB"
    scene.render.image_settings.color_depth = "16"

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = False

    scene.view_layers["ViewLayer"].use_pass_z              = True
    scene.view_layers["ViewLayer"].use_pass_normal         = True
    scene.view_layers["ViewLayer"].use_pass_object_index   = True
    scene.view_layers["ViewLayer"].use_pass_vector         = True

    camera_log: dict = {}
    bible      = _load_bible(scene_id)

    for entry in panels:
        pid     = entry["panel"].get("id")
        shot    = _synthesize_shot(entry, scene_id, narrative_meta)

        cam_obj = bpy.data.objects.get("Camera")
        if cam_obj is None:
            raise RuntimeError("No object named 'Camera' in scene .blend")

        cam_obj.data.clip_start = 0.01
        cam_obj.data.clip_end   = 15.0    # caps depth so Normalize has no infinite sky

        plans = _plan_cameras(shot, bible)
        if not isinstance(plans, list):
            plans = [plans]
        is_multi_cam = len(plans) > 1

        _position_mannequins(shot["characters"], scene_id)

        log_entry = {"cameras": []}
        for plan in plans:
            cam_id    = plan.get("cam_id", "A")
            cam_suffix = "" if not is_multi_cam else f"_cam{cam_id}"
            sid_cam   = f"{pid}{cam_suffix}"

            # ── Canonical (multi-pass: depth+normal+seg) ──
            _apply_camera_plan(cam_obj, plan)
            _setup_compositor(scene, OUT, sid_cam)
            scene.render.filepath = str(OUT / f"_render_{sid_cam}")
            bpy.ops.render.render(write_still=False)

            # ── Clean depth re-render: mannequins HIDDEN ──
            if not args.no_clean_depth:
                _set_visible(bpy.data.objects.get("Subject_A"), False)
                _set_visible(bpy.data.objects.get("Subject_B"), False)
                _setup_depth_only_compositor(scene, OUT, f"{sid_cam}__clean")
                scene.render.filepath = str(OUT / f"_render_{sid_cam}__clean")
                bpy.ops.render.render(write_still=False)
                clean_src = OUT / f"depth_{sid_cam}__clean_0001.png"
                clean_dst = OUT / f"depth_{sid_cam}_0001.png"
                if clean_src.exists():
                    clean_src.replace(clean_dst)
                _position_mannequins(shot["characters"], scene_id)

            cam_log = {
                "cam_id":     cam_id,
                "lens_mm":    cam_obj.data.lens,
                "fov_deg":    math.degrees(cam_obj.data.angle),
                "distance":   plan["subject_distance_m"],
                "location":   list(cam_obj.location),
                "rotation":   list(cam_obj.rotation_euler),
            }

            # ── Movement: render END frame too (multi-pass) ──
            mv = plan.get("movement")
            if mv:
                _apply_camera_xform(cam_obj, mv["end_position"],
                                    mv["end_rotation_deg"], mv["end_lens_mm"])
                _setup_compositor(scene, OUT, f"{sid_cam}_end")
                scene.render.filepath = str(OUT / f"_render_{sid_cam}_end")
                bpy.ops.render.render(write_still=False)
                cam_log["movement"] = {
                    "kind":         mv["kind"],
                    "end_lens_mm":  cam_obj.data.lens,
                    "end_location": list(cam_obj.location),
                    "end_rotation": list(cam_obj.rotation_euler),
                }
                print(f"[{pid}] cam={cam_id} movement={mv['kind']} "
                      f"→ depth_{sid_cam}_0001.png + depth_{sid_cam}_end_0001.png")
            else:
                print(f"[{pid}] cam={cam_id} → {OUT}/depth_{sid_cam}_0001.png")

            # ── VACE depth sequence: N eased keyframes via plan_camera_track ──
            if args.vace_frames > 0:
                num_frames = args.vace_frames
                planner_shot_for_track = _synthesize_shot(entry, scene_id, narrative_meta)
                label = (mv.get("kinds") or [mv.get("kind")]) if mv else ["static"]
                ease  = (mv or {}).get("easing", "linear")
                print(f"[{pid}] cam={cam_id} VACE: {num_frames} frames "
                      f"({'+'.join(label)} easing={ease})")
                _render_vace_sequence(cam_obj, scene, planner_shot_for_track,
                                       OUT, sid_cam, num_frames)
                cam_log["vace_frames"] = num_frames
                cam_log["vace_dir"]    = str(OUT / f"vace_{sid_cam}")
                cam_log["vace_easing"] = ease
                cam_log["vace_kinds"]  = list(label)
            log_entry["cameras"].append(cam_log)

        # Backward-compat flat fields for single-cam consumers
        if not is_multi_cam and log_entry["cameras"]:
            first = log_entry["cameras"][0]
            for k in ("lens_mm", "fov_deg", "distance", "location", "rotation", "movement"):
                if k in first:
                    log_entry[k] = first[k]
        camera_log[pid] = log_entry

    with open(OUT / "cameras.json", "w") as f:
        json.dump(camera_log, f, indent=2)
    print(f"Done — {len(panels)} panels rendered → {OUT}/cameras.json")


def _kernel_dispatch() -> None:
    """Blender entrypoint. Parse `-- --kernel <name> …` and run the kernel."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    kernel = None
    if "--kernel" in argv:
        ki = argv.index("--kernel")
        kernel = argv[ki + 1]
        argv = argv[:ki] + argv[ki + 2:]
    if kernel == "convert":
        _kernel_convert(argv)
    elif kernel == "list_parts":
        _kernel_list_parts(argv)
    elif kernel == "passes":
        _kernel_passes(argv)
    else:
        raise SystemExit(f"blender_box kernel: unknown --kernel {kernel!r}")


if __name__ == "__main__":
    _kernel_dispatch()
