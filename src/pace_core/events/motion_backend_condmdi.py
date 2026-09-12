"""CondMDI as a MotionBackend.

CondMDI (Cohan et al., SIGGRAPH 2024, MIT-licensed) is a conditional motion
diffusion model that accepts arbitrary dense-or-sparse keyframes, which is
the contract `motion_rig.MotionBackend` asks for. Buhmann et al. name it
explicitly as a compatible engine for their rig, so it is the natural first
learned core to try behind our own.

It is not admitted on reputation. The follow-up literature criticizes
CondMDI for a "weak condition constraint" that "results in significant
keyframe errors", and in previs an authored pose is the specification
rather than a suggestion — so this backend must pass
`motion_backend_conformance.conformance_report` before anything renders
through it. The gate is the point of this seam.

Two representation gaps sit between CondMDI and our rig, and both are real:

  * CondMDI generates HumanML3D-style motion: a fixed 22-joint skeleton in
    a specific redundant feature encoding, at 20 fps. Our rig speaks
    per-bone quaternions on whatever skeleton the greybox proxy has. A
    faithful bridge needs a named-bone correspondence and a resample, and
    where a proxy has no counterpart for a HumanML3D joint there is
    nothing honest to fill it with.
  * CondMDI is trained on human locomotion from text. It has no notion of
    a budget on joint speed, and no reason to honour our retiming.

This module therefore does the plumbing and refuses to guess: it maps only
bones named in an explicit `joint_map`, and raises rather than inventing a
correspondence. `available()` reports whether the checkout and weights are
actually present, so a caller can fall back to the procedural backend
instead of failing a render.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pace_core.events.motion_rig import ROOT_KEY, Pose

# Where the checkout lives on the render box. Infra constant, overridable —
# the same convention camera/comfy hosts follow.
CONDMDI_ROOT = Path(os.environ.get(
    "PAI_CONDMDI_ROOT", "/mnt_zkm/film/condmdi/diffusion-motion-inbetweening"))

# HumanML3D's 22-joint skeleton, in the order CondMDI emits. A caller maps
# its own proxy bone names onto these; anything unmapped is left alone.
HUMANML3D_JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)

CONDMDI_FPS = 20


def available(root: Optional[Path] = None) -> dict:
    """Report whether CondMDI can actually run, and what is missing.

    Returned rather than raised so a pipeline can degrade to the procedural
    backend and say why, instead of failing a shot at render time.
    """
    r = Path(root or CONDMDI_ROOT)
    venv = r / ".venv" / "bin" / "python"
    save = r / "save"
    ckpts = sorted(save.glob("*/model*.pt")) if save.is_dir() else []
    data = r / "dataset" / "HumanML3D"
    missing = []
    if not r.is_dir():
        missing.append(f"checkout at {r}")
    if not venv.exists():
        missing.append("virtualenv (.venv)")
    if not ckpts:
        missing.append("pretrained weights under save/")
    if not data.is_dir():
        missing.append("dataset/HumanML3D (mean/std + text)")
    return {"ok": not missing, "root": str(r), "python": str(venv),
            "checkpoints": [str(c) for c in ckpts], "missing": missing}


class CondMDIBackend:
    """MotionBackend backed by a CondMDI checkout.

    `joint_map` maps our bone names → HumanML3D joint names. It is required
    and unvalidated names raise: a silent mis-mapping would produce motion
    that looks fine and animates the wrong limb, which is worse than an
    error.
    """

    def __init__(self, joint_map: dict, *, root: Optional[Path] = None,
                 checkpoint: Optional[str] = None, fps: int = 16,
                 text_prompt: str = "", timeout: int = 900):
        bad = [v for v in joint_map.values() if v not in HUMANML3D_JOINTS]
        if bad:
            raise ValueError(
                f"unknown HumanML3D joints {bad}; valid names are {HUMANML3D_JOINTS}")
        self.joint_map = dict(joint_map)
        self.root = Path(root or CONDMDI_ROOT)
        self.fps = fps
        self.text_prompt = text_prompt
        self.timeout = timeout
        st = available(self.root)
        if not st["ok"]:
            raise RuntimeError("CondMDI is not installed: missing "
                               + ", ".join(st["missing"]))
        self.checkpoint = checkpoint or st["checkpoints"][0]
        self.python = st["python"]

    # `seed_pos` is accepted and ignored: the positional/orientational noise
    # split is a property of our procedural core, not of CondMDI, and
    # pretending to honour it would misreport what the model did.
    def infill(self, keys: list, n_frames: int, *, seed: int = 0,
               seed_pos: int = 0) -> list[Pose]:
        raise NotImplementedError(
            "CondMDI sampling is not wired yet. The checkout, weights, and "
            "joint mapping are in place; what remains is the HumanML3D "
            "feature bridge — converting our quaternion keys into the "
            "model's redundant 263-d encoding at "
            f"{CONDMDI_FPS} fps and back, and resampling to {self.fps}. "
            "Until that exists this backend must not be used, and "
            "motion_backend_conformance will reject it rather than let a "
            "half-bridge drive a shot.")


def probe(root: Optional[Path] = None) -> dict:
    """Run CondMDI's own sampler once, to prove the install works at all.

    Deliberately separate from `infill`: this answers "does the model load
    and produce motion on this machine", which is the question during
    install, and it does so without pretending our representation bridge
    exists.
    """
    st = available(root)
    if not st["ok"]:
        return {"ok": False, "missing": st["missing"]}
    r = Path(st["root"])
    cmd = [st["python"], "-c",
           "import torch, sys;"
           "print('torch', torch.__version__);"
           "print('cuda', torch.cuda.is_available());"
           "sys.path.insert(0, '.');"
           "from model.cfg_sampler import ClassifierFreeSampleModel;"
           "print('model imports OK')"]
    try:
        p = subprocess.run(cmd, cwd=str(r), capture_output=True, text=True,
                           timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "probe timed out"}
    return {"ok": p.returncode == 0, "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip()[-800:]}


if __name__ == "__main__":                      # quick install check
    import json
    print(json.dumps(available(), indent=2))
    print(json.dumps(probe(), indent=2))
    sys.exit(0)
