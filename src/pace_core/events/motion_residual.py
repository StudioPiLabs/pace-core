"""A learned residual over a deterministic base, so keyframe accuracy is
structural rather than tested for.

Both papers this rig follows decompose motion the same way: Goel et al.
emit a time warp *plus spatial residuals*, and the blocking-poses work
refines the input conditions around a retimer it does not replace. This
module takes that decomposition literally.

The deterministic core (`motion_rig.ProceduralBackend`) produces a base
track that hits every authored pose exactly. A learned model then predicts
a per-frame, per-bone *delta* on top, and that delta is multiplied by a
mask that is exactly zero at every keyframe. Two consequences follow, and
they are the reason to build it this way round:

  * A residual model **cannot** miss an authored pose. Not "is unlikely
    to" — the mask makes it arithmetically impossible, so the property the
    conformance gate exists to check is guaranteed by construction rather
    than by evaluation. That matters because the criticism levelled at
    conditioning-by-masking models (CondMDI's "weak condition constraint
    results in significant keyframe errors") is a property of the approach,
    not of any one implementation.
  * The model has a far easier job. It learns what a body does *between*
    poses, not what motions exist, so it needs orders of magnitude less
    data than a text-to-motion prior and can be trained on takes we own
    rather than on a research-licensed corpus.

Nothing here is trained. `ResidualBackend` with no model is exactly the
procedural backend, which is the honest default; `ResidualModel` is the
seam a torch module drops into. The mask, the composition, and the
guarantee are what this module provides, and they are testable now.
"""
from __future__ import annotations

import math
from typing import Optional, Protocol, Sequence

from pace_core.events.motion_rig import (ROOT_KEY, Pose, ProceduralBackend, slerp)


class ResidualModel(Protocol):
    """Predicts per-frame corrections to a base track.

    `base` is one pose per frame; `key_frames` are the frames the caller
    authored. The return is one delta per frame: for each bone a small
    rotation to compose onto the base, and optionally a root offset in
    metres. Deltas at or near keyframes are ignored by the caller, so a
    model need not learn to suppress them there.
    """

    def predict(self, base: list[Pose], key_frames: Sequence[int]) -> list[dict]: ...


def key_anchor_weights(key_frames: Sequence[int], n_frames: int, *,
                       falloff: int = 6) -> list[float]:
    """Zero at every keyframe, easing to one `falloff` frames away.

    This is the whole guarantee. A residual is scaled by this before it is
    applied, so at an authored frame the delta is multiplied by exactly
    0.0 and the base pose survives bit-for-bit. The ease is a raised
    cosine rather than a linear ramp so the residual arrives without a
    crease at the seam — the same reason the editing path blends.
    """
    if n_frames <= 0:
        return []
    keys = sorted({int(k) for k in key_frames if 0 <= int(k) < n_frames})
    if not keys:
        return [1.0] * n_frames
    out = []
    for f in range(n_frames):
        d = min(abs(f - k) for k in keys)
        if d == 0:
            out.append(0.0)
        elif d >= falloff:
            out.append(1.0)
        else:
            out.append(0.5 - 0.5 * math.cos(math.pi * d / falloff))
    return out


def _clamp_quat(q: Sequence[float], max_deg: float) -> tuple:
    """Limit a delta rotation's magnitude. A residual is detail, not a
    second animator: an unbounded delta could rewrite the motion between
    keys into something the author never asked for, which is the failure
    mode that makes learned motion untrustworthy in a shot."""
    w = max(-1.0, min(1.0, float(q[0])))
    ang = math.degrees(2.0 * math.acos(abs(w)))
    if ang <= max_deg or ang < 1e-9:
        return tuple(q)
    return slerp((1.0, 0.0, 0.0, 0.0), q, max_deg / ang)


def _compose(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Quaternion product a*b — apply delta `b` in `a`'s frame."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def apply_residual(base: list[Pose], deltas: list[dict], weights: Sequence[float],
                   *, max_deg: float = 25.0, max_root_m: float = 0.25) -> list[Pose]:
    """Compose weighted, clamped deltas onto a base track."""
    out: list[Pose] = []
    for f, pose in enumerate(base):
        w = weights[f] if f < len(weights) else 1.0
        d = deltas[f] if f < len(deltas) else {}
        if w <= 0.0 or not d:
            out.append(dict(pose))
            continue
        merged: Pose = {}
        for bone, q in pose.items():
            if bone == ROOT_KEY:
                off = d.get(ROOT_KEY)
                if off:
                    scale = min(1.0, max_root_m / max(1e-9, math.dist([0, 0, 0], off)))
                    merged[bone] = [q[i] + off[i] * scale * w for i in range(3)]
                else:
                    merged[bone] = list(q)
                continue
            dq = d.get(bone)
            if not dq:
                merged[bone] = q
                continue
            # Scale the delta by the anchor weight, then clamp, then compose.
            eased = slerp((1.0, 0.0, 0.0, 0.0), _clamp_quat(dq, max_deg), w)
            merged[bone] = _compose(q, eased)
        out.append(merged)
    return out


class ResidualBackend:
    """Deterministic base + optional learned residual.

    With `model=None` this is precisely `ProceduralBackend`, which is the
    correct behaviour before anything is trained: the pipeline keeps
    working and says nothing untrue about what produced the motion.
    """

    def __init__(self, model: Optional[ResidualModel] = None, *,
                 base: Optional[object] = None, falloff: int = 6,
                 max_deg: float = 25.0, max_root_m: float = 0.25):
        self.model = model
        self.base = base or ProceduralBackend()
        self.falloff = falloff
        self.max_deg = max_deg
        self.max_root_m = max_root_m

    def infill(self, keys: list, n_frames: int, *, seed: int = 0,
               seed_pos: int = 0) -> list[Pose]:
        try:
            base = self.base.infill(keys, n_frames, seed=seed, seed_pos=seed_pos)
        except TypeError:                       # a base that predates the split
            base = self.base.infill(keys, n_frames, seed=seed)
        if self.model is None:
            return base
        key_frames = [int(f) for f, _ in keys]
        deltas = self.model.predict(base, key_frames)
        w = key_anchor_weights(key_frames, n_frames, falloff=self.falloff)
        return apply_residual(base, deltas, w,
                              max_deg=self.max_deg, max_root_m=self.max_root_m)
