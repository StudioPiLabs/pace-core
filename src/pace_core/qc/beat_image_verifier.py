"""Did the delivered panel put the subject where the geometry staged it?

The design's image verifier compares a generated frame against the 3D anchor
that produced it. The full form needs a detector to find the subject in the
delivered pixels. This is the part that needs no detector at all, and it works
because the greybox and the panel share a camera AND an init latent: the panel
is denoised from the greybox at 0.55, so wherever the geometry put a body,
structure in the delivered frame should agree with the greybox's structure.

Every measurement here carries its own control. `anchor_margin` compares
alignment inside the staged silhouette against alignment when the same mask is
SHIFTED sideways across the frame. A number with no control is not evidence: if
a panel scored 0.62 aligned and 0.61 shifted, the metric is reading the room,
not the subject. The margin is what says the subject is where it was staged.
"""
from __future__ import annotations

import numpy as np

RESIZE = (256, 109)          # matches consistency_metric's aspect
EDGE_DENSITY = 0.20


def _edges(a: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(a, axis=1))[:-1, :]
    gy = np.abs(np.diff(a, axis=0))[:, :-1]
    g = gx + gy
    return g > np.quantile(g, 1.0 - EDGE_DENSITY)


def _iou(x, y) -> float:
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 0.0


def anchor_alignment(greybox: np.ndarray, delivered: np.ndarray,
                     mask: np.ndarray) -> float:
    """Edge agreement between the two frames, restricted to a region."""
    eg, ed = _edges(greybox), _edges(delivered)
    m = mask[:-1, :-1] if mask.shape != eg.shape else mask
    if m.sum() < 16:
        return 0.0
    return _iou(eg & m, ed & m)


def anchor_margin(greybox: np.ndarray, delivered: np.ndarray,
                  mask: np.ndarray, shifts=(-64, -32, 32, 64)) -> dict:
    """Alignment where the body was staged, against the same mask elsewhere.

    A positive margin says the delivered structure follows the staged
    silhouette specifically, rather than the panel simply being busy.
    """
    at = anchor_alignment(greybox, delivered, mask)
    ctrl = []
    for s in shifts:
        shifted = np.roll(mask, s, axis=1)
        # A shift that wraps the body onto itself is not a control.
        if (shifted & mask).sum() > 0.35 * mask.sum():
            continue
        ctrl.append(anchor_alignment(greybox, delivered, shifted))
    if not ctrl:
        return {"aligned": round(at, 4), "control": None, "margin": None,
                "note": "no disjoint shift available; subject too wide to control"}
    c = float(np.mean(ctrl))
    return {"aligned": round(at, 4), "control": round(c, 4),
            "margin": round(at - c, 4), "n_controls": len(ctrl)}


# ── binding detections to the bodies the geometry staged ──────────────────
#
# A detector returns boxes with no idea who is in them. Sorting them
# left-to-right and zipping against the staged order looks like an answer and
# is not one: on scene_02 the detector returned 4 and 5 boxes for 3 subjects
# (a car interior shows people on its screens and in its glass), so position
# order silently paired a subject with a reflection and widened the measured
# spread.
#
# Binding by overlap with the staged silhouette fixes that AND is what makes a
# per-subject identity check possible at all: the greybox knows which body is
# ryan, so a box bound to ryan's matte is the region an identity check should
# run on. Presence, position and identity stay three separate claims.


def _bbox_of(mask) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) in FRACTIONS of the frame, or None if empty."""
    import numpy as np
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def _box_iou(a, b) -> float:
    """Rectangle IoU. Named apart from the mask `_iou` above -- defining a
    second `_iou` silently replaced it, and anchor_alignment then handed two
    boolean ARRAYS to a function doing max() on their first elements."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def bind_detections(detections: list[dict], staged: dict, *,
                    min_iou: float = 0.10) -> dict:
    """Match each detection to the staged body it overlaps most.

    Greedy on IoU and one-to-one: a detection may claim at most one subject
    and a subject at most one detection, so a duplicate box beside a real one
    becomes an EXTRA rather than a second opinion about the same person.

    `staged` maps subject id -> boolean mask at any resolution; detections
    carry `box` in pixels plus the frame `size`, or `box_frac` already
    normalised.
    """
    boxes = []
    for i, d in enumerate(detections):
        bf = d.get("box_frac")
        if bf is None:
            w, h = d.get("size") or (0, 0)
            if not (w and h):
                continue
            x0, y0, x1, y1 = d["box"]
            bf = (x0 / w, y0 / h, x1 / w, y1 / h)
        boxes.append((i, tuple(bf)))

    sb = {k: _bbox_of(m) for k, m in staged.items()}
    sb = {k: v for k, v in sb.items() if v}

    pairs = sorted(((_box_iou(bf, s), i, k) for i, bf in boxes for k, s in sb.items()),
                   key=lambda t: -t[0])
    used_d, used_s, matched = set(), set(), {}
    for score, i, k in pairs:
        if score < min_iou or i in used_d or k in used_s:
            continue
        used_d.add(i); used_s.add(k)
        bf = dict(boxes)[i]
        matched[k] = {"detection": i, "iou": round(score, 4),
                      "delivered_cx": round((bf[0] + bf[2]) / 2, 4),
                      "staged_cx": round((sb[k][0] + sb[k][2]) / 2, 4),
                      "box_frac": [round(v, 4) for v in bf]}
    for k, v in matched.items():
        v["shift"] = round(v["delivered_cx"] - v["staged_cx"], 4)
    return {
        "matched": matched,
        # A box nothing staged: a reflection, a figure on a screen, a
        # duplicate. Reported, never counted as a subject.
        "extra_detections": [i for i, _ in boxes if i not in used_d],
        # A body the geometry staged and the detector did not find. This is
        # the one that means the panel is wrong, not the measurement.
        "missing_subjects": [k for k in sb if k not in used_s],
    }
