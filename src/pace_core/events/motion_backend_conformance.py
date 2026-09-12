"""Acceptance gate for a MotionBackend.

Swapping the rig's deterministic core for a learned motion prior is the
point of the `MotionBackend` seam, but not every model is admissible. In
previs the authored pose *is* the specification — a shot says the arm
reaches here, and a model that lands somewhere plausible-but-different has
not produced a variation, it has ignored an instruction. CondMDI is
criticized in the follow-up literature for exactly this ("weak condition
constraint results in significant keyframe errors"), so the property has to
be measured rather than assumed before any backend is wired in.

This module runs a fixed battery against any backend and reports pass/fail
per property. It is deliberately backend-agnostic and imports nothing
model-specific: the same battery gates the procedural default, a diffusion
model, or anything later.

    from pace_core.events.motion_backend_conformance import conformance_report
    print(conformance_report(MyBackend())["ok"])

The battery is not a benchmark. It does not score motion quality, which is
not measurable here and is the model author's concern; it checks the
contract the rig's callers already rely on.
"""
from __future__ import annotations

import math
from typing import Optional

from pace_core.events.motion_rig import ROOT_KEY, angle_between

# A pose the caller authored must be reproduced this closely. One degree is
# far looser than the procedural backend's exact hit and still tight enough
# that a viewer would not see the difference at previs fidelity.
KEYFRAME_TOL_DEG = 1.0


def _qz(deg: float) -> tuple:
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


# Fixed cases, chosen to span the regimes the rig actually meets: a
# comfortable two-key move, a multi-key sequence, and one with root motion.
_CASES = [
    ("two keys, comfortable", 33,
     [(0, {"j": _qz(0)}), (32, {"j": _qz(70)})]),
    ("four keys, varied rhythm", 49,
     [(0, {"j": _qz(0)}), (8, {"j": _qz(-30)}),
      (24, {"j": _qz(55)}), (48, {"j": _qz(10)})]),
    ("root translation", 25,
     [(0, {"j": _qz(0), ROOT_KEY: [0.0, 0.0, 0.0]}),
      (24, {"j": _qz(40), ROOT_KEY: [1.5, 0.0, 0.0]})]),
]


def _worst_key_error(poses: list, keys: list) -> tuple:
    """Largest angular and positional error at any authored key."""
    wa = wp = 0.0
    for f, want in keys:
        if f >= len(poses):
            return float("inf"), float("inf")
        got = poses[f]
        for b, q in want.items():
            if b == ROOT_KEY:
                if ROOT_KEY in got:
                    wp = max(wp, math.dist(got[ROOT_KEY], q))
            elif b in got:
                wa = max(wa, angle_between(got[b], q))
            else:
                return float("inf"), float("inf")   # dropped a bone
    return wa, wp


def conformance_report(backend, *, tol_deg: float = KEYFRAME_TOL_DEG,
                       tol_m: float = 0.01) -> dict:
    """Run the battery. Returns {"ok", "checks": [{name, ok, detail}], …}."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    worst_ang = worst_pos = 0.0
    length_ok = bones_ok = True
    for label, n, keys in _CASES:
        try:
            poses = backend.infill(list(keys), n, seed=0)
        except TypeError:
            poses = backend.infill(list(keys), n, seed=0, seed_pos=0)
        if len(poses) != n:
            length_ok = False
            add(f"length · {label}", False, f"returned {len(poses)}, expected {n}")
            continue
        wa, wp = _worst_key_error(poses, keys)
        if wa == float("inf"):
            bones_ok = False
        worst_ang, worst_pos = max(worst_ang, wa), max(worst_pos, wp)

    add("returns the requested frame count", length_ok, "one pose per frame")
    add("preserves every authored bone", bones_ok,
        "no bone present in a key may be missing from the output")
    # THE gate. Everything else is hygiene; this is the one that decides
    # whether a model may drive a shot at all.
    add("hits authored poses", worst_ang <= tol_deg and worst_pos <= tol_m,
        f"worst {worst_ang:.4f}° / {worst_pos:.4f} m "
        f"(tolerance {tol_deg}° / {tol_m} m)")

    # Determinism and variation, on the middle case. Compared over whatever
    # all three runs actually returned: a backend that returns the wrong
    # length has already failed above, and the gate must survive it rather
    # than crash — a harness that raises on a malformed backend reports
    # nothing, which is worse than reporting a rejection.
    _, n, keys = _CASES[1]
    a1 = backend.infill(list(keys), n, seed=5)
    a2 = backend.infill(list(keys), n, seed=5)
    b1 = backend.infill(list(keys), n, seed=6)
    m = min(len(a1), len(a2), len(b1))
    same = max((angle_between(a1[i]["j"], a2[i]["j"])
                for i in range(m) if "j" in a1[i] and "j" in a2[i]), default=0.0)
    diff = max((angle_between(a1[i]["j"], b1[i]["j"])
                for i in range(m) if "j" in a1[i] and "j" in b1[i]), default=0.0)
    add("same seed reproduces the take", same < 1e-6, f"max drift {same:.2e}°")
    add("different seed varies the take", diff > 1e-3, f"max difference {diff:.4f}°")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks,
            "worst_key_error_deg": worst_ang, "worst_key_error_m": worst_pos,
            "backend": type(backend).__name__}


