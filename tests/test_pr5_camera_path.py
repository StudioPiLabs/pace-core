"""PR-5 camera_path test — last active visual-pillar field.

Context
-------
camera.trajectory.camera_path is the artifact reference (assets:// URI) to
the computed 6-DoF keyframe JSON — the PRODUCT side of camera.program's
LAMP recipe (which PR-3 already registered). It is status=active /
tier=recommended in the registry; PR-3/PR-4 deferred it only to keep their
scope tight, not because it was gated. Adding it makes FIELD_TIER cover
EVERY active field of the 4 visual pillars (camera/setup/lighting/events).

Purely additive: CameraTrajectory.camera_path: Optional[str] = None.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from pace_core.types_v1 import FIELD_TIER, CameraTrajectory

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "src" / "pace_core" / "pace_fields.json"
PATH = "camera.trajectory.cameraPath"
VISUAL_PILLARS = ("camera", "setup", "lighting", "events")


def _registry() -> dict[str, dict]:
    reg = json.loads(REGISTRY.read_text())
    recs = reg["fields"] if isinstance(reg, dict) else reg
    return {it["path"]: it for it in recs}


def test_camera_path_tier_matches_registry_and_active() -> None:
    reg = _registry()
    assert FIELD_TIER.get(PATH) == "recommended"
    assert PATH in reg and reg[PATH].get("tier") == "recommended"
    assert reg[PATH].get("status") == "active"


def test_camera_path_backed_by_dataclass_field() -> None:
    assert "camera_path" in {f.name for f in fields(CameraTrajectory)}


def test_camera_path_additive() -> None:
    assert CameraTrajectory().camera_path is None
    assert CameraTrajectory(camera_path="assets://shot/cam.6dof.json").camera_path == (
        "assets://shot/cam.6dof.json"
    )


def test_all_active_visual_pillar_fields_covered() -> None:
    """Milestone guard: after PR-5, FIELD_TIER covers EVERY active field of the
    four visual pillars. If a future sync adds a new active visual-pillar
    field, this goes red — surfacing the gap instead of silent drift.
    (manifest.* / semantics.* are separate non-visual pillars, out of scope.)"""
    reg = _registry()
    missing = [
        p for p, it in reg.items()
        if it.get("status") == "active"
        and p.split(".")[0] in VISUAL_PILLARS
        and p not in FIELD_TIER
    ]
    assert not missing, f"active visual-pillar fields not in FIELD_TIER: {missing}"

# NOTE: the global FIELD_TIER-vs-registry no-drift invariant is covered once in
# test_pr2_tier_coverage.test_field_tier_matches_registry_on_shared_paths — not
# duplicated here.
