"""PR-2 tier-coverage completion test.

Context
-------
PR-1 aligned every FIELD_TIER path that was *already present* to the
pace-0.2 registry (0 mismatch). But 24 registry paths with status=active
were not yet listed in FIELD_TIER at all. PR-2 closes the lowest-risk
slice of that gap: the 6 paths whose dataclass field already exists in
types_v1 and only needed a FIELD_TIER entry (pure-additive, no new
struct, no rename):

    camera.trajectory.static          -> recommended
    setup.subjects[].cls              -> recommended
    setup.secondary_subjects          -> recommended
    setup.text_generation             -> advanced   (container of [].* leaves)
    events.change_in_environment      -> recommended
    lighting.notes                    -> advanced

This test pins (1) those 6 entries to the registry tier, (2) that each
maps to a real dataclass field (no phantom tier rows), and (3) a global
guard that FIELD_TIER never drifts from the registry on any shared path.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from pace_core import types_v1 as T
from pace_core.types_v1 import FIELD_TIER

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "src" / "pace_core" / "pace_fields.json"

# the 6 paths this PR adds, with the dataclass field that backs each
PR2_COVERAGE = {
    "camera.trajectory.static": (T.CameraTrajectory, "static"),
    "setup.subjects[].cls": (T.Subject, "cls"),
    "setup.secondarySubjects": (T.Setup, "secondary_subjects"),
    "setup.textGeneration": (T.Setup, "text_generation"),
    "events.changeInEnvironment": (T.Events, "change_in_environment"),
    "lighting.notes": (T.Lighting, "notes"),
}


def _registry_tiers() -> dict[str, str]:
    """path -> tier from the pace-0.2 registry copy.

    The registry is a ``{"fields": [ {path, tier, status, ...}, ... ]}``
    list-of-records; assert that shape rather than carrying untested
    fallbacks so a future restructure fails loudly here.
    """
    reg = json.loads(REGISTRY.read_text())
    records = reg["fields"] if isinstance(reg, dict) else reg
    assert isinstance(records, list) and records and isinstance(records[0], dict), (
        "unexpected registry shape; expected a list of field records"
    )
    return {it["path"]: it.get("tier") for it in records}


@pytest.mark.parametrize("path", sorted(PR2_COVERAGE))
def test_pr2_path_registered_at_registry_tier(path: str) -> None:
    """Each PR-2 path is in FIELD_TIER at exactly the registry's tier."""
    reg = _registry_tiers()
    assert path in FIELD_TIER, f"{path} not registered in FIELD_TIER"
    assert path in reg, f"{path} unexpectedly absent from registry"
    assert FIELD_TIER[path] == reg[path], (
        f"{path}: FIELD_TIER={FIELD_TIER[path]} != registry={reg[path]}"
    )


@pytest.mark.parametrize("path", sorted(PR2_COVERAGE))
def test_pr2_path_backed_by_dataclass_field(path: str) -> None:
    """No phantom tier rows: each added path maps to a real dataclass field
    (this PR is tier-only — the field must already exist)."""
    cls, field_name = PR2_COVERAGE[path]
    assert field_name in {f.name for f in fields(cls)}, (
        f"{path} -> {cls.__name__}.{field_name} does not exist"
    )


def test_field_tier_matches_registry_on_shared_paths() -> None:
    """Guard against *tier drift*: for every path present in BOTH FIELD_TIER
    and the registry, the tier must match.

    Scope note: this only covers the intersection. It deliberately does NOT
    assert FIELD_TIER is complete — ~18 active registry paths are still
    unlisted (camera.fps/frame_range/program, manifest.*, semantics.*, …),
    tracked as the remaining PR-3+ coverage work. So a green run here means
    'no wrong tiers', not 'every active field is registered'.
    """
    reg = _registry_tiers()
    mismatches = {
        k: (FIELD_TIER[k], reg[k]) for k in FIELD_TIER if k in reg and FIELD_TIER[k] != reg[k]
    }
    assert not mismatches, f"FIELD_TIER drifted from registry: {mismatches}"


def test_field_tier_has_no_duplicate_intent() -> None:
    """FIELD_TIER is a dict literal; a duplicate key would silently shadow.
    Re-parse the source to confirm the literal has no repeated keys."""
    import ast

    src = (REPO_ROOT / "src" / "pace_core" / "types_v1.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "FIELD_TIER":
            keys = [ast.literal_eval(k) for k in node.value.keys]
            dups = sorted({k for k in keys if keys.count(k) > 1})
            assert not dups, f"duplicate FIELD_TIER keys: {dups}"
            return
    pytest.fail("FIELD_TIER literal not found in types_v1.py")
