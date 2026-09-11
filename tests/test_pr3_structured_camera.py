"""PR-3 structured-camera fields test.

Context
-------
PR-1 added pure-scalar fields; PR-2 registered already-modeled fields into
FIELD_TIER. PR-3 adds the 3 genuinely-missing *structured* camera fields
from the pace-0.2 registry (all status=active), which need real dataclasses:

    camera.fps          -> FrameRate {num, denom}  (rational, floats banned)   recommended
    camera.frame_range  -> list[int] [start, end], both >= 0                    recommended
    camera.program      -> CameraProgram {lamp_dsl, source, narrative}          advanced

`camera.program` originated as PAILang's own studio field (studio writes
shot.camera.program = {lamp_dsl, source, narrative}); this PR gives it a
faithful dataclass. Purely additive: all 3 Camera fields default to None,
nothing constructs Camera from data, so existing scene docs are unaffected.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from pace_core import types_v1 as T
from pace_core.types_v1 import Camera, CameraProgram, FrameRate, FIELD_TIER

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "src" / "pace_core" / "pace_fields.json"

PR3_TIERS = {
    "camera.fps": "recommended",
    "camera.frameRange": "recommended",
    "camera.program": "advanced",
}


def _registry() -> dict[str, dict]:
    reg = json.loads(REGISTRY.read_text())
    recs = reg["fields"] if isinstance(reg, dict) else reg
    return {it["path"]: it for it in recs}


# ── FIELD_TIER registration ──────────────────────────────────────────────
@pytest.mark.parametrize("path,tier", sorted(PR3_TIERS.items()))
def test_pr3_tier_matches_registry_and_active(path: str, tier: str) -> None:
    reg = _registry()
    assert FIELD_TIER.get(path) == tier
    assert path in reg, f"{path} absent from registry"
    assert reg[path].get("tier") == tier
    assert reg[path].get("status") == "active"


# ── additive / backward-compat ───────────────────────────────────────────
def test_camera_new_fields_default_none() -> None:
    """Old callers (no new kwargs) still construct; new fields default None."""
    cam = Camera()
    assert cam.fps is None
    assert cam.frame_range is None
    assert cam.program is None
    cam_fields = {f.name for f in fields(Camera)}
    assert {"fps", "frame_range", "program"} <= cam_fields


# ── FrameRate (rational, floats banned) ──────────────────────────────────
def test_framerate_roundtrip() -> None:
    fr = FrameRate(num=24000, denom=1001)  # 23.976
    assert (fr.num, fr.denom) == (24000, 1001)


@pytest.mark.parametrize("num,denom", [(0, 1), (24, 0), (-1, 1), (24, -1)])
def test_framerate_rejects_nonpositive(num: int, denom: int) -> None:
    with pytest.raises(ValueError):
        FrameRate(num=num, denom=denom)


@pytest.mark.parametrize("num,denom", [(23.976, 1.0), (24.0, 1), (24, 1.0)])
def test_framerate_rejects_floats(num, denom) -> None:
    """The rational form exists to ban floats (schema: type=integer) — the
    exact mistake (passing 23.976 as a float) must be rejected."""
    with pytest.raises(ValueError):
        FrameRate(num=num, denom=denom)


# ── frame_range validation ───────────────────────────────────────────────
def test_frame_range_roundtrip() -> None:
    cam = Camera(frame_range=[0, 119])
    assert cam.frame_range == [0, 119]


@pytest.mark.parametrize("rng", [[0], [0, 1, 2], [-1, 5], [5, -1], [0.5, 10.5], [0, 10.0]])
def test_frame_range_rejects_invalid(rng) -> None:
    """Wrong length, negatives, or non-integer elements (schema: 2 ints >= 0)."""
    with pytest.raises(ValueError):
        Camera(frame_range=rng)


# ── CameraProgram matches the shape studio already writes on disk ─────────
def test_cameraprogram_matches_studio_written_shape() -> None:
    """studio (scenes_panels.py) writes shot.camera.program =
    {lamp_dsl, source, narrative}; the dataclass must accept exactly that."""
    studio_dict = {"lamp_dsl": "mx my mz ...", "source": "rule", "narrative": "slow push in"}
    cp = CameraProgram(**studio_dict)
    assert cp.lamp_dsl == studio_dict["lamp_dsl"]
    assert cp.source == "rule"
    assert cp.narrative == studio_dict["narrative"]


def test_program_source_literal_matches_schema_enum() -> None:
    """ProgramSource must equal the authoritative enum in
    pillar.camera.schema.json (program.source.enum) — cross-check against
    the schema, not a hardcoded set, so dataclass/schema drift is caught."""
    from typing import get_args

    schema = json.loads(
        (REPO_ROOT / "src" / "pace_core" / "pace_schema" / "pillar.camera.schema.json").read_text()
    )
    schema_enum = schema["properties"]["program"]["properties"]["source"]["enum"]
    assert set(get_args(T.ProgramSource)) == set(schema_enum)


def test_camera_full_construction() -> None:
    cam = Camera(
        fps=FrameRate(24, 1),
        frame_range=[0, 47],
        program=CameraProgram(lamp_dsl="...", source="llm", narrative="arc left"),
    )
    assert cam.fps.denom == 1
    assert cam.frame_range == [0, 47]
    assert cam.program.source == "llm"
