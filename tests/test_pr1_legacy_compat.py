"""PR-1 legacy-compat smoke test.

PR-1 adds purely-additive scalar fields to the types_v1 dataclasses
(Backdrop.weather/season, CameraIntrinsics.sensor_mm/focus_distance_m)
to align with pace-0.2. Two independent guarantees are tested:

1. **dict read path** (`test_legacy_scene_doc_reads`): PAILang consumes
   scene docs via `json.loads` + `.get()` (pai_compat.load_scene), never
   through the dataclasses, so existing on-disk data must keep reading.
   This loads every existing scene_*.json under main/test and exercises
   those `.get()` paths end-to-end.
2. **dataclass layer** (`test_new_fields_roundtrip` /
   `test_dataclasses_construct_without_new_fields`): the new fields are
   Optional with `=None` defaults, so old constructors (no new kwargs)
   must still work, and new kwargs must round-trip to the right values.
   This is what actually catches a future no-default field being added.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from pace_core.pai_compat import load_scene
from pace_core.types_v1 import Backdrop, CameraIntrinsics
import pace_core.types_v1 as t1

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = REPO_ROOT / "production" / "kb" / "on_scene" / "projects"

# canonical scene docs only — exclude enrichment artifacts like
# scene_01.enriched.json / scene_01_backup.json; match scene_<digits>.json
_CANONICAL = re.compile(r"^scene_\d+\.json$")


def _scene_docs() -> list[Path]:
    docs: list[Path] = []
    for project in ("main", "test"):
        docs.extend(
            sorted(
                p for p in (PROJECTS_DIR / project).glob("scene_*.json")
                if _CANONICAL.match(p.name)
            )
        )
    return docs


SCENE_DOCS = _scene_docs()
# Make an empty data dir an explicit skip rather than a silently-vacuous
# parametrised run (clean/sparse checkout without the production corpus).
_SCENE_PARAMS = SCENE_DOCS or [
    pytest.param(
        None,
        marks=pytest.mark.skip(reason=f"no canonical scene_*.json under {PROJECTS_DIR}"),
    )
]


def test_scene_docs_exist() -> None:
    """Guard: we actually found legacy data to exercise."""
    if not SCENE_DOCS:
        pytest.skip(f"no scene_*.json under {PROJECTS_DIR}")


@pytest.mark.parametrize(
    "path", _SCENE_PARAMS,
    ids=lambda p: (p.parent.name + "/" + p.name) if p is not None else "skip",
)
def test_legacy_scene_doc_reads(path: Path) -> None:
    """load_scene must not raise, shots iterate, and each shot's
    subjects/setup are reachable via the same .get() paths consumers use."""
    scene = load_scene(path)
    assert isinstance(scene, dict)

    shots = scene.get("shots") or []
    assert isinstance(shots, list)

    for shot in shots:
        assert isinstance(shot, dict)
        setup = shot.get("setup") or {}
        assert isinstance(setup, dict)

        # subjects read path (dict-of-list access used by consumers)
        subjects = setup.get("subjects") or []
        assert isinstance(subjects, list)
        for subj in subjects:
            assert isinstance(subj, dict)
            # representative key read — must not raise
            subj.get("character_id")

        # backdrop read path — where new weather/season would live
        backdrop = setup.get("backdrop") or {}
        assert isinstance(backdrop, dict)
        backdrop.get("weather")
        backdrop.get("season")

        # camera.intrinsics read path — where new sensor_mm/focus_distance_m live
        intrinsics = (shot.get("camera") or {}).get("intrinsics") or {}
        assert isinstance(intrinsics, dict)
        intrinsics.get("sensor_mm")
        intrinsics.get("focus_distance_m")


def test_dataclasses_construct_without_new_fields() -> None:
    """Old callers that pass none of the new kwargs must still construct,
    and the new fields must default to None (purely-additive guarantee).

    This is the regression guard for the dataclass layer: if a future
    change adds a non-defaulted field, these no-arg constructions raise
    TypeError and this test goes red.
    """
    assert CameraIntrinsics().sensor_mm is None
    assert CameraIntrinsics().focus_distance_m is None
    assert Backdrop().weather is None
    assert Backdrop().season is None

    # the new fields exist on the dataclass and carry a default (so they
    # may follow existing defaulted fields without a definition-order error)
    cam_fields = {f.name: f for f in fields(CameraIntrinsics)}
    bd_fields = {f.name: f for f in fields(Backdrop)}
    for name in ("sensor_mm", "focus_distance_m"):
        assert name in cam_fields and cam_fields[name].default is None
    for name in ("weather", "season"):
        assert name in bd_fields and bd_fields[name].default is None


def test_new_fields_roundtrip() -> None:
    """New kwargs round-trip to the expected values (behavioural coverage,
    not just no-raise) — proves the fields are wired, not no-ops."""
    cam = CameraIntrinsics(sensor_mm=[36.0, 24.0], focus_distance_m=2.5)
    assert cam.sensor_mm == [36.0, 24.0]
    assert cam.focus_distance_m == 2.5

    bd = Backdrop(weather="overcast", season="winter")
    assert bd.weather == "overcast"
    assert bd.season == "winter"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensor_mm": [36.0]},            # wrong length
        {"sensor_mm": [36.0, 24.0, 1.0]}, # wrong length
        {"sensor_mm": [-5.0, 24.0]},      # non-positive
        {"sensor_mm": [36.0, 0.0]},       # non-positive
        {"focus_distance_m": 0.0},        # must be > 0
        {"focus_distance_m": -1.0},       # must be > 0
    ],
)
def test_camera_intrinsics_rejects_invalid(kwargs: dict) -> None:
    """__post_init__ enforces the pace-0.2 schema constraints that dataclass
    typing can't express (sensor_mm len==2 & both>0, focus_distance_m>0),
    since `pace lint` is not on the in-process write path."""
    with pytest.raises(ValueError):
        CameraIntrinsics(**kwargs)


# ── FIELD_TIER alignment ──────────────────────────────────────────────────────

def test_prop_id_is_required() -> None:
    assert t1.FIELD_TIER["setup.props[].propId"] == "required"


def test_description_en_tier() -> None:
    assert t1.FIELD_TIER["events.actions[].descriptionEn"] in ("required", "recommended")


def test_new_intrinsics_fields_in_tier() -> None:
    assert "camera.intrinsics.sensorMm" in t1.FIELD_TIER
    assert "camera.intrinsics.focusDistanceM" in t1.FIELD_TIER


def test_no_duplicate_keys() -> None:
    """Dict literals silently de-dup — enumerate the source to catch it."""
    src = (REPO_ROOT / "src" / "pace_core" / "types_v1.py").read_text()
    keys: list[str] = []
    in_tier = False
    for line in src.splitlines():
        stripped = line.strip()
        if "FIELD_TIER" in stripped and "=" in stripped and "{" in stripped:
            in_tier = True
        if in_tier:
            if stripped.startswith('"') and '":' in stripped:
                key = stripped.split('"')[1]
                keys.append(key)
            if stripped == "}":
                in_tier = False
    dupes = [k for k in keys if keys.count(k) > 1]
    assert not dupes, f"Duplicate FIELD_TIER keys: {dupes}"
