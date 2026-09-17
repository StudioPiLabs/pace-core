"""Mesh3D schema — single source of truth for 3D mesh records attached to
KB entities (props, characters, locations).

Used by:
  * production/kb/on_scene/props.json — each prop's `model_3d` slot
  * production/kb/on_scene/characters.json — each character's `body_proxy_3d`
  * production/kb/on_scene/location_stubs.json — `meshes` map keyed by stub id

A mesh record is `None` until a file exists on disk. Once populated, callers
get a typed view via `Mesh3D.from_dict` and resolve paths via `.absolute_path(repo_root)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional


MeshSource = Literal[
    "polyhaven",        # CC0 download from polyhaven.com (source_ref = polyhaven slug)
    "hunyuan3d_v2",     # Hunyuan3D-2 image→3D via ComfyUI (source_ref = gen_id)
    "trellis",          # Microsoft Trellis (future)
    "manual",           # hand-modeled in Blender (source_ref = blend file)
    "photogrammetry",   # scanned (source_ref = scan dataset id)
    "imported",         # dropped in by user, provenance unknown
]

MeshOrigin = Literal[
    "base_center",      # pivot at base of mesh's footprint — RECOMMENDED for props that sit on surfaces
    "center",           # pivot at geometric center (good for hand-held / floating props)
    "custom",           # pivot already set in the source file — don't auto-adjust
]


@dataclass
class Mesh3D:
    """One 3D mesh record. All paths are relative to repo root unless stated."""
    mesh_file:    str                                  # e.g. "production/3d/props/oil_lamp.glb"
    source:       MeshSource                           # provenance
    source_ref:   Optional[str]      = None            # polyhaven slug / hunyuan gen_id / blend filename
    scale_factor: float              = 1.0             # multiplier from raw to scene meters (most assets ship at meters)
    origin:       MeshOrigin         = "base_center"   # pivot location
    bbox_m:       list[float]        = field(default_factory=list)  # [w, d, h] in meters — bake at import time
    version:      int                = 1
    created_at:   Optional[str]      = None            # ISO-8601 UTC
    license:      Optional[str]      = None            # "CC0" | "CC-BY-4.0" | "proprietary" | etc.
    note:         Optional[str]      = None            # human freeform

    # ─── dict serialization ──────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Mesh3D"]:
        """Lift a dict (or None) into a Mesh3D instance. Tolerates legacy keys."""
        if d is None:
            return None
        return cls(
            mesh_file=d["mesh_file"],
            source=d.get("source", "imported"),
            source_ref=d.get("source_ref"),
            scale_factor=float(d.get("scale_factor", 1.0)),
            origin=d.get("origin", "base_center"),
            bbox_m=list(d.get("bbox_m", [])),
            version=int(d.get("version", 1)),
            created_at=d.get("created_at"),
            license=d.get("license"),
            note=d.get("note"),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    # ─── path helpers ────────────────────────────────────────────────────
    def absolute_path(self, repo_root: Path | str) -> Path:
        """Resolve mesh_file against repo root."""
        return Path(repo_root) / self.mesh_file

    def exists(self, repo_root: Path | str) -> bool:
        return self.absolute_path(repo_root).is_file()


# ── conventional layout ─────────────────────────────────────────────────────
# Where mesh files live in the repo. CLI builders write here; consumers read here.
PROPS_MESH_DIR     = "production/3d/props"
CHARACTERS_MESH_DIR = "production/3d/characters"
LOCATIONS_MESH_DIR = "production/3d/locations"


def default_mesh_path(kind: Literal["prop", "character", "location"], stem: str,
                      ext: str = "glb") -> str:
    """Return the conventional relative path for a fresh mesh.
    `stem` is the entity id (e.g. "oil_lamp", "alice", "great_hall").
    """
    base = {
        "prop": PROPS_MESH_DIR,
        "character": CHARACTERS_MESH_DIR,
        "location": LOCATIONS_MESH_DIR,
    }[kind]
    return f"{base}/{stem}.{ext}"
