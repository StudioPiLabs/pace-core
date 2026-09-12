"""KB-tier scene loader for Stage A builders.

Reads `kb/on_scene/projects/<project>/scene_*.json` (PAI 1.1) and returns a
scenes_breakdown-shaped dict the build_* / extract_* modules consume. Only
the bulk `load_all_scenes` loader is in active use — single-scene read goes
through `pace_core.pai_compat.load_scene(path)`.
"""

from __future__ import annotations
import json
from pathlib import Path
from pace_core.paths import paths_for


def _resolve_dir(project: str | None = None) -> Path:
    """Map a project slug to its scenes directory. Project is required —
    no defaulting now that the studio is multi-tenant."""
    if not project:
        raise ValueError("project is required")
    return paths_for(project).scenes_dir


def load_all_scenes(project: str | None = None) -> dict:
    """Return a scenes_breakdown-compatible dict from one project's scene files.

    Output: {"_meta": {...}, "scenes": [{...}, {...}]}
    Each scene gets `scene_id`/`location_ref` synthesized if not present.
    """
    src_dir = _resolve_dir(project)
    if not src_dir.is_dir():
        return {"_meta": {"scene_count": 0, "project": project or "main"}, "scenes": []}

    from pace_core.paths import iter_canonical_scene_files
    files = iter_canonical_scene_files(src_dir)
    scenes = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        d.setdefault("scene_id",     f.stem)
        d.setdefault("location_ref", d.get("id") or f.stem)
        scenes.append(d)
    scenes.sort(key=lambda s: (s.get("scene_number") or 999, s.get("scene_id", "")))
    return {
        "_meta":  {
            "scene_count": len(scenes),
            "source_dir":  str(src_dir),
            "project":     project or "main",
        },
        "scenes": scenes,
    }
