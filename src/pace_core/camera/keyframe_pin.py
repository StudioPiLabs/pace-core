"""Pinning one render as a panel's canonical keyframe.

Everything downstream of a panel -- Wan I2V, VACE, the previs keyframe the 3D
control stage matches its GLB against -- is supposed to consume the render the
operator chose, not whichever file is newest in the directory. Without a pin
the fallback to "newest" is the only behaviour, which means a re-render
silently changes what the video stage consumes.

The write lives here rather than in the route that first needed it because it
now has two callers: an operator pinning from the gallery, and a batch render
asked to pin what it produced. Two copies of a KB write drift, and this one
has a shape it must keep -- the reader destructures `filename / seed /
lora_filename / lora_strength / base_model / authored_at / authored_by / note`,
so a pin written in any other shape reads back as a pin with null provenance.

The provenance is the point: a canonical keyframe records *how* it was made,
so it can be reproduced rather than merely pointed at.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(scene_file: Path) -> dict:
    if not Path(scene_file).exists():
        raise FileNotFoundError(f"scene document not found: {scene_file}")
    return json.loads(Path(scene_file).read_text())


def _save(scene_file: Path, doc: dict) -> None:
    Path(scene_file).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")


def has_panel(doc: dict, panel_id: str) -> bool:
    return any(p.get("id") == panel_id
               for sh in (doc.get("shots") or [])
               for p in (sh.get("panels") or []))


def pin_keyframe(scene_file: Path | str, panel_id: str, filename: str,
                 *, seed: Any = None, lora_filename: str | None = None,
                 lora_strength: Any = None, base_model: str | None = None,
                 authored_by: str = "studio", note: str | None = None) -> dict:
    """Record `filename` as `panel_id`'s keyframe. Returns the record written.

    The caller supplies what it rendered with; this owns the authorship stamp.
    Raises FileNotFoundError for a missing scene, LookupError for a panel the
    scene does not carry -- a pin on a panel that is not there would read back
    as no pin at all.
    """
    scene_file = Path(scene_file)
    doc = _load(scene_file)
    if not has_panel(doc, panel_id):
        raise LookupError(f"panel {panel_id!r} not found in {scene_file.name}")

    hints = doc.setdefault("compile_hints", [])
    entry = next((h for h in hints if h.get("panel_id") == panel_id), None)
    if entry is None:
        entry = {"panel_id": panel_id, "flux": {}, "gpt_image_2": {}, "wan_i2v": {}}
        hints.append(entry)
    record = {
        "filename": filename,
        "seed": seed,
        "lora_filename": lora_filename,
        "lora_strength": lora_strength,
        "base_model": base_model,
        "authored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authored_by": authored_by,
        "note": note,
    }
    entry.setdefault("flux", {})["keyframe"] = record
    _save(scene_file, doc)
    return record


def unpin_keyframe(scene_file: Path | str, panel_id: str) -> bool:
    """Drop the pin. Returns whether there was one. Downstream stages fall
    back to the newest render again."""
    scene_file = Path(scene_file)
    doc = _load(scene_file)
    entry = next((h for h in (doc.get("compile_hints") or [])
                  if h.get("panel_id") == panel_id), None)
    had = bool((entry or {}).get("flux", {}).pop("keyframe", None))
    if had:
        _save(scene_file, doc)
    return had
