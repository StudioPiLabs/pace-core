#!/usr/bin/env python3
"""Turn location bibles into the `location_stubs.json` everything reads.

Nothing has ever written this file. Every reference to it in the source is a
read -- the three prompt compilers, `panel_greybox`, `render_storyboard`, the
studio's compile context, `design_check` -- and the only writers are test
fixtures and `mesh_builder`, which attaches a mesh record to a stub that must
already exist. Where one was authored by hand, the
`location_stubs.json.pre-*.bak` files beside it are a person tuning belt,
seating, tray and wheel a field at a time. So the automated breakdown stops at
`kb/locations/*.bible.json`, and the step to the file that gates everything 3D
was manual. `panel_greybox` builds its shell from `scale_meters`, which only
the stub carries.

No model call. Every value here already exists on disk: the bible supplies
name, anchor and metric size, and the scenes supply era, region and culture.

**One stub per LOCATION, which is not one per scene.** `build_locations`
groups by `location_ref`, and `scenes_kb.load_all_scenes` synthesises that
field from the FILENAME when the scene does not carry one -- so a project
whose splitter left `narrative_meta.location_ref` null gets one bible per
scene rather than one per place. One office played in six scenes was
imagined twice over that way, once at 24x18 m and once at 16x11 m. The
same room in two sizes is the continuity break this pipeline exists to stop,
and it is invisible while each scene owns its own bible.

So the key here is the location as the scenes NAME it -- `location_raw`, which
is already what `setup.backdrop.location` holds, so a compiler resolves the
stub without anything being renamed. Bibles that describe one location are
merged, and a disagreement about size is reported rather than averaged: the
largest wins, because a room too small to hold its staged cast breaks the
greybox while a room too large only wastes space.

    uv run python -m pace_core.breakdown.build_location_stubs --project <slug>
    uv run python -m pace_core.breakdown.build_location_stubs --project <slug> --write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.paths import iter_canonical_scene_files, paths_for      # noqa: E402

SCHEMA_VERSION = "0.1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def location_key_of(scene: dict) -> str:
    """What this scene calls its location.

    `location_raw` first, because that is what `setup.backdrop.location` is
    set to and therefore what a compiler looks up. `location_ref` is checked
    too, but it is the field that is null in a fresh split.
    """
    nm = scene.get("narrative_meta") or {}
    b = ((scene.get("shot_defaults") or {}).get("setup") or {}).get("backdrop") or {}
    for v in (nm.get("location_ref"), nm.get("location_raw"), b.get("location")):
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _scale_list(bible: dict) -> list[float] | None:
    """[width, depth, height] in metres, the order the stub is read in.

    The bible writes `{length, width, height_open}`; `panel_greybox` unpacks
    the stub as W, D, H. Length is the depth of the room.
    """
    sl = (bible.get("spatial_layout") or {}).get("scale_meters")
    if isinstance(sl, list) and len(sl) == 3:
        try:
            return [float(x) for x in sl]
        except (TypeError, ValueError):
            return None
    if not isinstance(sl, dict):
        return None
    w = sl.get("width")
    d = sl.get("length") or sl.get("depth")
    h = sl.get("height_open") or sl.get("height")
    try:
        return [float(w), float(d), float(h)]
    except (TypeError, ValueError):
        return None


#: Words in a bible's own name or anchor that settle what kind of space it is.
#: Only the unambiguous ones: a shape guessed wrong is worse than none, because
#: `shape_from_stub` treats None as a real answer the caller must handle, while
#: a wrong shape builds a room nobody notices is wrong.
_OFFICE_WORDS = ("office", "open-plan", "open plan", "workstation", "cubicle",
                 "desk", "办公区", "工位")

#: Spaces that sit BESIDE an office and are not one. They match the office
#: words -- "Chairman's Office Doorway", "写字楼办公区 · 会议室外走廊" -- and an
#: open-plan shell is the wrong geometry for a corridor or a threshold, so a
#: name carrying one of these is left unclassified rather than guessed.
_NOT_A_ROOM = ("corridor", "doorway", "threshold", "lobby", "reception",
               "elevator", "走廊", "门口", "前台", "电梯")


def shape_of(bible: dict) -> str | None:
    """The greybox shell this bible describes, or None.

    Written as `shape` rather than as an `environment_type`, because the shape
    is what the builder consumes and inventing an environment vocabulary to be
    read back through a table adds a step that can only lose information.

    EXT is unambiguous. INT is not: `location_shape`'s interior family maps to
    `chamber`, a small room, and its own comment records that defaulting an
    unclassified location to chamber "turned 'this location is unclassified'
    into 'this location is a small room'". An open-plan floor of a hundred
    desks is not a chamber, so an interior is classified only when the bible
    says what kind, and otherwise left for a person to set.
    """
    t = (bible.get("type") or "").strip().upper()
    if t == "EXT":
        return "outdoor"
    hay = f"{bible.get('name') or ''}".lower()
    if any(w in hay for w in _NOT_A_ROOM):
        return None
    if any(w in hay for w in _OFFICE_WORDS):
        return "office"
    return None


def _backdrop_of(scene: dict) -> dict:
    return (((scene.get("shot_defaults") or {}).get("setup") or {})
            .get("backdrop") or {})


def _first(values):
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def build(scenes: list[dict], bibles: list[dict]) -> tuple[dict, list[str]]:
    """(stubs by location key, warnings). Pure: reads nothing, writes nothing."""
    warnings: list[str] = []

    # scene_id -> location key, and the reverse
    key_of_scene = {s.get("scene_id"): location_key_of(s) for s in scenes}
    scenes_by_key: dict[str, list[dict]] = {}
    for s in scenes:
        k = key_of_scene.get(s.get("scene_id")) or ""
        if not k:
            warnings.append(f"{s.get('scene_id')}: no location named anywhere; skipped")
            continue
        scenes_by_key.setdefault(k, []).append(s)

    # A bible belongs to the location of the scenes it was generated for.
    bibles_by_key: dict[str, list[dict]] = {}
    for b in bibles:
        refs = ((b.get("_meta") or {}).get("scene_refs")
                or ([b.get("id")] if b.get("id") else []))
        keys = {key_of_scene.get(r) for r in refs} - {None, ""}
        if len(keys) > 1:
            warnings.append(f"bible {b.get('id')!r} spans {sorted(keys)}; using the first")
        for k in sorted(keys)[:1]:
            bibles_by_key.setdefault(k, []).append(b)

    stubs: dict[str, dict] = {}
    for key, group in scenes_by_key.items():
        bs = bibles_by_key.get(key) or []
        if not bs:
            warnings.append(f"{key!r}: no bible; stub carries no geometry")

        sizes = {tuple(sc): b.get("id") for b in bs if (sc := _scale_list(b))}
        if len(sizes) > 1:
            pretty = ", ".join(f"{bid}={list(sz)}" for sz, bid in sizes.items())
            warnings.append(
                f"{key!r}: bibles disagree on size ({pretty}); using the largest")
        scale = max(sizes, key=lambda s: s[0] * s[1] * s[2]) if sizes else None

        bds = [_backdrop_of(s) for s in group]
        stub = {
            "name": _first([b.get("name") for b in bs]) or key,
            "scene_refs": [s.get("scene_id") for s in group],
            "anchor": _first([b.get("anchor") for b in bs]) or "",
            # The stub's `setting` is its prose description, which for a
            # hand-authored stub is usually the anchor verbatim.
            "setting": _first([b.get("anchor") for b in bs]) or "",
            "era": _first([b.get("era") for b in bds]),
            "region": _first([b.get("region") for b in bds]),
            "culture": _first([b.get("culture") for b in bds]),
            # NOT the bible's raw INT/EXT: that is not the environment
            # vocabulary `shape_from_stub` reads, and writing it there made
            # every location unclassified while looking populated.
            "shape": _first([shape_of(b) for b in bs]),
            "lighting": _first([(b.get("lighting_plan") or {}).get("key") for b in bs]),
            "_derived_from": [b.get("id") for b in bs],
        }
        if scale:
            stub["scale_meters"] = list(scale)
        # A null is a claim that the field is empty; absence is a claim that
        # nothing here knew. Only the second is true.
        stubs[key] = {k: v for k, v in stub.items() if v not in (None, "", [])}

    return stubs, warnings


def run(*, project: str, write: bool = False,
        set_location_ref: bool = True) -> dict:
    """Build the stubs, and optionally write them and backfill `location_ref`.

    Backfilling is additive: it is only ever set where it is null, which is
    the state a fresh split leaves it in. It matters because it is what
    `build_locations` groups on, so without it the next bible pass repeats the
    one-bible-per-scene mistake.
    """
    p = paths_for(project)
    scenes = [json.loads(f.read_text(encoding="utf-8"))
              for f in iter_canonical_scene_files(Path(p.scenes_dir))]
    bdir = Path(p.kb_dir) / "locations"
    bibles = [json.loads(f.read_text(encoding="utf-8"))
              for f in sorted(bdir.glob("*.bible.json"))] if bdir.is_dir() else []
    if not scenes:
        raise ValueError(f"project {project!r} has no scenes — split first")

    stubs, warnings = build(scenes, bibles)
    doc = {"_schema_version": SCHEMA_VERSION, "project": project,
           "updated_at": _now(), "stubs": stubs, "meshes": {}}

    existing = {}
    if Path(p.loc_stubs_file).exists():
        try:
            existing = (json.loads(Path(p.loc_stubs_file).read_text("utf-8"))
                        .get("stubs") or {})
        except (OSError, json.JSONDecodeError):
            existing = {}
    # Hand edits win. Stubs with hand-edit backups beside them record real
    # work, and regenerating over them would discard it.
    kept = sorted(set(existing) & set(stubs))
    for k in kept:
        doc["stubs"][k] = existing[k]

    refs_set = []
    if write:
        Path(p.loc_stubs_file).write_text(
            json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if set_location_ref:
            for f in iter_canonical_scene_files(Path(p.scenes_dir)):
                d = json.loads(f.read_text(encoding="utf-8"))
                nm = d.setdefault("narrative_meta", {})
                if nm.get("location_ref"):
                    continue
                key = location_key_of(d)
                if not key:
                    continue
                nm["location_ref"] = key
                f.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
                refs_set.append(d.get("scene_id"))

    return {"project": project, "stubs": len(stubs), "scenes": len(scenes),
            "bibles": len(bibles), "kept_existing": kept,
            "location_ref_set": refs_set, "warnings": warnings,
            "written": str(p.loc_stubs_file) if write else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--write", action="store_true",
                    help="write kb/location_stubs.json (default: preview)")
    ap.add_argument("--no-location-ref", action="store_true",
                    help="do not backfill narrative_meta.location_ref")
    a = ap.parse_args()
    out = run(project=a.project, write=a.write,
              set_location_ref=not a.no_location_ref)
    for w in out["warnings"]:
        print(f"  [warn] {w}", file=sys.stderr)
    print(json.dumps({k: v for k, v in out.items() if k != "warnings"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
