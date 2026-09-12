"""Give one location one world, in one language.

`design_check` reports `world_drift` and refuses to resolve it, correctly: which
side of a divergence is right is a directorial decision. This is the case where
there is no decision to make. Zheng's 公司办公区 is one room, and its shots tell
the generator three different things about it:

    scene_03  modern day / near future, 中国都市写字楼办公区,
              modern urban chinese corporate
    scene_09  modern day, modern chinese city office building,
              modern urban chinese office supernatural
    scene_13  modern day, 现代都市写字楼, modern urban chinese office

These are the same world in different words, and in two languages, written into
an English prompt. Nothing here is choosing between competing designs; it is
choosing one spelling for a design nobody disputes.

The canonical value is not invented. For each location and field it is the most
frequent value already in the corpus, restricted to ASCII when any ASCII value
exists -- the prompt around it is English, and a Chinese place phrase inside an
English clause is the reading that suffers most. Ties go to the stub, which is
the declared design. A location whose every value is CJK keeps it: this tool
normalizes, it does not translate.

    uv run python -m pace_core.qc.conform_world_fields --project <slug>
    uv run python -m pace_core.qc.conform_world_fields --project <slug> --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

from pace_core.pai_compat import resolve_shot
from pace_core.paths import PAI_PROJECTS_ROOT

WORLD_FIELDS = ("era", "region", "culture")


def _is_ascii(v: str) -> bool:
    return bool(v) and all(ord(c) < 128 for c in v)


def canonical(values: list[str], stub_value: str | None) -> str | None:
    """The one spelling this field should carry, or None if nothing to choose.

    Frequency first, because the corpus has already voted; ASCII first among
    those, because the clause around it is English; the stub breaks ties,
    because it is what the design declares.
    """
    seen = [v for v in values if v]
    if not seen:
        return None
    pool = [v for v in seen if _is_ascii(v)] or seen
    counts = collections.Counter(pool)
    top = max(counts.values())
    best = [v for v, n in counts.items() if n == top]
    if stub_value in best:
        return stub_value
    return sorted(best)[0]


def survey(scenes: list[dict], stubs: dict) -> dict:
    """Per location and field: every value the corpus carries, and the choice."""
    seen: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for sc in scenes:
        for shot in sc.get("shots") or []:
            b = ((resolve_shot(sc, shot).get("setup") or {}).get("backdrop")) or {}
            loc = b.get("location")
            if not loc:
                continue
            for f in WORLD_FIELDS:
                if b.get(f):
                    seen[loc][f].append(b[f])
    plan: dict = {}
    for loc, fields in seen.items():
        stub = stubs.get(loc) or {}
        for f in WORLD_FIELDS:
            vals = list(fields.get(f) or [])
            if stub.get(f):
                vals.append(stub[f])
            pick = canonical(vals, stub.get(f))
            if pick is None:
                continue
            distinct = sorted(set(vals))
            if len(distinct) > 1:
                plan.setdefault(loc, {})[f] = {"pick": pick, "was": distinct}
    return plan


def apply(scenes_dir: Path, stubs_file: Path, plan: dict) -> tuple[int, int]:
    """Write the chosen value onto every shot and stub it applies to."""
    shots_changed = 0
    for path in sorted(scenes_dir.glob("scene_*.json")):
        sc = json.loads(path.read_text())
        dirty = False
        # The scene's own defaults first. A field stated once there and
        # inherited by every shot is the field the compiler actually reads,
        # and conforming only the shots would leave the value that wins
        # untouched.
        dflt = ((sc.get("shot_defaults") or {}).get("setup") or {}).get("backdrop") or {}
        for f, ch in (plan.get(dflt.get("location")) or {}).items():
            if dflt.get(f) and dflt[f] != ch["pick"]:
                dflt[f] = ch["pick"]
                dirty = True
                shots_changed += 1
        for shot in sc.get("shots") or []:
            b = (shot.setdefault("setup", {}).setdefault("backdrop", {}))
            loc = b.get("location") or (((sc.get("shot_defaults") or {})
                                         .get("setup") or {}).get("backdrop")
                                        or {}).get("location")
            for f, ch in (plan.get(loc) or {}).items():
                # Only where the shot already states the field. A shot that is
                # silent is `world_unresolved`, a different finding with a
                # different fix -- filling it here would hide it.
                if b.get(f) and b[f] != ch["pick"]:
                    b[f] = ch["pick"]
                    dirty = True
                    shots_changed += 1
        if dirty:
            shutil.copy2(path, path.with_suffix(".json.pre-conform.bak"))
            path.write_text(json.dumps(sc, ensure_ascii=False, indent=2) + "\n")
    doc = json.loads(stubs_file.read_text())
    stubs = doc.get("stubs", doc)
    stubs_changed = 0
    for loc, fields in plan.items():
        st = stubs.get(loc)
        if not isinstance(st, dict):
            continue
        for f, ch in fields.items():
            if st.get(f) and st[f] != ch["pick"]:
                st[f] = ch["pick"]
                stubs_changed += 1
    if stubs_changed:
        shutil.copy2(stubs_file, stubs_file.with_suffix(".json.pre-conform.bak"))
        stubs_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return shots_changed, stubs_changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    kb = (a.projects_root or PAI_PROJECTS_ROOT) / a.project / "kb"
    paths = sorted((kb / "scenes").glob("scene_*.json"))
    if not paths:
        print(f"no scenes for {a.project!r}", file=sys.stderr)
        return 1
    scenes = [json.loads(p.read_text()) for p in paths]
    stubs_file = kb / "location_stubs.json"
    doc = json.loads(stubs_file.read_text()) if stubs_file.is_file() else {}
    stubs = doc.get("stubs", doc)

    plan = survey(scenes, stubs)
    if not plan:
        print("every location already carries one value per field")
        return 0
    for loc, fields in sorted(plan.items()):
        print(loc)
        for f, ch in sorted(fields.items()):
            print(f"    {f:8} -> {ch['pick']!r}")
            for w in ch["was"]:
                if w != ch["pick"]:
                    print(f"    {'':8}    was {w!r}")
    if not a.apply:
        print("\n(dry run; pass --apply to write)")
        return 0
    n_shots, n_stubs = apply(kb / "scenes", stubs_file, plan)
    print(f"\nconformed {n_shots} shot fields and {n_stubs} stub fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
