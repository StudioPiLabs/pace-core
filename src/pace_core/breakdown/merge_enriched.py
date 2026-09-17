#!/usr/bin/env python3
"""Fold `scene_NN_v1_enriched.json` back into `scene_NN.json`.

`enrich_scenes_scine` writes its output as a SIBLING of the scene it enriched,
and nothing reads a sibling. `paths.py` defines canonical as exactly
`scene_<digits>.json` and 29 call sites use that filter, deliberately: an
extra file in that directory doubles every scene in the studio sidebar. So the
enrichment pass has always produced files that no compiler, greybox or route
would open, and the merge was done by hand -- scenes merged that way carry
`_enriched_at` on the canonical files and no sidecars beside them.

What enrichment adds is not decoration. On a Chinese-language scene it filled
`events.actions[].description_en`, which was EMPTY -- so every prompt was
compiled from the Chinese beat because the English one did not exist yet --
plus `environment.mood`, `lighting.condition`, `lighting.color_temperature`,
`backdrop.setting`, `backdrop.time_of_day`, per-subject `gaze` and per-action
`emotions`.

The merge is a deep merge with the enriched document winning, and the
canonical keeping anything the enriched file does not mention. Not a
replacement: the sidecar was generated from a snapshot, and a field edited
after that snapshot -- a location_ref backfilled, a beat corrected by hand --
must survive a merge rather than be reverted by one. A `.pre-merge.bak` is
written beside every file this touches, the way every other edit to this
corpus has been.

    uv run python -m pace_core.breakdown.merge_enriched --project <slug>
    uv run python -m pace_core.breakdown.merge_enriched --project <slug> --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.paths import iter_canonical_scene_files, paths_for      # noqa: E402

SIDECAR_SUFFIX = "_v1_enriched.json"


def deep_merge(base: dict, over: dict) -> dict:
    """`over` wins, `base` keeps what `over` does not mention.

    Lists are merged element-wise by position rather than replaced, because
    the enriched document has the same shots in the same order and adds
    fields inside them -- replacing the list would drop anything added to a
    shot after the snapshot the enricher read.
    """
    out = dict(base)
    for k, v in (over or {}).items():
        b = out.get(k)
        if isinstance(v, dict) and isinstance(b, dict):
            out[k] = deep_merge(b, v)
        elif isinstance(v, list) and isinstance(b, list) and len(v) == len(b):
            out[k] = [deep_merge(x, y) if isinstance(x, dict) and isinstance(y, dict)
                      else y for x, y in zip(b, v)]
        elif v is None and k in out:
            # A null in the enriched file is the model declining to answer,
            # not an instruction to erase what is already known.
            continue
        else:
            out[k] = v
    return out


def _added_paths(base, over, p="") -> list[str]:
    """Leaf paths `over` sets that `base` did not have. For the preview."""
    out = []
    if isinstance(over, dict):
        for k, v in over.items():
            b = base.get(k) if isinstance(base, dict) else None
            if b is None and v not in (None, "", [], {}):
                out.append(f"{p}.{k}")
            else:
                out += _added_paths(b, v, f"{p}.{k}")
    elif isinstance(over, list) and isinstance(base, list):
        for i, v in enumerate(over[:1]):
            out += _added_paths(base[i] if i < len(base) else None, v, f"{p}[]")
    return out


def run(*, project: str, write: bool = False) -> dict:
    p = paths_for(project)
    sdir = Path(p.scenes_dir)
    merged, skipped, fields = [], [], set()

    for f in iter_canonical_scene_files(sdir):
        side = sdir / (f.stem + SIDECAR_SUFFIX)
        if not side.exists():
            skipped.append(f.stem)
            continue
        base = json.loads(f.read_text(encoding="utf-8"))
        over = json.loads(side.read_text(encoding="utf-8"))
        fields.update(_added_paths(base, over))
        if write:
            f.with_suffix(".json.pre-merge.bak").write_text(
                json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            f.write_text(json.dumps(deep_merge(base, over), ensure_ascii=False,
                                    indent=2) + "\n", encoding="utf-8")
            side.unlink()
        merged.append(f.stem)

    return {"project": project, "merged": merged, "no_sidecar": skipped,
            "fields_gained": sorted(fields), "written": write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--write", action="store_true",
                    help="merge, back up as .pre-merge.bak, delete the sidecar")
    a = ap.parse_args()
    out = run(project=a.project, write=a.write)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
