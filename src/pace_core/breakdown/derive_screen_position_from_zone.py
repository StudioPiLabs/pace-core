#!/usr/bin/env python3
"""Turn the declared zone into the numeric screen position the solver needs.

`screen_position` has four fields and the enrichment fills two of them. It
returns `zone` -- left, center_left, upper_right -- and `depth`, because those
are what a reader of a screenplay can say. It does not return `x` and `y`,
because a number is not in the text, and nothing else fills them: on another production all
190 subject placements carry `x: null`.

The consequences are downstream and silent. `panel_greybox._declared_x` reads
a null as 0.5, so every subject sorts to the same place and the seat assignment
spreads them evenly -- which is how a scene of a hundred people working late
staged as three figures standing in a row. The composition solve then has
nothing to aim at that the specification chose.

The paper's own corpus fills these by hand, and reports what that produced:
three distinct x values across 58 subjects and one y, corpus-wide. Evenly
spaced heads on a single line. The zone is a better source than a constant
precisely because a model reading the scene put nine different values in it.

The mapping is the rule of thirds, which is what `zone`'s own vocabulary
names. Symmetric about centre so a left and a right subject are mirror images,
and the inner pair sits at 0.39/0.61 rather than on the third lines
themselves, matching the values the existing corpus declares -- a shot
composed to this and a shot composed to that should not disagree by an
accident of rounding.

Existing numbers are never overwritten: a value someone authored is a decision
and this fills blanks.

    uv run python -m pace_core.breakdown.derive_screen_position_from_zone --project <slug>
    uv run python -m pace_core.breakdown.derive_screen_position_from_zone --project <slug> --write
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.paths import iter_canonical_scene_files, paths_for      # noqa: E402

#: Horizontal thirds. 0.39/0.61 rather than 0.33/0.67 so these agree with the
#: values the existing corpus declares by hand.
ZONE_X = {"left": 0.28, "center_left": 0.39, "center": 0.50,
          "center_right": 0.61, "right": 0.72}

#: Vertical. 0.52 for centre is the corpus's own declared height, kept so a
#: derived placement and an authored one are on the same line.
ZONE_Y = {"upper": 0.38, "center": 0.52, "lower": 0.66}


def split_zone(zone: str) -> tuple[str | None, str | None]:
    """A zone name into its horizontal and vertical halves.

    `upper_center` and `center_right` are one token each and mix the two axes,
    so the parse is by which half of the name names which axis rather than by
    splitting on the underscore.
    """
    z = (zone or "").strip().lower()
    if not z:
        return None, None
    vert = next((v for v in ("upper", "lower") if z.startswith(v) or f"_{v}" in z), None)
    rest = z.replace(f"{vert}_", "").replace(f"_{vert}", "") if vert else z
    horiz = rest if rest in ZONE_X else ("center" if rest in ("", "center") else None)
    return horiz, (vert or "center")


def xy_for(zone: str) -> tuple[float | None, float | None]:
    h, v = split_zone(zone)
    return (ZONE_X.get(h) if h else None), (ZONE_Y.get(v) if v else None)


def run(*, project: str, write: bool = False) -> dict:
    p = paths_for(project)
    filled = skipped = subjects = 0
    zones = collections.Counter()
    unknown = collections.Counter()

    for f in iter_canonical_scene_files(Path(p.scenes_dir)):
        doc = json.loads(f.read_text(encoding="utf-8"))
        touched = False
        for sh in doc.get("shots") or []:
            for s in ((sh.get("setup") or {}).get("subjects") or []):
                subjects += 1
                sp = s.get("screen_position")
                if not isinstance(sp, dict):
                    continue
                if sp.get("x") is not None:
                    skipped += 1          # authored: a decision, not a blank
                    continue
                x, y = xy_for(sp.get("zone") or "")
                if x is None:
                    unknown[sp.get("zone")] += 1
                    continue
                zones[sp.get("zone")] += 1
                filled += 1
                if write:
                    sp["x"], sp["y"] = x, y
                    touched = True
        if write and touched:
            f.with_suffix(".json.pre-screenpos.bak").write_text(
                json.dumps(json.loads(f.read_text(encoding="utf-8")),
                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            f.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    return {"project": project, "subjects": subjects, "filled": filled,
            "already_authored": skipped, "unreadable_zones": dict(unknown),
            "by_zone": dict(zones), "written": write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    out = run(project=a.project, write=a.write)
    for z, n in sorted(out["by_zone"].items(), key=lambda kv: -kv[1]):
        x, y = xy_for(z)
        print(f"  {z:16} -> x={x} y={y}   ({n} placements)")
    print(json.dumps({k: v for k, v in out.items() if k != "by_zone"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
