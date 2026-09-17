#!/usr/bin/env python3
"""Beat integrity for a project's shots.

The beat is the only thing that differs between panels of a scene -- cast,
framing, props, location and style are shared verbatim -- and the only field
that is free prose rather than a typed value. Everything else fails a schema
check when it is wrong; a beat fails silently and semantically, and the render
is the first place anyone notices.

Four rules, each from a failure a breakdown actually produced:

  unelaborated   the beat restates its key_action verbatim ("power loss"),
                 so the shot has a label where it needs staging and the
                 generator invents what the label looks like.
  multi-action   one beat coordinating two actions across clauses. Prose has
                 to hold them together and that is where it breaks: a beat
                 reading "look up at his mother ... while both parents stay
                 turned to the front" rendered four people in a three-person
                 car. One action per panel makes it structurally impossible.
  many-referents one character named under two labels ("his mother" and again
                 inside "both parents"), which reads as two people.
  empty          no beat at all; the panel is then indistinguishable from its
                 siblings, which is how three consecutive panels once compiled
                 to prompts 99.2% identical.

    uv run python python -m pace_core.qc.beat_check --project <slug>
    uv run python python -m pace_core.qc.beat_check --project <slug> --strict   # non-zero exit

Reports; never edits. What to write in place of a label is a directorial
decision, not something a checker should guess.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.paths import PAI_PROJECTS_ROOT  # noqa: E402

# A character named two ways in one sentence reads as two people.
_REFERENT = re.compile(
    r"\b(?:his|her|their) (?:mother|father|son|daughter)\b|\bboth parents\b"
    r"|\bthe (?:boy|girl|man|woman)\b|\bthe (?:two|three) adults\b", re.I)
_COORDINATOR = re.compile(r"\b(?:while|as|and then|meanwhile)\b", re.I)


def check_shot(beat: str, key_action: str) -> list[str]:
    """Rule names this shot's beat violates."""
    beat = (beat or "").strip()
    key = (key_action or "").strip()
    if not beat:
        return ["empty"]
    out = []
    if key and beat.lower() == key.lower():
        out.append("unelaborated")
    if _COORDINATOR.search(beat) and len(beat.split()) > 18:
        out.append("multi-action")
    if len(set(m.group(0).lower() for m in _REFERENT.finditer(beat))) > 2:
        out.append("many-referents")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when anything is flagged")
    a = ap.parse_args()

    scenes = sorted(((a.projects_root or PAI_PROJECTS_ROOT) / a.project
                     / "kb" / "scenes").glob("*.json"))
    if not scenes:
        print(f"no scenes for {a.project!r}", file=sys.stderr)
        return 1

    total = flagged = 0
    by_rule: dict[str, int] = {}
    for sp in scenes:
        doc = json.loads(sp.read_text(encoding="utf-8"))
        keys = (doc.get("narrative_meta") or {}).get("key_actions") or []
        for i, sh in enumerate(doc.get("shots") or []):
            total += 1
            act = ((sh.get("events") or {}).get("actions") or [{}])[0]
            beat = act.get("description_en") or act.get("description_zh") or ""
            rules = check_shot(beat, keys[i] if i < len(keys) else "")
            if not rules:
                continue
            flagged += 1
            for r in rules:
                by_rule[r] = by_rule.get(r, 0) + 1
            print(f"  {doc.get('scene_id')}/{sh.get('shot_id')}  "
                  f"{','.join(rules):28} {beat[:52]!r}")

    print(f"\n  {flagged} of {total} shots flagged")
    for r, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"    {r:16} {n}")
    return 1 if (a.strict and flagged) else 0


if __name__ == "__main__":
    raise SystemExit(main())
