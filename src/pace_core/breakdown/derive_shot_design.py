#!/usr/bin/env python3
"""Derive the director's beat groups from the film's own theme.

`kb/shot_design.json` is the typed half of 核心主题: four or five groups, each
naming the scenes it covers, the dramatic intent behind them, and the shot
size, angle and camera movement that intent asks for. It is the one document
that says WHY a shot is built as it is, and `camera._design.intent` is stamped
from it onto every shot -- read by the camera-movement deriver and admitted to
the prompt projection as grounds-not-words.

Nothing generated it. AutomaticDrive's was authored by hand, and a project
without one has no theme in the machine at all: `design_check` has nothing to
check a shot against, and a regenerated shot cannot rederive the intent it was
built for.

Now there is something to derive it FROM. `film.json` carries the treatment's
核心主题 and 世界观设定 verbatim, so the groups are grounded in what the author
wrote rather than invented from the scene list alone. That is the whole
argument for deriving this at all: without the theme it would be a guess about
coverage; with it, it is a reading of a stated intent.

What is deliberately NOT derived: the intent prose is written into
`camera._design.intent` and is never compiled into a prompt. The projection
already measures why -- writing a director's note produces a picture OF the
note, and "watching them like a surveillance feed" draws a surveillance feed.

    uv run python -m pace_core.breakdown.derive_shot_design --project <slug>
    uv run python -m pace_core.breakdown.derive_shot_design --project <slug> --execute
"""
from __future__ import annotations

import argparse
import json
import sys

from pace_scene_skills import load as _skill
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.compilers.compile_common import film_of            # noqa: E402
from pace_core.paths import (                                          # noqa: E402
    MODELS_FILE, iter_canonical_scene_files, paths_for,
)

GENERATOR_DEFAULT = "claude-opus-5-rb"

#: The schema's own vocabularies. The model picks from these rather than
#: inventing a label the compilers cannot read -- an unknown shot_size falls
#: through every framing table and silently renders as whatever the sampler
#: felt like.
# The vocabularies and the instruction both come from the skill. They were a
# module tuple and an f-string here, which meant the director's brief a person
# reads in `derive-shot-design/SKILL.md` and the one this module sends were
# two copies. Changing the brief in one of them left the other where it was.
_SKILL = _skill("derive-shot-design")
_VOCAB = _SKILL.reference("vocabulary.yaml")
SHOT_SIZES = tuple(_VOCAB["shot_size"])
ANGLES = tuple(_VOCAB["angle"])
#: Rig behaviour lives in trajectory.gear; framing moves in movement_3d. The
#: distinction this corpus already got wrong once, storing `handheld` where it
#: is not a legal value and losing it to a `tripod` default.
MOVEMENTS = tuple(_VOCAB["camera_movement"])

SYSTEM_PROMPT = _SKILL.instructions


def build_user_prompt(film, scenes: list[dict]) -> str:
    lines = [f"THEME (核心主题): {film.theme or '(not stated)'}",
             f"WORLD (世界观设定): {' '.join(film.design_language) or '(not stated)'}",
             f"GENRE (题材): {', '.join(film.genre) or '(not stated)'}", "", "SCENES:"]
    for s in scenes:
        nm = s.get("narrative_meta") or {}
        lines.append(
            f"  {s.get('scene_id')}  {s.get('scene_heading','')}  "
            f"[{len(s.get('shots') or [])} shots]  {nm.get('summary','')}")
    return "\n".join(lines)


def validate(groups: list[dict], scene_ids: list[str]) -> list[str]:
    """Warnings. A group naming a scene that does not exist, or a vocabulary
    the compilers cannot read, is worse than no group at all."""
    warn = []
    seen: list[str] = []
    for g in groups:
        for k, allowed in (("shot_size", SHOT_SIZES), ("angle", ANGLES),
                           ("camera_movement", MOVEMENTS)):
            if g.get(k) not in allowed:
                warn.append(f"{g.get('id')}: {k}={g.get(k)!r} is not in the schema")
        for s in g.get("scenes") or []:
            if s not in scene_ids:
                warn.append(f"{g.get('id')}: unknown scene {s!r}")
            seen.append(s)
    dupes = {s for s in seen if seen.count(s) > 1}
    if dupes:
        warn.append(f"scenes in more than one group: {sorted(dupes)}")
    missing = [s for s in scene_ids if s not in seen]
    if missing:
        warn.append(f"scenes in no group: {missing}")
    return warn


def run(*, project: str, model: str = GENERATOR_DEFAULT,
        execute: bool = False) -> dict:
    p = paths_for(project)
    scenes = [json.loads(f.read_text(encoding="utf-8"))
              for f in iter_canonical_scene_files(Path(p.scenes_dir))]
    if not scenes:
        raise ValueError(f"project {project!r} has no scenes — split first")
    film = film_of(p.film_file)
    if not film.theme:
        raise ValueError(
            f"{project!r} has no theme in kb/film.json — nothing to derive from. "
            "Run extract_film_doc, or write the theme by hand.")

    user = build_user_prompt(film, scenes)
    if not execute:
        return {"project": project, "model": model, "dry_run": True,
                "prompt_chars": len(SYSTEM_PROMPT) + len(user), "prompt": user}

    from pace_core.llm_client import call_model, strip_fences
    cfg = json.loads(MODELS_FILE.read_text())[model]
    raw, cost = call_model(cfg, [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": user}], api_key=None)
    try:
        groups = json.loads(strip_fences(raw)).get("groups") or []
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return JSON: {e}\nfirst 500: {raw[:500]}")

    warnings = validate(groups, [s.get("scene_id") for s in scenes])
    doc = {
        "_note": ("Director's shot design, derived from the film's own 核心主题 "
                  "by pace_core.breakdown.derive_shot_design. The screenplay "
                  "specifies no camera, so shot size and angle are AUTHORED by "
                  "dramatic function. Edit freely; nothing regenerates this."),
        "_source": f"derived from kb/film.json theme, {model}",
        "_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "groups": groups,
    }
    Path(p.shot_design_file).write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"project": project, "model": model, "groups": len(groups),
            "cost_usd": round(cost, 4), "warnings": warnings,
            "written": str(p.shot_design_file)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", default=GENERATOR_DEFAULT)
    ap.add_argument("--execute", action="store_true",
                    help="call the model (costs money). Without it, prints the prompt.")
    a = ap.parse_args()
    out = run(project=a.project, model=a.model, execute=a.execute)
    for w in out.get("warnings") or []:
        print(f"  [warn] {w}", file=sys.stderr)
    if out.get("dry_run"):
        print(out["prompt"])
        print(f"\n-- dry run: {out['prompt_chars']} prompt chars, model {out['model']}."
              f" Re-run with --execute.", file=sys.stderr)
    else:
        print(json.dumps({k: v for k, v in out.items() if k != "prompt"},
                         ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
