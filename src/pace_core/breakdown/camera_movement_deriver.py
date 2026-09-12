"""Derive each shot's camera movement automatically from its dramatic content.

Reads the same field a director would look at first — the shot's authored
`camera._design.intent`, falling back to its first action's description —
and classifies it against `camera_movement_kb`'s sourced (CITATION/MEASURED)
dramatic functions via `camera_skills.classify_dramatic_function`. A match
resolves to that entry's `movement_tags` and is written through
`movement_io.write_movement`, the same field the batch scene-render
pipeline (`camera_planner`, via `movement_io.read_movement`) actually reads
at render time — NOT `camera_skills`' own richer vocabulary, which today is
only ever compiled ephemerally for a one-off playground render job and
never persisted back to a shot's KB record.

A shot that does not classify — unsourced content, or already an explicit
authored move — is left untouched. This deriver ADDS a sourced movement
where the KB has grounds for one; it does not invent a plausible-looking
move for every shot regardless of evidence, and it does not overwrite an
already-authored `camera.trajectory`. That passively honours the KB's own
`coverage_reuse` policy too: a shot this deriver has no grounds to touch
keeps whatever camera position a previous pass (human or this tool) already
gave it, rather than being handed a fresh guess on every run.

Every write carries provenance at `shot.camera.trajectory._derived_by`:
the matched dramatic_function, its evidence tier and source, the
classifying model, and a timestamp — so a later reader can tell a
KB-sourced write from a hand-authored one without re-deriving it.

    uv run python -m pace_core.breakdown.camera_movement_deriver --project <slug> --scene <id>          # dry run
    uv run python -m pace_core.breakdown.camera_movement_deriver --project <slug> --scene <id> --apply
"""
from __future__ import annotations

from datetime import datetime, timezone

from pace_core.camera.camera_movement_kb import KB
from pace_core.camera.movement_io import write_movement

AUTHORED_BY = "camera_movement_deriver"


def _shot_text(shot: dict) -> str:
    """Text handed to the classifier: authored camera intent first — a
    director's own statement of what the shot is doing, when one exists —
    else the first action's description."""
    intent = ((shot.get("camera") or {}).get("_design") or {}).get("intent")
    if intent:
        return intent
    actions = ((shot.get("events") or {}).get("actions")) or []
    if actions:
        return actions[0].get("description_en") or ""
    return ""


def derive_shot(shot: dict, *, model: str = "gpt-4o-mini-rb") -> dict:
    """Classify one shot. Pure — never mutates `shot`. Returns a report:

        {"dramatic_function": None, "reason": "..."}                 unclassified
        {"dramatic_function": <fn>, "movement_tags": [...],
         "evidence": "citation"|"measured", "source": "..."}          classified

    `apply_shot` is the only function that writes; this one only decides.
    """
    text = _shot_text(shot)
    if not text.strip():
        return {"dramatic_function": None, "reason": "no action/intent text"}

    from pace_core.camera.camera_skills import classify_dramatic_function
    fn = classify_dramatic_function(text, model=model)
    if fn is None:
        return {"dramatic_function": None, "reason": "no sourced match"}

    entry = KB.get(fn)
    if entry is None or entry.is_policy or not entry.movement_tags:
        # Defensive: classify_dramatic_function only offers non-policy,
        # sourced entries to the model (_kb_summary_for_prompt excludes
        # policy/convention tiers) — a policy name here would mean the
        # model named something outside what it was grounded on.
        return {"dramatic_function": None,
               "reason": f"classifier named {fn!r}, not a resolvable movement"}

    return {"dramatic_function": fn, "movement_tags": list(entry.movement_tags),
           "evidence": entry.evidence.value, "source": entry.source}


def apply_shot(shot: dict, report: dict, *, model: str) -> bool:
    """Write a classified report onto `shot` in place. Returns False (no
    write) for an unclassified report."""
    fn = report.get("dramatic_function")
    if fn is None:
        return False
    write_movement(shot, report["movement_tags"])
    shot["camera"]["trajectory"]["_derived_by"] = {
        "authored_by": AUTHORED_BY,
        "dramatic_function": fn,
        "evidence": report["evidence"],
        "source": report["source"],
        "model": model,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return True


def derive_scene(scene: dict, *, model: str = "gpt-4o-mini-rb",
                 apply: bool = False) -> dict:
    """Walk every shot in `scene`, classify it, and — only if `apply` —
    write the result. Never touches `scene` when `apply=False`. Returns
    `{"scene_id": ..., "classified": [...], "skipped": [...]}`."""
    classified, skipped = [], []
    for shot in scene.get("shots") or []:
        shot_id = shot.get("shot_id")
        report = derive_shot(shot, model=model)
        if report.get("dramatic_function") is None:
            skipped.append({"shot": shot_id, "why": report.get("reason")})
            continue
        classified.append({"shot": shot_id, **report})
        if apply:
            apply_shot(shot, report, model=model)
    return {"scene_id": scene.get("scene_id"), "classified": classified,
           "skipped": skipped}


def main() -> int:
    """Dry run by default: this rewrites authored KB documents, and it
    keeps a `.json.pre-camera-deriver.bak` beside each one it changes. The
    suffix names the change rather than being a plain `.bak`, because a
    plain one is overwritten by the next tool to touch the same scene and
    a two-step repair then has no first rollback point. Each classification
    is a real LLM call — dry run reports what WOULD be written without
    spending anything past the classification calls themselves; --apply is
    required to write.
    """
    import argparse
    import json
    import shutil
    import sys
    from pathlib import Path

    from pace_core.paths import PAI_PROJECTS_ROOT

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", default=None, help="limit to one scene id")
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--model", default="gpt-4o-mini-rb")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    root = a.projects_root or PAI_PROJECTS_ROOT
    scenes = sorted((Path(root) / a.project / "kb" / "scenes").glob("*.json"))
    if a.scene:
        scenes = [p for p in scenes if p.stem == a.scene]
    if not scenes:
        print(f"no scenes under {Path(root) / a.project}", file=sys.stderr)
        return 1

    n_classified = n_skipped = 0
    for sp in scenes:
        doc = json.loads(sp.read_text(encoding="utf-8"))
        r = derive_scene(doc, model=a.model, apply=a.apply)
        n_skipped += len(r["skipped"])
        for rec in r["classified"]:
            n_classified += 1
            print(("  " if a.apply else "  [dry] ")
                  + f"{r['scene_id']}/{rec['shot']:8} {rec['dramatic_function']:24} "
                    f"tags={'+'.join(rec['movement_tags'])}")
        if a.apply and r["classified"]:
            shutil.copy2(sp, sp.with_suffix(".json.pre-camera-deriver.bak"))
            sp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{n_classified} shot(s) {'derived' if a.apply else 'would be derived'}; "
          f"{n_skipped} shot(s) had no sourced match and were left untouched"
          + ("" if a.apply else "  — re-run with --apply"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
