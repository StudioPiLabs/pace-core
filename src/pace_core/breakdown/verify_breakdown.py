"""The estimator: judge a breakdown against Script IR, and price the damage.

Deliberately a DIFFERENT vendor from the generator that produced the Script IR.
Two instances of one model share a prior, so they agree on whatever is
plausible -- and this corpus's `screen_position.x` of 0.38 is maximally
plausible while being a lookup-table constant chosen by list index. A critic
that shares the writer's taste rubber-stamps that. Splitting the vendors does
not make the estimator right, but it stops the two failing the same way.

The division of labour follows the design doc: the model does semantic event
matching, which needs judgement; the program computes coverage, counts and
severity, which do not. The model never returns a score.

Usage:
    uv run python -m pace_core.breakdown.verify_breakdown \
        --script-ir script_ir.json --project AutomaticDrive --dry-run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

# GPT-5.6 Terra: newer than 5.5 and cheaper in both directions ($2/$12 per M
# against $5/$30), so the check that runs on every scene is not the expensive
# half of the pair. Deliberately a different vendor from the generator.
ESTIMATOR_DEFAULT = "gpt-5.6-terra-rb"

SYSTEM = """You align a film BREAKDOWN against GROUND-TRUTH events extracted
from the screenplay. You are an auditor, not an author.

You are given SCRIPT EVENTS (each with an id, a canonical predicate, its
participants, and the exact screenplay words that support it) and BREAKDOWN
ACTIONS (free text the breakdown wrote for the same stretch of film).

For each SCRIPT EVENT decide whether the breakdown covers it, and how.

Return STRICT JSON, no prose, no fence:

{
  "matches": [
    {"script_event": "<script event local_id>",
     "breakdown_actions": ["<breakdown action id>", ...],
     "match": "EXACT" | "SEMANTIC" | "OVER_GENERALIZED" | "MISSING",
     "role_ok": true | false,
     "why": "<one short clause>"}
  ],
  "invented_events": [
    {"action": "<breakdown action id>", "why": "<the EVENT it asserts that the script never states>"}
  ],
  "over_specified": [
    {"action": "<breakdown action id>", "why": "<the added detail>"}
  ]
}

RULES
1. MISSING means no breakdown action states this event. Being implied by a
   neighbouring action is not coverage.
2. OVER_GENERALIZED means an action covers it but loses what mattered --
   "attack the wall" for "raise ladders against the wall".
3. role_ok is false when the actor and patient are swapped or wrong. Text
   similarity is irrelevant here; who did what to whom is the fact.
4. Separate two very different things, and do not merge them.
   invented_events: the action asserts something HAPPENED that the script
     does not state -- a character the script never puts there, an action
     nobody performs. This is a fabrication and it matters.
   over_specified: the action states a real event and adds craft detail the
     script does not dictate -- shot size, lens, "a close-up of", "bright
     faces", "violently". A breakdown is SUPPOSED to add these; they are the
     director's job, not a fault. Record them, do not treat them as errors.
   If an action realises a real event, it is never an invented_event no
   matter how much staging language it carries.
5. Judge only against the script events given. Do not invent script facts.
"""


def load_breakdown_actions(project: str, scene_ids: list[str],
                           scenes_dir: str | None = None) -> list[dict]:
    """Read a breakdown's actions. `scenes_dir` lets a CANDIDATE breakdown be
    judged by the same estimator, on the same prompt, without it being
    promoted into the project first -- which is the only way the before/after
    comparison means anything."""
    from pace_core.paths import paths_for
    d = pathlib.Path(scenes_dir) if scenes_dir else pathlib.Path(paths_for(project).scenes_dir)
    out = []
    for sid in scene_ids:
        f = d / f"{sid}.json"
        if not f.is_file():
            continue
        doc = json.loads(f.read_text())
        for sh in doc.get("shots") or []:
            for i, a in enumerate((sh.get("events") or {}).get("actions") or []):
                out.append({
                    "id": f"{sid}/{sh['shot_id']}/a{i}",
                    "text": a.get("description_en") or a.get("standalone") or "",
                })
            cie = (sh.get("events") or {}).get("change_in_environment")
            if cie:
                out.append({"id": f"{sid}/{sh['shot_id']}/env", "text": cie})
    return out


def build_messages(script_scene: dict, actions: list[dict]) -> list[dict]:
    evs = [{"id": e.get("local_id"), "predicate": e.get("predicate"),
            "actor": e.get("actor"), "patient": e.get("patient"),
            "order": e.get("order"),
            "script_words": (e.get("evidence") or {}).get("source_text")}
           for e in script_scene.get("events") or []]
    user = (f"SCENE: {script_scene['heading']}\n\n"
            f"SCRIPT EVENTS (ground truth):\n{json.dumps(evs, ensure_ascii=False, indent=1)}\n\n"
            f"BREAKDOWN ACTIONS:\n{json.dumps(actions, ensure_ascii=False, indent=1)}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def score(alignments: list[dict], script_scenes: list[dict]) -> dict:
    """Counts and rates, computed here and never asked of the model."""
    total = matched = semantic = overgen = missing = role_bad = 0
    unsupported = over_spec = 0
    missing_events: list[dict] = []
    # Keyed by (scene, local_id): scenes restart their ids at e1, so keying on
    # local_id alone made every scene's e11 the same event and reported one
    # dropped climax four times over.
    by_id = {(s["index"], e.get("local_id")): (s, e)
             for s in script_scenes for e in (s.get("events") or [])}
    for al in alignments:
        scene_i = al.get("script_scene")
        for m in al.get("matches") or []:
            total += 1
            k = m.get("match")
            if k == "EXACT":
                matched += 1
            elif k == "SEMANTIC":
                matched += 1; semantic += 1
            elif k == "OVER_GENERALIZED":
                overgen += 1
            else:
                missing += 1
                s_e = by_id.get((scene_i, m.get("script_event")))
                if s_e:
                    s, e = s_e
                    missing_events.append({
                        "scene": s["index"], "predicate": e.get("predicate"),
                        "importance": e.get("importance"),
                        "words": (e.get("evidence") or {}).get("source_text")})
            # Only a MATCHED event can have its roles wrong. The estimator
            # returns role_ok=false on MISSING events too, which is vacuous --
            # nothing matched, so nothing swapped actor for patient -- and
            # counting those turned 2 real role errors into 30.
            if m.get("role_ok") is False and k != "MISSING":
                role_bad += 1
        unsupported += len(al.get("invented_events") or [])
        over_spec += len(al.get("over_specified") or [])
    covered = matched + 0.6 * overgen
    return {
        "counts": {"script_events": total, "exact_or_semantic": matched,
                   "of_which_semantic": semantic, "over_generalized": overgen,
                   "missing": missing, "wrong_role": role_bad,
                   "invented_events": unsupported,
                   "over_specified": over_spec},
        "coverage": round(covered / total, 4) if total else None,
        "missing_events": sorted(missing_events,
                                 key=lambda m: -(m.get("importance") or 0)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script-ir", required=True)
    ap.add_argument("--project", default="AutomaticDrive")
    ap.add_argument("--scenes-dir", help="judge a candidate breakdown in this "
                    "directory instead of the project's own scenes")
    ap.add_argument("--map", required=True,
                    help="JSON {script_scene_index: [breakdown scene_id, ...]}")
    ap.add_argument("--model", default=ESTIMATOR_DEFAULT)
    ap.add_argument("--out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
    from pace_core.breakdown.extract_script_ir import load_model, estimate_tokens

    ir = json.loads(pathlib.Path(a.script_ir).read_text())
    mapping = json.loads(pathlib.Path(a.map).read_text())
    cfg = load_model(a.model)

    jobs = []
    for s in ir:
        ids = mapping.get(str(s["index"])) or []
        if not ids:
            continue
        jobs.append((s, load_breakdown_actions(a.project, ids, a.scenes_dir), ids))

    if a.dry_run:
        tin = sum(estimate_tokens(build_messages(s, acts)) for s, acts, _ in jobs)
        tout = 700 * len(jobs)
        cost = tin / 1000 * cfg["cost_per_1k_in"] + tout / 1000 * cfg["cost_per_1k_out"]
        print(f"estimator : {a.model} -> {cfg['model_name']}")
        for s, acts, ids in jobs:
            print(f"  script s{s['index']} ({len(s.get('events') or [])} events)"
                  f"  vs  {'+'.join(ids)} ({len(acts)} actions)")
        print(f"tokens    : ~{tin} in, ~{tout} out")
        print(f"EST COST  : ${cost:.4f}")
        return 0

    from pace_core.llm_client import call_model, strip_fences
    aligns, spent, failed = [], 0.0, []
    for s, acts, ids in jobs:
        # One scene's gateway hiccup used to abandon the whole run, throwing
        # away every scene already paid for -- a 502 on scene 6 of 8 cost the
        # five before it and bought nothing. Retry, then skip the scene and
        # keep what the run has, recording the gap rather than quietly
        # scoring a partial corpus as if it were whole.
        raw = None
        for attempt in range(3):
            try:
                raw, c = call_model(cfg, build_messages(s, acts))
                spent += c
                break
            except Exception as e:                            # noqa: BLE001
                if attempt == 2:
                    print(f"  s{s['index']} ESTIMATOR CALL FAILED (3 tries): "
                          f"{str(e)[:160]}", flush=True)
                    failed.append(s["index"])
                else:
                    print(f"  s{s['index']} retry {attempt + 1} after: "
                          f"{str(e)[:100]}", flush=True)
                    time.sleep(5 * (attempt + 1))
        if raw is None:
            continue
        try:
            al = json.loads(strip_fences(raw))
        except json.JSONDecodeError as e:
            print(f"  s{s['index']} ESTIMATOR PARSE FAIL: {e}", flush=True)
            continue
        al["script_scene"] = s["index"]; al["breakdown_scenes"] = ids
        aligns.append(al)
        ms = [m for m in al.get("matches") or [] if m.get("match") == "MISSING"]
        print(f"  s{s['index']} vs {'+'.join(ids):<20} "
              f"events={len(s.get('events') or []):>2} missing={len(ms):>2} "
              f"invented={len(al.get('invented_events') or []):>2}"
              f"  ${c:.4f}", flush=True)
    res = score(aligns, ir)
    res["alignments"] = aligns
    res["cost_usd"] = round(spent, 4)
    res["scenes_scored"] = sorted(a["script_scene"] for a in aligns)
    res["failed_scenes"] = failed
    print("\n", json.dumps(res["counts"], indent=1))
    print("coverage:", res["coverage"], f"  (${spent:.4f})")
    if failed:
        # Coverage is a ratio over the events of the scenes that ran, so a
        # skipped scene changes both halves and the number stays plausible.
        print(f"INCOMPLETE: {len(failed)} scene(s) never scored: {failed}. "
              f"Coverage above is over the {len(aligns)} that did.")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
