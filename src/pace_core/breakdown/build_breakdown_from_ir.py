"""Write a breakdown FROM Script IR, and make it cite what it used.

Auditing a breakdown after the fact found 59% of the screenplay's events
missing, including the climax. Auditing cannot fix that, because the generator
never knew what it was accountable to: it read prose and wrote whatever it
noticed.

Here it is handed the enumerated event list and required to name, for every
action it writes, the `script_event` ids that action realises. Two consequences
matter more than the prose it produces:

  * Coverage becomes checkable WITHOUT a model. An event whose id nobody cited
    is uncovered, and that is a set difference, not a judgement.
  * Every action inherits the evidence span of the events it cites, so the
    breakdown carries provenance for the first time -- which is the property
    whose absence let `screen_position.x` be a lookup-table constant for the
    life of the corpus.

Citing is not the same as covering: a generator can cite an id and write
something unfaithful to it. That is what the independent estimator is for.
This module only removes the failure mode where an event is dropped silently.

Usage:
    uv run python -m pace_core.breakdown.build_breakdown_from_ir \
        --script-ir script_ir.json --out-dir staging/ --dry-run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

GENERATOR_DEFAULT = "claude-opus-5-rb"

SYSTEM = """You are a first assistant director writing a shot breakdown for ONE
scene. You are given the scene's events, already extracted from the screenplay
and numbered. Your job is to group them into shots and describe each shot.

Return STRICT JSON, no prose, no fence:

{
  "shots": [
    {"shot_id": "shot_01",
     "actions": [
       {"description_en": "<what the camera sees, one sentence, present tense>",
        "script_events": ["e1", "e2"]}
     ],
     "subjects": ["<entity local_id>", ...],
     "state_notes": [
       {"entity": "<local_id>", "attribute": "power", "value": "off",
        "since_event": "<event id that caused it>"}
     ]}
  ]
}

RULES
1. EVERY event id you were given must appear in exactly one action's
   `script_events`. This is checked mechanically. If an event is hard to
   stage, it still gets an action -- a beat you cannot shoot is a note for
   the director, not something to drop.
2. Do not invent events. If you did not receive it, do not describe it.
3. Group events into shots the way coverage actually works: a continuous
   camera on a continuous action. A shot may carry several events; an event
   belongs to exactly one shot.
4. `state_notes` records what is true AFTER the shot because of it -- a
   panel that lost power stays off. Carry only states your events caused.
5. description_en describes the IMAGE, not the plot. "The car falls on top of
   him" not "Omar dies tragically".
"""


def build_messages(scene: dict) -> list[dict]:
    evs = [{"id": e.get("local_id"), "predicate": e.get("predicate"),
            "actor": e.get("actor"), "patient": e.get("patient"),
            "order": e.get("order"),
            "state_change": e.get("state_change") or [],
            "script_words": (e.get("evidence") or {}).get("source_text")}
           for e in scene.get("events") or []]
    ents = [{"id": e.get("local_id"), "type": e.get("type"),
             "name": e.get("canonical_name")} for e in scene.get("entities") or []]
    user = (f"SCENE: {scene['heading']}\n\n"
            f"ENTITIES:\n{json.dumps(ents, ensure_ascii=False, indent=1)}\n\n"
            f"EVENTS (every id must be cited exactly once):\n"
            f"{json.dumps(evs, ensure_ascii=False, indent=1)}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def check_citations(scene: dict, shots: list[dict]) -> dict:
    """Set difference, no model. Which events were cited, missed, invented,
    or cited more than once."""
    have = {e.get("local_id") for e in scene.get("events") or []}
    cited: dict[str, int] = {}
    for sh in shots:
        for a in sh.get("actions") or []:
            for eid in a.get("script_events") or []:
                cited[eid] = cited.get(eid, 0) + 1
    return {"events": len(have),
            "cited": len([e for e in have if e in cited]),
            "uncited": sorted(e for e in have if e not in cited),
            "invented": sorted(e for e in cited if e not in have),
            "duplicated": sorted(e for e, n in cited.items() if n > 1 and e in have)}


def to_scene_doc(scene: dict, shots: list[dict], project_scene_id: str) -> dict:
    """PACE-shaped, with every action carrying the spans it cites."""
    by_id = {e.get("local_id"): e for e in scene.get("events") or []}
    out_shots = []
    for i, sh in enumerate(shots, start=1):
        actions = []
        for a in sh.get("actions") or []:
            ids = a.get("script_events") or []
            ev = [by_id[e] for e in ids if e in by_id]
            actions.append({
                "description_en": a.get("description_en"),
                "standalone": a.get("description_en"),
                "temporal": "atomic", "foreground": "focal", "background": False,
                # The provenance that the old breakdown had nowhere to put.
                "provenance": {
                    "source": "screenplay",
                    "script_events": ids,
                    "evidence": [e.get("evidence") for e in ev if e.get("evidence")],
                },
            })
        out_shots.append({
            "shot_id": sh.get("shot_id") or f"shot_{i:02d}",
            "events": {"actions": actions,
                       "state_notes": sh.get("state_notes") or []},
            "setup": {"subjects": [{"character_id": s} for s in (sh.get("subjects") or [])]},
        })
    return {"_schema_version": "pai-1.1",
            "scene_id": project_scene_id,
            "scene_heading": scene["heading"],
            "script_scene_index": scene["index"],
            "source_span": {"start": scene["start"], "end": scene["end"]},
            "shots": out_shots}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script-ir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=GENERATOR_DEFAULT)
    ap.add_argument("--scene", type=int, action="append")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
    from pace_core.breakdown.extract_script_ir import load_model, estimate_tokens

    ir = json.loads(pathlib.Path(a.script_ir).read_text())
    want = ir if not a.scene else [s for s in ir if s["index"] in a.scene]
    cfg = load_model(a.model)

    if a.dry_run:
        tin = sum(estimate_tokens(build_messages(s)) for s in want)
        tout = 1300 * len(want)
        cost = tin / 1000 * cfg["cost_per_1k_in"] + tout / 1000 * cfg["cost_per_1k_out"]
        print(f"generator : {a.model} -> {cfg['model_name']}")
        print(f"scenes    : {len(want)} | events: {sum(len(s.get('events') or []) for s in want)}")
        print(f"tokens    : ~{tin} in, ~{tout} out")
        print(f"EST COST  : ${cost:.4f}")
        return 0

    from pace_core.llm_client import call_model, strip_fences
    out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    spent, report = 0.0, []
    for s in want:
        sid = f"scene_{s['index'] + 1:02d}"
        shots, last = None, None
        for _ in range(3):
            raw, c = call_model(cfg, build_messages(s))
            spent += c
            try:
                shots = json.loads(strip_fences(raw)).get("shots") or []
                break
            except json.JSONDecodeError as e:
                last = str(e)
        if shots is None:
            print(f"  {sid} GENERATE FAILED: {last}", flush=True)
            continue
        chk = check_citations(s, shots)
        doc = to_scene_doc(s, shots, sid)
        (out_dir / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        report.append({"scene": s["index"], **chk})
        print(f"  {sid} {s['heading'][:40]:<40} shots={len(shots):>2} "
              f"events={chk['events']:>2} cited={chk['cited']:>2} "
              f"uncited={len(chk['uncited'])} invented={len(chk['invented'])}"
              f"  ${spent:.4f}", flush=True)
    tot = sum(r["events"] for r in report); cit = sum(r["cited"] for r in report)
    print(f"\ncitation coverage (no model): {cit}/{tot} = {cit / tot:.3f}" if tot else "")
    print(f"TOTAL ${spent:.4f}")
    (out_dir / "_citation_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
