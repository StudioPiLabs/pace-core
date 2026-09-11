"""Pass 1: a screenplay scene into Script IR, with every fact tied to the page.

This is the ground-truth side of the breakdown verifier. It is a model call,
so it is wrong sometimes; the design that makes it usable anyway is that the
model is never asked for an offset. It is asked to QUOTE the words that support
each fact, and `ground()` then locates that quote in the source by string
search. A fact whose quote cannot be found did not come from the screenplay,
and we can say so without a second opinion.

That asymmetry is the whole point. A critic model shares the generator's
priors, so it agrees with whatever is plausible -- and the corpus's
`screen_position.x` of 0.38 is maximally plausible while being a lookup-table
constant selected by list index. Grounding is an instrument; plausibility is
not.

Usage:
    uv run python -m pace_core.breakdown.extract_script_ir SCRIPT.txt --dry-run
    uv run python -m pace_core.breakdown.extract_script_ir SCRIPT.txt --execute \
        --model claude-sonnet-5-rb --out script_ir.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

from pace_core.breakdown.screenplay_parser import parse, scenes

GENERATOR_DEFAULT = "claude-opus-5-rb"

SYSTEM = """You extract a Canonical Script IR from ONE screenplay scene.

You are building GROUND TRUTH for a verifier. Precision matters more than
richness: a fact you invent becomes a false accusation against a downstream
breakdown. Extract only what the scene's own words support.

Return STRICT JSON, no prose, no markdown fence, with this shape:

{
  "location": "<the place, as the heading names it>",
  "interior_exterior": "INT" | "EXT" | "INT/EXT",
  "time_of_day": "<DAY|NIGHT|CONTINUOUS|MOMENTS LATER|...>",
  "entities": [
    {"local_id": "old_soldier", "type": "CHARACTER|GROUP|OBJECT|LOCATION|VEHICLE|ANIMAL|ENVIRONMENT",
     "canonical_name": "OLD SOLDIER", "aliases": ["he"], "quote": "<exact substring of the scene>"}
  ],
  "events": [
    {"local_id": "e1",
     "predicate": "<SCREAMING_SNAKE canonical verb, e.g. HELP_STAND, RAISE_LADDER>",
     "predicate_text": "<the verb as written>",
     "actor": "<entity local_id or null>",
     "patient": "<entity local_id or null>",
     "target": "<entity local_id or null>",
     "instrument": "<entity local_id or null>",
     "order": <int, 1-based, the order the scene presents it>,
     "explicitness": "EXPLICIT" | "STRONG_INFERRED",
     "importance": <float 0..1>,
     "state_change": [
       {"entity": "<local_id>", "attribute": "posture|power|injured|on_fire|open|...",
        "from": "<or null>", "to": "<value>"}
     ],
     "quote": "<exact substring of the scene that states this event>"}
  ],
  "states": [
    {"entity": "<local_id>", "attribute": "injured", "value": "true",
     "quote": "<exact substring>"}
  ]
}

RULES
1. Every `quote` MUST be an EXACT substring of the scene text you were given,
   copied character for character. Do not paraphrase, do not fix typos, do not
   join two separated lines. It is checked by string search. If you cannot
   quote it, do not assert it.
2. Dialogue is evidence of speech, not of action. Do not turn a line of
   dialogue into a physical event unless the action lines say it happened.
3. explicitness EXPLICIT = the scene states it. STRONG_INFERRED = the scene
   entails it but does not write it. Do not emit anything weaker.
4. state_change is what the event leaves behind. A power loss turns a screen
   off and it STAYS off; say so with attribute/from/to.
5. Prefer several atomic events over one summarising event. Do not merge.
"""


def scene_text(source: str, sc: dict) -> str:
    return source[sc["start"]:sc["end"]]


def build_messages(source: str, sc: dict) -> list[dict]:
    body = scene_text(source, sc)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"SCENE HEADING: {sc['heading']}\n\nSCENE TEXT:\n{body}"}]


def ground(ir: dict, source: str, sc: dict) -> dict:
    """Locate every quote in the scene's own span. Adds `evidence` with real
    offsets, or marks the fact `grounded: false` and says what was claimed.

    Searching within the scene span rather than the whole script matters: a
    quote that only appears in a DIFFERENT scene is not support for a fact in
    this one, and a whole-script search would silently accept it.
    """
    lo, hi = sc["start"], sc["end"]
    window = source[lo:hi]
    ungrounded: list[dict] = []

    def locate(node: dict, kind: str) -> None:
        q = (node.get("quote") or "").strip()
        node.pop("quote", None)
        if not q:
            node["grounded"] = False
            ungrounded.append({"kind": kind, "claim": node.get("local_id"), "quote": None})
            return
        at = window.find(q)
        if at < 0:
            # A quote the model rewrote: try once with whitespace collapsed,
            # since the PDF's line breaks fall inside sentences.
            flat_w = " ".join(window.split())
            flat_q = " ".join(q.split())
            at2 = flat_w.find(flat_q)
            if at2 < 0:
                node["grounded"] = False
                node["claimed_quote"] = q
                ungrounded.append({"kind": kind, "claim": node.get("local_id"), "quote": q})
                return
            node["grounded"] = True
            node["evidence"] = {"approx": True, "source_text": flat_q}
            return
        node["grounded"] = True
        node["evidence"] = {"start": lo + at, "end": lo + at + len(q), "source_text": q}

    for e in ir.get("entities") or []:
        locate(e, "entity")
    for e in ir.get("events") or []:
        locate(e, "event")
    for s in ir.get("states") or []:
        locate(s, "state")
    ir["ungrounded"] = ungrounded
    return ir


def estimate_tokens(messages: list[dict]) -> int:
    """Rough, deliberately crude: characters/3.6. Used only to price a run
    before it happens, never to bill one after."""
    return int(sum(len(m["content"]) for m in messages) / 3.6)


def load_model(model_key: str) -> dict:
    """Resolve a model key against the one registry.

    This used to read `production/kb/behind_scene/models.json` by two
    hardcoded relative paths -- a second registry, and one the studio's model
    picker (which lists `paths.MODELS_FILE`) could not see. The three
    RouterBase entries the breakdown defaults to, this module's own
    GENERATOR_DEFAULT among them, existed only in that copy and so were
    unselectable in the UI. There is now one file and one lookup.
    """
    from pace_core.paths import MODELS_FILE
    if not MODELS_FILE.is_file():
        raise SystemExit(f"models registry not found at {MODELS_FILE}")
    reg = json.loads(MODELS_FILE.read_text())
    if model_key not in reg:
        raise SystemExit(f"model {model_key!r} not in {MODELS_FILE}")
    return reg[model_key]


def extract_scene(source: str, sc: dict, model_key: str,
                  attempts: int = 3) -> tuple[dict, float]:
    """Extract one scene, retrying a response that does not parse.

    A truncated or fenced reply is a transport accident, not a finding: the
    same prompt on the same scene parsed on the next attempt. Retrying is
    right, but silently returning an empty IR would not be -- a scene the
    extractor failed on must be visible as a failure, or it looks downstream
    exactly like a scene with nothing in it.
    """
    from pace_core.llm_client import call_model, strip_fences
    cfg = load_model(model_key)
    msgs = build_messages(source, sc)
    spent, last = 0.0, None
    for _ in range(attempts):
        raw, cost = call_model(cfg, msgs)
        spent += cost
        try:
            ir = json.loads(strip_fences(raw))
        except json.JSONDecodeError as e:                      # noqa: PERF203
            last = f"{e} (response {len(raw)} chars)"
            continue
        ir["index"] = sc["index"]
        ir["heading"] = sc["heading"]
        ir["start"], ir["end"] = sc["start"], sc["end"]
        ir["extract_ok"] = True
        return ground(ir, source, sc), spent
    return ({"index": sc["index"], "heading": sc["heading"],
             "start": sc["start"], "end": sc["end"],
             "extract_ok": False, "error": last,
             "entities": [], "events": [], "states": [], "ungrounded": []}, spent)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", help="plain-text screenplay (offsets index into it)")
    ap.add_argument("--model", default=GENERATOR_DEFAULT)
    ap.add_argument("--scene", type=int, action="append",
                    help="scene index to extract (repeatable); default all")
    ap.add_argument("--out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="price the run and print one prompt; calls nothing")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)

    source = pathlib.Path(a.script).read_text()
    sc_all = scenes(parse(source))
    want = sc_all if not a.scene else [s for s in sc_all if s["index"] in a.scene]

    if a.dry_run:
        cfg = load_model(a.model)
        tin = sum(estimate_tokens(build_messages(source, s)) for s in want)
        tout = 900 * len(want)
        cost = (tin / 1000) * cfg["cost_per_1k_in"] + (tout / 1000) * cfg["cost_per_1k_out"]
        print(f"model     : {a.model} -> {cfg['model_name']} @ {cfg['endpoint']}")
        print(f"scenes    : {len(want)} of {len(sc_all)}")
        print(f"tokens    : ~{tin} in, ~{tout} out (assumed 900/scene)")
        print(f"EST COST  : ${cost:.4f}")
        print("\n--- prompt for the first scene ---")
        print(build_messages(source, want[0])[1]["content"][:900])
        return 0

    out, total = [], 0.0
    for s in want:
        ir, cost = extract_scene(source, s, a.model)
        total += cost
        n_un = len(ir.get("ungrounded") or [])
        if not ir.get("extract_ok"):
            print(f"scene {s['index']} {s['heading'][:44]:<44} EXTRACT FAILED: "
                  f"{ir.get('error')}  ${cost:.4f}", flush=True)
            out.append(ir)
            continue
        print(f"scene {s['index']} {s['heading'][:44]:<44} "
              f"entities={len(ir.get('entities') or [])} "
              f"events={len(ir.get('events') or [])} "
              f"ungrounded={n_un}  ${cost:.4f}", flush=True)
        out.append(ir)
    print(f"\nTOTAL ${total:.4f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
