"""Beats: the smallest unit in which the world changes, not the smallest sentence.

A Beat boundary is where the story turns, so the signal for it is mostly
structural and can be measured rather than judged. This implements the design's
weighted boundary score with one honesty constraint: a feature the Script IR
cannot support is not faked.

Of the six features the design lists, three are computable from our IR:

    state_change      an event that leaves the world different -- we carry
                      structured `state_change`, so this is a lookup
    importance_delta  both events carry `importance`
    focus_shift       the actor changes; a proxy for the design's causal_shift,
                      named for what it actually measures

`spatial_shift` has no source: our events carry no location field, only the
scene's. Rather than invent one from text distance -- which measures how the
screenplay is typeset, not where anyone stands -- it is left unmeasured.
`goal_shift` and `reveal` need judgement and go to the adjudicator.

The measured weights are renormalised over what was actually measured, so the
score stays on [0,1] and a corpus missing one feature does not silently become
unable to reach the split threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# The design's weights, kept at their published values so the two can be
# compared. `spatial_shift` is declared and never emitted; see the docstring.
WEIGHTS = {
    "state_change":     0.30,
    "focus_shift":      0.20,   # design: causal_shift
    "goal_shift":       0.15,   # adjudicated
    "spatial_shift":    0.15,   # UNMEASURABLE from this IR
    "reveal":           0.10,   # adjudicated
    "importance_delta": 0.10,
}
MEASURABLE = ("state_change", "focus_shift", "importance_delta")
TAU_HIGH, TAU_LOW = 0.55, 0.20

# The design states two things that do not agree, and the difference decides
# most boundaries in a corpus like this one.
#
# Its prose says a new beat forms when ANY of the listed conditions holds --
# key entity state change, causal stage change, GOAL change, spatial change,
# REVEAL, new important entity. Its formula is a weighted sum against a single
# threshold, which requires SEVERAL at once.
#
# Measured on this screenplay the two diverge sharply. Under the formula alone,
# 40 of 41 adjudicated pairs merged: without a state change the sum tops out
# near 0.48 against a 0.55 threshold, so pairs the adjudicator scored 0.8 for
# goal shift ("Ryan shifts from grieving Emily to actively searching") and 0.8
# for reveal ("the arms swerve around Ryan, revealing they are ignoring him")
# were folded into a 13-event beat spanning a robot arm emerging through a car
# being dragged away. That is not the smallest unit in which the world changes.
#
# So the prose is implemented as well: a single strong signal splits on its own.
# The sum still decides everything below these.
STRONG_TRIGGERS = {"goal_shift": 0.7, "reveal": 0.7, "state_change": 1.0}


def strong_trigger(features: dict[str, float]) -> str | None:
    """The first condition that is on its own sufficient for a boundary."""
    for k, thr in STRONG_TRIGGERS.items():
        if float(features.get(k) or 0.0) >= thr:
            return k
    return None


def boundary_features(a: dict, b: dict) -> dict[str, float]:
    """What can be read off the pair, with no model and no guessing."""
    f = {}
    f["state_change"] = 1.0 if (b.get("state_change") or []) else 0.0
    f["focus_shift"] = 0.0 if (a.get("actor") == b.get("actor")) else 1.0
    ia, ib = a.get("importance") or 0.0, b.get("importance") or 0.0
    f["importance_delta"] = min(1.0, abs(float(ib) - float(ia)) / 0.5)
    return f


def boundary_score(features: dict[str, float]) -> float:
    """Weighted sum, renormalised over the weight actually measured.

    Without renormalising, a corpus that cannot measure spatial_shift caps at
    0.60 of the scale and the split threshold becomes quietly harder to reach
    -- the segmenter would under-split for a reason nobody could see.
    """
    tot = sum(WEIGHTS[k] for k in features if k in WEIGHTS)
    if not tot:
        return 0.0
    return sum(WEIGHTS[k] * v for k, v in features.items() if k in WEIGHTS) / tot


@dataclass
class Boundary:
    after: str                 # local_id of the event this boundary follows
    score: float
    decision: str              # SPLIT | MERGE | ADJUDICATE
    features: dict = field(default_factory=dict)
    why: str = ""


def propose(events: list[dict]) -> list[Boundary]:
    """One decision per adjacent pair, in script order."""
    evs = sorted(events, key=lambda e: e.get("order") or 0)
    out = []
    for a, b in zip(evs, evs[1:]):
        f = boundary_features(a, b)
        s = boundary_score(f)
        trig = strong_trigger(f)
        d = ("SPLIT" if (trig or s >= TAU_HIGH) else
             "MERGE" if s <= TAU_LOW else "ADJUDICATE")
        out.append(Boundary(after=a.get("local_id"), score=round(s, 4),
                            decision=d, features=f,
                            why=f"trigger:{trig}" if trig else ""))
    return out


def cut(events: list[dict], boundaries: list[Boundary]) -> list[list[dict]]:
    """Apply SPLIT decisions. MERGE and an unresolved ADJUDICATE both keep the
    events together: when in doubt a Beat stays whole, because over-splitting
    produces beats with no state change in them and those have nothing to
    depict."""
    evs = sorted(events, key=lambda e: e.get("order") or 0)
    by_after = {b.after: b for b in boundaries}
    groups, cur = [], []
    for e in evs:
        cur.append(e)
        b = by_after.get(e.get("local_id"))
        if b and b.decision == "SPLIT":
            groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    return groups


def build_beats(scene: dict, groups: list[list[dict]], timeline=None,
                scope: str | None = None) -> list[dict]:
    """Beat IR, with the state either side of the transition.

    `state_before` and `state_after` come from the world-state timeline rather
    than from the beat's own events, so a beat inherits everything true when it
    starts -- which is the property that stops a later beat quietly restoring
    an irreversible state.
    """
    out = []
    for i, g in enumerate(groups, start=1):
        first, last = g[0], g[-1]
        changes = [c for e in g for c in (e.get("state_change") or [])]
        before = after = {}
        if timeline is not None:
            def snap(order):
                return {f"{k[0]}.{k[1]}": t.to for k, t in
                        timeline.state_at(scene["index"], order, scope).items()}
            before = snap((first.get("order") or 1) - 1)
            after = snap(last.get("order") or 1)
        out.append({
            "id": f"scene_{scene['index'] + 1:02d}_beat_{i:02d}",
            "scene_id": f"scene_{scene['index'] + 1:02d}",
            "script_scene_index": scene["index"],
            "source_event_ids": [e.get("local_id") for e in g],
            "transition": [{"predicate": e.get("predicate"),
                            "actor": e.get("actor"), "patient": e.get("patient"),
                            "target": e.get("target")} for e in g],
            "state_before": before,
            "state_after": after,
            "state_changes": changes,
            "importance": max([float(e.get("importance") or 0) for e in g] or [0]),
            # A beat with nothing to show is still a beat, but the planner
            # needs to know before it tries to draw one.
            "visualizable": any((e.get("predicate_text") or "") for e in g),
            "evidence": [e.get("evidence") for e in g if e.get("evidence")],
        })
    return out


ADJUDICATOR = """You decide whether two adjacent screenplay events belong in the
SAME beat or start a NEW one.

A beat is the smallest unit in which the story's world meaningfully changes. It
is not a sentence and not a camera shot. A continuous action chain ("reaches
out -> grips the arm -> lifts") is ONE beat. A change of focus, goal, or of
what the audience now knows starts a NEW one.

You are being asked only about pairs a structural scorer could not settle. For
each, answer on two axes it cannot see:
  goal_shift : does the second event serve a different intention than the first?
  reveal     : does the second event tell the audience something new?

Return STRICT JSON, no prose, no fence:
{"decisions": [{"after": "<event id>", "goal_shift": 0.0-1.0,
                "reveal": 0.0-1.0, "why": "<one short clause>"}]}
"""


def adjudicate_messages(scene: dict, pending: list[Boundary],
                        events: list[dict]) -> list[dict]:
    by_id = {e.get("local_id"): e for e in events}
    pairs = []
    evs = sorted(events, key=lambda e: e.get("order") or 0)
    nxt = {a.get("local_id"): b for a, b in zip(evs, evs[1:])}
    for b in pending:
        a, c = by_id.get(b.after), nxt.get(b.after)
        if not a or not c:
            continue
        pairs.append({"after": b.after,
                      "first": {"predicate": a.get("predicate"),
                                "words": (a.get("evidence") or {}).get("source_text")},
                      "second": {"predicate": c.get("predicate"),
                                 "words": (c.get("evidence") or {}).get("source_text")}})
    return [{"role": "system", "content": ADJUDICATOR},
            {"role": "user", "content":
             f"SCENE: {scene['heading']}\n\nPAIRS:\n"
             f"{json.dumps(pairs, ensure_ascii=False, indent=1)}"}]


def apply_adjudication(boundaries: list[Boundary], decisions: list[dict]) -> None:
    """Fold the judged features back in and re-score against the same
    thresholds, so an adjudicated pair is decided by the same rule as a
    measured one rather than by the model's opinion of the outcome."""
    by_after = {b.after: b for b in boundaries}
    for d in decisions:
        b = by_after.get(d.get("after"))
        if b is None:
            continue
        b.features["goal_shift"] = float(d.get("goal_shift") or 0.0)
        b.features["reveal"] = float(d.get("reveal") or 0.0)
        b.score = round(boundary_score(b.features), 4)
        b.why = d.get("why") or ""
        trig = strong_trigger(b.features)
        b.decision = "SPLIT" if (trig or b.score >= TAU_HIGH) else "MERGE"
        if trig:
            b.why = f"trigger:{trig}; {b.why}"


def main(argv=None) -> int:
    import argparse, pathlib, sys
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script-ir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--model", default="claude-sonnet-5-rb")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
    from pace_core.breakdown.extract_script_ir import load_model, estimate_tokens
    from pace_core.breakdown.world_state import build as build_timeline
    import re

    ir = json.loads(pathlib.Path(a.script_ir).read_text())
    tl = build_timeline(ir)
    cfg = load_model(a.model)
    jobs = [(s, propose(s["events"])) for s in ir]
    pending = [(s, [b for b in bs if b.decision == "ADJUDICATE"])
               for s, bs in jobs]
    pending = [(s, p) for s, p in pending if p]

    if a.dry_run:
        tin = sum(estimate_tokens(adjudicate_messages(s, p, s["events"]))
                  for s, p in pending)
        tout = 90 * sum(len(p) for _, p in pending)
        cost = tin / 1000 * cfg["cost_per_1k_in"] + tout / 1000 * cfg["cost_per_1k_out"]
        print(f"adjudicator: {a.model} -> {cfg['model_name']}")
        print(f"pairs needing judgement: {sum(len(p) for _, p in pending)} "
              f"of {sum(len(bs) for _, bs in jobs)}")
        print(f"EST COST  : ${cost:.4f}")
        return 0

    from pace_core.llm_client import call_model, strip_fences
    spent = 0.0
    for s, p in pending:
        raw, c = call_model(cfg, adjudicate_messages(s, p, s["events"]))
        spent += c
        try:
            dec = json.loads(strip_fences(raw)).get("decisions") or []
        except json.JSONDecodeError as e:
            print(f"  s{s['index']} adjudication unparsed ({e}); pairs stay merged")
            continue
        bs = dict(jobs)[s["index"]] if False else next(b for sc, b in jobs if sc is s)
        apply_adjudication(bs, dec)

    beats, out = [], []
    for s, bs in jobs:
        scope = re.split(r"\s+-\s+", (s.get("heading") or "").upper())[0].strip()
        g_ = cut(s["events"], bs)
        b = build_beats(s, g_, timeline=tl, scope=scope)
        beats.extend(b)
        print(f"  s{s['index']} {s['heading'][:40]:<40} events={len(s['events']):>2} "
              f"beats={len(b):>2}  ${spent:.4f}")
        out.append({"scene": s["index"],
                    "boundaries": [vars(x) for x in bs]})
    print(f"\nbeats: {len(beats)} from {sum(len(s['events']) for s in ir)} events"
          f"   TOTAL ${spent:.4f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"beats": beats, "boundaries": out}, ensure_ascii=False, indent=1))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
