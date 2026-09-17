#!/usr/bin/env python3
"""Synthetic error injection, to measure the verifier instead of trusting it.

Every number this project reports about a breakdown -- coverage 0.37 before,
0.95 after -- comes out of `verify_breakdown`, and nothing has ever measured
`verify_breakdown`. The design says so directly (Script Breakdown Verifier,
sections 37-38): inject known errors, then score the verifier on whether it
found them, because "how far the score is from a human's" says much less than

    Error Detection Precision = true_detected / all_detected
    Error Detection Recall    = true_detected / all_ground_truth

This builds the mutated breakdowns. `score_detection` scores the runs.

WHICH MUTATIONS ARE MEASURABLE, AND WHY THAT IS THE POINT
---------------------------------------------------------
The design lists twelve mutations. Our estimator emits four signals -- MISSING,
`invented_events`, `role_ok`, OVER_GENERALIZED -- plus `over_specified`, so
only five of the twelve can land anywhere in its output:

    DeleteEvent           -> the event it covered should become MISSING
    InsertEvent           -> the new action should appear in invented_events
    SwapActor             -> role_ok should go false
    SwapTarget            -> role_ok should go false
    GeneralizePredicate   -> the match should become OVER_GENERALIZED
    SpecializePredicate   -> over_specified, and NOT invented_events

That last one is the sharpest test here. Separating "the breakdown invented an
event" from "the breakdown added craft detail, which is its job" is the
distinction that took this estimator's flagged-error rate from 97% of actions
to 4 events, and a mutation that adds staging language to a real event is
exactly the input that collapses the two back together.

The other seven need no API call to evaluate, and `verify_breakdown` cannot
see any of them. That is not a prediction: `load_breakdown_actions` reads
`shots[].events.actions[]` and `change_in_environment` and nothing else, so a
mutation to a location, a state or a relation leaves the estimator's prompt
byte-identical. `NO_CHANNEL` records which, and the test suite proves it by
comparing the compiled messages rather than by asserting it in prose.

    uv run python -m pace_core.breakdown.inject_errors \\
        --project <slug> --baseline eval_existing_breakdown.json \\
        --script-ir script_ir.json --map scene_map_existing.json \\
        --out-dir /tmp/mutated --manifest /tmp/manifest.json
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import re
import sys
from dataclasses import asdict, dataclass

# Mutations the estimator's output schema has somewhere to report.
WITH_CHANNEL = ("DeleteEvent", "InsertEvent", "SwapActor", "SwapTarget",
                "GeneralizePredicate", "SpecializePredicate")

# The rest of the design's list. These fail for two different reasons and the
# difference matters, so they are not pooled into one excuse.
#
# INVISIBLE_TO_PROMPT: the mutation edits a field the estimator is never shown.
# `load_breakdown_actions` reads `shots[].events.actions[]` and
# `change_in_environment`, and nothing else in a scene document reaches the
# prompt -- so the estimator is handed a byte-identical message and cannot
# possibly respond to the change. tests/test_inject_errors.py proves this by
# compiling the messages before and after rather than asserting it in prose,
# which is why none of these needs an API call to evaluate.
INVISIBLE_TO_PROMPT = {
    "ChangeLocation": "WRONG_LOCATION -- the prompt carries the SCRIPT's scene "
                      "heading and the script events; a breakdown's own "
                      "location field is never read",
    "RemoveState": "STATE_MISSING -- no state on a breakdown action reaches "
                   "the prompt",
    "ContradictState": "STATE_CONTRADICTION -- likewise. world_state detects "
                       "contradictions against the SCRIPT, which is a "
                       "different check, on a different input, in a different "
                       "module",
    "DropRelation": "MISSING_RELATION -- there is no Relation IR to drop from",
    "InvertRelation": "WRONG_RELATION -- likewise",
    "ReplaceCostume": "costume is not in either IR",
}

# NO_OUTPUT_FIELD: the mutation DOES change the prompt, and the estimator still
# cannot report it, because its response schema has nowhere to put it. Worth
# separating: closing these needs a new output field, not a new input.
NO_OUTPUT_FIELD = {
    "SwapOrder": "TEMPORAL_ORDER_ERROR -- reordering actions does change the "
                 "prompt, but the schema asks only whether each script event "
                 "is covered and by which actions. There is no field for "
                 "sequence, so a breakdown that tells the story backwards "
                 "scores exactly as well as one that does not",
}

NO_CHANNEL = {**INVISIBLE_TO_PROMPT, **NO_OUTPUT_FIELD}

# What SpecializePredicate appends: craft detail a director legitimately adds,
# which the estimator must record as over-specification and must not mistake
# for a fabricated event.
#
# Strictly camera and grade, with no pronoun and no new participant. An earlier
# draft ended one with "his face lit harshly from below" and appended it to a
# shot of vehicles on a highway, which introduces a person the scene does not
# contain -- and being flagged as an invented event would then have been
# CORRECT, quietly turning the one test that separates craft detail from
# fabrication into a test of nothing.
CRAFT_DETAIL = (
    ", shot in a tight close-up with the lens wide open",
    ", framed wide and low, the camera drifting slowly right",
    ", held in a lingering static frame with shallow focus",
    ", cut as a rapid handheld insert, colours pushed cold",
)

# What InsertEvent asserts: a whole event with a participant the scene never
# puts there. Drawn from the film's own world so it is plausible rather than
# absurd -- an estimator that only catches nonsense is not catching much.
INVENTED_ACTIONS = (
    "A uniformed police officer steps into frame and salutes the passengers.",
    "A paramedic kneels beside the road and opens a medical case.",
    "A news drone descends and hovers level with the windscreen.",
    "A tow-truck driver climbs out of his cab and waves the traffic past.",
)


@dataclass
class Mutation:
    """One injected error and the signal the verifier must produce for it."""
    kind: str
    scene_index: int            # script scene, so expectations can be keyed
    breakdown_scene: str
    action_id: str              # the action mutated, inserted or deleted
    script_event: str | None    # the event whose treatment should change
    expect: str                 # signal name: MISSING | INVENTED | ROLE | ...
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# ── locating things ───────────────────────────────────────────────────────

def _split(action_id: str) -> tuple[str, str, str]:
    """`scene_06/shot_02/a0` -> (scene, shot, a0)."""
    sid, shot, tail = action_id.split("/")
    return sid, shot, tail


def _actions_of(doc: dict, shot_id: str) -> list:
    for sh in doc.get("shots") or []:
        if sh.get("shot_id") == shot_id:
            return ((sh.setdefault("events", {})).setdefault("actions", []))
    raise KeyError(shot_id)


def _action_text(a) -> str:
    return (a.get("description_en") or a.get("standalone") or "") if isinstance(a, dict) else str(a)


def _set_action_text(a, text: str):
    if isinstance(a, dict):
        a["description_en" if a.get("description_en") else "standalone"] = text
        return a
    return text


def surface_names(entity: dict) -> list[str]:
    """Every way the script IR says this entity might be written."""
    out = [entity.get("canonical_name") or ""] + list(entity.get("aliases") or [])
    return [n for n in out if n and len(n) > 2]


def _find_name(text: str, names: list[str]) -> str | None:
    for n in names:
        if re.search(r"\b" + re.escape(n) + r"\b", text, re.I):
            return n
    return None


def swap_names(text: str, a: str, b: str) -> str:
    """Exchange two names in one pass, so the second does not undo the first.

    Each name is put back in the SURFACE form the text used, not in the script
    IR's canonical form. The IR spells characters in screenplay caps, so
    substituting it directly produced "wreckage limps toward the BOB" -- a
    sentence carrying a second, louder signal than the role swap being tested,
    and a mutation that is obvious for the wrong reason measures the wrong
    thing.
    """
    pa = re.compile(r"\b" + re.escape(a) + r"\b", re.I)
    pb = re.compile(r"\b" + re.escape(b) + r"\b", re.I)
    ma, mb = pa.search(text), pb.search(text)
    if not ma or not mb:
        return text
    sa, sb = ma.group(0), mb.group(0)
    tok = "\x00SWAP\x00"
    t = pa.sub(tok, text)
    t = pb.sub(sa, t)
    return t.replace(tok, sb)


# ── planning ──────────────────────────────────────────────────────────────

def plan(baseline: dict, script_ir: list[dict], actions: dict[str, str],
         mapping: dict, *, per_scene: int = 3, seed: int = 11,
         kinds: tuple[str, ...] = WITH_CHANNEL) -> list[Mutation]:
    """Choose mutations from what the baseline run actually matched.

    Mutations are defined against the verifier's OWN prior output, which is
    what makes the expectation checkable: DeleteEvent is only meaningful on an
    event the verifier currently says is covered, and SwapActor only on a match
    it currently says has its roles right. An injected error the verifier was
    already failing to see proves nothing about whether it can see it.
    """
    rng = random.Random(seed)
    by_index = {s["index"]: s for s in script_ir}
    baseline_signals = signals(baseline)
    out: list[Mutation] = []

    for al in baseline.get("alignments") or []:
        si = al.get("script_scene")
        scene = by_index.get(si)
        bscenes = mapping.get(str(si)) or []
        if scene is None or not bscenes:
            continue
        ents = {e["local_id"]: e for e in scene.get("entities") or []}
        evs = {e["local_id"]: e for e in scene.get("events") or []}
        covered = [m for m in (al.get("matches") or [])
                   if m.get("match") in ("EXACT", "SEMANTIC")
                   and m.get("breakdown_actions")]
        pool: list[Mutation] = []

        for m in covered:
            aid = m["breakdown_actions"][0]
            ev = evs.get(m.get("script_event") or "")
            if aid not in actions or ev is None:
                continue
            # DeleteEvent: only where this action is the sole cover of exactly
            # this event, so the expectation is unambiguous.
            sole = [c for c in covered if aid in (c.get("breakdown_actions") or [])]
            if len(sole) == 1:
                pool.append(Mutation("DeleteEvent", si, bscenes[0], aid,
                                     ev["local_id"], "MISSING",
                                     "the only action covering this event is removed"))
            # Role swap, where both participants are actually named in the text.
            text = actions[aid]
            for kind, other_key in (("SwapActor", "patient"), ("SwapTarget", "target")):
                other = ev.get(other_key)
                if not ev.get("actor") or not other or m.get("role_ok") is False:
                    continue
                na = _find_name(text, surface_names(ents.get(ev["actor"]) or {}))
                nb = _find_name(text, surface_names(ents.get(other) or {}))
                if na and nb and na.lower() != nb.lower():
                    pool.append(Mutation(kind, si, bscenes[0], aid,
                                         ev["local_id"], "ROLE",
                                         f"{na} <-> {nb}"))
            if m.get("match") != "OVER_GENERALIZED":
                pool.append(Mutation("GeneralizePredicate", si, bscenes[0], aid,
                                     ev["local_id"], "OVERGEN",
                                     "predicate replaced by a generic one, roles kept"))
            # Only where the baseline did NOT already call this action
            # over-specified. A delta cannot show a signal that was already
            # there, and the first run learned this the expensive way: all
            # four SpecializePredicate targets were already flagged, so the
            # kind scored 0.00 recall while the estimator had in fact done
            # nothing wrong. On a breakdown rich in craft detail that is nearly
            # every action, which is itself the finding: a breakdown already
            # saturated with craft detail leaves no headroom to inject more.
            if ("OVERSPEC", si, aid) not in baseline_signals:
                pool.append(Mutation("SpecializePredicate", si, bscenes[0], aid,
                                     ev["local_id"], "OVERSPEC",
                                     "craft detail appended to a real event"))

        # InsertEvent needs no baseline match: any shot can receive one.
        any_aid = next((a for a in actions if a.startswith(bscenes[0] + "/")), None)
        if any_aid:
            pool.append(Mutation("InsertEvent", si, bscenes[0], any_aid,
                                 None, "INVENTED",
                                 "an event with a participant the scene never places"))

        out.extend(m for m in pool if m.kind in kinds)

    # Select round-robin over kinds rather than taking whatever each scene
    # offers most of. Left to itself the pool is dominated by the mutations
    # every matched action admits -- a first pass drew 8 GeneralizePredicate
    # against 1 SwapActor -- and a recall figure averaged over a sample that
    # lopsided says almost nothing about the rare kind.
    #
    # At most one mutation per SHOT. Two edits to one text fight over it, and
    # a DeleteEvent renumbers every later action in its shot, which would
    # silently repoint both the manifest and the baseline alignment at a
    # different action than the one mutated. Where every shot carries exactly one
    # action, this is also one mutation per action.
    by_kind: dict[str, list[Mutation]] = {}
    for m in out:
        by_kind.setdefault(m.kind, []).append(m)
    for v in by_kind.values():
        rng.shuffle(v)

    chosen: list[Mutation] = []
    taken_shots: set[str] = set()
    per_scene_count: dict[int, int] = {}
    while any(by_kind.values()):
        for k in sorted(by_kind):
            queue = by_kind[k]
            while queue:
                m = queue.pop()
                shot = "/".join(m.action_id.split("/")[:2])
                if shot in taken_shots:
                    continue
                if per_scene_count.get(m.scene_index, 0) >= per_scene:
                    continue
                taken_shots.add(shot)
                per_scene_count[m.scene_index] = per_scene_count.get(m.scene_index, 0) + 1
                chosen.append(m)
                break
    return sorted(chosen, key=lambda m: (m.scene_index, m.action_id))


# ── applying ──────────────────────────────────────────────────────────────

def apply(mutations: list[Mutation], docs: dict[str, dict],
          script_ir: list[dict]) -> list[Mutation]:
    """Edit the scene documents in place. Returns the mutations that landed.

    Deletions are applied last and by identity, because removing an action
    renumbers every later action in its shot -- and the ids in the manifest,
    and in the baseline alignment, are positional.
    """
    by_index = {s["index"]: s for s in script_ir}
    landed: list[Mutation] = []
    seen: dict[str, int] = {}
    to_delete: list[tuple[dict, int]] = []

    for mu in mutations:
        sid, shot, tail = _split(mu.action_id)
        doc = docs.get(sid)
        if doc is None:
            continue
        try:
            acts = _actions_of(doc, shot)
        except KeyError:
            continue
        if tail == "env":
            continue
        i = int(tail[1:])
        if i >= len(acts):
            continue

        # Rotate WITHIN a kind. Counting all landed mutations made every
        # SpecializePredicate land on the same phrase, because the kinds
        # interleave at a fixed stride.
        n = seen.get(mu.kind, 0)
        seen[mu.kind] = n + 1
        if mu.kind == "DeleteEvent":
            to_delete.append((doc, i, shot))
            landed.append(mu)
        elif mu.kind == "InsertEvent":
            # Rotate rather than draw: independent draws gave three scenes
            # the same invented paramedic, and a verifier that has already
            # rejected that sentence once is not being asked a new question.
            text = INVENTED_ACTIONS[n % len(INVENTED_ACTIONS)]
            acts.append(_set_action_text(copy.deepcopy(acts[i]), text)
                        if isinstance(acts[i], dict) else text)
            # The inserted action's id is its new position in the shot.
            mu = Mutation(mu.kind, mu.scene_index, mu.breakdown_scene,
                          f"{sid}/{shot}/a{len(acts) - 1}", None,
                          mu.expect, mu.note)
            landed.append(mu)
        elif mu.kind in ("SwapActor", "SwapTarget"):
            a, b = mu.note.split(" <-> ")
            acts[i] = _set_action_text(acts[i],
                                       swap_names(_action_text(acts[i]), a, b))
            landed.append(mu)
        elif mu.kind == "GeneralizePredicate":
            scene = by_index.get(mu.scene_index) or {}
            ents = {e["local_id"]: e for e in scene.get("entities") or []}
            ev = next((e for e in scene.get("events") or []
                       if e["local_id"] == mu.script_event), None)
            who = ((ents.get((ev or {}).get("actor") or "") or {})
                   .get("canonical_name") or "someone").lower()
            acts[i] = _set_action_text(acts[i], f"{who} does something here.")
            landed.append(mu)
        elif mu.kind == "SpecializePredicate":
            extra = CRAFT_DETAIL[n % len(CRAFT_DETAIL)]
            t = _action_text(acts[i]).rstrip().rstrip(".")
            acts[i] = _set_action_text(acts[i], t + extra + ".")
            landed.append(mu)

    for doc, i, shot in sorted(to_delete, key=lambda d: -d[1]):
        del _actions_of(doc, shot)[i]
    return landed


# ── scoring the verifier ──────────────────────────────────────────────────

def signals(result: dict) -> set[tuple]:
    """Every error the verifier reported, as comparable tuples.

    Each is (KIND, script_scene, key), where key is the script event for
    event-scoped signals and the action id for action-scoped ones. Tagging
    both with the scene is what lets a run that lost a scene be compared
    against one that did not.
    """
    out: set[tuple] = set()
    for al in result.get("alignments") or []:
        si = al.get("script_scene")
        for m in al.get("matches") or []:
            k, ev = m.get("match"), m.get("script_event")
            if k == "MISSING":
                out.add(("MISSING", si, ev))
            elif k == "OVER_GENERALIZED":
                out.add(("OVERGEN", si, ev))
            # Mirrors verify_breakdown.score: role_ok on a MISSING event is
            # vacuous -- nothing matched, so nothing was swapped -- and
            # counting those once turned 2 role errors into 30.
            if m.get("role_ok") is False and k != "MISSING":
                out.add(("ROLE", si, ev))
        for x in al.get("invented_events") or []:
            out.add(("INVENTED", si, x.get("action")))
        for x in al.get("over_specified") or []:
            out.add(("OVERSPEC", si, x.get("action")))
    return out


def expected_signal(mu: Mutation) -> tuple:
    key = mu.action_id if mu.expect in ("INVENTED", "OVERSPEC") else mu.script_event
    return (mu.expect, mu.scene_index, key)


def score_detection(baseline: dict, mutated: dict,
                    mutations: list[Mutation]) -> dict:
    """Precision and recall of the verifier against known injected errors.

    Scored on the DELTA between the two runs, not on the mutated run alone.
    The baseline breakdown already contains real errors -- genuinely missing
    events -- so an absolute count would credit the verifier
    for finding faults nobody injected. What an injected error must do is make
    a signal APPEAR that was not there before.

    The model is not deterministic, so some signals appear and vanish on their
    own. That is what a control run with no mutations measures, and it is the
    only thing that makes these numbers readable: run this with `mutations=[]`
    over two unmutated runs and everything in `spurious` is drift.
    """
    scored = ({a.get("script_scene") for a in baseline.get("alignments") or []}
              & {a.get("script_scene") for a in mutated.get("alignments") or []})
    dropped = sorted({m.scene_index for m in mutations} - scored)
    # A run that lost a scene to a gateway failure has no signals from it,
    # which would read as every error injected there going undetected -- an
    # outage reported as a verifier blind spot.
    mutations = [m for m in mutations if m.scene_index in scored]

    keep = lambda sig: sig[1] in scored              # noqa: E731
    base = {s for s in signals(baseline) if keep(s)}
    mut = {s for s in signals(mutated) if keep(s)}
    expect = {expected_signal(m) for m in mutations}
    appeared = mut - base
    found = expect & appeared
    spurious = appeared - expect

    # A signal already present in the baseline can never "appear", so a
    # mutation aimed at one is unmeasurable by this method rather than missed.
    # Reported separately: scoring it as a failure blames the estimator for an
    # error in the experiment.
    blind = [m for m in mutations if expected_signal(m) in base]
    blind_ids = {id(m) for m in blind}

    def any_new_signal(m: Mutation) -> bool:
        """Did ANY new error land on this mutation's own target?

        The strict measure demands the exact code, and the estimator often
        grades a mutation differently but correctly -- calling a generalised
        predicate MISSING rather than OVER_GENERALIZED. That is a detection
        under a different name, and conflating it with a miss understates the
        verifier badly.
        """
        key = expected_signal(m)
        return any(s[1] == key[1] and s[2] == key[2] for s in appeared)

    per_kind: dict[str, dict] = {}
    for m in mutations:
        row = per_kind.setdefault(m.kind, {"injected": 0, "detected": 0,
                                           "detected_any_signal": 0,
                                           "unmeasurable": 0})
        row["injected"] += 1
        row["detected"] += int(expected_signal(m) in appeared)
        row["detected_any_signal"] += int(any_new_signal(m))
        row["unmeasurable"] += int(id(m) in blind_ids)
    for row in per_kind.values():
        row["recall"] = round(row["detected"] / row["injected"], 4)

    measurable = [m for m in mutations if id(m) not in blind_ids]
    strict_m = sum(expected_signal(m) in appeared for m in measurable)
    any_m = sum(any_new_signal(m) for m in measurable)

    # The specific confusion this estimator was fixed for: craft detail added
    # to a real event must be over-specification, never a fabricated event.
    confused = [m.action_id for m in mutations
                if m.kind == "SpecializePredicate"
                and ("INVENTED", m.scene_index, m.action_id) in appeared]

    return {
        "scenes_scored": sorted(x for x in scored if x is not None),
        "scenes_dropped": dropped,
        "injected": len(expect),
        "signals_that_appeared": len(appeared),
        "true_detected": len(found),
        "recall": round(len(found) / len(expect), 4) if expect else None,
        "precision": round(len(found) / len(appeared), 4) if appeared else None,
        # The same numbers over the mutations this method can actually see.
        "measurable": len(measurable),
        "unmeasurable_already_flagged": [m.action_id for m in blind],
        "recall_measurable": (round(strict_m / len(measurable), 4)
                              if measurable else None),
        # Detection under any error code, which is what "did the verifier
        # notice" means when the codes themselves overlap.
        "recall_any_signal": (round(any_m / len(measurable), 4)
                              if measurable else None),
        "undetected_entirely": sorted(
            f"{m.kind} {m.scene_index} {expected_signal(m)[2]}"
            for m in measurable if not any_new_signal(m)),
        "per_kind": per_kind,
        "craft_detail_misread_as_invention": confused,
        "missed": sorted(str(s) for s in expect - appeared),
        "spurious": sorted(str(s) for s in spurious),
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def _load_docs(scenes_dir: pathlib.Path, ids: set[str]) -> dict[str, dict]:
    out = {}
    for sid in sorted(ids):
        f = scenes_dir / f"{sid}.json"
        if f.is_file():
            out[sid] = json.loads(f.read_text())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write a mutated breakdown + manifest")
    b.add_argument("--project", required=True, help="project slug")
    b.add_argument("--baseline", required=True, help="a prior verify_breakdown result")
    b.add_argument("--script-ir", required=True)
    b.add_argument("--map", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--manifest", required=True)
    b.add_argument("--per-scene", type=int, default=3)
    b.add_argument("--seed", type=int, default=11)
    b.add_argument("--control", action="store_true",
                   help="copy the breakdown unmutated, for the noise floor")

    s = sub.add_parser("score", help="precision/recall against a manifest")
    s.add_argument("--baseline", required=True)
    s.add_argument("--mutated", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--out")

    a = ap.parse_args(argv)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))

    if a.cmd == "score":
        mus = [Mutation(**m) for m in json.loads(
            pathlib.Path(a.manifest).read_text())["mutations"]]
        res = score_detection(json.loads(pathlib.Path(a.baseline).read_text()),
                              json.loads(pathlib.Path(a.mutated).read_text()), mus)
        print(json.dumps(res, indent=1))
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
        return 0

    from pace_core.breakdown.verify_breakdown import load_breakdown_actions
    from pace_core.paths import paths_for

    baseline = json.loads(pathlib.Path(a.baseline).read_text())
    script_ir = json.loads(pathlib.Path(a.script_ir).read_text())
    mapping = json.loads(pathlib.Path(a.map).read_text())
    ids = {sid for v in mapping.values() for sid in v}
    scenes_dir = pathlib.Path(paths_for(a.project).scenes_dir)
    docs = _load_docs(scenes_dir, ids)

    actions = {x["id"]: x["text"]
               for v in mapping.values()
               for x in load_breakdown_actions(a.project, v)}

    landed: list[Mutation] = []
    if not a.control:
        wanted = plan(baseline, script_ir, actions, mapping,
                      per_scene=a.per_scene, seed=a.seed)
        landed = apply(wanted, docs, script_ir)

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid, doc in docs.items():
        (out_dir / f"{sid}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1))
    pathlib.Path(a.manifest).write_text(json.dumps(
        {"control": bool(a.control), "seed": a.seed,
         "no_channel": NO_CHANNEL,
         "mutations": [m.as_dict() for m in landed]}, indent=1))

    kinds: dict[str, int] = {}
    for m in landed:
        kinds[m.kind] = kinds.get(m.kind, 0) + 1
    print(f"{'control (no mutations)' if a.control else str(len(landed)) + ' mutations'}"
          f" -> {out_dir}")
    for k, n in sorted(kinds.items()):
        print(f"  {n:>2} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
