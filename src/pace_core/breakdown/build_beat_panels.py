"""Beat IR -> a PACE panel, staged as geometry rather than described in prose.

One panel per beat. The panel is not a camera shot: per the design's technical
camera, it picks only coarse framing sufficient to read the beat's facts, and
does not decide lens aesthetics or motion.

Three things are NOT left to the model, because each is a defect this pipeline
already measured:

  subjects   come from the beat's own actors. The corpus's cast lists were
             assembled from prose and rendered four people for an enumerated
             three-person scene, so the count is geometry, not a request.
  state      comes from the world state, written onto the props. A prop whose
             state contradicts its registry appearance is now rewritten by the
             compiler, so a beat after a blackout cannot compile a glowing
             panel. The beat that causes the change stages what it starts
             from, so the change falls between two panels rather than before
             both of them (see `prop_states`).
  provenance every action keeps the beat and the screenplay span it came from.

What the model does decide is blocking and framing, which need judgement.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

GENERATOR_DEFAULT = "claude-opus-5-rb"

# A beat names entities in the screenplay's words ("the panels"); the registry
# names props by id. The project supplies the map (--prop-map), kept apart by
# kind: "state" for what a state_after names, "event" for the actor of an event
# that moves a prop. A name with no entry is taken to be a prop id.
DARK_VALUES = {"off", "closed"}

# A beat's events name everyone involved, including people who are only HEARD.
# A scream from someone trapped inside a car, staged like any other actor, put
# a second person on their feet beside it, whom the frame reads as the
# protagonist. A vocal sound aimed at no one is heard, not seen; one aimed at
# someone ("he SOBS over her body") places its actor beside them.
SOUND_WORDS = {"SCREAM", "SCREAMS", "SHOUT", "SHOUTS", "YELL", "YELLS",
               "WAIL", "WAILS", "MOAN", "MOANS", "CRY", "CRIES", "SOB", "SOBS"}


def heard_only(t: dict) -> bool:
    """A vocal sound with no patient or target: evidence of a voice, not a body."""
    words = str(t.get("predicate") or "").upper().split("_")
    return (any(w in SOUND_WORDS for w in words)
            and not t.get("patient") and not t.get("target"))

SYSTEM = """You stage ONE beat as a single storyboard panel.

You get the beat's events, the state of the world before and after it, and the
location. Decide only what needs judgement: where the people stand relative to
each other and the camera, and how wide the frame is.

Return STRICT JSON, no prose, no fence:

{
  "framing": "establishing" | "wide" | "full" | "medium_full" | "medium"
             | "medium_close_up" | "close_up",
  "angle": "eye_level" | "high" | "low" | "overhead",
  "camera_position": "front" | "three_quarter",
  "subjects": [
    {"character_id": "<one of the human actors given>",
     "screen_x": 0.0-1.0, "depth": "foreground"|"midground"|"background",
     "pose": "<plain words: standing, kneeling, lying, crouching, pushing>",
     "gaze_at": "<entity id or null>"}
  ],
  "action_en": "<one sentence, present tense, what the CAMERA SEES>",
  "why_framing": "<one short clause>"
}

RULES
1. `subjects` contains exactly the `human_actors` given. Do not add a
   bystander, and do not drop one of them. Anyone in `heard_actors` is only
   heard in this beat -- a scream from inside a car, a voice off -- and is in
   neither `subjects` nor the picture action_en describes.
2. screen_x is where the subject sits across the frame, 0 at frame left. Give
   distinct values; two people do not occupy one position.
3. `pose` describes the body, not the mood. "kneeling" not "grief-stricken".
4. action_en describes the image. "He lies still under the collapsed shelf",
   not "He dies tragically".
5. Choose framing for LEGIBILITY of the beat's own event, not for drama.
"""


def human_actors(beat: dict, cast: set[str]) -> list[str]:
    """The people the beat shows: anyone in an event that is not only a sound."""
    out = []
    for t in beat.get("transition") or []:
        if heard_only(t):
            continue
        for k in ("actor", "patient", "target"):
            v = t.get(k)
            if v in cast and v not in out:
                out.append(v)
    return out


def heard_actors(beat: dict, cast: set[str]) -> list[str]:
    """The people the beat only hears."""
    shown = set(human_actors(beat, cast))
    return [t["actor"] for t in beat.get("transition") or []
            if heard_only(t) and t.get("actor") in cast and t["actor"] not in shown]


def build_messages(beat: dict, cast: set[str], location: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({
                "location": location,
                "human_actors": human_actors(beat, cast),
                "heard_actors": heard_actors(beat, cast),
                "events": beat.get("transition"),
                "state_before": beat.get("state_before"),
                "state_after": beat.get("state_after"),
                "screenplay_words": [e.get("source_text")
                                     for e in beat.get("evidence") or []],
            }, ensure_ascii=False, indent=1)}]


def prop_states(beat: dict, state_map: dict | None = None) -> dict[str, str]:
    """World state -> the props that carry it, in the schema's flat form.

    A beat that CHANGES a state stages the state it starts in, not the one it
    ends in. Taking `state_after` everywhere made the panel depicting "the
    lamps go out" already dark, and the next panel dark as well, so the
    one thing the two panels existed to show -- the change -- appeared in
    neither: a storyboard that cannot show a change is not doing its job. The
    beat that turns the lamps off therefore stages them on, and the beat
    after it inherits them off, which is what puts the change between the two
    frames. A beat that only inherits a state stages it as it stands.
    """
    before = beat.get("state_before") or {}
    after = beat.get("state_after") or {}
    out = {}
    for key, val in after.items():
        was = before.get(key)
        changed_here = key in before and str(was).lower() != str(val).lower()
        shown = was if changed_here else val
        ent, _, attr = key.partition(".")
        if str(shown).lower() not in DARK_VALUES:
            continue
        for pid in (state_map or {}).get(ent, (ent,)):
            out[pid] = "powered_off" if attr in ("power", "display") else "closed"
    return out


# Events that leave a prop lying on a person. Such an event says where the
# prop ends up; a panel that parks it where the registry keeps it leaves the
# generator to invent one at the person's size, fused to them.
REST_ON_PREDICATES = {"FALL_ON_TOP_OF", "FALL_ON", "LAND_ON", "CRUSH", "PIN_UNDER"}


def props_on_subjects(beat: dict, shown: set[str],
                      entity_map: dict | None = None) -> dict[str, str]:
    """prop_id -> the shown character the beat leaves it lying on."""
    out = {}
    for t in beat.get("transition") or []:
        if (str(t.get("predicate") or "").upper() in REST_ON_PREDICATES
                and t.get("patient") in shown):
            for pid in (entity_map or {}).get(t.get("actor"), (t.get("actor"),)):
                out[pid] = t["patient"]
    return out


def to_panel(beat: dict, plan: dict, template: dict, n: int,
             scene_id: str, prop_map: dict | None = None) -> dict:
    """A shot carrying one panel, in the shape the greybox stage reads.

    `prop_map` maps the beat's state and event names to registry prop ids.
    """
    pm = prop_map or {}
    setup = json.loads(json.dumps(template["setup"]))       # deep copy
    states = prop_states(beat, pm.get("state"))
    for p in setup.get("props") or []:
        if p.get("prop_id") in states:
            p["state"] = states[p["prop_id"]]

    # The cast is the beat's to say, not the plan's: a subject the plan names
    # outside the people the beat shows is dropped rather than trusted.
    shown = set(human_actors(beat, set(template["_ages"] or {})))
    resting = props_on_subjects(beat, shown, pm.get("event"))
    for p in setup.get("props") or []:
        if p.get("prop_id") in resting:
            p["rests_on"] = resting[p["prop_id"]]

    subs = []
    for s in plan.get("subjects") or []:
        if s.get("character_id") not in shown:
            continue
        subs.append({
            "character_id": s.get("character_id"),
            "age_state": (template["_ages"] or {}).get(s.get("character_id")),
            "cls": "human_character",
            "pose": s.get("pose"),
            "screen_position": {"x": float(s.get("screen_x") or 0.5),
                                "y": 0.52, "depth": s.get("depth") or "midground",
                                "zone": None},
            "gaze": {"target_type": "object", "target_ref": s.get("gaze_at")},
        })
    setup["subjects"] = subs

    cam = json.loads(json.dumps(template["camera"]))
    ci = cam.setdefault("creative_intent", {})
    ci["shot_size"] = plan.get("framing") or "wide"
    # The framing PATTERN follows the cast this panel has. Copied from the
    # template it said "two_shot" over one man under a car, and the compiler
    # writes the pattern into the prompt as a composition to fill.
    if subs:
        ci["framing"] = {1: "single", 2: "two_shot"}.get(len(subs), "crowd")
    else:
        ci.pop("framing", None)
    cam.setdefault("extrinsics", {})["angle"] = plan.get("angle") or "eye_level"
    # Only a position the greybox can build. Anything else fails at the
    # geometry stage, after the model call that chose it has been paid for.
    from pace_core.node.panel_greybox import BUILDABLE_POSITIONS
    pos = plan.get("camera_position")
    cam["extrinsics"]["position"] = pos if pos in BUILDABLE_POSITIONS else "three_quarter"

    # Ids follow the SCENE THIS DOC IS, not the beat's own scene label. A
    # panel whose id names a different scene than the file it lives in cannot
    # be looked up by id, which is how every downstream stage finds it.
    sid = f"shot_{n:02d}"
    pid = f"{scene_id}_{sid}_panel_{n:04d}"
    return {
        "shot_id": sid, "camera": cam, "setup": setup,
        "lighting": json.loads(json.dumps(template.get("lighting") or {})),
        "events": {"actions": [{
            "standalone": plan.get("action_en"),
            "description_en": plan.get("action_en"),
            "temporal": "atomic", "foreground": "focal", "background": False,
            "provenance": {"source": "screenplay", "beat_id": beat["id"],
                           "script_events": beat.get("source_event_ids"),
                           "evidence": beat.get("evidence")},
        }]},
        "panels": [{"id": pid, "panel_number": n,
                    "scene_id": scene_id,
                    "primary_focus": {"type": "character", "coverage_pct": 55,
                                      "ref": (subs[0]["character_id"] if subs else "")}}],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--scene", type=int, required=True, help="script scene index")
    ap.add_argument("--project", required=True, help="project slug")
    ap.add_argument("--template-scene", required=True,
                    help="an existing scene doc to inherit location/props from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-scene-id", required=True,
                    help="the scene id the built panels are numbered under")
    ap.add_argument("--prop-map", default=None, metavar="JSON",
                    help='{"state": {entity: [prop_id]}, "event": {actor: [prop_id]}}; '
                         "a name with no entry is taken to be a prop id")
    ap.add_argument("--model", default=GENERATOR_DEFAULT)
    ap.add_argument("--only", default=None, metavar="BEAT_ID",
                    help="re-plan this one beat and keep every other shot in --out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    pm = json.loads(pathlib.Path(a.prop_map).read_text()) if a.prop_map else {}

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
    from pace_core.paths import paths_for
    from pace_core.breakdown.extract_script_ir import load_model, estimate_tokens

    beats = [b for b in json.loads(pathlib.Path(a.beats).read_text())["beats"]
             if b["script_scene_index"] == a.scene]
    src = json.loads((pathlib.Path(paths_for(a.project).scenes_dir)
                      / f"{a.template_scene}.json").read_text())
    t_shot = src["shots"][0]
    ages = {s.get("character_id"): s.get("age_state")
            for sh in src["shots"] for s in (sh.get("setup") or {}).get("subjects") or []}
    template = {"setup": t_shot["setup"], "camera": t_shot["camera"],
                "lighting": t_shot.get("lighting"), "_ages": ages}
    cast = set(ages)
    location = (t_shot["setup"].get("backdrop") or {}).get("location") or ""
    cfg = load_model(a.model)

    todo = [b for b in beats if not a.only or b["id"] == a.only]
    if not todo:
        print(f"no beat {a.only!r} in script scene {a.scene}", file=sys.stderr)
        return 1
    if a.dry_run:
        tin = sum(estimate_tokens(build_messages(b, cast, location)) for b in todo)
        tout = 400 * len(todo)
        cost = tin / 1000 * cfg["cost_per_1k_in"] + tout / 1000 * cfg["cost_per_1k_out"]
        print(f"generator : {a.model} -> {cfg['model_name']}")
        print(f"beats     : {len(beats)} | cast: {sorted(cast)} | location: {location}")
        for b in todo:
            print(f"   {b['id']}  actors={human_actors(b, cast)}  "
                  f"heard={heard_actors(b, cast) or '-'}  "
                  f"states={prop_states(b, pm.get('state')) or '-'}")
        print(f"EST COST  : ${cost:.4f}")
        return 0

    from pace_core.llm_client import call_model, strip_fences
    shots, spent = [], 0.0
    for i, b in enumerate(beats, start=1):
        if b not in todo:
            continue          # its position still numbers the shot
        raw, c = call_model(cfg, build_messages(b, cast, location))
        spent += c
        try:
            plan = json.loads(strip_fences(raw))
        except json.JSONDecodeError as e:
            print(f"  {b['id']} PLAN FAILED: {e}"); continue
        sh = to_panel(b, plan, template, i, a.out_scene_id, pm)
        # The count is geometry, so say when the plan lost an actor.
        want, got = set(human_actors(b, cast)), {s["character_id"] for s in sh["setup"]["subjects"]}
        note = "" if want == got else f"  CAST MISMATCH want={sorted(want)} got={sorted(got)}"
        shots.append(sh)
        print(f"  {b['id']} -> {sh['panels'][0]['id']} {plan.get('framing'):<12}"
              f" subj={len(sh['setup']['subjects'])} "
              f"state_applied={len(prop_states(b, pm.get('state'))) and sum(1 for p in sh['setup'].get('props') or [] if p.get('prop_id') in prop_states(b, pm.get('state')))}"
              f"{note}  ${spent:.4f}")
    out = pathlib.Path(a.out)
    if a.only:
        # Every other shot is kept exactly as it was planned: the planner is a
        # sampled model, so re-running the scene to fix one panel would
        # re-roll the other five.
        def _beat(sh):
            return sh["events"]["actions"][0]["provenance"]["beat_id"]
        new = {_beat(sh): sh for sh in shots}
        doc = json.loads(out.read_text())
        doc["shots"] = [new.get(_beat(sh), sh) for sh in doc["shots"]]
    else:
        doc = {"_schema_version": "pai-1.1", "scene_id": a.out_scene_id,
               "scene_number": 20, "scene_heading": src.get("scene_heading"),
               "narrative_meta": src.get("narrative_meta"),
               "shot_defaults": src.get("shot_defaults"),
               "_from_beats": [b["id"] for b in beats],
               "shots": shots, "compile_hints": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}   TOTAL ${spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
