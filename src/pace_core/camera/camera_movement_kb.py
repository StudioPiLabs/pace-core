"""camera_movement_kb — what dramatic function maps to what camera
movement, and on what grounds.

Before this file, the entire mapping lived as one unsourced clause inside
`lamp_dsl.SYSTEM_PROMPT`: "reveal -> crane or pull-out; tension -> slow
push-in; intimate close-up -> slow push-in or hold." Three rules, no
citation, no evidence tier, read once per LLM call with nothing checking
whether the model actually followed them or invented something else. That
is exactly the shape `prompt_projection.py` existed to fix for prose
fields, and the fix is the same shape here: a typed, sourced table instead
of prose a model may or may not attend to.

Targets `camera_skills.py`, not `lamp_dsl.py` — checked, not assumed. Three
compilers converge on the same render sink (BlenderBox depth pass / Wan
VACE track-json): `camera_planner` (the default, geometric), `camera_skills`
(named-operation authoring, UI-wired in the canvas Trajectory node, the
newer of the two motion vocabularies), and `lamp_dsl` (a compressed 24-token
DSL adapted from LaMP, "a third compiler over the same camera block, and
deliberately not the default" per the canvas UI's own comment). An earlier
draft of this file targeted `lamp_dsl`'s vocabulary because that is what
`PACE.tex` happens to cite, without first checking which compiler the
product actually exposes as its manual-authoring path. `camera_skills` is
richer at exactly the level a dramatic-function mapping needs — orbit, arc,
track, crane, zoom, push_in/pull_out are first-class named operations with
real parameters, not six compressed axis tokens — so it is the right
target regardless of which compiler a given render happens to run through.

Evidence is graded the same way `prompt_projection.Rule` grades it, because
the failure mode is the same one: a claim that reads as established fact
and is actually someone's plausible guess. Three tiers:

    citation     traceable to a named, citable source (katz2019, bowen2023,
                 the P1/P3 pedagogy this paper already cites) — verifiable
                 by anyone who opens the book.
    measured     traceable to something THIS project actually observed —
                 a real scene's own declared intent, a rendered result.
    convention   common craft usage with no single citable source found for
                 it yet — usable, but weaker, and marked as such rather than
                 dressed up as either of the above.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Evidence(enum.Enum):
    CITATION = "citation"
    MEASURED = "measured"
    CONVENTION = "convention"


@dataclass(frozen=True)
class MovementEntry:
    """One dramatic function's mapping to camera movement.

    `skills` is a list of `camera_skills.compile_shot`-shaped dicts —
    `{"skill": <name>, ...params}`, the same shape a caller passes straight
    through to `compile_shot`. Validated against `camera_skills.KNOWN_SKILLS`
    at import time (`_validate`), so a typo fails at load rather than
    silently becoming an unknown move `compile_shot` holds on. Numeric
    params are given only where the choice is semantic (a crane's up/down
    sign, a track's side) and left to `compile_shot`'s own defaults
    otherwise — this table decides WHICH operation and WHICH direction, not
    exact distances, which are a per-shot scale question this table has no
    evidence to answer.

    An empty `skills` list with `is_policy=False` is a genuine, deliberate
    recommendation to hold — `camera_skills` has an explicit `hold`
    operation, so this table always says so explicitly rather than relying
    on an absence being read as "hold" by whatever consumes it.

    `is_policy` marks an entry that is not a movement recommendation at all
    but a rule about WHEN to apply one — coverage reuse, cut timing. Kept in
    the same table because both answer "what should this shot's camera do",
    but audit() and any caller picking a movement need to tell the two apart.

    `counter` names when NOT to reach for this movement despite the function
    fitting, because a mapping with no stated exception reads as
    unconditional and gets applied where the source itself would not have
    applied it. Optional only because not every entry has one on record.

    `movement_tags` is the SAME recommendation restated in the coarser flat
    Movement vocabulary `movement_io.write_movement` accepts (push_in,
    pull_out, tracking, orbit, crane_up/down, pan_lr/rl, tilt_up/down,
    handheld, steadicam, from_behind, reverse_shot, static) — this is the
    vocabulary the batch scene-render pipeline (camera_planner, via
    movement_io.read_movement) actually reads at render time. `skills` is
    NOT read there today: camera_skills.compile_shot's richer output is
    currently ephemeral, compiled in-memory for a playground render job and
    never written back to a shot's KB record. A caller writing a shot's
    persistent, batch-rendered camera therefore wants `movement_tags`, not
    `skills`; a caller compiling a one-off camera_skills track (the
    Trajectory node, nl_to_skills) wants `skills`. Two columns instead of
    one field with two meanings, because collapsing them would silently
    pick one destination for both use cases. Round-tripped through
    movement_io at import time (`_validate`) so an unrecognised or
    silently-dropped tag fails at load, not at the first shot that uses it.
    """
    dramatic_function: str
    skills: list[dict]
    reason: str
    evidence: Evidence
    source: str
    counter: str = ""
    is_policy: bool = False
    movement_tags: list[str] = field(default_factory=list)


KB: dict[str, MovementEntry] = {
    # ── the three that already existed, now sourced ──────────────────────
    "reveal": MovementEntry(
        "reveal", [{"skill": "crane", "height_delta": 1.2}, {"skill": "pull_out"}],
        "Rising and widening is how new information enters a shot without "
        "a cut — the reveal IS the movement's payload, not a fact stated "
        "before it.",
        Evidence.CONVENTION,
        "Standing craft usage; carried over from the pre-existing LAMP "
        "system-prompt clause with no citable source found for it yet.",
        movement_tags=["crane_up", "pull_out"]),
    "tension": MovementEntry(
        "tension", [{"skill": "push_in", "ease": "ease_in"}],
        "A slow push tightens the frame around a subject over time, which "
        "reads as mounting pressure without any cut to signal it.",
        Evidence.CONVENTION,
        "Standing craft usage; carried over from the pre-existing LAMP "
        "system-prompt clause with no citable source found for it yet.",
        movement_tags=["push_in"]),
    "intimate_close": MovementEntry(
        "intimate_close", [{"skill": "push_in", "ease": "ease_in_out"}],
        "Closing distance over a held shot, rather than cutting to a "
        "closer size, keeps the performance continuous through the "
        "approach instead of skipping to its result.",
        Evidence.CONVENTION,
        "Standing craft usage; carried over from the pre-existing LAMP "
        "system-prompt clause with no citable source found for it yet.",
        movement_tags=["push_in"]),

    # ── new, each with an actual source ──────────────────────────────────
    "objective_surveillance": MovementEntry(
        "objective_surveillance", [{"skill": "hold"}],
        "A camera that does not move reads as a fixed observation point "
        "rather than a participant — exactly the reading a scene wants "
        "when the point is that the cast is being watched, not accompanied.",
        Evidence.MEASURED,
        "AutomaticDrive scene_01's own camera._design.intent: 'the camera "
        "is fixed and objective, watching them like a surveillance feed, "
        "which implies the system has been watching all along' — an "
        "authored intent this project already declared, not invented for "
        "this table.",
        counter="Do not apply to a scene whose _design.intent names "
                "participation or subjectivity instead of observation; the "
                "static hold is doing narrative work only for the reading "
                "this specific intent describes.",
        movement_tags=["static"]),
    "coverage_reuse": MovementEntry(
        "coverage_reuse", [],
        "Camera position is expensive to invent and cheap to reuse: "
        "cutting back to an already-established setup reads as returning "
        "to the same observation point, where a new position for no "
        "reason reads as arbitrary.",
        Evidence.CITATION,
        "Katz (2019), on 3-subject dialogue coverage: when a new speaking "
        "pair does not obviously demand a new camera position, 'the basic "
        "idea is to reuse existing positions as much as possible, rather "
        "than keep creating new ones after every new axis.'",
        counter="Prefer an already-used setup over a new one unless the "
                "axis of action has actually changed.",
        is_policy=True),
    "action_coverage": MovementEntry(
        "action_coverage", [{"skill": "track", "side": "rear"}],
        "A subject in sustained motion is covered by moving the camera "
        "with them, which holds them in frame without repeatedly "
        "re-centring through cuts.",
        Evidence.CITATION,
        "Katz (2019), Ch.19-20 on tracking-shot construction for sustained "
        "action: the camera's path is planned as its own continuous "
        "action alongside the subject's, not as a series of static "
        "re-framings.",
        counter="'side' defaults to a rear chase angle, matching the "
                "moving-subject example camera_skills.py's own docstring "
                "uses; override for a lead/lateral track when staging "
                "calls for the subject entering rather than being "
                "followed.",
        movement_tags=["tracking", "from_behind"]),
    "state_change_cut": MovementEntry(
        "state_change_cut", [],
        "A panel boundary is warranted by a change in dramatic state, not "
        "by elapsed shot duration or a change in shot size alone.",
        Evidence.CITATION,
        "This paper's own P1 (Section 2.3), from Bowen (2023) and Simon "
        "(2012): editing motive is new information, a phase change in an "
        "action, a reaction assigning meaning, or a rhythm accent — never "
        "duration by itself.",
        counter="Listed here because a movement recommendation that "
                "changes on every beat with no stated state change is the "
                "same error P1 already names for cuts.",
        is_policy=True),
}


def _validate() -> None:
    """Every skill name must exist in camera_skills.KNOWN_SKILLS — a KB
    naming one it does not recognise would fail silently downstream
    (compile_shot holds state unchanged on an unrecognised skill, per its
    own 'hold / unknown -> carry state unchanged' branch) rather than at
    load time. Every movement_tag must survive a real round-trip through
    movement_io.write_movement/read_movement — that module's own comment
    warns 'unknowns silently dropped', so name-checking against a hardcoded
    copy of its vocabulary would go stale the moment movement_io's forward
    maps change; round-tripping through the real function catches that."""
    from pace_core.camera.camera_skills import KNOWN_SKILLS
    from pace_core.camera.movement_io import read_movement, write_movement
    for fn, entry in KB.items():
        assert entry.dramatic_function == fn, f"key/field mismatch: {fn}"
        for s in entry.skills:
            assert s.get("skill") in KNOWN_SKILLS, (
                f"{fn}: skill {s.get('skill')!r} is not in "
                f"camera_skills.KNOWN_SKILLS — compile_shot would silently "
                f"hold rather than perform it")
        if entry.movement_tags:
            shot = write_movement({}, entry.movement_tags)
            back, _easing = read_movement(shot)
            for tag in entry.movement_tags:
                assert tag in back, (
                    f"{fn}: movement_tag {tag!r} did not survive "
                    f"movement_io's write/read round-trip — it is not a "
                    f"real Movement token, it would be silently dropped")
        elif not entry.is_policy:
            assert False, (
                f"{fn}: non-policy entry has no movement_tags — the batch "
                f"render pipeline reads movement_tags, not skills; an "
                f"empty list here would silently write nothing")


_validate()


def movement_for(dramatic_function: str) -> MovementEntry | None:
    return KB.get(dramatic_function)


def citable(dramatic_function: str) -> bool:
    """Whether this entry may be stated as fact in the paper (a citation)
    versus reported only as this project's own measurement or as
    unattributed convention. Mirrors prompt_projection's values_checked
    distinction: what IS true and what is merely PLAUSIBLE are different
    claims, and conflating them is the exact failure this file exists to
    stop repeating."""
    entry = KB.get(dramatic_function)
    return bool(entry and entry.evidence is Evidence.CITATION)


def audit() -> str:
    """The table as a readable report, grouped by evidence tier — nothing
    is worth having a sourced table if no one can see, at a glance, which
    entries are actually sourced and which are still convention standing
    in for it."""
    rows = []
    for tier in Evidence:
        entries = [e for e in KB.values() if e.evidence is tier]
        if not entries:
            continue
        rows.append(f"\n{tier.value.upper()}  ({len(entries)})")
        for e in entries:
            if e.is_policy:
                target = "(policy, not a movement)"
            else:
                skills = [s.get("skill") for s in e.skills] or "(unresolved)"
                target = f"{skills}  tags={e.movement_tags}"
            rows.append(f"  {e.dramatic_function:24} -> {target}")
            rows.append(f"      {e.source}")
            if e.counter:
                rows.append(f"      counter: {e.counter}")
    return "\n".join(rows)
