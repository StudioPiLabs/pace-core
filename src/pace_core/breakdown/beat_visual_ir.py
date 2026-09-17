"""What must be VISIBLE in a beat's panel, stated so it can be checked.

Between a beat and its geometry there was nothing. `build_beat_panels` went
straight from "these events happened" to a staged panel, so the facts the
image is supposed to carry were never written down -- and a fact nobody wrote
down is a fact no verifier can check. The image verifier could compare
delivered pixels against staged geometry and nothing else, because geometry
was the only claim on record.

Two ideas do the work.

REQUIRED vs OPTIONAL. A verifier that penalises every unstated detail fails a
panel for not drawing a ladder in the background. Only `required` facts carry
a hard penalty; the rest are recorded and reported.

PROVENANCE, on every fact. SCRIPT means the screenplay states it. INFERRED
means the beat's own state entails it -- a body that was crushed is still
crushed. VISUALIZATION means the planner invented a way to SHOW something
otherwise undepictable ("food is running out" -> "one sack in an empty
granary"), and those must never be written back as script truth. The corpus
already has one field that turned out to be a table constant read as
direction; the distinction exists so that cannot recur here.

A beat is a span and an image is an instant, so a visual moment is chosen too
-- PRE_ACTION through POST_ACTION -- rather than pushing a whole action chain
into one frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCRIPT, INFERRED, VISUALIZATION = "SCRIPT", "INFERRED", "VISUALIZATION"

# A beat is a time span; a panel is one instant of it. Which instant reads
# best is a property of the predicate, so the default comes from the verb and
# only the ambiguous ones need judgement.
MOMENTS = ("PRE_ACTION", "ACTION_START", "MID_ACTION", "ACTION_PEAK", "POST_ACTION")

_MOMENT_BY_PREDICATE_HINT = (
    # (substring of the canonical predicate, moment) -- first match wins.
    ("FALL", "ACTION_PEAK"), ("CRUSH", "ACTION_PEAK"), ("EXPLO", "ACTION_PEAK"),
    ("SCREAM", "ACTION_PEAK"), ("HIT", "ACTION_PEAK"), ("STRIKE", "ACTION_PEAK"),
    ("DRAG", "MID_ACTION"), ("PUSH", "MID_ACTION"), ("HELP", "MID_ACTION"),
    ("CARRY", "MID_ACTION"), ("PULL", "MID_ACTION"), ("GRAB", "MID_ACTION"),
    ("LIE_DEAD", "POST_ACTION"), ("CLOSE", "POST_ACTION"),
    ("LOSE_POWER", "POST_ACTION"), ("HALT", "POST_ACTION"),
    ("OPEN", "ACTION_START"), ("EMERGE", "ACTION_START"), ("RAISE", "ACTION_START"),
    ("LOOK", "PRE_ACTION"), ("GLANCE", "PRE_ACTION"), ("WATCH", "PRE_ACTION"),
    ("HOVER", "PRE_ACTION"),
)


def moment_for(beat: dict) -> tuple[str, str]:
    """Which instant of the beat the panel depicts, and why.

    Keyed on the beat's most important event: a beat that ends in a death is
    about that, not about the limp that preceded it.
    """
    evs = beat.get("transition") or []
    if not evs:
        return "POST_ACTION", "no action to be mid-way through"
    lead = max(evs, key=lambda e: (e.get("predicate") or "") and 1)
    # Prefer the predicate that carries a state change; that is what the beat
    # exists to show.
    changed = [e for e in evs if any(
        c for c in (beat.get("state_changes") or [])
        if c.get("entity") and e.get("predicate"))]
    for e in (changed or evs):
        p = (e.get("predicate") or "").upper()
        for frag, m in _MOMENT_BY_PREDICATE_HINT:
            if frag in p:
                return m, f"{p} reads at {m.lower().replace('_', ' ')}"
    p = (lead.get("predicate") or "").upper()
    return "MID_ACTION", f"{p or 'the action'} has no stronger reading than its middle"


@dataclass
class VisualFact:
    subject: str
    predicate: str
    value: str | None = None
    provenance: str = SCRIPT
    required: bool = True
    source_event: str | None = None
    # Who the action lands on. A canonical predicate can end in a preposition
    # -- FALL_ON_TOP_OF -- and without the patient the question dangles: "Is
    # the car falling on top of?" is not answerable, and a VQA model asked an
    # unanswerable question still answers.
    patient: str | None = None
    evidence: dict | None = None
    depiction: str | None = None      # only for VISUALIZATION

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BeatVisualIR:
    beat_id: str
    scene_id: str
    visual_moment: str
    moment_reason: str
    focal_event: str | None
    visible_facts: list[dict] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    verification_questions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def required(self) -> list[dict]:
        return [f for f in self.visible_facts if f.get("required")]


def facts_from_beat(beat: dict, cast: set[str]) -> list[VisualFact]:
    """Every fact the panel must carry, derived without a model.

    Script facts come from the beat's events; inferred facts come from the
    state the beat starts in, which is what carries an injury or a blackout
    forward into a later panel that never restates it.
    """
    out: list[VisualFact] = []
    seen = set()
    for e, eid in zip(beat.get("transition") or [],
                      beat.get("source_event_ids") or []):
        for role in ("actor", "patient", "target"):
            who = e.get(role)
            if not who or who in seen:
                continue
            seen.add(who)
            out.append(VisualFact(
                subject=who, predicate="present", value=None,
                provenance=SCRIPT, source_event=eid,
                # A human the beat acts on must be in frame; a location or an
                # off-screen object it merely names need not be.
                required=who in cast))
        if e.get("actor"):
            out.append(VisualFact(
                subject=e["actor"], predicate="performs",
                value=e.get("predicate"), provenance=SCRIPT,
                source_event=eid, required=True,
                patient=e.get("patient") or e.get("target")))
    for c in beat.get("state_changes") or []:
        out.append(VisualFact(
            subject=c.get("entity"), predicate=c.get("attribute"),
            value=str(c.get("to")), provenance=SCRIPT, required=True))
    # Inherited state: true when the beat starts, not restated by it.
    changed_now = {(c.get("entity"), c.get("attribute"))
                   for c in beat.get("state_changes") or []}
    for key, val in (beat.get("state_before") or {}).items():
        ent, _, attr = key.partition(".")
        if (ent, attr) in changed_now:
            continue
        out.append(VisualFact(
            subject=ent, predicate=attr, value=str(val),
            provenance=INFERRED,
            # Inherited state is real but not what this panel is ABOUT, so it
            # is recorded and not penalised. A verifier that hard-failed every
            # panel for an off-screen entity's carried state would fail most.
            required=False))
    return out


def _readable(x: str | None) -> str:
    return (x or "").replace("_", " ").strip()


# Canonical predicates are SCREAMING_SNAKE verb phrases in the infinitive.
# Asking "Is the car fall on top of?" is not a question anyone can answer, and
# a VQA model asked it will answer something.
# Prepositions a canonical predicate can end on, leaving the question hanging.
_DANGLING = {"of", "on", "at", "to", "toward", "towards", "into", "onto",
             "against", "with", "for", "from", "over", "under", "around"}

_VERB_FORMS = {
    "fall": "falling", "lie": "lying", "drag": "dragging", "push": "pushing",
    "pull": "pulling", "grab": "grabbing", "hold": "holding", "run": "running",
    "sit": "sitting", "stand": "standing", "kneel": "kneeling",
    "close": "closing", "open": "opening", "hover": "hovering",
    "scream": "screaming", "look": "looking", "glance": "glancing",
    "emerge": "emerging", "clamp": "clamping", "swerve": "swerving",
    "limp": "limping", "bang": "banging", "press": "pressing",
    "lose": "losing", "fade": "fading", "help": "helping", "watch": "watching",
}


def _gerund(word: str) -> str:
    """English -ing, well enough to ask a question with.

    Naive `word + "ing"` produced "pauseing", "changeing" and -- because
    `endswith("ing")` is true of the word SING -- "Is alice sing?". A VQA model
    asked a malformed question still answers, so the grammar is not cosmetic.
    """
    w = word.lower()
    if w in _VERB_FORMS:
        return _VERB_FORMS[w]
    # Already a gerund. Length guards the short words that merely end in
    # those letters: sing, ring, bring, cling.
    if w.endswith("ing") and len(w) > 6:
        return w
    if w.endswith("ie"):
        return w[:-2] + "ying"                     # lie -> lying
    if w.endswith("e") and not w.endswith(("ee", "oe", "ye")):
        return w[:-1] + "ing"                      # pause -> pausing
    # Consonant-vowel-consonant on a single syllable doubles: sit -> sitting.
    if (len(w) == 3 and w[0] not in "aeiou" and w[1] in "aeiou"
            and w[2] not in "aeiouwxy"):
        return w + w[-1] + "ing"
    return w + "ing"


def _verb(predicate: str | None) -> str:
    """FALL_ON_TOP_OF -> "falling on top of"."""
    words = _readable(predicate).lower().split()
    if not words:
        return "doing anything"
    return " ".join([_gerund(words[0]), *words[1:]])


# Subjects the screenplay names as groups. "Is vehicles traveling fast?" is
# not a question; number agreement is part of being answerable.
_PLURAL_HINTS = ("s", "vehicles", "panels", "arms", "seats", "androids",
                 "family", "crowd", "people")


def _is_plural(subject: str) -> bool:
    w = _readable(subject).lower()
    if w in ("family", "crowd", "traffic"):
        return False
    return w.endswith("s") and not w.endswith("ss")


def _be(subject: str) -> str:
    return "Are" if _is_plural(subject) else "Is"


def questions_for(facts: list[dict], subjects: list[str],
                  resolve=None) -> list[dict]:
    """Atomic questions with their dependencies.

    Dependency order is the point: asking "is the man kneeling" of a frame
    that contains no man produces an answer, and the answer is noise. A
    presence question gates every question about that subject.

    `resolve(entity_id) -> (detect_prompt, identity_bound)` says what a
    detector should be sent and whether a hit still leaves WHO unanswered. An
    entity it cannot see -- a voice, a location -- yields no prompt and gets
    NO question at all: an unanswerable question still gets answered.
    """
    resolve = resolve or (lambda e: (_readable(e), False))
    qs: list[dict] = []
    qid = {}
    for s in subjects:
        prompt, ident = resolve(s)
        if not prompt:
            continue
        i = f"q{len(qs) + 1}"
        qid[s] = i
        qs.append({"id": i, "ask": f"{_be(s)} {_readable(s)} visible in the image?",
                   "expect": "yes", "depends_on": [], "about": s,
                   "answered_by": "detector", "detect_prompt": prompt,
                   # A box round a man does not say which man. Presence is a
                   # detection problem and identity is a separate one; reading
                   # the first as the second is the actor/target confusion in
                   # another form.
                   "identity_check_required": ident})
    for f in facts:
        if f.get("predicate") == "present" or not f.get("required"):
            continue
        s = f.get("subject")
        # An actor that is not a cast member -- a car, a robot arm -- still
        # needs its own presence gate, or "is the car falling on him" is asked
        # of a frame that may contain no car and the answer is noise.
        if s not in qid:
            prompt, ident = resolve(s)
            if not prompt:
                continue            # nothing to detect, so nothing to ask
            g = f"q{len(qs) + 1}"
            qid[s] = g
            qs.append({"id": g, "ask": f"{_be(s)} {_readable(s)} visible in the image?",
                       "expect": "yes", "depends_on": [], "about": s,
                       "answered_by": "detector", "detect_prompt": prompt,
                       "identity_check_required": ident})
        i = f"q{len(qs) + 1}"
        val = f.get("value") or ""
        if f.get("predicate") == "performs":
            v = _verb(val)
            tgt = f.get("patient")
            # Only append the patient when the verb phrase needs an object --
            # a trailing preposition is the reliable signal.
            if tgt and v.split()[-1] in _DANGLING:
                v = f"{v} {_readable(tgt)}"
            ask = f"{_be(s)} {_readable(s)} {v}?"
        else:
            ask = f"{_be(s)} {_readable(s)} {_readable(f.get('predicate'))} {_readable(val)}?"
        qs.append({"id": i, "ask": ask, "expect": "yes",
                   "depends_on": [qid[s]], "about": s,
                   "answered_by": "vlm"})
    return qs


def plan(beat: dict, cast: set[str], resolve=None) -> BeatVisualIR:
    facts = [f.as_dict() for f in facts_from_beat(beat, cast)]
    # Only people THIS beat acts on. Inherited state names entities that are
    # true elsewhere in the film -- a late beat can carry "alice.alive =
    # false" from an event two scenes earlier -- and asking whether she is
    # visible in a frame she is not in produces a confident wrong answer.
    subjects = [s for s in dict.fromkeys(
        f["subject"] for f in facts
        if f["subject"] in cast and f["provenance"] == SCRIPT
        and f["predicate"] == "present")]
    m, why = moment_for(beat)
    lead = (beat.get("transition") or [{}])[0]
    ir = BeatVisualIR(beat_id=beat.get("id"), scene_id=beat.get("scene_id"),
                      visual_moment=m, moment_reason=why,
                      focal_event=lead.get("predicate"),
                      visible_facts=facts, subjects=subjects)
    ir.verification_questions = questions_for(facts, subjects, resolve)
    return ir
