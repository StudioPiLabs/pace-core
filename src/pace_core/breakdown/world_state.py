"""A world state that survives the event that changed it.

PACE carries static entity descriptors and per-shot transient events, and
nothing in between. So a beat can assert "the lamps go out" while the
prop registry keeps saying "casting warm light", and the compiler
concatenates both into one prompt: the assembled text states its own beat and
contradicts it in the same paragraph. That defect took a render campaign and a
figure to find.

This module makes it a set of dictionary lookups instead. Events carry
`state_change`; applying them in script order gives, for any point in the
film, what is true of every entity at that point. A descriptor that disagrees
with the state at its own panel is then a contradiction anyone can compute.

Two things it deliberately does NOT do:

  * It does not merge entities across locations. "the panels" in one car are
    not the panels in another, and a timeline that carried a power cut from
    one vehicle into a different one would invent continuity rather than
    check it. Scope is the scene's location-continuity group.
  * It does not guess when a state ends. A state persists until an event
    changes it, because that is what the screenplay's silence means; if the
    film never turns the panels back on, they are off.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


def normalise_entity(name: str) -> str:
    """`all_panels`, `the_panels`, `panels` are one entity; `main_console` is
    not. Only leading articles and quantifiers are stripped -- anything
    cleverer starts merging things the screenplay keeps apart."""
    s = (name or "").strip().lower().replace("-", "_")
    s = re.sub(r"^(the|a|an|all|both|some)_", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class Transition:
    scene: int
    order: int
    entity: str
    attribute: str
    to: str
    frm: str | None = None
    predicate: str = ""
    evidence: dict | None = None
    scope: str = ""            # location-continuity group this applies within

    def key(self) -> tuple[str, str]:
        return (self.entity, self.attribute)


@dataclass
class Timeline:
    transitions: list[Transition] = field(default_factory=list)

    def state_at(self, scene: int, order: int | None = None,
                 scope: str | None = None) -> dict[tuple[str, str], Transition]:
        """Everything true at (scene, order), latest write wins.

        `order=None` means "at the end of this scene", which is what a panel
        belonging to a whole scene should be judged against.
        """
        out: dict[tuple[str, str], Transition] = {}
        for t in self.transitions:
            if scope is not None and t.scope != scope:
                continue
            if t.scene > scene:
                continue
            if t.scene == scene and order is not None and t.order > order:
                continue
            out[t.key()] = t
        return out

    def value(self, scene: int, entity: str, attribute: str,
              order: int | None = None, scope: str | None = None) -> str | None:
        t = self.state_at(scene, order, scope).get((normalise_entity(entity), attribute))
        return t.to if t else None


def build(script_ir: list[dict], scope_of=None) -> Timeline:
    """Walk the IR in script order and collect every state change.

    `scope_of(scene) -> str` groups scenes that share a physical space; the
    default keys on the scene heading with its time qualifier removed, so
    `INT. ANOTHER CAR - MOMENTS LATER` and `INT. ANOTHER CAR - CONTINUOUS`
    are one car and `INT. CAR` is a different one.
    """
    if scope_of is None:
        def scope_of(sc):                                    # noqa: ANN001
            h = (sc.get("heading") or "").upper()
            return re.split(r"\s+-\s+", h)[0].strip()
    tl = Timeline()
    for sc in sorted(script_ir, key=lambda s: s["index"]):
        sco = scope_of(sc)
        for ev in sorted(sc.get("events") or [], key=lambda e: e.get("order") or 0):
            for ch in ev.get("state_change") or []:
                tl.transitions.append(Transition(
                    scene=sc["index"], order=ev.get("order") or 0,
                    entity=normalise_entity(ch.get("entity")),
                    attribute=(ch.get("attribute") or "").strip().lower(),
                    to=str(ch.get("to")), frm=(None if ch.get("from") is None
                                               else str(ch.get("from"))),
                    predicate=ev.get("predicate") or "",
                    evidence=ev.get("evidence"), scope=sco))
    return tl


# ── contradiction between a live state and the words a prompt uses ────────
#
# Deliberately a small, explicit table rather than an embedding: a word that
# means "this thing is on" is a short, closed list per attribute, and a
# lexicon that can be read is one a reviewer can argue with.
STATE_WORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "power": {
        "off": ("glowing", "glows", "lit by", "illuminated", "backlit",
                "shining", "bright screen", "active display", "displaying"),
    },
    "display": {
        "off": ("glowing", "glows", "showing", "displaying", "depicting",
                "active display", "interface graphics"),
    },
    "open": {
        "closed": ("open panel", "opens up", "gaping"),
    },
    "alive": {
        "false": ("smiling", "laughing", "walks", "runs", "speaks"),
    },
}


def contradictions(text: str, state: dict[tuple[str, str], Transition],
                   entity_aliases: dict[str, tuple[str, ...]] | None = None) -> list[dict]:
    """Words in `text` that assert the opposite of a live state.

    `text` MUST already be scoped to one entity -- a single prop phrase, or a
    lighting clause. Do not hand it a whole compiled prompt. Scanning the
    assembled prompt cannot tell which noun a word belongs to: "panels ...,
    with a handheld slab, its entire face a single glowing screen" mentions
    panels and glows separately, and reads as a panel contradiction that is
    not one. Proximity does not rescue it either -- a true positive can sit
    further apart than a false one (120 characters against 89, measured).
    Scope the input; do not guess at the seam.

    Use `audit_props` for a shot, which does the scoping correctly.
    """
    low = (text or "").lower()
    out = []
    for (ent, attr), t in state.items():
        words = STATE_WORDS.get(attr, {}).get(t.to)
        if not words:
            continue
        names = (entity_aliases or {}).get(ent, (ent.replace("_", " "),))
        if not any(n.lower() in low for n in names):
            continue
        for w in words:
            if w in low:
                out.append({
                    "entity": ent, "attribute": attr, "state": t.to,
                    "contradicting_phrase": w, "since_predicate": t.predicate,
                    "since_scene": t.scene,
                    "script_words": (t.evidence or {}).get("source_text"),
                })
    return out


def audit_props(shot: dict, props_kb: dict | None,
                expected: dict[str, list[tuple[str, str]]] | None = None) -> list[dict]:
    """Per-prop state audit for one shot -- the scoped form of the check.

    Two distinct faults, deliberately named apart:

      CONTRADICTION  the prop declares a state and its phrase still says the
                     opposite. After the substitution repair this should be
                     unreachable, so if it fires the lexicon has a gap.
      STATE_MISSING  the world state says this prop should be in some state
                     and the prop declares none. This is the corpus's actual
                     condition and the one worth reporting.
    """
    from pace_core.pai_compat import prop_phrase
    out = []
    for p in (shot.get("setup") or {}).get("props") or []:
        pid = p.get("prop_id")
        declared = states_for_prop(p)
        phrase = prop_phrase(p, props_kb)
        for attr, val in declared:
            st = {(normalise_entity(pid), attr): Transition(
                scene=-1, order=0, entity=normalise_entity(pid),
                attribute=attr, to=val)}
            for c in contradictions(phrase, st, {normalise_entity(pid): (" ",)}):
                out.append({"kind": "CONTRADICTION", "prop_id": pid,
                            "phrase": phrase, **c})
        want = (expected or {}).get(pid) or []
        for attr, val in want:
            if (attr, val) not in declared:
                out.append({"kind": "STATE_MISSING", "prop_id": pid,
                            "attribute": attr, "expected": val,
                            "declared": declared or None})
    return out


# ── turning a state into words, by substitution ───────────────────────────
#
# Flux runs cfg=1.0, so the negative prompt is never evaluated: a thing cannot
# be removed from an image by asking for its absence. The only way a dead panel
# stops glowing is for the word "glowing" never to be written. So state is
# applied by REWRITING the appearance clause, not by appending a denial to it.
#
# Longest phrase first, so "glowing with interface graphics" is rewritten as a
# unit rather than leaving "with interface graphics" attached to "dark".
STATE_SUBSTITUTIONS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("power", "off"): (
        ("glowing with interface graphics", "dark, its interface graphics extinguished"),
        ("glowing softly", "dark and unlit"),
        ("a single glowing screen", "a single dead black screen"),
        ("its entire face a single glowing screen", "its entire face a dead black screen"),
        ("glowing", "dark"),
        ("illuminated", "unlit"),
        ("backlit", "unlit"),
        ("showing television channels, games and camera views",
         "showing nothing, their surfaces black"),
    ),
    ("display", "off"): (
        ("glowing with interface graphics", "dark, showing nothing"),
        ("glowing softly", "dark and blank"),
        ("displaying", "blank, no longer displaying"),
        ("glowing", "dark"),
    ),
    ("open", "closed"): (
        ("opens up", "sits shut"),
        ("open panel", "closed panel"),
    ),
}


def rewrite_for_state(text: str, states) -> tuple[str, list[dict]]:
    """Apply every live state to an appearance clause.

    `states` is an iterable of (attribute, value). Returns the rewritten text
    and a record of what was changed, so a caller can report the substitution
    rather than making it silently -- a prompt that quietly says the opposite
    of its registry is as hard to debug as one that contradicts itself.
    """
    out, changes = text or "", []
    for attr, val in states:
        for phrase, replacement in STATE_SUBSTITUTIONS.get(
                (str(attr).lower(), str(val).lower()), ()):
            if phrase.lower() in out.lower():
                # Case-preserving only at the start; appearance clauses are
                # lower-case prose in this schema.
                i = out.lower().index(phrase.lower())
                out = out[:i] + replacement + out[i + len(phrase):]
                changes.append({"attribute": attr, "value": val,
                                "was": phrase, "now": replacement})
    return out, changes


# States that mean "as built" -- these never rewrite anything, so a prop that
# simply says `active` behaves exactly as it always has.
NEUTRAL_STATES = {"active", "normal", "default", "on", "intact", "", "none"}


def states_for_prop(prop: dict) -> list[tuple[str, str]]:
    """Read a Prop's declared state as (attribute, value) pairs.

    Accepts the schema's flat `state: "powered_off"` as well as a structured
    `state: {"power": "off"}`, because the flat form is what the corpus has
    and the structured form is what the world-state timeline produces.
    """
    st = prop.get("state")
    if isinstance(st, dict):
        return [(k, str(v)) for k, v in st.items()
                if str(v).lower() not in NEUTRAL_STATES]
    s = str(st or "").strip().lower().replace("-", "_")
    if not s or s in NEUTRAL_STATES:
        return []
    # Flat vocabulary the schema already uses, mapped onto attributes.
    flat = {"powered_off": ("power", "off"), "power_off": ("power", "off"),
            "unpowered": ("power", "off"), "dead": ("power", "off"),
            "dark": ("power", "off"), "off": ("power", "off"),
            "blank": ("display", "off"), "closed": ("open", "closed"),
            "shut": ("open", "closed")}
    hit = flat.get(s)
    return [hit] if hit else []
