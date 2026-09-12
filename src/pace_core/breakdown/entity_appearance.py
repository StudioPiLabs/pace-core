"""What to ask a detector to look for, given an entity the screenplay named.

The verification questions were being generated against entity IDS -- `ryan`,
`the_car`, `all_panels`, `computer_voice`. An open-vocabulary detector does not
know who Ryan is; it knows what a man looks like. Sending it an id gets a
confident answer to a prompt that means nothing, which is worse than no answer.

Three things this decides.

WHAT IT LOOKS LIKE. A character resolves through the project's own character
KB to a plain visual noun -- age and sex give "man", "woman", "young man" --
rather than to the long appearance anchor, because a detector prompt is a noun
phrase and a paragraph of hair and wardrobe makes it worse, not better.

WHETHER IT CAN BE SEEN AT ALL. `computer_voice` is typed CHARACTER by the
extractor and is a voice: it has no body in any frame. A LOCATION or an
ENVIRONMENT is not an object in the picture either -- it IS the picture. These
get no detection prompt and must generate no question.

WHAT A DETECTION DOES NOT SETTLE. "man" matches Ryan and Ethan both. Presence
is a detection problem; WHICH man this is, is an identity problem, and the two
are kept apart here so a verifier cannot quietly read one as the other.
"""
from __future__ import annotations

import re

# Types that can appear as an object in a frame. A LOCATION or ENVIRONMENT is
# the frame's setting rather than something in it, so neither is a detection
# target -- "is the accident scene visible" is not a question a box answers.
DETECTABLE_TYPES = {"CHARACTER", "GROUP", "OBJECT", "VEHICLE", "ANIMAL"}

# Entities that are typed as characters but have no body: a voice, an
# announcer, a narrator. The extractor cannot be blamed for calling a speaking
# part a character; the detector still cannot find one.
_DISEMBODIED = re.compile(
    r"\b(voice|voiceover|vo|narrator|announcer|radio|intercom|pa|caller)\b")


def _readable(x: str) -> str:
    return (x or "").replace("_", " ").strip()


def _person_noun(age: float | None, sex: str | None) -> str:
    s = (sex or "").strip().lower()
    if age is not None:
        if age < 13:
            return "boy" if s == "male" else "girl" if s == "female" else "child"
        if age < 25:
            return ("young man" if s == "male" else
                    "young woman" if s == "female" else "young person")
    return "man" if s == "male" else "woman" if s == "female" else "person"


def detect_prompt(entity_id: str, etype: str | None,
                  chars: dict | None = None,
                  props: dict | None = None) -> str | None:
    """A noun phrase a detector can look for, or None if nothing can be.

    `chars` / `props` are the project's registries; both are optional so this
    stays usable on a corpus that has neither.
    """
    eid = (entity_id or "").strip()
    if not eid:
        return None
    t = (etype or "").upper()
    if t and t not in DETECTABLE_TYPES:
        return None
    if _DISEMBODIED.search(eid.lower().replace("_", " ")):
        return None

    ent = (chars or {}).get(eid)
    if isinstance(ent, dict):
        x = ent.get("_extracted") or {}
        age = x.get("age_estimate")
        return _person_noun(float(age) if age is not None else None,
                            x.get("gender"))

    p = (props or {}).get(eid)
    if isinstance(p, dict):
        # The registry's short NAME, not its anchor: the anchor is a
        # production's agreed appearance and runs to a paragraph, which makes
        # an open-vocabulary prompt worse rather than more precise.
        n = p.get("name") or _readable(eid)
        return n.strip().lower()

    word = _readable(eid).lower()
    word = re.sub(r"^(the|a|an|all)\s+", "", word)
    if t == "VEHICLE":
        return word if "car" in word or "vehicle" in word else f"{word} vehicle"
    if t == "GROUP":
        return word if word.endswith("s") else f"{word} group"
    return word or None


def is_identity_bound(etype: str | None) -> bool:
    """Whether a positive detection still leaves WHO unanswered.

    "man" matches every man in frame, so a character's presence question is
    only half-answered by a box; §10.4's identity check on the detected region
    answers the rest. Saying so is the point -- a verifier that reads a
    detection as an identity is the actor/target confusion in another form.
    """
    return (etype or "").upper() == "CHARACTER"


def load_registries(project: str) -> tuple[dict, dict]:
    """(characters, props) for a project, empty dicts when absent."""
    import glob
    import json
    import pathlib
    from pace_core.paths import paths_for
    p = paths_for(project)
    chars: dict = {}
    props: dict = {}
    try:
        raw = json.loads(pathlib.Path(p.chars_file).read_text())
        chars = {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:                                          # noqa: BLE001
        pass
    for f in glob.glob(str(pathlib.Path(p.storage) / "kb/**/props*.json"),
                       recursive=True):
        try:
            raw = json.loads(pathlib.Path(f).read_text())
            src = raw.get("props", raw)
            props = {k: v for k, v in src.items() if not k.startswith("_")}
            break
        except Exception:                                      # noqa: BLE001
            continue
    return chars, props


def resolver_for(project: str, entity_types: dict[str, str]):
    """The callable `beat_visual_ir.plan` wants, wired to one project.

    Exists because every caller was assembling it by hand -- load the two
    registries, close over the type map, return a two-tuple -- and a
    composition rebuilt at each call site is one that drifts between them.
    """
    chars, props = load_registries(project)

    def resolve(entity_id: str):
        t = entity_types.get(entity_id)
        return detect_prompt(entity_id, t, chars, props), is_identity_bound(t)

    return resolve
