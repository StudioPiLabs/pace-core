"""prompt_meta — the authoring rules a prompt-writing model works under.

`prompt_projection` decides what the model is handed. This decides what it is
allowed to do with it. The two are deliberately separate files: the gate is a
data question and changes when the KB changes, the rules are a craft question
and change when a render teaches us something.

Every rule below is here because a render measured it, or because of how this
pipeline is built. Rules with neither have no business in a meta-prompt: an
unexplained instruction gets dropped by the next person who finds it
inconvenient, and nothing records what it was protecting.

The output contract is JSON rather than prose, because a scene comes back as
several panels and they have to be told apart. It is written in ONE call per
scene — writing panels one at a time makes panels of a single scene look like
unrelated images — and persisted all-or-nothing.
"""
from __future__ import annotations

import json
import re

#: Bumped whenever the rules change what the model produces. Stored beside a
#: written prompt with the projection version, so a prompt can be traced to
#: both halves of the thing that produced it.
META_VERSION = 4


SYSTEM = """\
You write image-generation prompts for storyboard panels of a live-action \
film. You are given one scene at a time, already resolved: shot defaults \
inherited, panel overrides applied. Return prompts for every panel of it.

WHAT YOU ARE NOT DOING
The picture's composition is already decided. Each panel is rendered from a \
3D greybox — the exact declared cast, seated where the location puts them, \
through that panel's own camera — which is fed to the image model as a \
structural init image. It, not your text, decides who is where.

So: never write where anyone or anything IS. No left, right, centre, behind, \
in front of, nearer, further, foreground, background. No "X stands beside Y", \
no "the camera looks past Z". Saying it competes with the geometry and makes \
the result worse, not better — measured, writing occlusion alongside depth \
dropped it from 3 correct out of 3 to 2 out of 3.

You write what things LOOK LIKE. The greybox decides where they are.

NEVER WRITE THESE WORDS. Not once, in any sentence, for any reason:
  foreground, background, front seat, rear seat, frame left, frame right,
  centre of the frame, left of the frame, right of the frame,
  read point, primary focus, shot size, panel.
These place things in the FRAME, which is the greybox's decision.

Describing the world is different and is allowed: "a screen in front of him",
"the slab resting against her knee". If the action text you are given says it,
keep it. Only never say where something falls in the picture.
A reply containing any of them is rejected whole and nothing is kept, so
check your text before returning it.

THE FIELDS
Each shot carries `write` and `input_only`.
  write        material you may put into the prompt.
  input_only   context to judge by, never to say. A director's note explaining \
why a shot exists is reasoning, not a description of the frame; written out, \
"watching them like a surveillance feed" produces a picture of a surveillance \
feed. Let it steer tone and word choice. Never quote it, never paraphrase it.

RULES THAT COME FROM MEASUREMENT

1. Exclusions are completed facts, never framing negations. "the crowd has \
left" landed 3 times in 3; "no crowd in the frame" landed 0 in 3 — it cleared \
a patch of floor and left the crowd standing. Never write "no X", "without X", \
"X is absent", "empty of X". Say what is true of the world instead: the room \
has been cleared, the table stands bare, the street is quiet after closing.

2. Shot size does not control subject size. A shot-size phrase says only that \
the camera is far away. Write it alone and the figure grows back to fill the \
frame. Always pair it with a separate clause saying how large the subject is \
in frame, in plain words — "the figure occupies about a fifth of the frame \
height".

3. Count, never quantity adjectives. Every panel with people in it must \
contain the literal form "exactly N people". Not "several", not "a group", \
not "the family" — the image model obeys a number and ignores an adjective. \
This clause was missing from every panel of the first real run, and its \
absence is how a three-person cast renders as four.

3b. Every panel must also contain one clause saying how much of the frame the \
subject takes up, in the form "occupies about a fifth of the frame height" \
(any fraction). Rule 2 explains why; it was likewise missing from every panel \
of the first run. Both clauses are required, both are checked.

4. Material binds to the prop you attached it to, and bleeds if loose. Writing \
"wooden" of a door turned the floor and the furniture wooden too. Name the \
prop and its material in the same phrase, and do not use that material word \
anywhere else in the panel.

5. A prop's state gets its own sentence, never folded into the action. An \
action sentence freezes the prop mid-motion — a chair being knocked over stays \
in the air. A state sentence lands it: "the chair lies on its side on the \
floor".

6. Name the action, do not elaborate it. "coil slung up onto the shoulder" \
produced a coil lying flat against a neck; "carries the coil slung diagonally \
over one shoulder" got the carry, the loops and the orientation right. Plain \
verb, plain object, one clause.

7. Light is one sentence naming the source AND where the shadows fall. Both \
together measured 6 in 6; the source alone 6 in 9; neither, 0 in 3.

8. Intensity attaches to the nearest forceful verb, not to the one you meant. \
Keep an intensity word adjacent to its own action or leave it out.

9. Never name a compositional shape or guide. "read the frame as three \
vertical thirds", "arrange them in a triangle" — the model draws the guide \
lines. Say nothing about thirds, diagonals, triangles, framing devices.

10. Emotion has to be stated, or the beat plays backwards. A posture with no \
emotional word attached renders as the wrong feeling — asked for anger, \
returned a sad face. The KB does not currently carry an emotion field, so read \
the emotion out of the action and the scene and write it explicitly, as a word \
about the face or body: "her jaw sets", "he laughs, easy and unguarded".

RULES THAT COME FROM THIS PIPELINE

11. Do not write style, medium, film stock, grain, lens character or colour \
grading. A constant style anchor is appended after you, and a prompt that \
also describes style competes with it — one stray sentence of lighting logic \
moved black coverage from 25.4% to 1.3% by contradicting the style pack.

12. Do not write camera angle unless the field says high or low. Eye level is \
where the model already stands; a clause spent saying so buys nothing.

13. Anchor phrases (`anchor` on a subject or prop, `location_anchor`) are \
identity and go in VERBATIM, unedited, every panel they appear in. They are \
how the same person and place stay the same across a film. Never fold a \
momentary state into one.

14. Snake_case values are identifiers, not English. Render them as language: \
`near_future_autonomous_mobility` becomes "a near-future city built around \
autonomous transport".

15. Assume no text, signage or lettering appears in frame unless a panel \
explicitly calls for it.

SCENE VERSUS PANEL
Material true of the whole scene — location, period, light, the cast's \
appearance — is written once in `scene_common` and applies to every panel. \
Per-panel text carries only what changes: the action, the beat, the read \
point. Do not repeat scene_common inside a panel, and do not put a panel's \
own action into scene_common.

WHAT THE PANEL IS ABOUT
Each panel names one person or thing it is about. Describe that one more \
closely than anything else — it has to stay recognisable when the panel is \
shrunk to a thumbnail. Do not announce it. Write "her face is open, caught \
mid-laugh", never "she is the subject of this panel".

OUTPUT
Return JSON only, no commentary:

{
  "scene_common": "<prose true of every panel in this scene>",
  "panels": [
    {"panel_id": "<exact id given>", "prompt": "<this panel's own prose>"}
  ]
}

One entry per panel given, ids copied exactly. Prose is plain English \
sentences — no lists, no field names, no markdown.

A GOOD PANEL, for shape — note what it never says:

  "Exactly three people ride together in the cabin. She wears a plain cream \
  cotton t-shirt, her long brown hair loose, and she laughs — open, \
  unguarded, caught mid-breath. The handheld slab rests dark and inert \
  against her knee. Hard afternoon sun comes through the tinted glass and \
  throws crisp shadows down the seat backs. She occupies about a third of \
  the frame height."

It states a count, gives the emotion a word, binds the material to its own \
prop, names the light source and where its shadows fall, and says how large \
she is. It never says where anyone sits, and it never names a field.
"""


#: Prose that re-decides what the greybox already decided. Rejecting is the
#: point: on the first real call, gpt-4o wrote "foreground", "in front of"
#: and "rear seat" despite the rule, so the rule alone does not hold and the
#: output has to be checked rather than trusted.
_FRAME_POSITION = re.compile(
    r"\b(foreground|background|front seats?|rear seats?|"
    r"frame (?:left|right)|(?:centre|center) of the frame|"
    r"(?:left|right) (?:side )?of the frame)\b", re.I)

#: Object-relative phrasing — "a screen in front of him". It describes the
#: world, not the frame, and the authored action text often contains it: the
#: KB's own beat for scene_01_shot_03_panel_0003 reads "a screen in front of
#: him", so banning it outright put rule 6 (transcribe the action) in direct
#: conflict with the position rule, and the model correctly obeyed rule 6
#: three times out of three. Warned about, never rejected — the measured
#: harm was frame-relative placement and occlusion, not this.
_RELATIVE_POSITION = re.compile(
    r"\b(beside|behind|in front of|next to)\b", re.I)

#: The projection's own vocabulary, leaking into the picture as words. Every
#: panel of that same first call contained the literal phrase "read point".
_FORBIDDEN_META = re.compile(
    r"\b(read point|primary focus|panel \d|shot size|input_only|scene_common)\b",
    re.I)

#: Rules 2 and 3 — both measured, both silently omitted on the first call.
#: Absence is reported rather than rejected: a prompt without them is weaker,
#: not actively fighting the geometry.
_HAS_SIZE = re.compile(
    r"(occupies|fills|takes up|of the frame\b|frame (?:height|width))", re.I)
_HAS_COUNT = re.compile(r"\bexactly \w+\b", re.I)


def validate_prose(text: str, *, expect_count: bool = False,
                   source: str = "") -> tuple[list[str], list[str]]:
    """(errors, warnings) for one composed panel prompt.

    `source` is the authored text the panel was written from — its action,
    setting and anchors. Placement the model INVENTED is an error; the same
    words TRANSCRIBED out of authored text are a warning against the data,
    because rule 6 tells the model to name the action as given and the KB's
    own beats carry staging: scene_01_shot_02_panel_0002 reads "the two
    adults in the front seats watch the wraparound screens", and
    panel_0003's says "a screen in front of him". Rejecting those punished
    the model for obedience, 3 times out of 3, twice over.

    That the beats contain staging at all is the finding: it is the same
    two-sources-for-one-decision problem as writing position prose, sitting
    upstream in the KB where no prompt rule can reach it.
    """
    errors, warnings = [], []
    src = (source or "").lower()
    for m in sorted(set(m.group(0).lower() for m in _FRAME_POSITION.finditer(text))):
        if m in src:
            warnings.append(f"placement {m!r} carried over from the authored "
                            f"text — the beat itself stages the shot")
        else:
            errors.append(f"invents frame position: {m!r} — the greybox decides placement")
    for m in sorted(set(m.group(0).lower() for m in _RELATIVE_POSITION.finditer(text))):
        warnings.append(f"object-relative placement: {m!r} — fine if the action "
                        f"says so, competing with geometry if invented")
    for m in sorted(set(m.group(0).lower() for m in _FORBIDDEN_META.finditer(text))):
        errors.append(f"leaks field vocabulary into the prose: {m!r}")
    if not _HAS_SIZE.search(text):
        warnings.append("no subject-size clause; a shot-size phrase alone lets "
                        "the figure grow back to fill the frame")
    if expect_count and not _HAS_COUNT.search(text):
        warnings.append("no explicit count; the model obeys a number and "
                        "ignores a quantity adjective")
    return errors, warnings


def build_messages(payload: dict) -> list[dict]:
    """(system, user) messages for one scene's projected payload.

    The payload goes in as JSON rather than being flattened into prose: it is
    already structured, and re-narrating it here would be a second author
    competing with the one being instructed."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            "Write the prompts for this scene.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def parse_reply(text: str) -> dict:
    """The model's JSON, or a clear failure.

    Deliberately strict about shape and deliberately forgiving about wrapping:
    models fence JSON in ```json blocks about half the time, and refusing that
    would fail a reply that is otherwise exactly right."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return JSON: {e}") from e
    if not isinstance(data, dict) or "panels" not in data:
        raise ValueError("reply has no `panels`")
    for p in data["panels"]:
        if not isinstance(p, dict) or not p.get("panel_id") or not p.get("prompt"):
            raise ValueError(f"panel entry missing panel_id/prompt: {p!r}")
    return data


def compose(scene_common: str, panel_prompt: str) -> str:
    """Scene-wide prose plus one panel's own, in the order the image model
    reads them: the world first, then what is happening in it."""
    parts = [s.strip().rstrip(".") for s in (scene_common, panel_prompt) if s and s.strip()]
    return ". ".join(parts) + "." if parts else ""
