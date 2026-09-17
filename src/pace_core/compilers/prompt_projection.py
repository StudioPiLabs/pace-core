"""prompt_projection — which KB fields a prompt-writing model may see, and
which of those it may put into words.

A prompt written by a model needs a gate, and the gate has to be ADDITIVE.
Under a subtractive list — everything goes except a deny-list — any field
added to the KB later reaches the model the day it lands, with nobody
alerted: output degrades from some date onward and the cause is untraceable,
because nothing anywhere records that the model was being handed dozens of
fields no one had ever weighed. So every field here is named, with a reason,
and anything not named is withheld by construction (`_UNLISTED_ARE_WITHHELD`).

Two gates, not one — `Mode` is the distinction:

    WRITE        the model sees it and may put it into the prose
    INPUT_ONLY   the model sees it, and may not say it. Grounds for a
                 judgment, not words to say: a director's note explaining
                 why a shot is framed as it is should steer the tone of
                 what gets written without ever appearing as a sentence.
    WITHHELD     the model never sees it
    NEVER_FILLED withheld, but for a different reason: nothing in the KB
                 has ever set it. Not "tested and useless" — untested. The
                 distinction matters, because collapsing the two is how a
                 field gets deleted on the strength of evidence that was
                 never gathered.
    QUARANTINED  wanted, populated, and currently holding WRONG values. A
                 data bug, not a verdict — it comes back when the KB is
                 repaired. Two fields landed here on first contact with
                 real data, both of which a fill-rate audit had passed:
                 see `Rule.values_checked`, which exists because of them.

**Geometry owns composition here, and the model does not.** This is where
this projection departs from the system it borrows from: theirs has a
prompt-writing model and no 3D, so its model has to write the blocking. Ours
renders a greybox — exactly the declared cast, in the seats the location
describes, through the panel's own camera — and feeds it to `structure_flux2`
as an init image. Two sources for one decision measurably degrades it
(occlusion written alongside depth: 3/3 correct alone, 2/3 written together),
so screen positions, coordinates and camera geometry are WITHHELD. The model
writes what things look like. The greybox decides where they are.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field as _field

from pace_core.compilers.compile_common import _character_hint
from pace_core.pai_compat import (
    action0_of, angle_of, excluded_of, location_of, primary_focus_of,
    resolve_character, resolve_shot, shot_size_of, subjects_of, time_of_day_of,
)
from pace_core.paths import paths_for


class Mode(enum.Enum):
    WRITE = "write"
    INPUT_ONLY = "input_only"
    WITHHELD = "withheld"
    NEVER_FILLED = "never_filled"
    #: Populated, wanted, and currently holding values that are WRONG. Not a
    #: design verdict like WITHHELD — a data bug, and the field should come
    #: back the day the KB is repaired. Kept distinct precisely so nobody
    #: reads it later as "we decided against this".
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class Rule:
    """One field's verdict. `reason` is why, and it is not optional: a rule
    with no reason gets re-litigated by whoever reads it next.

    `evidence` says what kind of ground the verdict stands on, kept apart
    from `reason` on purpose — "we measured this" and "this follows from
    how the pipeline is built" are not the same claim, and a reader has to
    be able to tell which one they are looking at before overturning it.

    `values_checked` records whether anyone has looked at what this field
    actually CONTAINS in a real KB, which is not the same question as
    whether it is populated. A fill-rate audit says a field is set; it does
    not say the value is true. Two fields here passed the fill audit at
    27/27 and 58/58 and were admitted on that basis, and both turned out to
    be holding wrong data — an identity anchor calling a 50-year-old a
    "young adult", and every calm interior in the film labelled a
    "crowded_crash_site". An additive gate that admits on fill rate alone
    admits nonsense, confidently. Anything WRITE with values_checked=False
    is an admission nobody has verified."""
    mode: Mode
    reason: str
    evidence: str = "reasoned"          # reasoned | measured | untested
    values_checked: bool = False


# ── The table ────────────────────────────────────────────────────────────
#
# Paths are dotted, relative to a resolved shot. "[]" marks a list element.

FIELDS: dict[str, Rule] = {
    # ── camera: framing language yes, geometry no ────────────────────────
    "camera.creative_intent.shot_size": Rule(
        Mode.WRITE, "The framing term the image model actually responds to. "
        "Note it says only that the camera is FAR, never that the subject is "
        "small — the meta-prompt pairs it with an explicit subject-size line, "
        "without which the figure grows back to fill the frame.", "measured"),
    "camera.creative_intent.framing": Rule(
        Mode.WRITE, "Plain-language framing ('crowd', 'two_shot'). Measured to "
        "work as plain words and to fail as jargon.", "measured"),
    "camera.extrinsics.angle": Rule(
        Mode.WRITE, "high/low read plainly and change the picture. eye_level is "
        "inert — the model's default eye already sits there — so the "
        "meta-prompt drops it rather than spending a clause on it.", "measured"),
    "camera._design.intent": Rule(
        Mode.INPUT_ONLY, "A director's note ('the camera keeps its distance, like a "
        "stranger at the window'). It "
        "explains why the shot is built as it is; it is not a description of "
        "the frame, and writing it produces a picture of a stranger at a window."),
    "camera_brief": Rule(
        Mode.INPUT_ONLY, "Prose written for a human reader about the camera. "
        "Same reasoning as _design.intent."),
    "camera.intrinsics.lens_mm": Rule(
        Mode.WITHHELD, "The greybox solves the camera from this. Writing '28mm' "
        "as well is a second source for a decision geometry has already made."),
    "camera.intrinsics.sensor_width_mm": Rule(
        Mode.WITHHELD, "As lens_mm — solver input, not description."),
    "camera.intrinsics.fov_class": Rule(
        Mode.WITHHELD, "Restates shot_size in jargon."),
    "camera.extrinsics.position": Rule(
        Mode.WITHHELD, "Which side the lens stands on is geometry; the greybox "
        "already places it."),
    "camera.trajectory": Rule(
        Mode.WITHHELD, "A still panel has no motion. The video stage owns this."),

    # ── location and period: the anchor ──────────────────────────────────
    "setup.backdrop.location": Rule(
        Mode.WRITE, "The location's fixed identity phrase, written verbatim as "
        "an anchor so the same place reads the same across panels."),
    "setup.backdrop.setting": Rule(
        Mode.WRITE, "The location's own prose description."),
    "setup.backdrop.era": Rule(Mode.WRITE, "Period adjective."),
    "setup.backdrop.culture": Rule(Mode.WRITE, "Period/place adjective."),
    "setup.backdrop.region": Rule(Mode.WRITE, "Period/place adjective."),

    # ── environment ──────────────────────────────────────────────────────
    "setup.environment.lighting": Rule(
        Mode.WRITE, "Light belongs in the prose, and belongs as ONE sentence "
        "naming the source and where the shadows fall — that form measured "
        "6/6, source alone 6/9, nothing 0/3.", "measured"),
    "setup.environment.background": Rule(
        Mode.WRITE, "What sits behind the cast."),
    "setup.environment.density": Rule(
        Mode.QUARANTINED, "Found holding one template value on most shots of "
        "a measured production, including interiors it contradicts. A leaked "
        "default is not a description, and a model told a frame is crowded "
        "will draw a crowd. Restore when the field describes the shot it is "
        "on.", "measured", True),
    "setup.environment.scale": Rule(
        Mode.WRITE, "How large the space reads, and distinct from shot_size, "
        "which is only where the camera stands — writing scale stretched a "
        "courtyard from 14 gate-widths to 30-40 with the shot size unchanged. "
        "Note it can hold one value on every shot of a production, so "
        "it differentiated nothing there; harmless, but do not read a "
        "difference into it.", "measured", True),
    "setup.environment.style": Rule(
        Mode.WITHHELD, "The style anchor is a constant this pipeline owns and "
        "the model may not touch a word of. A model free to restate the style "
        "competes with it: one sentence of lighting logic moved black coverage "
        "25.4% -> 1.3% by contradicting the style pack.", "measured"),
    "setup.environment.elements": Rule(
        Mode.NEVER_FILLED, "Never filled by the breakdown."),

    # ── the film's world, which is not a per-shot field ──────────────────
    # Not read off a shot at all: it comes from `kb/shot_design.json` and is
    # the same for every panel in the film. Named here because this gate is
    # additive and anything unlisted is withheld by construction, so a
    # reader looking for where the world went should find the answer in the
    # one place that records these verdicts.
    "film.genre": Rule(
        Mode.WITHHELD, "题材. Compiled, but by the pipeline rather than by the "
        "writer, and into one slot only: it is the floor under a panel that "
        "declares no atmosphere -- 8 of 36 here, which until now rendered "
        "with whichever mood the sampler chose."),
    "film.source": Rule(
        Mode.WITHHELD, "核心溯源. Where the concept came from -- a Shan Hai "
        "Jing passage, a news item. Provenance for a human reader; compiling "
        "it draws the source, not the film."),
    "film.worldview.premise": Rule(
        Mode.INPUT_ONLY, "世界观设定. The world's RULES, not its look: what the "
        "system in the story can and cannot do bounds the plot and describes "
        "no frame. Grounds for a judgment, like a director's note."),
    "film.theme": Rule(
        Mode.INPUT_ONLY, "核心主题. A theme is abstract by definition. Writing "
        "one produces a picture OF it, the failure already measured on "
        "camera._design.intent -- 'like a stranger at the window' "
        "draws a stranger at a window."),
    "film.worldview.design_language": Rule(
        Mode.WITHHELD, "The world anchor is a constant this pipeline owns, "
        "appended by the compilers next to the style anchor and for the same "
        "measured reason: a model free to restate a constant competes with "
        "it. It was five byte-identical copies of `visual_constraints`, one "
        "per location stub, and read by nothing at all.", "measured"),
    "film.worldview.invariants": Rule(
        Mode.WITHHELD, "States something ACROSS shots -- 'physical spatial "
        "continuity across reused shots' -- which no single frame can honour "
        "or violate, so there is nothing for a prompt to say. Checked "
        "instead, by a design check against the shot design."),

    # ── the room itself: geometry ────────────────────────────────────────
    "setup.space.scale_meters": Rule(
        Mode.WITHHELD, "The greybox builds the shell from this. Prose about "
        "room dimensions competes with a room that has already been built."),
    "setup.space.depth": Rule(Mode.WITHHELD, "As scale_meters."),
    "setup.space.orientation": Rule(Mode.WITHHELD, "As scale_meters."),

    # ── subjects: appearance yes, position no ────────────────────────────
    "setup.subjects[].continuity_anchor": Rule(
        Mode.QUARANTINED, "Meant to be the character's fixed identity phrase, "
        "written verbatim — and it has been found contradicting "
        "characters.json for most of a cast, anchors opening 'young adult' for "
        "characters registered middle_aged and 'child' for one registered "
        "young_adult. The deterministic compiler never read this field, so "
        "the contradiction has been free until now; writing it verbatim as the "
        "identity authority would make it expensive. Restore once the anchors "
        "agree with the registry. (Also: never let a momentary state into one "
        "— an anchor reading 'a table broken in two' contaminated every later "
        "scene where the table was intact.)", "measured", True),
    "setup.subjects[].costume": Rule(
        Mode.QUARANTINED, "Found to contradict the character registry's own "
        "costumes.default for the same character in the same shot, one a "
        "genre phrase and the other a specific garment. "
        "character.costume (the registry) is now WRITE in its place; this "
        "field returns once the two are reconciled or this one is retired.",
        "measured", True),
    "setup.subjects[].hair": Rule(Mode.WRITE, "Appearance."),
    "setup.subjects[].pose": Rule(Mode.WRITE, "What the body is doing."),
    "setup.subjects[].character_id": Rule(
        Mode.INPUT_ONLY, "Resolves to a LoRA trigger the compiler inserts "
        "verbatim. The model needs to know who is present in order to write "
        "about them; the id itself is not English."),
    "setup.subjects[].age_state": Rule(
        Mode.INPUT_ONLY, "Selects the mesh and the LoRA; the age reads out of "
        "the anchor phrase, not out of this."),
    "setup.subjects[].cls": Rule(Mode.INPUT_ONLY, "Schema bookkeeping."),
    "setup.subjects[].screen_position": Rule(
        Mode.WITHHELD, "Blocking is the greybox's. This is the field whose "
        "mirror bug produced a cast staged against its own declaration — "
        "fixed in geometry, and it stays fixed only if prose does not "
        "re-decide it. Withheld from the *writer* for that reason, which is "
        "not the same as keeping it out of the prompt: the deterministic "
        "compiler now names each subject in screen order and anchors its "
        "description to this declared position "
        "(compile_common.subjects_left_to_right), because leaving the text "
        "silent about place let the sampler pair descriptions with bodies "
        "freely — count and positions correct, the people in the wrong "
        "seats. Transcribing the declaration cannot contradict the geometry; "
        "a model choosing a side can. When this projection starts feeding "
        "live renders, the writer's output needs that same anchoring applied "
        "to it deterministically, or the writer path reintroduces the bug the "
        "compiler path just lost."),
    "setup.subjects[].gaze": Rule(
        Mode.NEVER_FILLED, "0 of 58 subjects. Untested here, and flagged in "
        "the source system as one of the hardest rules with the emptiest "
        "data — not evidence that it does not matter.", "untested"),

    # ── character registry (characters.json — a different file from the
    #    scene) ────────────────────────────────────────────────────────
    # Found by comparing a written prompt against the compiled one: the
    # compiler pulls a character's physical description from a SECOND KB
    # file this projection never read, so every written prompt had been
    # missing the only identity signal that exists — no LoRA is trained for
    # any of the three leads (lora.path is null), so this prose is the sole
    # thing telling the model who is on screen.
    "character.descriptor": Rule(
        Mode.WRITE, "One character's canonical physical description — build, "
        "ethnicity, hair, expression, age, costume — resolved through "
        "compile_common._character_hint(), the SAME function compile_flux2 "
        "calls, so this can never drift from what the deterministic compiler "
        "would have written. Not reimplemented here on purpose: age-state "
        "lookup, costume-append and trigger-stripping are three independent "
        "small decisions, and a second copy of them is a second place for "
        "the two to disagree. Distinct from setup.subjects[].costume (now "
        "QUARANTINED — found to contradict this registry's own costume for "
        "the same character in the same shot) and from the scene-level "
        "continuity_anchor (QUARANTINED for contradicting this registry's "
        "age_band on 4 of 5 cast). This field IS the registry, not a copy of "
        "it, and carries neither contradiction.", "measured", True),
    "character.trigger": Rule(
        Mode.INPUT_ONLY, "A LoRA activation word for a model that, for this "
        "cast, does not exist yet (lora.path is null for all three leads). "
        "Kept as context rather than dropped, since it costs nothing and the "
        "day a LoRA lands here it should already know to expect a trigger "
        "convention — but writing an inert activation word into prose reads "
        "as a name, and this cast is not named that in dialogue."),

    # ── props ────────────────────────────────────────────────────────────
    "setup.props[].name": Rule(Mode.WRITE, "What the thing is."),
    "setup.props[].continuity_anchor": Rule(
        Mode.WRITE, "The prop's fixed identity phrase, verbatim."),
    "setup.props[].material": Rule(
        Mode.WRITE, "Bound to the named prop only — material bleeds: writing "
        "'wooden' of a door turned the floor wooden too. Set on 8 of 76 props, "
        "as snake_case tokens the writer has to render into English.",
        "measured", True),
    "setup.props[].size": Rule(Mode.WRITE, "Physical size of the prop."),
    "setup.props[].state": Rule(
        Mode.WRITE, "Its own sentence, never folded into the action: an action "
        "sentence freezes a prop mid-fall, where a state sentence lands it "
        "('the chair lies on its side on the floor'). Thin here — set on 8 of "
        "76 props, and the only value is the token 'active', which is not a "
        "described state. Worth authoring properly.", "measured", True),
    "setup.props[].cls": Rule(Mode.INPUT_ONLY, "Schema bookkeeping."),
    "setup.props[].prop_id": Rule(Mode.INPUT_ONLY, "Identity, not English."),
    "setup.props[].screen_position": Rule(
        Mode.WITHHELD, "Blocking is the greybox's."),

    # ── the read point ───────────────────────────────────────────────────
    "setup.primary_focus.ref": Rule(
        Mode.WRITE, "One panel carries one primary read point; naming it is "
        "how the prose agrees with the frame."),
    "setup.primary_focus.type": Rule(
        Mode.WRITE, "Decides the phrasing branch — a feature reads differently "
        "from a character."),
    "setup.primary_focus.coverage_pct": Rule(
        Mode.INPUT_ONLY, "Grounds for how large to write the subject, and a "
        "coarse knob at that: asked 55% the model gives 20-40%, asked 15% it "
        "gives 8-15%. Direction is reliable, magnitude is not, so it informs "
        "a judgment and is never quoted as a number.", "measured"),
    "setup.primary_focus.camera_target": Rule(
        Mode.WITHHELD, "Geometry."),
    "setup.primary_focus.screen_position": Rule(
        Mode.WITHHELD, "Geometry."),

    # ── the beat ─────────────────────────────────────────────────────────
    "events.actions[].description_en": Rule(
        Mode.WRITE, "The action. Named, not elaborated — 'coil slung up onto "
        "the shoulder' produced a coil lying flat against the neck, where "
        "'carries the coil slung diagonally over one shoulder' got carry, "
        "loops and orientation right.", "measured"),
    "events.actions[].description_zh": Rule(
        Mode.INPUT_ONLY, "The same beat in the authoring language; useful "
        "context, but the prompt is written in English."),
    "events.actions[].intensity": Rule(
        Mode.WRITE, "'dramatic' works, but it attaches to the nearest forceful "
        "action rather than the one named — the meta-prompt keeps it adjacent "
        "to its own verb.", "measured"),
    "events.actions[].foreground": Rule(
        Mode.INPUT_ONLY, "Which beat leads. Sequencing, not description — and "
        "a background flag written as prose pushes the subject away, shrinking "
        "the face until the emotion is unreadable.", "measured"),
    "events.actions[].standalone": Rule(Mode.INPUT_ONLY, "Sequencing."),
    "events.actions[].temporal": Rule(Mode.INPUT_ONLY, "Sequencing."),
    "events.advanced.pace": Rule(Mode.INPUT_ONLY, "Scene rhythm; steers tone."),
    "events.advanced.regularity": Rule(Mode.INPUT_ONLY, "As pace."),
    "events.emotions": Rule(
        Mode.NEVER_FILLED, "Never filled by the breakdown — and the one field measured to be "
        "the ONLY source of the emotional adjective: posture with no emotion "
        "word plays the beat backwards, asking for anger and getting a sad "
        "face. Until it is authored, the meta-prompt has the model read the "
        "emotion out of the action text instead.", "untested"),
    "events.dialogues": Rule(Mode.NEVER_FILLED, "Never filled by the breakdown."),
    "events.change_in_environment": Rule(Mode.NEVER_FILLED, "Never filled by the breakdown."),
    "events.advanced.story_structure": Rule(Mode.NEVER_FILLED, "Never filled by the breakdown."),

    # ── exclusions ───────────────────────────────────────────────────────
    "setup.excluded": Rule(
        Mode.WRITE, "Written as a COMPLETED FACT, never as framing negation: "
        "'the crowd has left' landed 3/3, 'no crowd in the frame' 0/3 — the "
        "latter cleared a patch of floor and left the crowd. Doubly true here, "
        "where cfg=1.0 means the negative prompt is never evaluated at all, so "
        "positive substitution is the only channel that exists.", "measured"),
    "setup.secondary_subjects": Rule(
        Mode.NEVER_FILLED, "Never filled by the breakdown."),
    "setup.text_generation": Rule(Mode.NEVER_FILLED, "Never filled by the breakdown."),

    # ── panel identity and authoring notes ───────────────────────────────
    "panels[].id": Rule(Mode.INPUT_ONLY, "Addresses the panel in the reply."),
    "panels[].panel_number": Rule(Mode.INPUT_ONLY, "Ordering within the shot."),
    "panels[].notes": Rule(
        Mode.INPUT_ONLY, "Notes an author wrote to a human. Steers what to "
        "emphasise; is not itself a description of the frame."),
}

# Anything the KB grows later and nobody has weighed is not shown to the
# model. This is the whole point of the file.
_UNLISTED_ARE_WITHHELD = True


def visible(path: str) -> bool:
    """Does the model get to see this field at all?"""
    r = FIELDS.get(path)
    return bool(r) and r.mode in (Mode.WRITE, Mode.INPUT_ONLY)


def writable(path: str) -> bool:
    """May the model put this field into the prose?"""
    r = FIELDS.get(path)
    return bool(r) and r.mode is Mode.WRITE


def _gate(pairs: list[tuple[str, str, object]]) -> tuple[dict, dict]:
    """(out_key, table_path, value) triples -> (write_block, input_block).

    Every value in the payload passes through here, so the table is the gate
    rather than a description of one. Building the payload by hand alongside
    a table that says what it should contain is two implementations of one
    rule, and they drift: the first version of this file quarantined two
    fields and went on emitting both, because the dict literal never asked.

    Empty values are dropped — a key with nothing behind it is a line the
    model has to read and cannot use."""
    write, inputs = {}, {}
    for key, path, value in pairs:
        rule = FIELDS.get(path)
        if rule is None or value in (None, "", [], {}):
            continue
        if rule.mode is Mode.WRITE:
            write[key] = value
        elif rule.mode is Mode.INPUT_ONLY:
            inputs[key] = value
    return write, inputs


def project_scene(scene: dict, *, project: str = "") -> dict:
    """The payload a prompt-writing model is given for ONE scene.

    One scene, not one panel: writing panels one at a time makes panels of a
    single scene look like unrelated images, so the whole scene goes in one
    call and comes back all-or-nothing.

    Values are resolved (panel overrides applied, shot defaults inherited)
    through the same helpers the deterministic compiler uses, so the model
    sees the value that would actually be rendered rather than the raw
    field. Everything WITHHELD by the table is absent; everything
    INPUT_ONLY is present and marked, so the meta-prompt can forbid writing
    it without having to hide it.
    """
    # characters.json is a SEPARATE file from the scene, keyed by character
    # id, holding the canonical anchor/costume every render is meant to hold
    # to — as opposed to setup.subjects[], which re-types a description per
    # shot and can (and did) drift from it.
    characters_kb = {}
    if project:
        try:
            characters_kb = json.loads(paths_for(project).chars_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    shots_out = []
    for shot in scene.get("shots") or []:
        sh = resolve_shot(scene, shot)
        setup = sh.get("setup") or {}
        env = setup.get("environment") or {}
        backdrop = setup.get("backdrop") or {}
        loc_ref, loc_setting = location_of(scene, sh)

        panels_out = []
        for panel in sh.get("panels") or []:
            pf = primary_focus_of(panel, sh)
            act = action0_of(sh, panel)
            pan_w, pan_i = _gate([
                ("action", "events.actions[].description_en", act.get("description_en")),
                ("intensity", "events.actions[].intensity", act.get("intensity")),
                ("read_point_ref", "setup.primary_focus.ref", pf.get("ref")),
                ("read_point_type", "setup.primary_focus.type", pf.get("type")),
                ("excluded", "setup.excluded", excluded_of(panel, sh)),
                ("action_zh", "events.actions[].description_zh", act.get("description_zh")),
                ("read_point_coverage_pct", "setup.primary_focus.coverage_pct",
                 pf.get("coverage_pct")),
                ("foreground", "events.actions[].foreground", act.get("foreground")),
                ("panel_number", "panels[].panel_number", panel.get("panel_number")),
                ("notes", "panels[].notes", panel.get("notes")),
            ])
            panels_out.append({
                "panel_id": panel.get("id"),
                "write": pan_w,
                "input_only": pan_i,
            })

        chars, ages = subjects_of(sh)
        # The model gets the English framing phrase the deterministic compiler
        # would have used, imported rather than restated so the two cannot
        # drift apart. Both sides key on the ShotSize Literal now; this used to
        # need a note explaining that shot_size_of returned a Chinese table key.
        from pace_core.compilers.compile_flux2 import _SHOT_SIZE_FRAMING_EN
        size_key = shot_size_of(sh)
        ang = angle_of(sh)

        shot_w, shot_i = _gate([
            ("shot_size", "camera.creative_intent.shot_size",
             _SHOT_SIZE_FRAMING_EN.get(size_key or "", "")),
            ("framing", "camera.creative_intent.framing",
             ((sh.get("camera") or {}).get("creative_intent") or {}).get("framing")),
            # eye_level is dropped rather than withheld: the model's default
            # eye already sits there, so a clause spent on it buys nothing.
            ("camera_angle", "camera.extrinsics.angle", ang if ang != "eye_level" else ""),
            # location_of()'s second value is backdrop.location — an id
            # ("sedan_interior"), not a description. The prose is backdrop.setting.
            ("location_anchor", "setup.backdrop.location", loc_ref or loc_setting),
            ("setting", "setup.backdrop.setting", backdrop.get("setting")),
            ("era", "setup.backdrop.era", backdrop.get("era")),
            ("culture", "setup.backdrop.culture", backdrop.get("culture")),
            ("region", "setup.backdrop.region", backdrop.get("region")),
            ("lighting", "setup.environment.lighting", env.get("lighting")),
            ("background", "setup.environment.background", env.get("background")),
            ("density", "setup.environment.density", env.get("density")),
            ("scale", "setup.environment.scale", env.get("scale")),
            ("design_intent", "camera._design.intent",
             ((sh.get("camera") or {}).get("_design") or {}).get("intent")),
            ("camera_brief", "camera_brief", sh.get("camera_brief")),
            ("pace", "events.advanced.pace",
             ((sh.get("events") or {}).get("advanced") or {}).get("pace")),
        ])
        # time_of_day is derived (shot, falling back to scene) rather than a
        # single field, so it rides on the backdrop's admission.
        if visible("setup.backdrop.setting"):
            shot_w["time_of_day"] = time_of_day_of(sh, scene)
        shot_i["character_ids"], shot_i["age_states"] = chars, ages

        subs_out, triggers = [], []
        for s in setup.get("subjects") or []:
            cid, age = s.get("character_id"), s.get("age_state")
            ref = f"{cid}@{age}" if cid and age else (cid or "")
            descriptor = _character_hint(ref, characters_kb) if ref else ""
            char_kb = resolve_character(cid, characters_kb) if cid else {}
            if visible("character.trigger") and char_kb.get("trigger"):
                triggers.append(char_kb["trigger"])
            sw, _si = _gate([
                ("anchor", "setup.subjects[].continuity_anchor", s.get("continuity_anchor")),
                ("costume", "setup.subjects[].costume", s.get("costume")),
                ("hair", "setup.subjects[].hair", s.get("hair")),
                ("pose", "setup.subjects[].pose", s.get("pose")),
                ("descriptor", "character.descriptor",
                 descriptor if descriptor != ref else ""),  # unchanged = no KB entry
            ])
            if sw:
                subs_out.append(sw)
        if subs_out:
            shot_w["subjects"] = subs_out
        if triggers:
            shot_i["character_trigger_words"] = triggers

        props_out = []
        for pr in setup.get("props") or []:
            pw, _pi = _gate([
                ("name", "setup.props[].name", pr.get("name")),
                ("anchor", "setup.props[].continuity_anchor", pr.get("continuity_anchor")),
                ("material", "setup.props[].material", pr.get("material")),
                ("size", "setup.props[].size", pr.get("size")),
                ("state", "setup.props[].state", pr.get("state")),
            ])
            if pw:
                props_out.append(pw)
        if props_out:
            shot_w["props"] = props_out

        shots_out.append({
            "shot_id": sh.get("shot_id"),
            "write": shot_w,
            "input_only": shot_i,
            "panels": panels_out,
        })

    return {
        "project": project,
        "scene_id": scene.get("scene_id"),
        "projection_version": PROJECTION_VERSION,
        "shots": shots_out,
    }


# Bumped whenever the table changes what reaches the model. Stored beside a
# written prompt, so a prompt can be traced to the gate that produced it —
# otherwise "the prompts got worse in September" has no way back to a cause.
PROJECTION_VERSION = 1


def audit() -> str:
    """The table as a readable report. Exists because the gate is worth
    nothing if no one can see what it is currently letting through."""
    rows = []
    for mode in Mode:
        paths = sorted(p for p, r in FIELDS.items() if r.mode is mode)
        if not paths:
            continue
        rows.append(f"\n{mode.value.upper()}  ({len(paths)})")
        for p in paths:
            r = FIELDS[p]
            rows.append(f"  {p:44} [{r.evidence}]")
            rows.append(f"      {r.reason}")
    return "\n".join(rows)
