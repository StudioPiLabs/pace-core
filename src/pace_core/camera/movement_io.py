"""Movement I/O — single source of truth for camera-movement adapter logic.

All consumers (camera_planner, studio_server, blender_render, Wan VACE)
delegate movement reads here. One import = one canonical interpretation.

Surface:

  read_movement(shot)              → (movements, easing)
  write_movement(shot, mvs, ease)  → shot (mutated in place, returned for chaining)
  shot_to_planner_dict(shot, ...)  → planner-shape dict
  compile_wan_motion(movements, …) → natural-language motion directive

Operates on pai-1.0 shot dicts (the on-disk shape). Reads
`camera.trajectory.{movement_2d, movement_3d, gear, easing}` and
`camera.extrinsics.position`; SCINE English → planner Movement vocab
mapping lives in pai_compat.

Pure functions, no I/O. Tested by tests/test_movement_io.py.
"""

from __future__ import annotations

from typing import Optional

from pace_core.pai_compat import (
    movement_of, easing_of, shot_size_of, angle_of, dig,
)


EASING_KINDS = ("linear", "ease_in", "ease_out", "ease_in_out")
DEFAULT_EASING = "linear"


# ── Read / write ──────────────────────────────────────────────────────


def read_movement(shot: dict) -> tuple[list[str], str]:
    """Canonical reader. Returns (movements, easing) for a pai-1.0 shot dict.

    `movements` is the flat Movement Literal list (compiled by
    `pai_compat.movement_of` from camera.trajectory.movement_2d/3d + gear
    + extrinsics.position). `easing` is the temporal sampling curve.
    """
    easing = easing_of(shot)
    if easing not in EASING_KINDS:
        easing = DEFAULT_EASING
    return movement_of(shot), easing


def write_movement(shot: dict, movements: list[str],
                   easing: str = DEFAULT_EASING) -> dict:
    """Canonical writer. Translates a flat Movement Literal list back into
    SCINE Camera.trajectory.movement_2d/movement_3d/gear + Extrinsics.position
    and writes it onto the pai-1.0 shot dict in place. Mutually exclusive:
    a token landing in `static`, `movement_2d`, `movement_3d`, `gear`, or
    `position` clears the other slots it conflicts with."""
    if easing not in EASING_KINDS:
        easing = DEFAULT_EASING
    cam = shot.setdefault("camera", {})
    traj = cam.setdefault("trajectory", {})
    ex   = cam.setdefault("extrinsics", {})

    # Forward map: planner Movement Literal → SCINE buckets.
    fwd_2d = {
        "pan_lr": "pan_right", "pan_rl": "pan_left",
        "tilt_up": "tilt_up", "tilt_down": "tilt_down",
    }
    fwd_3d = {
        "push_in": "push_in", "pull_out": "pull_out",
        "tracking": "tracking", "orbit": "arc",
        "crane_up": "crane", "crane_down": "crane",
    }
    fwd_gear = {"handheld": "handheld", "steadicam": "steadicam"}
    fwd_pos  = {"from_behind": "behind", "reverse_shot": "ots"}

    m2d, m3d = [], []
    gear = None
    pos  = None
    is_static = False
    for m in movements:
        if m == "static":
            is_static = True
        elif m in fwd_2d:
            m2d.append(fwd_2d[m])
        elif m in fwd_3d:
            m3d.append(fwd_3d[m])
        elif m in fwd_gear:
            gear = fwd_gear[m]
        elif m in fwd_pos:
            pos = fwd_pos[m]
        # unknowns silently dropped

    traj["movement_2d"] = m2d
    traj["movement_3d"] = m3d
    traj["easing"]      = easing
    traj["static"]      = bool(is_static and not (m2d or m3d))
    # Always (re)write gear + position, even when the new list omits them.
    # Otherwise a previous save's gear/position leaks into the next response
    # and the user thinks their unchecked chip "didn't save".
    traj["gear"]     = gear      # None when no gear chip is in `movements`
    ex["position"]   = pos       # None when no position chip is in `movements`
    return shot


# ── PAI → planner adapter ─────────────────────────────────────────────


def shot_to_planner_dict(shot: dict,
                         scene: Optional[dict] = None,
                         scene_id: Optional[str] = None) -> dict:
    """Adapt a pai-1.0 shot dict to the planner's input shape.

    The planner wants flat fields:
      camera.shot_size / camera.angle / camera.lens_mm / camera.aperture
      frame.movement / frame.movement_easing
    Walk pai-1.0 paths and emit that shape. `scene_ref` is derived from
    narrative_meta.location_ref so the planner's location-aware
    eye-level lookup hits the right entry.
    """
    movements, easing = read_movement(shot)
    intr = dig(shot, "camera", "intrinsics") or {}
    scene_ref = dig(scene, "narrative_meta", "location_ref") or "" if scene else ""
    if not scene_ref and scene_id:
        scene_ref = scene_id
    return {
        "scene_ref": scene_ref,
        "camera": {
            "shot_size": shot_size_of(shot) or "medium",
            "angle":     angle_of(shot)     or "eye_level",
            # Two key vintages for the same quantity: the schema name
            # (`focal_length_mm`, `t_stop`) and the one the scene documents
            # actually carry (`lens_mm`, `aperture_f`). Reading only the schema
            # name returned None for every real scene, and the planner then
            # substituted its default -- a shot declaring 35 mm was planned,
            # framed and rendered at 50 mm with nothing reporting the swap.
            # Accept both, schema name first.
            "lens_mm":   intr.get("focal_length_mm") or intr.get("lens_mm"),
            "aperture":  intr.get("t_stop") or intr.get("aperture_f"),
        },
        "frame": {
            "movement":        movements,
            "movement_easing": easing,
        },
    }


def _shot_size_of_raw(shot: dict):
    """shot_size from a pai-1.0 shot (creative_intent first, then compat)."""
    return (((shot.get("camera") or {}).get("creative_intent") or {}).get("shot_size")
            or shot_size_of(shot))


def shot_to_narrative(shot: dict, scene: Optional[dict] = None,
                      prev_shot: Optional[dict] = None) -> str:
    """Compose a natural-language camera-intent description for a pai-1.0 shot —
    the prompt the LAMP language→DSL planner consumes when planning **from the
    scene directly** (no hand-typed text).

    Assembles, in one string:
      - the human-authored action beat  (events.actions[0].description_en|zh)
      - scene setting (when `scene` given): narrative_meta location_ref /
        interior_exterior / time_of_day
      - framing: shot_size / angle / framing / intensity
      - continuity (when `prev_shot` given): the previous shot's size, so the
        LLM can vary or match the cut.

    Returns "" when the shot carries no usable text. Canonical single source of
    truth — the per-shot + scene-batch camera-plan endpoints (and CLI) all call
    this, so every path derives the same intent.
    """
    cam = shot.get("camera") or {}
    ci  = cam.get("creative_intent") or {}
    ex  = cam.get("extrinsics") or {}
    acts = ((shot.get("events") or {}).get("actions")) or []
    beat = intensity = ""
    if acts and isinstance(acts[0], dict):
        a = acts[0]
        beat = (a.get("description_en") or a.get("description_zh") or "").strip()
        intensity = (a.get("intensity") or "").strip()

    # Scene setting from narrative_meta (location / INT-EXT / time-of-day).
    nm = (scene or {}).get("narrative_meta") or {}
    setting = [str(v) for v in (nm.get("location_ref") or nm.get("location_raw"),
                                nm.get("interior_exterior"),
                                nm.get("time_of_day")) if v]
    setting_str = ", ".join(setting)

    # Framing + mood + continuity → the parenthetical.
    ctx: list[str] = []
    for v in (ci.get("shot_size") or shot_size_of(shot),
              ex.get("angle") or angle_of(shot),
              ci.get("framing")):
        if v:
            ctx.append(str(v))
    if intensity:
        ctx.append(f"{intensity} intensity")
    if prev_shot:
        psize = _shot_size_of_raw(prev_shot)
        if psize:
            ctx.append(f"follows a {psize} shot")
    framing_str = ", ".join(ctx)

    head = beat
    if setting_str:
        head = f"{head} — {setting_str}" if head else setting_str
    if head and framing_str:
        return f"{head} ({framing_str})"
    return head or framing_str


# ── Wan VACE motion directive ─────────────────────────────────────────


# Each Movement Literal → its English motion phrase for Wan I2V / VACE
# prompts. Wan's text encoder understands fluent English motion language
# better than the Literal token itself.
_WAN_MOTION_PHRASE = {
    "push_in":      "slow camera push-in",
    "push_in_slow": "gentle camera push-in",
    "pull_out":     "slow camera pull-out",
    "tracking":     "tracking shot following the subject",
    "pan_lr":       "slow pan from left to right",
    "pan_rl":       "slow pan from right to left",
    "tilt_up":      "camera tilt upward",
    "tilt_down":    "camera tilt downward",
    "crane_up":     "rising crane shot",
    "crane_down":   "descending crane shot",
    # Stylistic — no geometric delta but still affects how Wan animates
    "handheld":     "subtle handheld camera shake",
    "steadicam":    "smooth steadicam motion",
    "rapid":        "rapid handheld camera movement",
    # Wan-only — planner stays static; only manifest through Wan's
    # text-to-motion path.
    "orbit":        "slow orbital camera move around the subject",
    "from_behind":  "view from behind the subject",
    "reverse_shot": "reverse-angle shot of the subject",
}


# Gear that a movement token already carries (movement_of folds handheld and
# steadicam into the flat list, so naming them again would say it twice).
_GEAR_IMPLIED = {"handheld", "steadicam"}

# How the remaining gear reads in a sentence. A tripod is worth saying even
# though it produces no movement token: "locked off on a tripod" is a camera
# decision, and an LLM asked for skills should hear it rather than infer a
# move from silence.
_GEAR_PHRASE = {
    "tripod":    "locked off on a tripod",
    "dolly":     "on a dolly",
    "crane":     "on a crane",
    "jib":       "on a jib",
    "gimbal":    "on a gimbal",
    "drone":     "from a drone",
    "slider":    "on a slider",
}


def _words(v) -> str:
    """`extreme_close_up` -> `extreme close up`. The JSON stores enum tokens;
    the brief is read by a language model, so it gets language."""
    return str(v or "").replace("_", " ").strip()


def camera_brief(shot: dict, scene: Optional[dict] = None,
                 prev_shot: Optional[dict] = None) -> str:
    """The camera in words: how it moves, on top of what the shot is about.

    ``shot_to_narrative`` already composes the canonical intent text — action
    beat, setting, framing, continuity — and deliberately says nothing about
    the move, because the LAMP planner takes ``movements`` as its own separate
    argument. The skills route (``/api/camera/skills/from-nl``) takes text and
    nothing else, so it needs both halves in one string.

    This is the half that was missing — movement, rig, and the
    ``camera_space_goal`` the shotlist generator wrote as prose — in front of
    the narrative. Framing and angle are not repeated here: they are
    ``shot_to_narrative``'s to say.
    """
    traj = ((shot or {}).get("camera") or {}).get("trajectory") or {}
    movements, easing = read_movement(shot)

    parts: list[str] = []
    motion = compile_wan_motion(movements, easing)
    gear = str(traj.get("gear") or "").strip().lower()
    if motion:
        parts.append(motion)
        # A tripod under a moving camera is the data contradicting itself —
        # crane shots have been found declaring gear "tripod" too.
        # The move is the more specific claim, so the rig stays quiet rather
        # than asking a language model to reconcile the two.
        if gear == "tripod":
            gear = ""
    elif gear in _GEAR_PHRASE:
        parts.append(_GEAR_PHRASE[gear])          # says the stillness and the rig at once
        gear = ""
    else:
        parts.append("static camera, no movement")
    if gear and gear not in _GEAR_IMPLIED:
        parts.append(_GEAR_PHRASE.get(gear, f"on a {gear}"))

    brief = ", ".join(parts)
    narrative = shot_to_narrative(shot, scene=scene, prev_shot=prev_shot)
    if narrative:
        brief = f"{brief}. {narrative}"
    goal = _words(traj.get("camera_space_goal"))
    if goal:
        brief = f"{brief}. {goal[0].upper()}{goal[1:]}"
    return brief


def compile_wan_motion(movements: list[str], easing: str = DEFAULT_EASING) -> str:
    """Build a natural-language motion directive for a Wan I2V / VACE prompt.

    Empty / static-only movement list → `""` (caller decides how to
    handle the empty case — usually by skipping the motion clause).
    """
    phrases = [_WAN_MOTION_PHRASE[m] for m in movements
               if m in _WAN_MOTION_PHRASE]
    if not phrases:
        return ""
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]} combined with {phrases[1]}"
    else:
        body = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    suffix = {
        "ease_in":     ", starting slowly then accelerating",
        "ease_out":    ", starting quickly then settling",
        "ease_in_out": ", easing in and out smoothly",
    }.get(easing, "")
    return body + suffix


# ── MiniMax H3 camera-motion directive ─────────────────────────────────

# Each Movement Literal → H3's own motion-type vocabulary. Source: H3's
# official prompt-writing guide (skills/h3-prompt-writing/references/
# base-en.txt, MiniMax-AI/MiniMax-H3 on GitHub — mirrored at
# huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/
# VIDEO_PROMPT_WRITING_GUIDE_base_en.md). Camera motion is written as
# natural English action embedded in the shot ("The camera pushes in..."),
# NOT as bracket tags stacked at the end of the sentence — that `[Pan
# left]` convention belongs to the older Hailuo-2.3/02 API
# (platform.minimax.io/docs/api-reference/video-generation-i2v) and is a
# different, earlier syntax, not what H3's own skill guide asks for.
# H3's full named motion-type list: Zoom In/Out, Push In/Pull Out, Pan
# Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot,
# Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll
# Clockwise/Counterclockwise — only the subset our Movement Literal
# vocab actually has a token for is mapped below.
_H3_MOTION_VERB = {
    "push_in":    "pushes in",
    "pull_out":   "pulls out",
    "tracking":   "tracks alongside the subject",
    "pan_lr":     "pans right",
    "pan_rl":     "pans left",
    "tilt_up":    "tilts up",
    "tilt_down":  "tilts down",
    "crane_up":   "pedestals up",
    "crane_down": "pedestals down",
    "orbit":      "arcs around the subject",
    "static":     "holds a static shot",
    # Gear/style tokens H3 names a dedicated motion type for.
    "handheld":   "shakes slightly",
    "rapid":      "shakes strongly",
}

# Same motion types, gerund form — for the second-and-later movement when
# two are joined with "while" ("pans right while tracking...", not the
# ungrammatical "pans right while tracks...").
_H3_MOTION_GERUND = {
    "push_in":    "pushing in",
    "pull_out":   "pulling out",
    "tracking":   "tracking alongside the subject",
    "pan_lr":     "panning right",
    "pan_rl":     "panning left",
    "tilt_up":    "tilting up",
    "tilt_down":  "tilting down",
    "crane_up":   "pedestaling up",
    "crane_down": "pedestaling down",
    "orbit":      "arcing around the subject",
    "static":     "holding a static shot",
    "handheld":   "shaking slightly",
    "rapid":      "shaking strongly",
}


def compile_h3_motion(movements: list[str], easing: str = DEFAULT_EASING, *,
                      intensity: str | None = None) -> str:
    """Build a MiniMax H3 camera-motion sentence — natural English action
    embedded in the shot, per H3's own prompt-writing guide (see the
    module comment above `_H3_MOTION_VERB` for the source), not the
    bracket-tag `[Pan left]` syntax that belongs to an older, different
    Hailuo API.

    H3's motion framework has three independent dimensions: motion type
    (what moves — this function), amplitude ("with small/large
    amplitude", omitted = medium) and speed ("at slow/fast speed",
    omitted = normal). `intensity` — a shot's own
    `events.actions[].intensity` ("subtle" or "dramatic") when the caller
    has it — supplies amplitude+speed together; every other value omits
    them rather than guessing a magnitude the data does not actually
    support. Katz (2019 CN ed.) Ch.19.7 (p.297) backs the same pairing
    this project's compile_flux2._INTENSITY_CUES already makes for stills:
    a push-in at a character-realization beat reads as the classic
    "look-cut" setup, which is exactly a `dramatic`-intensity push.

    A shot can carry more than one movement (e.g. push_in + pan_lr);
    joined with "while" rather than stacked as separate clauses, since
    H3's guide asks for one natural sentence, not a list of tags.

    `static` alongside a real motion token is a contradiction, not a
    combination — some corpus shots carry both (trajectory.static
    = true AND movement_3d = ["crane"] on disk, stale data from a shot
    that was set to hold after crane was declared, or vice versa; which
    side is the true intent isn't decidable from the data alone). Rather
    than guess, `static` is dropped whenever anything else is present —
    "the camera holds a static shot while pedestaling up" is nonsense
    regardless of which token is the leftover one, but a real movement
    stated alone is never wrong to state.

    Empty / no-recognised-movement list → `""`, same contract as
    `compile_wan_motion` — the caller decides how to handle no motion.
    """
    tokens = [m for m in movements if m in _H3_MOTION_VERB]
    if len(tokens) > 1 and "static" in tokens:
        tokens = [m for m in tokens if m != "static"]
    if not tokens:
        return ""
    verbs = [_H3_MOTION_VERB[tokens[0]]] + [_H3_MOTION_GERUND[m] for m in tokens[1:]]
    action = " while ".join(verbs)
    qualifier = {
        "dramatic": " with large amplitude at fast speed",
        "subtle":   " with small amplitude at slow speed",
    }.get(intensity, "")
    sentence = f"The camera {action}{qualifier}."
    if qualifier:
        # Amplitude+speed already states the pacing; an easing clause on
        # top of that would talk about speed twice, sometimes
        # contradictorily ("at fast speed. ... starts slowly...").
        return sentence
    ease_clause = {
        "ease_in":     " The move starts slowly then accelerates.",
        "ease_out":    " The move starts quickly then settles.",
        "ease_in_out": " The move eases in and out smoothly.",
    }.get(easing, "")
    return sentence + ease_clause
