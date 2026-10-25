"""camera_skills — a composable vocabulary of camera moves ("skills") that
compiles to a camera_planner track.

Authoring a shot as raw per-frame positions + eulers is hard. Instead a shot
is an ordered list of named operations, each with a duration:

    [{"skill":"orbit","degrees":360,"frames":14},
     {"skill":"track","side":"rear","distance":7,"frames":13},
     {"skill":"zoom","from_mm":35,"to_mm":75,"frames":13},
     {"skill":"pan","degrees":-22,"frames":14},
     {"skill":"push_in","to":2.5,"frames":14},
     {"skill":"tilt","degrees":28,"frames":13}]

`compile_shot(skills, subject_track=...)` walks a camera state through the
segments and emits the N-frame track that BlenderBox.render_depth / a video control pass
consume (the `--track-json` shape). The camera always looks AT the subject;
"relative" skills (orbit / track / push_in) read the subject's per-frame
position, so they automatically track a MOVING subject (e.g. a driving car).

State carried between segments is the camera's **offset relative to the
subject** + lens + accumulated in-place rotation (pan/tilt/roll) + an aim
offset (moved by `truck`) + the subject's screen position (set by `frame`).
That makes the moves compose: `track rear` then `push_in` continues from
where the track left off; a `pan` persists into later segments.

Handheld jitter is an overlay, not state: a segment's `shake` level (or a
`handheld` segment) adds seeded, frame-rate-independent noise that ramps
across segment boundaries and never accumulates into later segments.

This is the deterministic core. An optional LLM layer (`nl_to_skills`) maps
a natural-language shot description onto this list — sugar on top, not
required.
"""
from __future__ import annotations
from pace_core.llm_client import default_model, strip_fences

import math
from typing import Optional, Sequence

# ── camera convention (matches camera_planner) ──────────────────────────
# rotation_deg = [rx (pitch, about X), ry (roll, about Y), rz (yaw, about Z)]
# applied XYZ; the camera looks along  (-sin rx · sin rz, sin rx · cos rz, -cos rx).
# So aiming at a target with direction d (normalised):
#   rx = acos(-dz)              rz = atan2(-dx, dy)
# (Verified: reproduces the planner's static side-cam [82, 0, 180].)


def look_at(cam: Sequence[float], target: Sequence[float]) -> list[float]:
    """Euler [rx, 0, rz] that points the camera at `target`."""
    dx, dy, dz = (target[0]-cam[0], target[1]-cam[1], target[2]-cam[2])
    n = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    dx, dy, dz = dx/n, dy/n, dz/n
    rx = math.degrees(math.acos(max(-1.0, min(1.0, -dz))))
    s = math.sqrt(max(0.0, 1.0 - dz*dz))
    rz = math.degrees(math.atan2(-dx, dy)) if s > 1e-6 else 0.0
    return [rx, 0.0, rz]


def _ease(t: float, kind: str) -> float:
    t = max(0.0, min(1.0, t))
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if kind in ("ease_in_out", "smooth"):
        return 2*t*t if t < 0.5 else 1.0 - 2*(1.0-t)**2
    return t                                   # linear


# The full skill vocabulary compile_shot understands. Anything else is an
# unknown move (the from-nl route filters these out, the compiler holds).
KNOWN_SKILLS = frozenset({
    "hold", "orbit", "arc", "track", "dolly",
    "push_in", "pull_out", "crane", "zoom", "pan", "tilt", "roll",
    "dolly_zoom", "truck", "frame", "handheld",
})

# Handheld noise at shake level 1.0: positional metres and rotational degrees,
# each channel a sum of three sines (Hz) so it reads as body sway plus jitter.
_SHAKE_POSITION_M = 0.012
_SHAKE_ROTATION_DEG = (0.45, 0.25, 0.45)            # pitch, roll, yaw
_SHAKE_PARTIALS = ((0.6, 0.7), (0.3, 1.9), (0.1, 4.3))   # (weight, Hz)
_SHAKE_RAMP_S = 0.25                                 # blend between segment levels
_LENS_RANGE_MM = (6.0, 600.0)

_SIDE_DIR = {                                   # subject faces +X (wheel convention)
    "rear":  (-1.0, 0.0),
    "front": (1.0, 0.0),
    "left":  (0.0, 1.0),
    "right": (0.0, -1.0),
}

# Aim point above the subject origin (roughly the body centre).
_DEFAULT_AIM_Z = 0.6


def static_pose(azimuth_deg: float, elevation_deg: float, distance: float, *,
                aim_z: float = _DEFAULT_AIM_Z, lens_mm: float = 35.0) -> dict:
    """One static camera pose at (azimuth, elevation, distance) around a
    subject at the origin, looking at (0, 0, aim_z) — for a single reference
    shot (e.g. "shoot this KB asset's GLB from a 3/4-front angle"), not a
    multi-frame move. Reuses look_at() and the exact same spherical
    parameterization compile_shot's orbit/arc skills already use
    (`r*cos(az), r*sin(az)` relative to the subject) — exposed directly by
    absolute angle instead of "current offset + sweep by N degrees", since
    reverse-engineering a specific target angle out of an N-frame orbit's
    relative-degrees param is indirect for a single static shot.

    Returns the same {position, rotation_deg, lens_mm} shape as one
    compile_shot track entry (minus frame_index — a static shot has none).
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    r_horiz = distance * math.cos(el)
    z = distance * math.sin(el) + aim_z
    cam = [r_horiz * math.cos(az), r_horiz * math.sin(az), z]
    aim = [0.0, 0.0, aim_z]
    rot = look_at(cam, aim)
    return {"position": [round(c, 4) for c in cam],
            "rotation_deg": [round(c, 3) for c in rot],
            "lens_mm": round(lens_mm)}


def _seg_frames(skill: dict, fps: int) -> int:
    if "frames" in skill:
        return max(1, int(skill["frames"]))
    if "seconds" in skill:
        return max(1, round(float(skill["seconds"]) * fps))
    return max(1, fps)                           # default 1 second


def lens_value(lens_mm: float) -> float | int:
    """Track lens: whole millimetres stay ints (the historical shape), a
    continuously solved lens (dolly zoom) keeps two decimals."""
    lens = round(float(lens_mm), 2)
    return int(lens) if lens.is_integer() else lens


def _shake_phases(seed: int) -> list[list[float]]:
    """Per-channel sine phases; seeded so a compile is reproducible."""
    import random
    rng = random.Random(seed)
    return [[rng.uniform(0.0, 2 * math.pi) for _ in _SHAKE_PARTIALS] for _ in range(6)]


def _shake_offsets(time_s: float, phases: list[list[float]]) -> tuple[list[float], list[float]]:
    """Unit-level handheld noise at `time_s`: ([dx, dy, dz] m, [pitch, roll, yaw] deg)."""
    def channel(c: int) -> float:
        return sum(w * math.sin(2 * math.pi * hz * time_s + phases[c][j])
                   for j, (w, hz) in enumerate(_SHAKE_PARTIALS))
    return ([_SHAKE_POSITION_M * channel(c) for c in range(3)],
            [_SHAKE_ROTATION_DEG[c] * channel(3 + c) for c in range(3)])


def _smoothed_levels(levels: list[float], radius: int) -> list[float]:
    """Box-filter per-frame shake levels so a level change ramps, not steps."""
    n = len(levels)
    out = []
    for f in range(n):
        lo, hi = max(0, f - radius), min(n, f + radius + 1)
        out.append(sum(levels[lo:hi]) / (hi - lo))
    return out


def compile_shot(skills: list[dict], *,
                 subject_track: Optional[list[dict]] = None,
                 n_frames: Optional[int] = None,
                 fps: int = 16,
                 start_offset: Sequence[float] = (0.0, -6.0, 2.0),
                 lens_mm: float = 35.0,
                 aim_z: float = _DEFAULT_AIM_Z,
                 seed: int = 0) -> list[dict]:
    """Compile an ordered skill list into a camera_planner track.

    `subject_track` is the per-frame subject world position (the
    object-tracks "subject" entry). If omitted the subject sits at the
    origin. The total length is the sum of the skills' frames; if `n_frames`
    is given the segments are scaled proportionally to fit it. `seed` fixes
    the handheld noise phases.
    """
    if not skills:
        raise ValueError("no skills supplied")

    # Resolve per-segment frame counts (optionally scaled to n_frames).
    counts = [_seg_frames(s, fps) for s in skills]
    total = sum(counts)
    if n_frames and n_frames != total:
        scaled = [max(1, round(c * n_frames / total)) for c in counts]
        # fix rounding drift so they sum exactly to n_frames
        drift = n_frames - sum(scaled)
        scaled[-1] += drift
        counts = [max(1, c) for c in scaled]
        total = sum(counts)

    def subj(f: int) -> list[float]:
        if subject_track and 0 <= f < len(subject_track):
            return list(subject_track[f]["position"])
        if subject_track:
            return list(subject_track[-1]["position"])
        return [0.0, 0.0, 0.0]

    from pace_core.setup.composition_solver import solve_rotation_for_screen_position

    # Persistent camera state, relative to the subject.
    off = list(start_offset)            # camera position - subject position
    lens = float(lens_mm)
    pan = tilt = roll = 0.0             # accumulated in-place rotation (deg)
    aim_off = [0.0, 0.0, 0.0]           # aim point - (subject + aim_z), moved by truck
    frame = (0.5, 0.5)                  # subject's normalised screen position

    raw: list[tuple[list[float], list[float], float, float]] = []   # cam, rot, lens, shake
    f = 0
    for skill, K in zip(skills, counts):
        kind = skill.get("skill", "hold")
        ease = skill.get("ease", "ease_in_out")
        off0, lens0, pan0, tilt0, roll0 = list(off), lens, pan, tilt, roll
        aim0, frame0 = list(aim_off), frame
        shake = float(skill.get("shake", skill.get("amplitude", 1.0) if kind == "handheld" else 0.0))

        # End-of-segment targets (computed from the start state + params).
        r0 = math.hypot(off0[0], off0[1])
        az0 = math.atan2(off0[1], off0[0])
        aim_rel0 = [aim0[0], aim0[1], aim_z + aim0[2]]
        d0 = math.dist(off0, aim_rel0) or 1e-6
        fwd = [aim_rel0[0] - off0[0], aim_rel0[1] - off0[1]]
        fwd_n = math.hypot(*fwd)
        right = [fwd[1] / fwd_n, -fwd[0] / fwd_n] if fwd_n > 1e-9 else [1.0, 0.0]

        for i in range(K):
            t = _ease(i / max(1, K - 1), ease)
            o = list(off0)
            ln, pn, tl, rl = lens0, pan0, tilt0, roll0
            ao, fr = list(aim0), frame0

            if kind == "orbit" or kind == "arc":
                r = float(skill.get("radius", r0)) or r0 or 6.0
                hz = float(skill["height"]) if "height" in skill else off0[2]
                az = az0 + math.radians(float(skill.get("degrees", 360))) * t
                o = [r*math.cos(az), r*math.sin(az), off0[2] + (hz-off0[2])*t]
            elif kind == "track":
                sx, sy = _SIDE_DIR.get(skill.get("side", "rear"), (-1.0, 0.0))
                d = float(skill.get("distance", 7.0))
                hz = float(skill.get("height", off0[2]))
                tgt = [sx*d, sy*d, hz]
                o = [off0[k] + (tgt[k]-off0[k])*t for k in range(3)]
            elif kind == "dolly":
                ax = {"x": 0, "y": 1, "z": 2}.get(skill.get("axis", "x"), 0)
                dist = float(skill.get("distance", 0.0))
                o[ax] = off0[ax] + dist*t
            elif kind in ("push_in", "pull_out"):
                cur = r0 or 1e-6
                to = float(skill.get("to", cur*0.5 if kind == "push_in" else cur*1.6))
                frm = float(skill.get("from", cur))
                r = frm + (to-frm)*t
                scale = r / cur
                o = [off0[0]*scale, off0[1]*scale, off0[2]]
            elif kind == "crane":
                o[2] = off0[2] + float(skill.get("height_delta", 0.0))*t
            elif kind == "zoom":
                frm = float(skill.get("from_mm", lens0))
                to = float(skill.get("to_mm", lens0))
                ln = frm + (to-frm)*t
            elif kind == "pan":
                pn = pan0 + float(skill.get("degrees", 0.0))*t
            elif kind == "tilt":
                tl = tilt0 - float(skill.get("degrees", 0.0))*t   # +deg = look up
            elif kind == "roll":
                rl = roll0 + float(skill.get("degrees", 0.0))*t
            elif kind == "dolly_zoom":
                # Dolly radially like push_in while the lens holds the aim
                # point's frame height (distance / lens) constant: Vertigo.
                cur = r0 or 1e-6
                to = float(skill.get("to", cur*0.5))
                frm = float(skill.get("from", cur))
                scale = (frm + (to-frm)*t) / cur
                o = [off0[0]*scale, off0[1]*scale, off0[2]]
                ln = min(max(lens0 * math.dist(o, aim_rel0) / d0, _LENS_RANGE_MM[0]),
                         _LENS_RANGE_MM[1])
            elif kind == "truck":
                # Sideways along the camera's horizontal right (+ = right);
                # `follow` is the share of the move the aim point travels too
                # (1 = parallel truck, 0 = re-aim at the subject).
                d = float(skill.get("distance", 2.0)) * t
                follow = float(skill.get("follow", 1.0))
                o = [off0[0] + right[0]*d, off0[1] + right[1]*d, off0[2]]
                ao = [aim0[0] + right[0]*d*follow, aim0[1] + right[1]*d*follow, aim0[2]]
            elif kind == "frame":
                fx, fy = float(skill.get("x", frame0[0])), float(skill.get("y", frame0[1]))
                if not (0.0 < fx < 1.0 and 0.0 < fy < 1.0):
                    raise ValueError("frame x and y must lie strictly inside the frame (0..1)")
                fr = (frame0[0] + (fx-frame0[0])*t, frame0[1] + (fy-frame0[1])*t)
            # "hold" / "handheld" / unknown → carry state unchanged

            sp = subj(f)
            cam = [sp[k] + o[k] for k in range(3)]
            aim = [sp[0] + ao[0], sp[1] + ao[1], sp[2] + aim_z + ao[2]]
            rot = look_at(cam, aim)
            if abs(fr[0] - 0.5) > 1e-9 or abs(fr[1] - 0.5) > 1e-9:
                solved = solve_rotation_for_screen_position(cam, rot, ln, aim, fr[0], fr[1])
                if not solved["converged"]:
                    raise ValueError(f"frame ({fr[0]:.3f}, {fr[1]:.3f}) is unreachable at frame {f}")
                rot = solved["rotation_deg"]
            rot = [rot[0] + tl, rot[1] + rl, rot[2] + pn]
            # Only a solved lens keeps decimals; zoom keeps its historical whole mm.
            raw.append((cam, rot, ln if kind == "dolly_zoom" else round(ln), shake))
            f += 1

        # persist the end-of-segment state
        off, lens, pan, tilt, roll = o, ln, pn, tl, rl
        aim_off, frame = ao, fr

    levels = _smoothed_levels([r[3] for r in raw], max(1, round(_SHAKE_RAMP_S * fps)))
    phases = _shake_phases(seed)
    track: list[dict] = []
    for f, ((cam, rot, ln, _), level) in enumerate(zip(raw, levels)):
        if level > 1e-9:
            dp, dr = _shake_offsets(f / fps, phases)
            cam = [cam[k] + level*dp[k] for k in range(3)]
            rot = [rot[k] + level*dr[k] for k in range(3)]
        track.append({"frame_index": f,
                      "position": [round(c, 4) for c in cam],
                      "rotation_deg": [round(c, 3) for c in rot],
                      "lens_mm": lens_value(ln)})
    return track


# ── optional LLM convenience (deterministic core works without it) ──────

_SKILLS_SYSTEM_BASE = """You translate a natural-language camera description into a \
JSON list of camera "skills". Output ONLY JSON, no prose.

If the text names an EXPLICIT camera move ("pan left 20 degrees", "slow push \
in to 2m"), output a JSON array. Each item: {{"skill": <name>, "frames": <int>, \
...params}}. Skills + params:
  hold {{}}
  orbit {{degrees, radius?, height?}}
  arc {{degrees, radius?}}
  track {{side: rear|left|right|front, distance, height?}}
  dolly {{axis: x|y|z, distance}}
  push_in {{to}}      pull_out {{to}}
  pan {{degrees}}     tilt {{degrees}}   roll {{degrees}}
  crane {{height_delta}}
  zoom {{from_mm, to_mm}}
  dolly_zoom {{to}}   (Vertigo: dolly to `to` m while the lens holds the subject's size)
  truck {{distance}}  (sideways along the camera's right; negative = left)
  frame {{x, y}}      (move the subject to normalised screen x, y; 0.5, 0.5 = centre)
  handheld {{amplitude?}}  (hold with handheld jitter; any skill may add "shake": 0..3)
Distribute frames so they sum to the requested clip length (default 81). \
Subject faces +X; the camera always looks at it.

If the text instead describes a DRAMATIC FUNCTION with no explicit move named \
— what the beat is doing, not how the camera should move — and it matches one \
of the sourced functions below, output a single JSON object naming it: \
{{"dramatic_function": "<name>"}}. Do not invent your own skills for these; the \
caller resolves the sourced answer deterministically. Matching one of these \
beats guessing a plausible-looking move:
{kb_summary}

If the text is about style, setting, or the subject's action and names \
neither an explicit move nor a sourced dramatic function (e.g. "simple \
sketch, city street, running"), output an empty array: []."""


def _kb_summary_for_prompt() -> str:
    """The KB's own citable/measured entries, rendered into the prompt —
    generated FROM camera_movement_kb at call time rather than hand-typed
    here a second time, so this text cannot silently drift out of sync
    with the table that is actually authoritative. Convention-tier entries
    (reveal/tension/intimate_close) are deliberately excluded: they carry
    no source, so offering them as "the sourced answer" here would be
    exactly the unearned-confidence problem this file exists to fix.
    Policy entries (coverage_reuse, state_change_cut) are also excluded —
    they are not movements this function resolves."""
    from pace_core.camera.camera_movement_kb import KB, Evidence
    lines = []
    for fn, entry in KB.items():
        if entry.is_policy or entry.evidence is Evidence.CONVENTION:
            continue
        lines.append(f'  "{fn}": {entry.reason}')
    return "\n".join(lines) if lines else "  (none currently sourced)"


def _llm_classify(text: str, *, n_frames: int, model: str) -> list | dict:
    """The one LLM call both `nl_to_skills` and `classify_dramatic_function`
    need — kept in one place so the prompt and its parsing cannot drift
    between the two callers. Returns either a skills array (explicit move
    or no move) or `{"dramatic_function": "<name>"}` (sourced beat, not
    yet resolved to either vocabulary — that is each caller's own job)."""
    import json
    from pace_core.camera.lamp_dsl import default_llm_call
    call = default_llm_call(model)
    system = _SKILLS_SYSTEM_BASE.format(kb_summary=_kb_summary_for_prompt())
    user = f"Clip length: {n_frames} frames (16 fps). Shot: {text.strip()}"
    out, _cost = call([{"role": "system", "content": system},
                       {"role": "user", "content": user}])
    return json.loads(strip_fences(out))


def nl_to_skills(text: str, *, n_frames: int = 81,
                 model: str = default_model("classifier"),
                 dramatic_function: str | None = None) -> list[dict]:
    """Map a shot description to a skills list.

    `dramatic_function`, when given, is a deterministic bypass: no LLM call,
    the KB's own sourced skills for that function come back directly. This
    is what an automated per-shot deriver should pass once it has already
    classified a beat — the classification is one decision (which this
    function does not need to remake), and the skills for a classified
    function are not a judgment call at all once classified, they are a
    table lookup.

    Without it, `text` goes to the LLM. An explicit move ("pan left 20
    degrees") returns a skills array as before. Text naming a dramatic
    function GROUNDED IN THE KB (the prompt lists only citable/measured
    entries, never the unsourced convention-tier ones) returns that
    function's sourced skills via the same deterministic path, not the
    model's own improvisation — the model classifies, the KB answers.
    Anything else — no move, no sourced function — returns [], unchanged
    from this function's original behaviour.
    """
    if dramatic_function is not None:
        return _resolve_dramatic_function(dramatic_function)

    parsed = _llm_classify(text, n_frames=n_frames, model=model)
    if isinstance(parsed, dict) and "dramatic_function" in parsed:
        return _resolve_dramatic_function(parsed["dramatic_function"])
    return parsed


def classify_dramatic_function(text: str, *, n_frames: int = 81,
                               model: str = default_model("classifier")) -> str | None:
    """The classification half of `nl_to_skills`, without resolving into
    camera_skills' own vocabulary — for a caller (camera_movement_deriver)
    that writes a different field (movement_io's flat Movement tags) and
    needs the matched KB key itself, not this module's skills for it.

    Returns the dramatic_function name on a sourced match, `None` for an
    explicit move, no move, or unsourced text — the same three non-matching
    cases `nl_to_skills` folds into a skills array, collapsed here to "not a
    classified beat" since a caller of this function has nothing else to
    do with them."""
    parsed = _llm_classify(text, n_frames=n_frames, model=model)
    if isinstance(parsed, dict) and "dramatic_function" in parsed:
        return parsed["dramatic_function"]
    return None


def _resolve_dramatic_function(dramatic_function: str) -> list[dict]:
    """The deterministic half of the wiring: a KB key in, that entry's
    validated skills out. Raises rather than silently falling back to []
    on an unknown or policy-only key — a caller (human or the LLM
    classification path) that names one deserves to know it does not
    resolve to a movement, not to get a quietly empty track."""
    from pace_core.camera.camera_movement_kb import KB
    entry = KB.get(dramatic_function)
    if entry is None:
        raise ValueError(f"{dramatic_function!r} is not in camera_movement_kb")
    if entry.is_policy:
        raise ValueError(f"{dramatic_function!r} is a policy entry (no "
                         f"movement), not something nl_to_skills resolves")
    return [dict(s) for s in entry.skills]
