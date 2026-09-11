"""LAMP camera-motion DSL — language → DSL planner (Phase 1).

Ports the *motion DSL* and the *language→DSL* step from LAMP
(Language-Assisted Motion Planning, CVPR 2026, arXiv 2512.03619) — but drives
the translation with a general LLM via our own token manager instead of LAMP's
finetuned Qwen2.5-VL-7B. See docs/LAMP_MOTION_PLANNER.md.

The DSL (ground truth from LAMP/src/scripts/generate_camera_trajectory.py and
the repo's cam_tag.txt examples): a camera shot = 4 segments × 6 tokens, one
space-separated line of 24 tokens. Each segment is

    mx my mz yaw tilt roll

  mx (truck) : no | [near_|far_]left  | [near_|far_]right
  my (boom)  : no | [near_|far_]up    | [near_|far_]down
  mz (dolly) : no | [near_|far_]in    | [near_|far_]out      (in = toward subject)
  yaw  (pan) : integer degrees, + = right, − = left
  tilt(pitch): integer degrees, + = up,    − = down
  roll(dutch): integer degrees, + = CW,    − = CCW

  magnitude prefix: near_ = small (⅓), bare = medium (⅔), far_ = large (full).
  The 4 segments play in sequence over ~21 frames; repeat a segment to make
  that motion dominate/last. `no no no 0 0 0` = hold.

This module is pure/offline except `plan_dsl`, which takes an injectable `call`
callable (defaulting to the llm_client) so the parse/validate/fallback logic is
unit-testable with no network. Downstream (Phase 3 `lamp_to_track`, the box
compiler) consumes the validated 24-token line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# ── DSL shape ──────────────────────────────────────────────────────────
N_SEGMENTS = 4
TOKENS_PER_SEGMENT = 6
N_TOKENS = N_SEGMENTS * TOKENS_PER_SEGMENT  # 24

_MAGS = ("", "near_", "far_")


def _axis_vocab(*directions: str) -> frozenset[str]:
    """{'no'} ∪ {<prefix><dir>} for every magnitude prefix × direction."""
    out = {"no"}
    for d in directions:
        for m in _MAGS:
            out.add(f"{m}{d}")
    return frozenset(out)


MOVE_X_VOCAB = _axis_vocab("left", "right")
MOVE_Y_VOCAB = _axis_vocab("up", "down")
MOVE_Z_VOCAB = _axis_vocab("in", "out")
_AXIS_VOCABS = (MOVE_X_VOCAB, MOVE_Y_VOCAB, MOVE_Z_VOCAB)

STATIC_DSL = " ".join(["no no no 0 0 0"] * N_SEGMENTS)


@dataclass(frozen=True)
class Segment:
    mx: str
    my: str
    mz: str
    yaw: int
    tilt: int
    roll: int

    def tokens(self) -> list[str]:
        return [self.mx, self.my, self.mz, str(self.yaw), str(self.tilt), str(self.roll)]


class LampDslError(ValueError):
    """Raised when an LLM response cannot be coerced into a valid DSL line."""


# ── Parse / validate ───────────────────────────────────────────────────
def parse_dsl(line: str) -> list[Segment]:
    """Parse a 24-token DSL line into 4 Segments. Raises ValueError on any
    structural problem — mirrors LAMP's own check
    (generate_camera_trajectory.py: `len(parts) != PARAMS_PER_SEGMENT*SEGMENT_COUNT`)
    but with explicit per-token vocabulary + integer validation."""
    parts = line.strip().split()
    if len(parts) != N_TOKENS:
        raise ValueError(f"expected {N_TOKENS} tokens, got {len(parts)}: {line!r}")
    segs: list[Segment] = []
    for s in range(N_SEGMENTS):
        base = s * TOKENS_PER_SEGMENT
        moves = [parts[base + i].lower() for i in range(3)]
        for i, (tok, vocab) in enumerate(zip(moves, _AXIS_VOCABS)):
            if tok not in vocab:
                axis = "xyz"[i]
                raise ValueError(f"segment {s} move_{axis}={tok!r} not in vocabulary")
        try:
            yaw, tilt, roll = (int(parts[base + 3]), int(parts[base + 4]), int(parts[base + 5]))
        except ValueError as e:
            raise ValueError(f"segment {s} yaw/tilt/roll must be integers: {e}") from e
        segs.append(Segment(moves[0], moves[1], moves[2], yaw, tilt, roll))
    return segs


def is_valid_dsl(line: str) -> bool:
    try:
        parse_dsl(line)
        return True
    except ValueError:
        return False


def normalize_dsl(line: str) -> str:
    """Canonical form: lower-cased moves, collapsed whitespace, ints re-rendered."""
    return " ".join(tok for seg in parse_dsl(line) for tok in seg.tokens())


# ── Coerce messy LLM output → a clean DSL line ─────────────────────────
_FREEFORM_RE = re.compile(
    r"t_x:\s*(\S+)\s+t_y:\s*(\S+)\s+t_z:\s*(\S+)\s+"
    r"yaw:\s*(-?\d+)\s+tilt:\s*(-?\d+)\s+roll:\s*(-?\d+)",
    re.IGNORECASE,
)


def _freeform_to_line(text: str) -> Optional[str]:
    """Convert LAMP's raw `free-form t_x: .. t_y: ..` format (4 reps) into the
    24-token line — mirrors LAMP/src/scripts/qwen_to_tag.py."""
    matches = _FREEFORM_RE.findall(text)
    if len(matches) < N_SEGMENTS:
        return None
    toks: list[str] = []
    for mx, my, mz, yaw, tilt, roll in matches[:N_SEGMENTS]:
        toks += [mx, my, mz, yaw, tilt, roll]
    return " ".join(toks)


def extract_dsl_line(text: str) -> Optional[str]:
    """Pull a valid DSL line out of a (possibly chatty) LLM response.

    Tries, in order: code-fence contents, the free-form format, each line, and
    the whole blob. Returns the normalized 24-token line, or None."""
    if not text:
        return None
    candidates: list[str] = []
    # strip ``` fences, keep their inner text as candidates too
    fenced = re.findall(r"```[a-zA-Z]*\n?(.*?)```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.append(text)
    for blob in candidates:
        ff = _freeform_to_line(blob)
        if ff and is_valid_dsl(ff):
            return normalize_dsl(ff)
        for raw in blob.splitlines():
            ln = raw.strip().strip("`").lstrip("-•").strip()
            ln = re.sub(r"^(dsl|output|tags?)\s*[:=]\s*", "", ln, flags=re.IGNORECASE)
            if is_valid_dsl(ln):
                return normalize_dsl(ln)
    # last resort: whole text as one line (handles no-newline responses)
    flat = " ".join(text.split())
    return normalize_dsl(flat) if is_valid_dsl(flat) else None


# ── Rule-based fallback (no LLM): PAI Movement Literals → DSL ───────────
# Maps our camera_planner Movement vocabulary onto DSL deltas for ONE segment;
# the segment is then repeated ×4 (steady move across the shot). Lets the LAMP
# path always yield a valid DSL even when the LLM is unavailable.
_RULE_MOVE = {  # token -> (axis, value)  axis in {x,y,z}
    "push_in": ("z", "far_in"), "push_in_slow": ("z", "near_in"), "pull_out": ("z", "far_out"),
    "dolly_in": ("z", "far_in"), "dolly_out": ("z", "far_out"),
    "dolly_left": ("x", "far_left"), "dolly_right": ("x", "far_right"), "tracking": ("x", "far_left"),
    "crane_up": ("y", "far_up"), "crane_down": ("y", "far_down"),
    # zoom has no lens in the DSL — approximate with a gentle dolly (documented).
    "zoom_in": ("z", "near_in"), "zoom_out": ("z", "near_out"),
}
_RULE_ANGLE = {  # token -> (field, degrees)
    "pan_left": ("yaw", -15), "pan_right": ("yaw", 15),
    "pan_lr": ("yaw", 15), "pan_rl": ("yaw", -15),
    "tilt_up": ("tilt", 10), "tilt_down": ("tilt", -10),
    "orbit": ("yaw", 30),
}


def rule_based_dsl(movements: list[str]) -> str:
    """Deterministic PAI Movement list → DSL line. Stylistic/Wan-only moves
    (handheld, steadicam, from_behind, …) carry no geometry → contribute
    nothing; an empty/unknown list yields a static hold."""
    mv = {"x": "no", "y": "no", "z": "no"}
    ang = {"yaw": 0, "tilt": 0, "roll": 0}
    for m in (movements or []):
        m = str(m).lower().replace("-", "_")
        if m in _RULE_MOVE:
            axis, val = _RULE_MOVE[m]
            mv[axis] = val
        if m in _RULE_ANGLE:
            field, deg = _RULE_ANGLE[m]
            ang[field] += deg
    seg = f"{mv['x']} {mv['y']} {mv['z']} {ang['yaw']} {ang['tilt']} {ang['roll']}"
    return normalize_dsl(" ".join([seg] * N_SEGMENTS))


# ── Language → DSL prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a cinematography motion planner. Convert a natural-language CAMERA-move \
description into a LAMP motion program.

OUTPUT: exactly ONE line of 24 space-separated tokens — four segments of six:
  mx my mz yaw tilt roll
No prose, no labels, no code fences. The four segments play in order over the \
shot; repeat a motion across segments to make it dominate/last; use \
`no no no 0 0 0` to hold.

AXES (camera-local):
  mx: no | [near_|far_]left | [near_|far_]right        (truck)
  my: no | [near_|far_]up   | [near_|far_]down         (boom)
  mz: no | [near_|far_]in    | [near_|far_]out          (dolly; in = toward subject)
  magnitude prefix: near_=small(1/3), bare=medium(2/3), far_=large(full)
  yaw +right/-left   tilt +up/-down   roll +CW/-CCW    (integer degrees, ~5-30)

MAP: push/dolly in->mz in; pull out->mz out; pan->yaw; tilt->tilt; \
crane/boom->my; truck/track->mx; orbit right->yaw + steady; dutch->roll; \
hold->all no/0.

If the input describes a SCENE/ACTION/MOOD rather than an explicit camera move, \
choose a cinematographically fitting move for that beat + framing (e.g. intimate \
or dramatic close-up -> slow push-in or hold; a reveal -> crane or pull-out; \
tension -> slow push-in).

Return ONLY the 24-token line."""

# Few-shot as message turns — strongest instruction-following. Two examples are
# authentic LAMP outputs (cam_tag.txt 0000 / 0001).
_FEWSHOT: list[tuple[str, str]] = [
    ("Slow orbit to the right around the subject.",
     "no no no 30 0 0 no no no 30 0 0 no no no 30 0 0 no no no 30 0 0"),
    ("Track right past the subject while turning to follow.",
     "near_right no no 30 0 0 near_right no no 30 0 0 near_right no no 30 0 0 near_right no no 30 0 0"),
    ("Push in slowly, then settle.",
     "no no near_in 0 0 0 no no near_in 0 0 0 no no near_in 0 0 0 no no no 0 0 0"),
    ("Crane up and tilt down to reveal the courtyard.",
     "no far_up no 0 -10 0 no far_up no 0 -10 0 no near_up no 0 -5 0 no no no 0 0 0"),
    ("Hold steady.", STATIC_DSL),
]


# Object/subject motion: SAME 24-token grammar, but the tokens describe how the
# SUBJECT moves in frame (not the camera). Compiled with base_distance 54 and
# emitted as a translating bbox path (see lamp_compile.compile_object_dsl).
SYSTEM_PROMPT_OBJECT = """\
You are a blocking/animation planner. Convert a natural-language description of \
how a SUBJECT (person/object) MOVES IN THE FRAME into a LAMP motion program.

OUTPUT: exactly ONE line of 24 space-separated tokens — four segments of six:
  mx my mz yaw tilt roll
No prose, no labels, no code fences. The four segments play in order over the \
shot; repeat a motion across segments to make it dominate/last; use \
`no no no 0 0 0` for a subject that stays put.

AXES (screen-relative, from the camera's view):
  mx: no | [near_|far_]left | [near_|far_]right   (moves screen-left / -right)
  my: no | [near_|far_]up    | [near_|far_]down    (rises / stands / falls / sits)
  mz: no | [near_|far_]in     | [near_|far_]out     (in = deeper into scene; out = toward camera)
  magnitude prefix: near_=small(1/3), bare=medium(2/3), far_=large(full)
  yaw/tilt/roll: the subject's own turn, integer degrees (usually 0)

MAP: walks/steps left|right->mx; rises/stands/lifts->my up; falls/sits/kneels/ \
collapses->my down; approaches camera->mz out; recedes/walks away->mz in; \
turns->yaw; still->all no/0.

Return ONLY the 24-token line."""

_FEWSHOT_OBJECT: list[tuple[str, str]] = [
    ("The man walks steadily to the right.",
     "right no no 0 0 0 right no no 0 0 0 right no no 0 0 0 right no no 0 0 0"),
    ("He rises slowly to his feet.",
     "no far_up no 0 0 0 no far_up no 0 0 0 no near_up no 0 0 0 no no no 0 0 0"),
    ("She steps toward the camera.",
     "no no far_out 0 0 0 no no far_out 0 0 0 no no near_out 0 0 0 no no no 0 0 0"),
    ("He collapses to the ground.",
     "no far_down no 0 0 0 no far_down no 0 0 0 no no no 0 0 0 no no no 0 0 0"),
    ("The subject stays still.", STATIC_DSL),
]

# kind → (system prompt, few-shot pairs).
_PROMPTS = {
    "camera": (SYSTEM_PROMPT, _FEWSHOT),
    "object": (SYSTEM_PROMPT_OBJECT, _FEWSHOT_OBJECT),
}


def build_messages(narrative: str, *, kind: str = "camera") -> list[dict]:
    """OpenAI/Anthropic-style message list for `llm_client.call_model`. The
    `narrative` is the complete prompt — framing/scene context is already baked
    in by `movement_io.shot_to_narrative`, so nothing extra is appended here
    (avoids duplicating shot_size/angle). `kind` ∈ {"camera","object"} selects
    the system prompt + few-shot."""
    system, fewshot = _PROMPTS.get(kind, _PROMPTS["camera"])
    msgs: list[dict] = [{"role": "system", "content": system}]
    for u, a in fewshot:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": narrative.strip()})
    return msgs


# ── LLM layer (thin; `call` is injectable for tests) ───────────────────
DEFAULT_MODEL = "claude-sonnet-5-rb"  # in-house by default — keeps film IP off
                                     # external services. Override per call.

# An LLM call: takes a messages list, returns (text, cost_usd).
LlmCall = Callable[[list[dict]], "tuple[str, float]"]


def default_llm_call(model: str = DEFAULT_MODEL) -> LlmCall:
    """Build a `call` bound to a model key from the registry (paths.MODELS_FILE).
    Imports are lazy so the pure layer never needs llm_client/paths/network."""
    import json
    from pace_core import paths
    from pace_core.llm_client import call_model

    registry = json.loads(paths.MODELS_FILE.read_text())
    if model not in registry:
        raise LampDslError(f"model {model!r} not in {paths.MODELS_FILE}; "
                           f"have: {sorted(k for k in registry if not k.startswith('_'))}")
    cfg = registry[model]

    def _call(messages: list[dict]) -> tuple[str, float]:
        return call_model(cfg, messages, api_key=None)  # llm_tokens resolves the key

    return _call


def plan_dsl(narrative: str, *, model: str = DEFAULT_MODEL, kind: str = "camera",
             retries: int = 1, call: Optional[LlmCall] = None) -> dict:
    """Language → DSL via an LLM. Returns
    {dsl, segments, source:'llm', model, cost, attempts}.
    Raises LampDslError if no attempt yields a valid 24-token line.

    `call` lets tests inject a fake LLM; production passes None → the registry
    model. NOTE: a real `call` performs a (possibly paid) API request — callers
    gate that on user approval."""
    call = call or default_llm_call(model)
    messages = build_messages(narrative, kind=kind)
    total_cost = 0.0
    last_text = ""
    for attempt in range(1, retries + 2):
        text, cost = call(messages)
        total_cost += (cost or 0.0)
        last_text = text
        line = extract_dsl_line(text)
        if line:
            return {"dsl": line, "segments": [s.tokens() for s in parse_dsl(line)],
                    "source": "llm", "model": model, "cost": round(total_cost, 6),
                    "attempts": attempt}
        # corrective nudge for the next attempt
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                f"That was not valid. Return ONLY one line of exactly {N_TOKENS} "
                f"space-separated tokens (4 segments of: mx my mz yaw tilt roll)."},
        ]
    raise LampDslError(f"no valid DSL after {retries + 1} attempts; "
                       f"last response: {last_text[:200]!r}")


def plan_dsl_or_rule(narrative: str, *, movements: Optional[list[str]] = None,
                     model: str = DEFAULT_MODEL, kind: str = "camera",
                     retries: int = 1, call: Optional[LlmCall] = None) -> dict:
    """Safe entry point: try the LLM, fall back to the rule-based mapping of the
    shot's existing PAI Movement list. Always returns a valid DSL dict.
    For kind="object" the fallback is a static hold (no object-movement vocab)."""
    try:
        return plan_dsl(narrative, model=model, kind=kind, retries=retries, call=call)
    except (LampDslError, Exception) as e:  # network/model errors included
        line = rule_based_dsl(movements or [])
        return {"dsl": line, "segments": [s.tokens() for s in parse_dsl(line)],
                "source": "rule", "model": None, "cost": 0.0, "attempts": 0,
                "fallback_reason": str(e)[:200]}


# ── CLI (dry-run by default — never an accidental paid call) ────────────
def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="LAMP language->DSL planner. Default = dry-run (prints the "
                    "prompt + rule-based DSL; no API call). Pass --execute to "
                    "actually call the LLM (incurs cost).")
    ap.add_argument("narrative", help="Natural-language camera-move description")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model key (kb models.json)")
    ap.add_argument("--shot-size", default=None)
    ap.add_argument("--angle", default=None)
    ap.add_argument("--movements", default="", help="comma-sep PAI Movement list (rule fallback)")
    ap.add_argument("--execute", action="store_true", help="actually call the LLM")
    args = ap.parse_args()

    movements = [m.strip() for m in args.movements.split(",") if m.strip()]
    if not args.execute:
        msgs = build_messages(args.narrative, shot_size=args.shot_size, angle=args.angle)
        print("DRY RUN — no API call. Re-run with --execute to call", args.model)
        print("\n--- SYSTEM ---\n" + msgs[0]["content"])
        print("\n--- USER ---\n" + msgs[-1]["content"])
        print("\n--- rule-based DSL (LLM-free fallback) ---")
        print(rule_based_dsl(movements))
        return

    result = plan_dsl_or_rule(args.narrative, movements=movements, model=args.model,
                              shot_size=args.shot_size, angle=args.angle)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
