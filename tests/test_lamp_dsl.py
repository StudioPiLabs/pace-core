"""Regression tests for pace_core/camera/lamp_dsl.py — the LAMP camera-motion DSL
parser/validator, the messy-LLM-output extractor, the rule-based fallback, and
the LLM planner (with an injected fake call — no network).

Run: uv run python tests/test_lamp_dsl.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from pace_core.camera.lamp_dsl import (
    STATIC_DSL, Segment, LampDslError,
    parse_dsl, is_valid_dsl, normalize_dsl, extract_dsl_line,
    rule_based_dsl, build_messages, plan_dsl, plan_dsl_or_rule,
    MOVE_X_VOCAB, MOVE_Y_VOCAB, MOVE_Z_VOCAB,
)


def _expect(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


# Authentic LAMP outputs (blender_vis/vis/example_traj/*/cam_tag.txt)
EX_0000 = "no no no 30 0 0 no no no 30 0 0 no no no 30 0 0 no no no 30 0 0"
EX_0001 = "near_right no no 30 0 0 near_right no no 30 0 0 near_right no no 30 0 0 near_right no no 30 0 0"


def test_vocab(fails):
    _expect("no" in MOVE_X_VOCAB and "far_left" in MOVE_X_VOCAB and "near_right" in MOVE_X_VOCAB,
            "MOVE_X_VOCAB missing expected tokens", fails)
    _expect("far_up" in MOVE_Y_VOCAB and "near_down" in MOVE_Y_VOCAB, "MOVE_Y_VOCAB bad", fails)
    _expect("far_in" in MOVE_Z_VOCAB and "out" in MOVE_Z_VOCAB, "MOVE_Z_VOCAB bad", fails)
    _expect("up" not in MOVE_X_VOCAB, "MOVE_X_VOCAB must not accept y-axis tokens", fails)


def test_parse_valid(fails):
    segs = parse_dsl(EX_0001)
    _expect(len(segs) == 4, "EX_0001 should parse to 4 segments", fails)
    _expect(segs[0] == Segment("near_right", "no", "no", 30, 0, 0),
            f"EX_0001 seg0 wrong: {segs[0]}", fails)
    _expect(is_valid_dsl(EX_0000) and is_valid_dsl(STATIC_DSL), "valid lines rejected", fails)
    # negative angles + mixed case + extra whitespace
    line = "NO no FAR_in  0 -10 0  no no no 0 0 0 no no no 0 0 0 no no no 0 0 0"
    _expect(is_valid_dsl(line), "case/whitespace/negative should be valid", fails)
    _expect(normalize_dsl(line).split()[2] == "far_in", "normalize should lowercase moves", fails)


def test_parse_invalid(fails):
    bad = [
        ("too few", "no no no 0 0 0"),
        ("too many", EX_0000 + " no no no 0 0 0"),
        ("bad vocab", "diagonal no no 0 0 0 " + " ".join(["no no no 0 0 0"] * 3)),
        ("x got y-token", "up no no 0 0 0 " + " ".join(["no no no 0 0 0"] * 3)),
        ("non-int angle", "no no no x 0 0 " + " ".join(["no no no 0 0 0"] * 3)),
    ]
    for name, line in bad:
        _expect(not is_valid_dsl(line), f"should reject ({name}): {line!r}", fails)
        try:
            parse_dsl(line)
            _expect(False, f"parse_dsl should raise for ({name})", fails)
        except ValueError:
            pass


def test_extract(fails):
    # chatty prose around the line
    chatty = "Sure! Here is the motion program:\n\n" + EX_0000 + "\n\nHope that helps."
    _expect(extract_dsl_line(chatty) == EX_0000, "extract from prose failed", fails)
    # fenced code block
    fenced = "```\n" + EX_0001 + "\n```"
    _expect(extract_dsl_line(fenced) == EX_0001, "extract from fence failed", fails)
    # label prefix
    labelled = "DSL: " + EX_0000
    _expect(extract_dsl_line(labelled) == EX_0000, "extract with label prefix failed", fails)
    # LAMP raw free-form format
    ff = ("free-form t_x: near_right t_y: no t_z: no yaw: 30 tilt: 0 roll: 0 " * 4)
    _expect(extract_dsl_line(ff) == EX_0001, f"extract from free-form failed: {extract_dsl_line(ff)}", fails)
    # nothing valid
    _expect(extract_dsl_line("I cannot help with that.") is None, "should return None on garbage", fails)


def test_rule_based(fails):
    _expect(rule_based_dsl([]) == STATIC_DSL, "empty should be static", fails)
    _expect(is_valid_dsl(rule_based_dsl(["push_in"])), "push_in invalid", fails)
    push = rule_based_dsl(["push_in"]).split()
    _expect(push[2] == "far_in", f"push_in should set mz far_in, got {push[2]}", fails)
    pan = rule_based_dsl(["pan_right"]).split()
    _expect(pan[3] == "15", f"pan_right should set yaw 15, got {pan[3]}", fails)
    # composite: push_in + pan_right + crane_up
    comp = parse_dsl(rule_based_dsl(["push_in", "pan_right", "crane_up"]))[0]
    _expect(comp.mz == "far_in" and comp.my == "far_up" and comp.yaw == 15,
            f"composite mapping wrong: {comp}", fails)
    # stylistic-only → static
    _expect(rule_based_dsl(["handheld", "steadicam"]) == STATIC_DSL,
            "stylistic-only should be static", fails)
    # hyphen normalisation
    _expect(is_valid_dsl(rule_based_dsl(["pan-right"])), "hyphenated token not handled", fails)


def test_messages(fails):
    msgs = build_messages("Push in. (close_up, low_angle)")
    _expect(msgs[0]["role"] == "system", "first message must be system", fails)
    _expect(any(m["role"] == "assistant" for m in msgs), "few-shot assistant turns missing", fails)
    # final user message is the narrative verbatim — no duplicated framing hint.
    _expect(msgs[-1]["content"] == "Push in. (close_up, low_angle)",
            f"final user msg should be the narrative verbatim: {msgs[-1]['content']!r}", fails)


def test_plan_dsl_injected(fails):
    # fake LLM returns a chatty but valid response
    def good(messages):
        return ("Here you go:\n" + EX_0000, 0.001)
    r = plan_dsl("orbit right", call=good)
    _expect(r["dsl"] == EX_0000 and r["source"] == "llm" and r["attempts"] == 1,
            f"plan_dsl happy path wrong: {r}", fails)
    _expect(abs(r["cost"] - 0.001) < 1e-9, "cost not accumulated", fails)

    # first call invalid, second valid → retry path
    state = {"n": 0}
    def flaky(messages):
        state["n"] += 1
        return ((("garbage", 0.0)) if state["n"] == 1 else (EX_0001, 0.002))
    r2 = plan_dsl("track right", retries=1, call=flaky)
    _expect(r2["dsl"] == EX_0001 and r2["attempts"] == 2, f"retry path wrong: {r2}", fails)

    # always invalid → raises
    try:
        plan_dsl("x", retries=1, call=lambda m: ("nope", 0.0))
        _expect(False, "plan_dsl should raise on persistent invalid", fails)
    except LampDslError:
        pass


def test_plan_or_rule_fallback(fails):
    def boom(messages):
        raise RuntimeError("network down")
    r = plan_dsl_or_rule("push in", movements=["push_in"], call=boom)
    _expect(r["source"] == "rule" and r["dsl"].split()[2] == "far_in",
            f"fallback should use rule_based_dsl: {r}", fails)
    _expect("network down" in r.get("fallback_reason", ""), "fallback_reason missing", fails)


def main() -> int:
    fails: list[str] = []
    for fn in (test_vocab, test_parse_valid, test_parse_invalid, test_extract,
               test_rule_based, test_messages, test_plan_dsl_injected,
               test_plan_or_rule_fallback):
        fn(fails)
    if fails:
        print(f"FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("OK — all lamp_dsl tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
