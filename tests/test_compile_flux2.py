"""compile_flux2 regression: Flux 2-tuned compiler MUST fix the
prompt-engineering issues catalogued in the project_compile_flux audit
(snake_case bleed, redundant interior, coverage_pct numeric noise, "in
the air" suffix on non-airborne elements, "at intimate scale").

Also pins the dispatcher (compile_flux_for_base): it always resolves to
compile_flux2 now that Flux 1 (compile_flux.py) has been retired."""
from __future__ import annotations

import pytest

from pace_core.compilers.compile_flux2 import compile_flux2, compile_flux_for_base
from pace_core.compilers.compile_common import CompileContext


# ── fixture: same scene shape as test_compile_scine.populated_scene
#           with a few fields tuned to surface Flux 2-specific fixes.

@pytest.fixture
def scene_with_flux2_traps():
    return {
        "scene_id": "scene_subway_test",
        "narrative_meta": {
            "location_raw": "interior of a modern urban subway car",
            "time_of_day": "day",
        },
        "shots": [{
            "shot_id": "shot_01",
            "setup": {
                "backdrop": {
                    "setting":      "int",
                    "location":     "interior of a modern urban subway car",
                    "time_of_day":  "day",
                    # Snake-case period fields — compile_flux leaks them raw,
                    # compile_flux2 must normalize.
                    "era":          "modern_day",
                    "region":       "modern_china",
                    "culture":      "modern_urban_chinese",
                },
                "environment": {
                    "style":      "photoreal",
                    "mood":       "tense",
                    # "intimate" must be DROPPED by compile_flux2 (vs.
                    # compile_flux which emits "at intimate scale").
                    "scale":      "intimate",
                    # Non-airborne element — compile_flux2 must NOT append
                    # "in the air"; compile_flux blindly does.
                    "elements":   ["cold LED ceiling lights"],
                },
                "subjects": [
                    {"character_id": "modern_subway_girl", "age_state": "20s"},
                ],
                "primary_focus": {
                    "type": "character", "ref": "modern_subway_girl",
                    # Numeric coverage — compile_flux emits "occupies roughly
                    # 35% of the frame", compile_flux2 must drop it.
                    "coverage_pct": 35,
                },
                "excluded": [],
            },
            "camera": {
                "extrinsics":      {"angle": "eye_level"},
                "creative_intent": {"shot_size": "medium_close"},
            },
            "events": {
                "actions": [{
                    "description_en": "the woman gazes out the window, vulnerable and distant",
                    "intensity":      "subtle",
                }],
            },
            "panels": [
                {"id": "scene_subway_test_shot_01_panel_0001", "panel_number": 1,
                 "scene_id": "scene_subway_test", "shot_id": "shot_01"},
            ],
        }],
    }


@pytest.fixture
def ctx():
    return CompileContext()


# ─── fix-by-fix assertions ──────────────────────────────────────────

def test_flux2_normalizes_snake_case_period(scene_with_flux2_traps, ctx):
    """era / region / culture must lose their underscores."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    # No raw snake_case identifiers.
    assert "modern_china" not in pos
    assert "modern_urban_chinese" not in pos
    assert "modern_day" not in pos
    # But the natural-language forms ARE there.
    assert "modern day" in pos.lower()
    assert "modern china" in pos.lower()
    assert "modern urban chinese" in pos.lower()


def test_flux2_panel_events_override_wins_over_shot_action(scene_with_flux2_traps, ctx):
    """A panel's own `events_override.actions` must reach the prompt instead
    of the shot's baseline action -- the same "panel wins over shot" rule
    primary_focus_of/excluded_of already followed. It didn't: every compiler
    called action0_of(shot) alone, so a panel like "the man disappears
    beneath the car" silently rendered its shot's default action instead,
    with nothing saying so. One corpus panel is
    the real panel this was caught on."""
    scene = scene_with_flux2_traps
    shot = scene["shots"][0]
    panel = dict(shot["panels"][0])
    panel["events_override"] = {"actions": [{
        "description_en": "he grips the pole as the train lurches, feet braced",
        "intensity": "medium",
    }]}
    pos, _ = compile_flux2(scene, shot, panel, ctx)
    assert "he grips the pole as the train lurches" in pos
    assert "gazes out the window, vulnerable and distant" not in pos


def test_flux2_panel_events_override_wins_over_shot_emotions(scene_with_flux2_traps, ctx):
    """A two-shot split into one close-up per subject keeps one emotion list
    on the shot, and it describes whoever acts. The reaction panel must be
    able to replace it -- with the cue that is its own subject's, or with
    none -- or the listener is drawn with the speaker's trembling hands."""
    scene = scene_with_flux2_traps
    shot = dict(scene["shots"][0])
    shot["events"] = {**shot["events"], "emotions": [
        {"implicit": "trembling hands, ragged breathing"}]}
    panel = dict(shot["panels"][0])
    pos, _ = compile_flux2(scene, shot, panel, ctx)
    assert "trembling hands, ragged breathing" in pos

    panel["events_override"] = {"emotions": [{"implicit": "brow furrowing in confusion"}]}
    pos, _ = compile_flux2(scene, shot, panel, ctx)
    assert "brow furrowing in confusion" in pos
    assert "trembling hands" not in pos

    panel["events_override"] = {"emotions": []}
    pos, _ = compile_flux2(scene, shot, panel, ctx)
    assert "trembling hands" not in pos


def test_flux2_drops_intimate_scale(scene_with_flux2_traps, ctx):
    """`scale: intimate` is redundant noise — compile_flux2 skips it."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    assert "intimate scale" not in pos.lower()
    assert "at intimate" not in pos.lower()


def test_flux2_drops_coverage_percentage(scene_with_flux2_traps, ctx):
    """Flux 2 doesn't comply with numeric percentages — drop the line."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    assert "35%" not in pos
    assert "occupies roughly" not in pos.lower()


def test_flux2_no_airborne_suffix_for_LED_lights(scene_with_flux2_traps, ctx):
    """`elements: ["cold LED ceiling lights"]` — NOT airborne. compile_flux
    blindly appends "in the air"; compile_flux2 uses "in frame"."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    assert "lights in the air" not in pos.lower()
    assert "cold LED ceiling lights in frame" in pos.lower() \
        or "cold led ceiling lights in frame" in pos.lower()


def test_flux2_does_add_airborne_for_smoke(ctx):
    """Smoke / dust / mist / etc. SHOULD still get the airborne suffix."""
    scene = {
        "scene_id": "s",
        "shots": [{
            "shot_id": "x",
            "setup": {
                "backdrop": {"setting": "ext", "location": "battlefield"},
                "environment": {"style": "photoreal", "elements": ["smoke", "ash"]},
                "subjects": [], "primary_focus": {}, "excluded": [],
            },
            "camera": {"extrinsics": {}, "creative_intent": {"shot_size": "wide"}},
            "events": {"actions": [{"description_en": "the battlefield falls quiet"}]},
            "panels": [{"id": "p", "panel_number": 1, "scene_id": "s", "shot_id": "x"}],
        }],
    }
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    assert "in the air" in pos.lower()


def test_flux2_no_duplicate_interior(scene_with_flux2_traps, ctx):
    """Location says `interior of a modern urban subway car` so
    setting_kind_of's `interior` MUST be suppressed (compile_flux emits
    both)."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    # "interior" should appear exactly once (inside the location text).
    assert pos.lower().count("interior") == 1


def test_flux2_style_anchor_drops_flux1_memes(scene_with_flux2_traps, ctx):
    """`photoreal` Flux 1 anchor includes "film grain", "museum-quality",
    "period-accurate" — Flux 2 anchor drops them all."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)
    assert "film grain" not in pos.lower()
    assert "museum-quality" not in pos.lower()
    assert "period-accurate" not in pos.lower()
    # The Flux 2 anchor uses "widescreen" instead of "2.35:1 anamorphic widescreen".
    assert "anamorphic widescreen" not in pos.lower()


# ─── dispatcher routing ─────────────────────────────────────────────
#
# Flux 1 (compile_flux.py) was retired project-wide — compile_flux_for_base
# now always resolves to compile_flux2 regardless of base_model. These pin
# that invariant across the base_model values the studio actually sends
# (a named flux2 file, its underscore variant, and the empty/None case the
# UI sends for "(default — based on fp8 toggle)"), rather than re-deriving
# from a Flux 1 baseline that no longer exists.

@pytest.mark.parametrize("base_model", [
    "flux2-dev.safetensors", "flux2_dev_fp8mixed.safetensors", None, "",
])
def test_dispatcher_always_routes_to_flux2(scene_with_flux2_traps, ctx, base_model):
    scene = scene_with_flux2_traps
    shot, panel = scene["shots"][0], scene["shots"][0]["panels"][0]
    pos_disp, _ = compile_flux_for_base(base_model, scene, shot, panel, ctx)
    pos_direct, _ = compile_flux2(scene, shot, panel, ctx)
    assert pos_disp == pos_direct



def _three_subject_scene():
    """Same shape as the fixture above, with three declared subjects."""
    return {
        "scene_id": "scene_car_test",
        "narrative_meta": {"location_raw": "interior of a family autonomous car",
                           "time_of_day": "day"},
        "shots": [{
            "shot_id": "shot_01",
            "setup": {
                "backdrop": {"setting": "int",
                             "location": "interior of a family autonomous car",
                             "time_of_day": "day"},
                "subjects": [
                    {"character_id": "fay", "age_state": "adult_50"},
                    {"character_id": "gus",  "age_state": "adult_50"},
                    {"character_id": "hal", "age_state": "adult_18"},
                ],
                "primary_focus": {"type": "character", "ref": "fay"},
                "excluded": [],
            },
            "camera": {"extrinsics": {"angle": "eye_level"},
                       "creative_intent": {"shot_size": "wide", "framing": "crowd"}},
            "events": {"actions": [{"description_en": "the two adults sing together",
                                    "intensity": "subtle"}]},
            "panels": [{"id": "scene_car_test_shot_01_panel_0001", "panel_number": 1,
                        "scene_id": "scene_car_test", "shot_id": "shot_01"}],
        }],
    }


def test_multi_person_cast_is_enumerated_with_the_count_first(ctx):
    """A comma-run of appearance descriptions leaves the encoder to guess
    where one person's description ends and the next begins, and a family-car
    prompt resolves that ambiguity by adding a body.

    Measured on one corpus shot (three declared subjects,
    seed 1693641188, everything else held identical): the run-on form
    rendered FOUR people with 8 reference plates, with 7, and with none at
    all — so the extra person came from the text, not the reference channel.
    Count-first plus enumeration, with the per-character hints left word for
    word the same, rendered three."""
    scene = _three_subject_scene()
    shot = scene["shots"][0]
    pos, _ = compile_flux2(scene, shot, shot["panels"][0], ctx)

    assert "exactly three people and no other people in frame:" in pos
    assert all(n in pos for n in ("(1)", "(2)", "(3)"))
    # Stated BEFORE the descriptions, not only in the framing clause far below.
    assert pos.index("exactly three people") < pos.index("(1)")


def test_single_subject_is_not_enumerated(scene_with_flux2_traps, ctx):
    """One person needs no list — enumerating a cast of one would put "(1)"
    in front of every close-up in the film."""
    scene = scene_with_flux2_traps
    pos, _ = compile_flux2(scene, scene["shots"][0], scene["shots"][0]["panels"][0], ctx)

    assert "(1)" not in pos
    assert "and no other people in frame" not in pos


def test_every_declared_prop_reaches_the_prompt(ctx):
    """The reference channel attaches a plate for EVERY declared prop, so a
    capped prompt hands the model a picture of an object it was never told
    about. One corpus scene declares five; the swivel seat sat past the
    old cap of four in all four panels."""
    scene = _three_subject_scene()
    shot = scene["shots"][0]
    shot["setup"]["props"] = [{"prop_id": f"p{i}", "name": f"prop {i}"} for i in range(6)]
    ctx.props_kb = {f"p{i}": {"id": f"p{i}", "name": f"prop {i}",
                              "anchor": f"a distinctive object number {i}"}
                    for i in range(6)}

    pos, _ = compile_flux2(scene, shot, shot["panels"][0], ctx)

    for i in range(6):
        assert f"distinctive object number {i}" in pos, f"prop {i} dropped"


def test_environment_insert_keeps_its_props_but_not_its_people(ctx):
    """"No people or animals in frame" is not the same claim as "no objects".
    scene_04's highway inserts declare the car console and the reference
    channel attaches its plate to every one of them."""
    scene = _three_subject_scene()
    shot = scene["shots"][0]
    shot["camera"]["creative_intent"]["framing"] = "empty"
    shot["setup"]["props"] = [{"prop_id": "console", "name": "console"}]
    ctx.props_kb = {"console": {"id": "console", "name": "console",
                                "anchor": "a full-width glass dashboard console"}}

    pos, _ = compile_flux2(scene, shot, shot["panels"][0], ctx)

    assert "environment-only insert shot with no people" in pos
    assert "full-width glass dashboard console" in pos


def test_environment_insert_does_not_get_facial_close_up_prose(ctx):
    """Shot sizes describe how much of a PERSON is in frame. An
    environment-only insert has none, so the person-shaped ladder contradicts
    the shot rather than merely failing to help it.

    Measured: scene_04's three highway inserts compiled to "an environment-only
    insert shot with no people or animals in frame" AND "an extreme close-up, a
    single facial feature filling 70-90% of the frame", and all three rendered
    an enormous eye over the traffic. The model resolved the contradiction the
    only way it could."""
    scene = _three_subject_scene()
    shot = scene["shots"][0]
    shot["setup"]["subjects"] = []
    shot["setup"]["primary_focus"] = {"type": "environment", "ref": "HIGHWAY",
                                      "coverage_pct": 70}
    shot["camera"]["creative_intent"] = {"shot_size": "extreme_close_up",
                                        "framing": "empty"}

    pos, _ = compile_flux2(scene, shot, shot["panels"][0], ctx)

    assert "no people or animals in frame" in pos
    assert "facial feature" not in pos
    assert "a single detail filling" in pos      # same scale, told about the subject matter


def test_a_person_close_up_keeps_the_facial_vocabulary(ctx):
    """The subject-free ladder must not leak into shots that do have a cast —
    a close-up of a face is still a close-up of a face."""
    scene = _three_subject_scene()
    shot = scene["shots"][0]
    shot["camera"]["creative_intent"] = {"shot_size": "extreme_close_up"}

    pos, _ = compile_flux2(scene, shot, shot["panels"][0], ctx)

    assert "facial feature" in pos
