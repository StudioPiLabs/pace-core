"""PAI 1.0 (scene, shot, panel) triple → Flux **2.dev** prompt + negative.

The sole Flux compiler — Flux 1/T5 support (compile_flux.py) was retired
project-wide once Flux 2 replaced it as the render backend. Mistral-
Small-3.1-24B is a true LLM and reacts differently to the prompt-
engineering tricks the T5-based Flux 1 needed; the choices below are kept
as the record of what that difference required, now that there is nothing
left to diff against:

  - **No snake_case bleed**: era / region / culture / element strings all
    get `_` → space normalization before joining (Flux 1's T5 encoder
    needed this only on season / weather / tod, leaving period_bits raw —
    Mistral reads `modern_china` as a malformed identifier).
  - **No coverage % line**: numerical "subject occupies 35% of the frame"
    distracts Mistral with no compliance payoff.
  - **No "in the air" suffix** unless the element is actually airborne
    (dust / smoke / mist / snow / leaves / embers / sparks / ash).
  - **No duplicated "interior"** when the location string already
    contains "interior of" / "exterior of" / "inside" / "outside".
  - **No "at <scale>"** when scale is "intimate" — redundant with shot
    size + framing prose; epic/vast still useful.
  - **De-duplicated gaze**: if `lora_trigger` matches the gaze subject,
    the gaze clause uses "the subject" instead of repeating the trigger.
    Emotion clauses that start with "looking…" / "gaze…" are dropped
    when a gaze clause is already present.
  - **Tighter style anchors**: Flux 2 over-cooks "film grain",
    "museum-quality", "period-accurate" — they're all dropped. The
    verbose "2.35:1 anamorphic widescreen" phrasing is replaced by a
    single "widescreen" hint (aspect ratio is set via EmptyLatentImage,
    not via text).
  - **Shorter negative**: Mistral handles negation better than T5;
    fewer redundant terms keep more of the 512-token context for
    positive content.

`compile_flux_for_base` is the dispatch helper every render endpoint calls
rather than this function directly.
"""

from __future__ import annotations

from pace_core.compilers.compile_common import (
    CompileContext, _character_back_hint, _character_hint, _focus_ordered_characters,
    _same_scope, _same_text, location_text as _location_text, seen_from_behind, sentence,
    subjects_left_to_right,
)
from pace_core.pai_compat import (
    dig,
    shot_size_of, angle_of, primary_focus_of, excluded_of,
    style_family_of, is_monochrome_style, action0_of,
    time_of_day_of, location_of, subjects_of,
    gaze_clauses_of, is_empty_framing,
    resolve_shot,
    lens_str, aperture_str, iso_str, shutter_str, color_temp_str,
    compile_hints_for_panel, lora_trigger_for,
    emotion_clauses_of, props_of, prop_phrase, prop_placement,
    prop_left_unstaged_away_from_cast, props_out_of_frame, props_not_in_frame,
    in_frame_extent_phrase, secondary_subjects_of,
    framing_of, aspect_ratio_of, setting_kind_of, weather_of, season_of,
    change_in_environment_of,
)


# ─── framing / angle prose tables (same shape as compile_flux,
# slightly tightened — Mistral doesn't need the verbose anchors) ───

# Shot sizes describe how much of a PERSON is in frame -- a facial feature, a
# head, the waist up. An environment-only insert has no person, so that prose
# does not merely fail to help, it contradicts the shot: one scene's three
# highway inserts compiled to "an environment-only insert shot with no people
# or animals in frame" AND "an extreme close-up, a single facial feature
# filling 70-90% of the frame", and all three rendered an enormous eye over
# the traffic. The model resolved the contradiction the only way it could.
#
# Same scale ladder, told about the subject matter instead of a body.
_SHOT_SIZE_FRAMING_NO_SUBJECT_EN = {
    # Keyed on the ShotSize Literal, tightest to widest. The Chinese 景别 term
    # rides along as a comment: it is the professional vocabulary these entries
    # were written against, and it used to be the key itself.
    "extreme_close_up": (                                       # 大特写
        "an extreme close-up, a single detail filling 70-90% of the frame, "
        "the surroundings reduced to soft blur"
    ),
    "close_up":         "a close-up, one element dominating the frame",              # 特写
    "medium_close_up":  "a tight framing on part of the scene",                      # 中近景
    "medium":           "a medium framing of the scene",                             # 中景
    "medium_full":      "a medium-full framing of the scene",                        # 中全景
    "full":             "a wide shot showing the whole space, visible to the edges",  # 全景
    "master":           "a master framing covering the whole space in one setup",    # 主镜头
    "wide":             "a distant wide shot, the space small within a larger environment",  # 远景
    "establishing":     "an extreme establishing wide shot, a vast dominating environment",  # 大远景 / 定场镜头
}


_SHOT_SIZE_FRAMING_EN = {
    "extreme_close_up": (                                       # 大特写
        "an extreme close-up, a single facial feature filling 70-90% of the frame, "
        "the background reduced to soft blur or a simple wall"
    ),
    "close_up": (                                               # 特写
        "a close-up of the head only, the face filling almost the entire frame, "
        "no shoulders visible"
    ),
    "medium_close_up": (                                        # 中近景
        "a medium close-up framing the subject from the chest up, head in the upper third"
    ),
    "medium": (                                                 # 中景
        "a medium shot framing the subject from the waist up, posture clear"
    ),
    "medium_full": (                                            # 中全景
        "a medium-full shot framing the subject from the knees up, stance visible"
    ),
    "full": (                                                   # 全景
        "a wide shot showing the full figure head to feet, environment visible to the edges"
    ),
    "master": (                                                 # 主镜头
        "a master shot covering the whole scene in one setup, every subject visible"
    ),
    "wide": (                                                   # 远景
        "a distant wide shot, the figure small within a substantial environment"
    ),
    "establishing": (                                           # 大远景 / 定场镜头
        "an extreme establishing wide shot, the figure tiny against a dominating environment"
    ),
}


_SHOT_SIZE_NEGATIVE = {
    "extreme_close_up": "shoulders, chest, torso, full body, wide framing",  # 大特写
    "close_up":         "shoulders, chest, torso, full body, wide framing",  # 特写
    "medium_close_up":  "full body, feet visible",                           # 中近景
    "medium":           "feet visible",                                      # 中景
    "medium_full":      "feet visible",                                      # 中全景
    "full":             "",                                                  # 全景
    "master":           "",                                                  # 主镜头
    "wide":             "",                                                  # 远景
    "establishing":     "",                                                  # 大远景 / 定场镜头
}


# Camera position relative to the subject. This is an authored, typed,
# controlled-vocabulary field (CameraExtrinsics.position) that no compiler read:
# movement_of() consumed only "behind" and "ots", and no compiler calls it, so
# "three_quarter", "front" and "profile" resolved and reached nothing. The
# symptom was a scene whose dashboard could not be placed, because a front-on
# camera looks the passengers in the face with the front of the cabin behind it,
# and the field that would have said otherwise was never emitted.
_COUNT_EN = ("zero", "one", "two", "three", "four", "five", "six",
             "seven", "eight", "nine", "ten")


def _count_word(n: int) -> str:
    """Small counts as words. Flux 2 reads "exactly three people" more
    reliably than "exactly 3 people" — digits in a prompt full of measurements
    ("20 cm by 10 cm", "35mm") read as another dimension."""
    return _COUNT_EN[n] if 0 <= n < len(_COUNT_EN) else str(n)


_CAMERA_POSITION_EN = {
    "front":         "seen from the front",
    "three_quarter": "seen from a three-quarter angle",
    "profile":       "seen in profile",
    "ots":           "an over-the-shoulder view past the foreground subject",
    "behind":        "seen from behind",
}

_ANGLE_FRAMING_EN = {
    "eye_level":    None,
    "high":         "shot from above looking down at a high angle",
    "low":          "shot from below looking up at a low angle",
    "aerial":       "aerial drone perspective high above, the ground spreading out below",
    "dutch":        "on a dutch angle, the horizon canted",
    "low_position": "low camera position close to the ground",
    "overhead":     None,
}


# Flux 2 over-cooks "film grain" + "museum-quality" + "period-accurate".
# The verbose 2.35:1 phrasing is replaced by a single "widescreen" cue —
# the actual aspect ratio comes from EmptyLatentImage.width/height.
_STYLE_ANCHOR = {
    "sketch_bw": (
        "rendered as a black-and-white graphite pencil drawing on white paper, "
        "charcoal shading where dramatic, traditional life-drawing aesthetic"
    ),
    "inkwash_bw": (
        # 白描 baimiao (fine-line contour) + 水墨 shuimo (ink wash)
        "rendered as a Chinese ink-on-rice-paper drawing in the baimiao fine-line "
        "and shuimo ink-wash tradition, "
        "calligraphic brush strokes, generous negative space"
    ),
    "chiaroscuro": (
        "rendered as a chiaroscuro oil painting with a single warm light source, "
        "dramatic shadow over most of the frame, painterly brushwork"
    ),
    "photoreal": (
        "cinematic 35mm photographic still, widescreen, natural lighting, shallow depth of field"
    ),
    "ink_color":      None,
    "concept_art":    None,
    "line_art_clean": (
        "professional production storyboard art in the Toon Boom Storyboard Pro house "
        "style: clean, confident tapered lines from a pressure-sensitive vector brush, "
        "pointed at both ends and thick through the middle, precise industrial linework "
        "with no loose scribbles or waste lines, every form clearly and economically "
        "defined; flat cel-shaded grayscale wash for depth, one or two layers of "
        "semi-transparent gray shadow with the foreground darker and the background "
        "lighter; strictly black, white and gray only, no color anywhere in the image, "
        "including no color tint on screens, lights, or interface graphics; a thin black "
        "safety-frame border around the panel"
    ),
}


_INTENSITY_CUES = {
    "subtle":   "quiet, restrained beat, micro-expression scale, stillness emphasized",
    "dramatic": "charged, intense beat, decisive gesture or expression",
}


# Mistral handles negation cleanly — keep the negative tight.
_BASE_NEGATIVE = (
    "deformed anatomy, distorted proportions, mutated body, mangled hands, "
    "fused fingers, malformed face"
)


_FEATURE_CLOSEUP_NEGATIVE = (
    "head and shoulders portrait, half-body portrait, body shot, "
    "full face visible, environment, room interior, wide framing, "
    "shoulders visible, chest, torso, arms, multi-subject composition"
)


_COLOR_NEGATIVE = (
    "color, painted, photorealistic, any chromatic hue"
)


_STYLE_NEGATIVE: dict[str, str] = {
    "sketch_bw":       "color, painted, photorealistic, finished illustration",
    "inkwash_bw":      "color, painted, photorealistic, finished illustration",
    "line_art_clean":  "color, colored, warm tint, blue tint, red tint, green tint, "
                        "any chromatic hue, photorealistic, painted, photographic "
                        "lighting, sketchy loose scribbles, gestural hand-drawn marker "
                        "texture, messy overlapping lines",
    "chiaroscuro":     "flat lighting, bright frame, color cartoon",
    "photoreal":       "anime, cartoon, manga, painterly, illustration, film grain texture",
}


# Words that legitimately float in air. Anything else gets the
# "<element> in frame" phrasing instead of "<element> in the air".
_AIRBORNE = {
    "dust", "smoke", "mist", "fog", "haze", "snow", "leaves", "petals",
    "embers", "sparks", "ash", "rain", "droplets", "pollen", "feathers",
    "cinders", "vapor", "steam",
}


# Framing vocabulary rendered as prose. The enum names are internal
# categories, not English: `crowd` means "three or more subjects" in the
# schema (types_v1), but an image model reads it as a throng of people and
# fills a family car with strangers. A controlled value has to be
# translated before it reaches a text encoder, never passed through.
_CI_FRAMING_EN = {
    "single":    "a single-subject composition",
    "two_shot":  "a two shot composition",
    "three_shot": "a three-subject group composition",
    "group":     "a small group composition",
    # NOT "three or more": the enum name is a category boundary in the schema,
    # but a generator reads the phrase as licence to add people, and did —
    # a three-person panel rendered five. The exact count is substituted from
    # the panel's own cast below, and this string is only the fallback for a
    # panel that declares the framing without a resolvable cast.
    "crowd":     "a group composition",
    "over_the_shoulder": "an over-the-shoulder composition",
    "insert":    "an insert composition",
    "establishing": "an establishing composition",
}


def _looks_airborne(s: str) -> bool:
    lo = (s or "").lower()
    return any(tok in lo for tok in _AIRBORNE)


def _loc_already_says_kind(loc: str) -> bool:
    """True when the location string already conveys interior/exterior,
    in which case adding `setting_kind_of` is duplication."""
    lo = (loc or "").lower()
    return any(t in lo for t in ("interior of", "exterior of", "inside ", "outside ",
                                 "inside the ", "outside the ", "indoor", "outdoor"))


def _dedup_emotion_against_gaze(emos: list[str], gaze_bits: list[str]) -> list[str]:
    """If a gaze clause is already present, drop emotion clauses that
    just re-state the gaze (anything starting with looking / gaze)."""
    if not gaze_bits:
        return emos
    out = []
    for e in emos:
        lo = (e or "").strip().lower()
        if lo.startswith(("looking", "gaze", "eyes look")):
            continue
        out.append(e)
    return out


def _strip_trigger_from_gaze(gaze_bits: list[str], trigger: str) -> list[str]:
    """Mistral reads `modern_subway_girl` as a malformed identifier.
    When the LoRA trigger is already in identity_bits, the gaze clause
    shouldn't repeat it — swap for `the subject` so the prose flows."""
    if not trigger or not gaze_bits:
        return gaze_bits
    return [g.replace(trigger, "the subject") for g in gaze_bits]



def compile_flux2(scene: dict, shot: dict, panel: dict, ctx: CompileContext) -> tuple[str, str]:
    """Compile a (scene, shot, panel) triple into (positive, negative)
    Flux 2 prompts. See module docstring for the diffs vs compile_flux.

    Hand-tuned overrides in `compile_hints[i].flux.prompt_override` still
    short-circuit when `positive` is non-empty (unchanged from Flux 1)."""
    hints = compile_hints_for_panel(scene, panel.get("id") or "")
    # Props the greybox staged where its camera does not see them (see
    # props_out_of_frame); the panel's own declaration outranks this record
    # once the shot is resolved below.
    staged_out = props_out_of_frame(getattr(ctx, "greyboxes_dir", None), panel.get("id") or "")
    flux_hints = (hints.get("flux") or {})
    override = flux_hints.get("prompt_override")
    if isinstance(override, dict) and override.get("positive"):
        return override["positive"], override.get("negative") or ""

    shot = resolve_shot(scene, shot, panel)
    # Neither the prop list nor an eyeline names a prop that is not in the
    # picture: declared out of frame, or staged out of it and not declared in.
    offscreen = props_not_in_frame(shot, staged_out)

    style       = style_family_of(shot)
    shot_size   = shot_size_of(shot)
    angle       = angle_of(shot)
    pf          = primary_focus_of(panel, shot)
    excluded    = excluded_of(panel, shot)
    action      = action0_of(shot, panel)
    chars, ages = subjects_of(shot)
    loc_ref, loc_raw = location_of(scene, shot)
    tod         = time_of_day_of(shot, scene)

    parts: list[str] = []
    empty_shot = is_empty_framing(shot)
    is_feature_extreme_closeup = (
        pf.get("type") == "feature"
        and shot_size == "大特写"
        and bool(pf.get("ref"))
    )

    trigger = ""  # populated below; needed for gaze de-dup

    # 1. Subject identity + (optional) gaze, opening sentence.
    if empty_shot:
        parts.append(
            "an environment-only insert shot with no people or animals in frame, "
            "atmospheric and transitional"
        )
    elif not is_feature_extreme_closeup:
        trigger = lora_trigger_for(_focus_ordered_characters(chars, pf), ages, ctx.characters_kb) or ""
        identity_bits: list[str] = []
        if trigger:
            identity_bits.append(trigger)
        of_char = pf.get("of_character")
        cast_bits: list[str] = []
        if of_char:
            cast_bits.append(_character_hint(of_char, ctx.characters_kb))
        elif chars:
            # Every declared subject, not the first two. A cap here silently
            # contradicts the panel: the framing field can say "three or more
            # subjects" while the prompt names two, and the reference channel
            # can carry a third identity the text never mentions. A generator
            # given that disagreement resolves it by inventing someone, which
            # is how a three-person family car rendered four adults and no
            # child. Panels declare a handful of subjects, so the prompt-length
            # concern the cap addressed does not arise in practice.
            # Named in the order they stand on screen, each anchored to where
            # the greybox stages it. The registry order this used to follow
            # carries no spatial information, so the text left it to the
            # sampler to decide which description belonged to which body: the
            # count and the positions came out right and the *people* moved
            # between seats from render to render. Both halves now read the
            # same declared screen_position.
            placed = subjects_left_to_right(shot)
            by_id = {p["character_id"]: p for p in placed}
            ordered = [p["character_id"] for p in placed]
            ordered += [c for c in chars if c not in by_id]
            # A subject the lens sees from behind is described without a face;
            # see _character_back_hint for what describing the face did.
            backs = seen_from_behind(shot, ordered, pf)
            for c in ordered:
                age = ages.get(c)
                ref = f"{c}@{age}" if age else c
                hint = (_character_back_hint(ref, ctx.characters_kb) if c in backs
                        else _character_hint(ref, ctx.characters_kb))
                phrase = (by_id.get(c) or {}).get("phrase")
                # Only worth saying when there is more than one body to tell
                # apart; on a single-subject panel the composition solver has
                # already put the one subject where it belongs, and naming a
                # side in the text would compete with it.
                if phrase and len(chars) > 1:
                    hint = f"{phrase}, {hint}"
                cast_bits.append(hint)
        # Enumerate a multi-person cast instead of running the descriptions
        # together with commas, and say the count before them rather than only
        # in the framing clause further down.
        #
        # Measured on one corpus shot (three declared
        # subjects, seed 1693641188, everything else held identical): the
        # comma-run form rendered FOUR people with 8 reference plates, with 7,
        # and with none at all — so the extra person was coming from the text,
        # not the reference channel. Switching only these two things, with the
        # per-character hints left word-for-word the same, rendered three.
        # Without delimiters the encoder has to guess where one person's
        # description ends and the next begins, and a family car full of
        # comma-separated features resolves that ambiguity by adding a body.
        if len(cast_bits) > 1:
            identity_bits.append(
                f"exactly {_count_word(len(cast_bits))} people and no other "
                "people in frame: "
                + "; ".join(f"({i}) {b}" for i, b in enumerate(cast_bits, 1)))
        else:
            identity_bits += cast_bits
        if identity_bits:
            opening = ". ".join(identity_bits)
            gaze_bits = _strip_trigger_from_gaze(gaze_clauses_of(shot, offscreen), trigger)
            if gaze_bits:
                opening = f"{opening}, with {'; '.join(gaze_bits)}"
            parts.append(opening)

    # 3. Action, placed directly after the subjects rather than after the
    # framing/prop/location/lighting block. Measured on one
    # corpus scene, three consecutive panels compiled to prompts that were 99.2%
    # word-for-word identical: 250 words of cast, framing, props, location and
    # style shared verbatim, with the beat as a single trailing clause. The
    # renders were correspondingly indistinguishable. The beat is the only
    # thing a panel does not share with its neighbours, so it leads.
    if not is_feature_extreme_closeup and not empty_shot:
        beat_en = action.get("description_en") or ""
        beat_zh = action.get("description_zh") or ""
        if beat_en:
            parts.append(beat_en)
        elif beat_zh:
            parts.append(beat_zh)
        cue = _INTENSITY_CUES.get(action.get("intensity"))
        if cue:
            parts.append(cue)

    # 2. Framing + angle.
    if shot_size and not is_feature_extreme_closeup:
        # An empty shot takes the subject-free ladder: the person-shaped
        # vocabulary would put a face in a frame the same prompt has just said
        # holds no people.
        _table = (_SHOT_SIZE_FRAMING_NO_SUBJECT_EN if empty_shot
                  else _SHOT_SIZE_FRAMING_EN)
        framing = _table.get(shot_size, "")
        angle_str = _ANGLE_FRAMING_EN.get(angle) if angle else None
        # A canted horizon on a high or low camera: the roll is its own
        # field, so the elevation keeps its words and the cant is added.
        roll = dig(shot, "camera", "extrinsics", "roll_deg")
        if angle != "dutch" and isinstance(roll, (int, float)) and roll:
            angle_str = (f"{angle_str}, " if angle_str else "") + _ANGLE_FRAMING_EN["dutch"]
        if framing and angle_str:
            parts.append(f"Captured as {framing}, {angle_str}")
        elif framing:
            parts.append(f"Captured as {framing}")
        elif angle_str:
            parts.append(angle_str.capitalize())

    cam_pos = (dig(shot, "camera", "extrinsics", "position") or "").strip()
    if cam_pos and not is_feature_extreme_closeup:
        phrase = _CAMERA_POSITION_EN.get(cam_pos)
        if phrase:
            parts.append(phrase)

    ci_framing = framing_of(shot)
    if ci_framing and not is_feature_extreme_closeup:
        phrase = _CI_FRAMING_EN.get(
            ci_framing, f"{ci_framing.replace('_', ' ')} composition")
        # State how many people are in frame. Leaving it to a category name
        # lets subject count drift render to render regardless of the cast.
        n_cast = len(chars or [])
        if ci_framing in ("crowd", "group") and n_cast:
            phrase = f"a group composition of exactly {n_cast} people"
        parts.append(phrase)

    # 2b. Aspect ratio — only when non-standard. Standard 2.35:1 comes
    # from EmptyLatentImage and the style anchor's "widescreen" hint.
    ar = aspect_ratio_of(shot)
    if ar and ar != "2.35:1":
        parts.append(f"{ar} aspect ratio")

    # 2c. Feature extreme close-up lockdown (rare path — kept verbose
    # because Flux 2 also needs the explicit framing here).
    if is_feature_extreme_closeup:
        feat = (pf.get("ref") or "").replace("_", " ")
        feat_owner = pf.get("of_character") or ""
        owner_clause = f" of {feat_owner}" if feat_owner else ""
        lockdown = (
            f"a cinematic extreme close-up of the {feat}{owner_clause} at 20x macro "
            f"magnification, the {feat} filling the entire frame edge to edge, every pore, "
            f"crack and texture razor-sharp at macro depth-of-field, the surrounding area "
            f"falling away into abstract out-of-focus blur"
        )
        parts.append(lockdown)

        beat_features = action.get("beat_features") or []
        if beat_features:
            texture_bits = [tag.replace("_", " ") for tag in beat_features]
            parts.append(f"with visible {', '.join(texture_bits)}")

        beat_en = action.get("description_en") or ""
        beat_zh = action.get("description_zh") or ""
        if beat_en:
            parts.append(beat_en)
        elif beat_zh:
            parts.append(beat_zh)

        cue = _INTENSITY_CUES.get(action.get("intensity"))
        if cue:
            parts.append(cue)

    # 3a. Emotional cues — but skip the ones that are just re-stating
    # gaze ("looking off-frame right" / "gaze out the window").
    if not empty_shot:
        emos = _dedup_emotion_against_gaze(emotion_clauses_of(shot, panel), gaze_clauses_of(shot, offscreen))
        if emos:
            parts.append("with " + ", ".join(emos))

    # 3a-bis. What the bodies are doing. `Subject.pose` is declared on every
    # subject of every shot and drives the 3D proxy staging; without also
    # reading it here, a cast declared "seated inside the car" could still
    # render standing on a motorway with nothing in the pipeline disagreeing.
    #
    # Stated once when the cast shares a posture, which is the usual case for
    # a group in a vehicle; repeating it per character reads as three
    # separate people to a text encoder and wastes prompt budget.
    if not is_feature_extreme_closeup and not empty_shot:
        poses = [str((s or {}).get("pose") or "").strip()
                 for s in (dig(shot, "setup", "subjects") or [])]
        poses = [p for p in poses if p]
        if poses:
            uniq = list(dict.fromkeys(poses))
            if len(uniq) == 1:
                parts.append(uniq[0] if len(poses) == 1 else f"all {uniq[0]}")
            else:
                parts.append(", ".join(uniq[:3]))

    # 3b. Props + secondary subjects.
    #
    # Props survive an environment-only insert; people do not. The empty-shot
    # branch says "no people or animals in frame", which is not the same claim
    # as "no objects" — one scene's highway inserts declare the car console,
    # and the reference channel duly attaches its plate to every one of them.
    # Skipping the prop text there meant the render was handed a picture of
    # the console and a prompt describing an empty highway.
    if not is_feature_extreme_closeup:
        # A large prop declared away from the cast and staged nowhere is not
        # named: with nothing in the control image to be drawn on, it is drawn
        # on the cast (see prop_left_unstaged_away_from_cast).
        props = [p for p in props_of(shot)
                 if not prop_left_unstaged_away_from_cast(p, getattr(ctx, "props_kb", None))
                 and (p.get("prop_id") or "") not in offscreen]
        if props:
            # Every declared prop, not the first four. The same reasoning as
            # the subject list above, with a sharper edge: the reference
            # channel attaches a plate for EVERY prop the shot declares, so a
            # capped prompt hands the model a picture of an object it was
            # never told about. One scene declares five and the swivel seat sat
            # past the cap in all four panels — its plate arrived, its anchor
            # did not, and the seats came out different every render.
            # Each with its declared place when that place is away from the
            # cast (see prop_placement).
            parts.append("with " + ", ".join(
                prop_phrase(p, getattr(ctx, "props_kb", None)) + prop_placement(p)
                + in_frame_extent_phrase(p)
                for p in props) + " in frame")
    if not is_feature_extreme_closeup and not empty_shot:
        secondary = secondary_subjects_of(shot)
        if secondary:
            parts.append("with " + ", ".join(_character_hint(s, ctx.characters_kb) for s in secondary[:3])
                         + " visible in the background")

    # 4. Setting / time / period / atmosphere — one sentence.
    if not is_feature_extreme_closeup:
        setting_bits: list[str] = []
        loc = _location_text(
            ctx.location_stubs.get(loc_ref) if loc_ref and loc_ref in ctx.location_stubs
            else loc_raw)
        bg_preview = ((shot.get("setup") or {}).get("environment") or {}).get("background")
        # A shot-specific background naming the SAME INT/EXT scope as the
        # location anchor is a refinement or correction of that setting (a
        # close-up's own authored exclusion of something the wide generic
        # anchor still mentions), not an additional fact, so the generic
        # anchor clause is dropped in favour of it below rather than
        # stacked alongside it -- stacking produced a self-contradictory
        # prompt (one clause placing a wrecked car in frame, the "backed
        # by" clause explicitly excluding it) for a close-up whose
        # shot-level override had been corrected without touching the
        # location anchor its accident_scene location shares with two
        # other scenes. A background naming a DIFFERENT scope (the ext.
        # view through this int. cabin's own windows) is genuinely
        # additive and both clauses are kept, unchanged from before.
        loc_superseded = bool(
            loc and bg_preview and not _same_text(bg_preview, loc)
            and _same_scope(loc, bg_preview))
        if loc and not loc_superseded:
            setting_bits.append(f"set in {loc}")
        # Only add interior/exterior if the location text doesn't already say so.
        kind = setting_kind_of(shot)
        if kind and not _loc_already_says_kind(loc):
            setting_bits.append(kind)
        season = season_of(shot)
        if season:
            setting_bits.append(f"in {season.replace('_', ' ')}")
        weather = weather_of(shot)
        if weather:
            setting_bits.append(f"during a {weather.replace('_', ' ')}")
        if tod:
            setting_bits.append(f"at {tod.replace('_', ' ')}")
        # FIX: normalize underscores in era / region / culture (compile_flux
        # leaves these raw, producing strings like "modern_china").
        era     = dig(shot, "setup", "backdrop", "era")
        region  = dig(shot, "setup", "backdrop", "region")
        culture = dig(shot, "setup", "backdrop", "culture")
        # Era is a period. Region and culture are a place and a milieu, and
        # putting all three behind "during the period of" said "during the
        # period of modern day, modern chinese city office building" -- a
        # building is not a period, and the phrase reads as a date the whole
        # way through. Each takes the preposition it is.
        if era:
            setting_bits.append("during the period of " + era.replace("_", " "))
        place_bits = [s.replace("_", " ") for s in (region, culture) if s]
        if place_bits:
            setting_bits.append("in " + ", ".join(place_bits))
        env = (shot.get("setup") or {}).get("environment") or {}
        # A panel that declares no atmosphere used to render with whichever
        # one the sampler chose -- 8 of this corpus's 36. The film's genre is
        # the floor under that: coarser than a declared mood and never a
        # substitute for one, but it is what the treatment says the picture
        # is, and it beats an unstated adjective.
        mood = env.get("mood") or (", ".join(ctx.film.genre) if ctx.film else "")
        if mood:
            setting_bits.append(f"with a {mood} atmosphere")
        # FIX: "intimate scale" is meaningless to the model — keep epic/vast/etc.
        scale = env.get("scale")
        if scale and scale not in ("intimate",):
            setting_bits.append(f"at {scale} scale")
        bg = env.get("background")
        # The location anchor and environment.background are frequently the
        # same authored paragraph (see _same_text's docstring) — emitting
        # both repeats it verbatim inside one prompt.
        if bg and not _same_text(bg, loc):
            setting_bits.append(f"backed by {bg}")
        elements = env.get("elements") or []
        if elements:
            # FIX: "in the air" only when the elements actually float.
            joined = ", ".join(elements)
            if any(_looks_airborne(e) for e in elements):
                setting_bits.append(joined + " in the air")
            else:
                setting_bits.append(joined + " in frame")
        if setting_bits:
            parts.append("The scene is " + ", ".join(setting_bits))

        change = change_in_environment_of(shot)
        if change:
            parts.append(change[:1].upper() + change[1:])

        # 5. Lighting + optics.
        natural   = dig(shot, "lighting", "natural") or []
        position  = dig(shot, "lighting", "position")
        condition = dig(shot, "lighting", "condition")
        light_bits: list[str] = []
        if condition == "blackout":
            # OVERRIDES the natural source rather than joining it. A shot that
            # has lost its power keeps whatever `natural` it was authored
            # with, and merging the two produced "all power in the car goes
            # out suddenly. Lit by sunlight from a key light" in one prompt.
            # Substitution, not denial: cfg=1.0 leaves the negative prompt
            # unevaluated, so "sunlight" has to simply not be written.
            light_bits.append("in near-total darkness, unlit, shapes barely "
                              "separable from the black")
        elif natural and position:
            light_phrase = f"{natural[0].replace('_', ' ')} from a {position.replace('_', ' ')}"
            if condition == "candlelight":
                light_phrase += " with dramatic chiaroscuro shadowing"
            light_bits.append(f"lit by {light_phrase}")
        ct = color_temp_str(shot)
        if ct:
            light_bits.append(ct.lower())
        optics = [s for s in (lens_str(shot), aperture_str(shot),
                              iso_str(shot), shutter_str(shot)) if s]
        if optics:
            light_bits.append("shot " + ", ".join(o.replace("shot ", "") for o in optics))
        if light_bits:
            parts.append("; ".join(light_bits).capitalize())

    # 6b. The film: genre and world. A constant the pipeline owns, appended here
    # rather than offered to the prompt writer, for the reason recorded on
    # the style anchor in prompt_projection: a model free to restate a
    # constant competes with it.
    if ctx.film and ctx.film.compiled:
        parts.append(sentence(", ".join(ctx.film.compiled)))

    # 6. Style anchor (Flux 2 variant — trimmed of the over-cooking memes).
    style_anchor = _STYLE_ANCHOR.get(style) or _STYLE_ANCHOR["sketch_bw"]
    parts.append(style_anchor)

    prompt = ". ".join(p.rstrip(".") for p in parts if p) + "."

    # ─── Negative ───
    neg_parts = [_BASE_NEGATIVE]
    if is_monochrome_style(shot):
        neg_parts.append(_COLOR_NEGATIVE)
    sn = _STYLE_NEGATIVE.get(style)
    if sn:
        neg_parts.append(sn)
    if shot_size:
        szn = _SHOT_SIZE_NEGATIVE.get(shot_size)
        if szn:
            neg_parts.append(szn)
    if flux_hints.get("avoid_full_body_negative"):
        neg_parts.append("full body, wide shot, head and shoulders")
    if is_feature_extreme_closeup:
        neg_parts.append(_FEATURE_CLOSEUP_NEGATIVE)
    if excluded:
        neg_parts.append(", ".join(e.replace("_", " ") for e in excluded))

    return prompt, ", ".join(neg_parts)


# ─── dispatcher used by render endpoints ─────────────────────────────

def compile_flux_for_base(base_model: str | None,
                          scene: dict, shot: dict, panel: dict,
                          ctx: CompileContext) -> tuple[str, str]:
    """Every render endpoint calls this rather than compile_flux2 directly,
    so a future second Flux-family compiler has one place to route from.
    `base_model` is currently unused: Flux 1/T5 support (the only other
    branch this ever had) was retired project-wide and compile_flux.py
    deleted, leaving compile_flux2 as the sole Flux compiler. Kept as a
    named parameter rather than dropped so call sites don't need editing
    if that changes again."""
    return compile_flux2(scene, shot, panel, ctx)
