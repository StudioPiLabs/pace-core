"""Pillar accessors for PAI 1.0 (SCINE 4-pillar) scene docs.

Read-only helpers that lift the most-used per-shot / per-panel fields
out of the pillar tree. Every function takes the raw on-disk dict (or
sub-dict) and returns the value the compilers want — no synthesised
flat-shape dataclasses, no compat layer.

Two kinds of helpers:

  - `dig(...)` — defensive walk through nested dicts.
  - `*_of(shot)` / `*_of(panel)` — named extractors that encapsulate one
    pillar lookup. Some apply small vocab translations the Flux prompt
    assembler still expects (SCINE English `shot_size` → Chinese label
    used in the Flux table; SCINE Movement2D/3D → planner Movement list).

Writes go through studio_server's pillar-native PATCH endpoints, not
this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


# ─────────────────  vocab translation tables  ─────────────────

# SCINE Movement2D / Movement3D + gear + extrinsics.position  →  flat
# Movement Literal vocab the planner / Wan-VACE compiler grew up with.
_V10_MOVEMENT_REVERSE: dict[str, str] = {
    "push_in":   "push_in",
    "pull_out":  "pull_out",
    "tracking":  "tracking",
    "trucking":  "tracking",
    "arc":       "orbit",
    "crane":     "crane_up",
    "pan_left":  "pan_rl",
    "pan_right": "pan_lr",
    "tilt_up":   "tilt_up",
    "tilt_down": "tilt_down",
    "zoom_in":   "push_in",
    "zoom_out":  "pull_out",
}


# ─────────────────  generic helpers  ─────────────────


def dig(d: Any, *keys: str) -> Any:
    """Walk a nested dict defensively. `dig(x, "a", "b", "c")` returns
    `x["a"]["b"]["c"]`, with `None` for any missing or non-dict link."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_scene(path: str | Path) -> dict:
    """Read a scene_*.json off disk. Returns the raw pai-1.0 dict."""
    return json.loads(Path(path).read_text())


def iter_panels(scene: dict) -> Iterator[tuple[dict, dict, dict]]:
    """Yield (scene, shot, panel) triples for every panel in the scene."""
    for shot in scene.get("shots") or []:
        for panel in (shot or {}).get("panels") or []:
            yield scene, shot, panel


def find_panel(scene: dict, panel_id: str) -> tuple[dict | None, dict | None]:
    """Locate (shot, panel) by panel_id within the scene. (None, None) if absent."""
    for _scene, shot, panel in iter_panels(scene):
        if panel.get("id") == panel_id:
            return shot, panel
    return None, None


def find_shot(scene: dict, shot_id: str) -> dict | None:
    """Locate the shot dict by shot_id, or None."""
    for shot in scene.get("shots") or []:
        if (shot or {}).get("shot_id") == shot_id:
            return shot
    return None


def dedupe_shots(shots: list[dict]) -> list[dict]:
    """Collapse duplicate shot_ids, keeping the richest entry per id
    (most panels; ties → first-seen). Preserves first-occurrence order.

    Defends the UI/compilers against legacy data where a scene's shots[]
    array carried the same shot_id twice (e.g. an empty 0-panel skeleton
    appended after the real shot during a migration). Without this, the
    storyboard sidebar renders the shot header twice and the panel
    lookup — keyed by shot_id — shows the same panels under both."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for s in shots or []:
        sid = (s or {}).get("shot_id")
        if sid is None:
            continue
        if sid not in best:
            best[sid] = s
            order.append(sid)
        else:
            # Keep whichever has more panels.
            cur_n = len((best[sid].get("panels") or []))
            new_n = len((s.get("panels") or []))
            if new_n > cur_n:
                best[sid] = s
    return [best[sid] for sid in order]


# ─────────────────  pillar extractors  ─────────────────


def shot_size_of(shot: dict) -> str | None:
    """Camera.creative_intent.shot_size as the schema declares it, or None.

    This used to translate the field into a Chinese label, because the prompt
    tables were written against the pre-SCINE vocabulary and were keyed that
    way. The tables are keyed on the ShotSize Literal now, so there is nothing
    left to translate — and the translation was lossy in a way nothing
    reported: `medium_full` and `medium` both arrived as one key, as did
    `master` and `full`, so two authored distinctions were silently spent.
    """
    return dig(shot, "camera", "creative_intent", "shot_size") or None


def angle_of(shot: dict) -> str | None:
    return dig(shot, "camera", "extrinsics", "angle")


def movement_of(shot: dict) -> list[str]:
    """Flat Movement Literal list, derived from
    Camera.trajectory.movement_2d + movement_3d + gear + extrinsics.position.
    Empty list = no movement (caller may mark the shot static)."""
    traj = dig(shot, "camera", "trajectory") or {}
    pos  = dig(shot, "camera", "extrinsics", "position")
    out: list[str] = []
    if traj.get("static"):
        out.append("static")
    for m in (traj.get("movement_2d") or []) + (traj.get("movement_3d") or []):
        mapped = _V10_MOVEMENT_REVERSE.get(m)
        if mapped and mapped not in out:
            out.append(mapped)
    gear = traj.get("gear")
    if gear in ("handheld", "steadicam") and gear not in out:
        out.append(gear)
    if pos == "behind" and "from_behind" not in out:
        out.append("from_behind")
    elif pos == "ots" and "reverse_shot" not in out:
        out.append("reverse_shot")
    return out


def easing_of(shot: dict) -> str:
    return dig(shot, "camera", "trajectory", "easing") or "linear"


def subjects_of(shot: dict) -> tuple[list[str], dict[str, str]]:
    """`(character_ids, age_states_by_id)` from Setup.subjects[]."""
    chars: list[str] = []
    ages:  dict[str, str] = {}
    for sub in (dig(shot, "setup", "subjects") or []):
        cid = (sub or {}).get("character_id")
        if not cid:
            continue
        chars.append(cid)
        age = sub.get("age_state")
        if age:
            ages[cid] = age
    return chars, ages


# ── Character reference convention ──────────────────────────────────────
# All character references across the schema use ONE canonical string form:
#
#     "<character_id>"                — character w/o age qualifier
#     "<character_id>@<age_state>"    — character at a specific life stage
#
# Applies to:
#   - setup.primary_focus.of_character
#   - setup.secondary_subjects[]
#   - events.actions[].interactive / standalone (when referring to a person)
#   - subjects[].gaze.target_ref (when gaze.target_type == "character")
#
# Uniqueness rule: same character_id + different age_state ⇒ different
# identity (different LoRA / PuLID reference, different visual continuity).
# compile_flux2's _character_hint() resolves this ref form against the KB's
# characters_kb[char].generic_anchors[age_state].

def ref_to_id_age(ref: str | None) -> tuple[str, str | None]:
    """Parse a canonical character reference.

    >>> ref_to_id_age("nina@adult_50")
    ('nina', 'adult_50')
    >>> ref_to_id_age("modern_office_man")
    ('modern_office_man', None)
    >>> ref_to_id_age("") == ref_to_id_age(None) == ("", None)
    True
    """
    if not ref:
        return "", None
    if "@" in ref:
        cid, age = ref.split("@", 1)
        return cid, age
    return ref, None


def id_age_to_ref(character_id: str, age_state: str | None) -> str:
    """Inverse of ref_to_id_age — assemble the canonical string form.

    >>> id_age_to_ref("nina", "adult_50")
    'nina@adult_50'
    >>> id_age_to_ref("modern_office_man", None)
    'modern_office_man'
    """
    return f"{character_id}@{age_state}" if age_state else character_id


# ── Scene-defaults deep-merge ───────────────────────────────────────────
# PAI 1.1: scene_defaults at scene level + sparse override at shot level.
# `resolve_shot(scene, shot)` returns a virtual shot dict that combines
# the two. Used by compile_flux2 and any reader that wants the "as-if"
# full shot without authors having to duplicate scene-wide constants.

def _deep_merge(base: dict | None, override: dict | None) -> dict:
    """Recursively merge override onto base. Override wins. None values
    in override DON'T overwrite filled values in base — they're treated as
    "not specified at this level" so the parent default leaks through.
    Lists are atomic (override replaces base entirely if non-empty)."""
    if base is None:
        return dict(override) if override else {}
    if override is None:
        return dict(base) if base else {}
    out: dict = dict(base)
    for k, v in override.items():
        if v is None:
            continue                               # skip — keep base's value
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif isinstance(v, list) and not v:
            continue                               # empty list = "no override"
        else:
            out[k] = v
    return out


# ── Camera dual-track: precise EXIF-style values → string clauses ──────
# PAI 1.0 / 1.1 keeps both categorical buckets (lens_size, aperture, …) and
# exact numbers (focal_length_mm, aperture_f, …). compile_flux2 historically
# only read the categorical layer. These helpers let it prefer exact values
# when present and fall back to the bucket — "shot at f/1.4 with 85mm lens"
# is much more useful to Flux than "wide aperture, long_lens lens".

_LENS_CAT_TO_STR = {
    "fisheye":    "fisheye lens",
    "wide":       "wide-angle lens",
    "medium":     "medium-length lens",
    "standard":   "50mm normal lens",
    "long_lens":  "85-135mm long lens (portrait compression)",
    "telephoto":  "200mm+ telephoto lens (deep compression)",
}
_APERTURE_CAT_TO_STR = {
    "wide":   "wide aperture, shallow depth of field, bokeh background",
    "medium": "moderate aperture, balanced depth of field",
    "narrow": "narrow aperture, deep focus, everything sharp",
}
_ISO_CAT_TO_STR = {
    "low":    "low ISO, clean image, fine grain",
    "medium": "moderate ISO",
    "high":   "high ISO, visible film grain / noise",
}
_SHUTTER_CAT_TO_STR = {
    "slow":   "slow shutter, motion blur on moving subjects",
    "medium": "standard shutter",
    "fast":   "fast shutter, frozen action, no motion blur",
}
_COLORTEMP_CAT_TO_STR = {
    "warm": "warm color temperature (≈3200K tungsten / candlelight feel)",
    "cool": "cool color temperature (≈5600K daylight balanced)",
    "cold": "cold color temperature (≈7000K+ blue hour / overcast)",
}


def lens_str(shot: dict) -> str | None:
    """Render a lens-description clause. Prefer focal_length_mm if set;
    fall back to lens_size; else None."""
    mm  = dig(shot, "camera", "intrinsics", "focal_length_mm")
    if mm:
        return f"shot on a {mm:g}mm lens"
    cat = dig(shot, "camera", "intrinsics", "lens_size")
    return _LENS_CAT_TO_STR.get(cat) if cat else None


def aperture_str(shot: dict) -> str | None:
    """Prefer aperture_f / t_stop if set; fall back to bucket."""
    f = dig(shot, "camera", "intrinsics", "aperture_f")
    if f:
        return f"shot at f/{f:g}"
    t = dig(shot, "camera", "intrinsics", "t_stop")
    if t:
        return f"shot at T/{t:g} (cinema lens)"
    cat = dig(shot, "camera", "intrinsics", "aperture")
    return _APERTURE_CAT_TO_STR.get(cat) if cat else None


def iso_str(shot: dict) -> str | None:
    """Prefer iso_value if set; fall back to bucket."""
    v = dig(shot, "camera", "intrinsics", "iso_value")
    if v:
        return f"ISO {v}"
    cat = dig(shot, "camera", "intrinsics", "iso")
    return _ISO_CAT_TO_STR.get(cat) if cat else None


def shutter_str(shot: dict) -> str | None:
    """Prefer shutter_angle_deg if set; fall back to bucket."""
    deg = dig(shot, "camera", "intrinsics", "shutter_angle_deg")
    if deg:
        # 180° = standard cinematic motion blur; <180 sharp; >180 smeared
        qualifier = ""
        if deg < 90:    qualifier = " — very sharp, almost no motion blur"
        elif deg > 270: qualifier = " — heavy motion blur"
        return f"{deg:g}° shutter angle{qualifier}"
    cat = dig(shot, "camera", "intrinsics", "shutter_speed")
    return _SHUTTER_CAT_TO_STR.get(cat) if cat else None


def color_temp_str(shot: dict) -> str | None:
    """Prefer exact Kelvin from lighting.color_temp_k; fall back to band."""
    k = dig(shot, "lighting", "color_temp_k")
    if k:
        # Add a feel descriptor based on Kelvin range
        if k <= 3200:    feel = "warm tungsten"
        elif k <= 4500:  feel = "neutral incandescent"
        elif k <= 5600:  feel = "daylight balanced"
        elif k <= 6500:  feel = "cool daylight"
        else:            feel = "cold blue hour"
        return f"{k}K color temperature ({feel})"
    cat = dig(shot, "lighting", "color_temperature")
    return _COLORTEMP_CAT_TO_STR.get(cat) if cat else None


def resolve_shot(scene: dict, shot: dict, panel: dict | None = None) -> dict:
    """Return the shot deep-merged with scene_defaults, then with `panel`'s
    own overrides when one is given.

    The inheritance ladder the schema documents is scene defaults → shot →
    panel, and only the first two rungs were built. `Panel.camera_override`
    ("sparse Camera-shaped diff vs the parent shot's camera") was declared,
    initialised to None by the splitter, and read by nothing: a panel could
    not change the camera at all, however the field was filled in.
    `setup_override` was half-wired — `excluded_of` reads one leaf of it and
    the rest was ignored.

    That gap is why a shot's two panels render the same frame. Densification
    adds a moving shot's end framing as a second panel, and the only place
    the end of a move could be written is a camera override, so the second
    panel had nowhere to differ and the pair came out identical.

    Passing `panel` is optional so every existing caller keeps working; a
    caller that has a panel in hand should pass it.
    """
    defaults = scene.get("shot_defaults") or {}
    overrides = {}
    if panel:
        for pillar, key in (("camera", "camera_override"),
                            ("setup", "setup_override"),
                            ("lighting", "lighting_override")):
            diff = panel.get(key)
            if isinstance(diff, dict) and diff:
                overrides[pillar] = diff
    if not defaults and not overrides:
        return shot
    out = dict(shot)   # shallow copy — we'll deep-merge the 3 visual pillars
    for pillar in ("camera", "setup", "lighting"):
        merged = _deep_merge(defaults.get(pillar), shot.get(pillar))
        if pillar in overrides:
            merged = _deep_merge(merged, overrides[pillar])
        out[pillar] = merged
    return out


def gaze_clauses_of(shot: dict, offscreen: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """English clauses describing each subject's eyeline. Empty list when
    no subject has a `gaze` field filled in. Compilers append these to
    the prompt so the storyboard's eyeline gets faithfully rendered.

    An object in `offscreen` (see props_out_of_frame) is not named: "gazing
    at the car console" asks for a console the frame does not hold.

    Examples produced:
      "nina looking at omar"
      "theo gazing at the tablet"
      "omar looking off-frame right"
      "nina looking directly into camera"
    """
    out: list[str] = []
    for sub in (dig(shot, "setup", "subjects") or []):
        gaze = (sub or {}).get("gaze")
        if not gaze:
            continue
        who = sub.get("character_id") or sub.get("cls") or "subject"
        target_type = gaze.get("target_type")
        target_ref  = gaze.get("target_ref")
        direction   = gaze.get("direction")
        if target_type == "character" and target_ref:
            out.append(f"{who} looking at {target_ref}")
        elif target_type == "object" and target_ref in offscreen:
            out.append(f"{who} gazing at something out of frame")
        elif target_type == "object" and target_ref:
            out.append(f"{who} gazing at the {target_ref.replace('_', ' ')}")
        elif target_type == "feature" and target_ref:
            owner_clause = ""
            of_char = gaze.get("of_character")
            if of_char:
                owner_clause = f"{of_char}'s "
            out.append(f"{who} focused on {owner_clause}{target_ref.replace('_', ' ')}")
        elif target_type == "camera" or direction == "into_camera":
            out.append(f"{who} looking directly into camera")
        elif direction in ("off_left", "off_right", "off_up", "off_down"):
            d = direction.replace("off_", "off-frame ")
            out.append(f"{who} looking {d}")
        elif direction in ("left", "right", "up", "down"):
            out.append(f"{who} looking {direction}")
        elif direction == "averted":
            out.append(f"{who} averting gaze")
    return out


def is_empty_framing(shot: dict) -> bool:
    """True when this is a 空镜 — environment-only shot, no human/animal
    subjects. Either explicitly marked via `framing == 'empty'`, or
    inferred when Setup.subjects[] is empty AND framing isn't 'insert'."""
    framing = dig(shot, "camera", "creative_intent", "framing")
    if framing == "empty":
        return True
    return False


def primary_focus_of(panel: dict, shot: dict) -> dict:
    """Effective primary_focus: panel override wins over shot baseline."""
    return panel.get("primary_focus") or dig(shot, "setup", "primary_focus") or {}


def excluded_of(panel: dict, shot: dict) -> list[str]:
    """Effective excluded list: panel `setup_override.excluded` wins over shot."""
    panel_exc = dig(panel.get("setup_override") or {}, "excluded")
    if panel_exc is not None:
        return list(panel_exc)
    return list(dig(shot, "setup", "excluded") or [])


def style_family_of(shot: dict) -> str:
    """Setup.environment.style — Flux style anchor key, with fall-back to
    `sketch_bw` when unset or unknown."""
    raw = dig(shot, "setup", "environment", "style") or ""
    return raw if raw in (
        "sketch_bw", "inkwash_bw", "chiaroscuro", "photoreal",
        "ink_color", "concept_art", "line_art_clean",
    ) else "sketch_bw"


def is_monochrome_style(shot: dict) -> bool:
    """True for styles that must render with no color at all.

    line_art_clean briefly excluded itself here to make room for Storyboard
    Pro's spot-color-highlight convention (red/blue/green marking one special
    element), but Flux2 didn't respect "one element" — it applied color to
    every screen/console surface in frame instead, which read as MORE
    colorful, not a controlled accent. Reverted: hard color suppression wins
    over an unreliable spot-color effect."""
    fam = style_family_of(shot)
    return fam in ("sketch_bw", "inkwash_bw", "chiaroscuro", "line_art_clean")


def action0_of(shot: dict, panel: dict | None = None) -> dict:
    """First action in effect for this beat, or empty dict. Compilers read
    description_zh / description_en / beat_features / intensity from this.

    Panel `events_override.actions` wins over the shot baseline, same rule
    as `primary_focus_of`/`excluded_of` — it exists precisely for "a
    different action for this beat" (see Panel.events_override in
    types_v1.py). Every compiler called this with `shot` alone until now,
    so the override was captured in the KB and never read: a panel like
    "the man disappears beneath the car, only one hand still visible"
    rendered its shot's first/default action instead, with nothing saying
    so.
    """
    panel_actions = dig(panel or {}, "events_override", "actions")
    if panel_actions:
        return panel_actions[0]
    actions = dig(shot, "events", "actions") or []
    return actions[0] if actions else {}


def time_of_day_of(shot: dict, scene: dict) -> str | None:
    """Shot Setup.backdrop.time_of_day overrides scene narrative_meta.time_of_day."""
    return (
        dig(shot, "setup", "backdrop", "time_of_day")
        or dig(scene, "narrative_meta", "time_of_day")
    )


def location_of(scene: dict, shot: dict) -> tuple[str | None, str | None]:
    """(location_ref, location_raw). Shot Setup.backdrop.location overrides
    scene narrative_meta.location_raw."""
    raw = dig(shot, "setup", "backdrop", "location") or dig(scene, "narrative_meta", "location_raw")
    ref = dig(scene, "narrative_meta", "location_ref")
    return ref, raw


def compile_hints_for_panel(scene: dict, panel_id: str) -> dict:
    """Compile-hints sidecar entry for one panel, or empty dict."""
    for h in (scene.get("compile_hints") or []):
        if isinstance(h, dict) and h.get("panel_id") == panel_id:
            return h
    return {}


# ── character-name resolution ────────────────────────────────────────
# Shot documents name people the way the screenplay does ("李明"); the
# registry is keyed on an ASCII slug ("li_ming"). A plain dict lookup
# misses silently, and every caller reads a miss as "this character has
# no identity data" — so the panel renders with nothing locking
# appearance and the character comes out as a different person in each
# scene. Resolution is by exact key, then declared alias, then display
# name, then trigger word; an unknown name still misses, because the
# point is to follow real aliases, not to match loosely.

def _character_entries(kb: dict):
    """Registry entries only. Schema metadata sits at the same level."""
    for key, entry in (kb or {}).items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if isinstance(entry, dict):
            yield key, entry


def resolve_character(name: str, characters_kb: dict) -> dict:
    """The registry entry a character reference denotes, or {} if none."""
    if not name or not characters_kb:
        return {}
    ref = name.partition("@")[0].strip()
    if not ref:
        return {}
    entry = characters_kb.get(ref)
    if isinstance(entry, dict):
        return entry
    for _key, ent in _character_entries(characters_kb):
        if ref in (ent.get("aliases") or []):
            return ent
        if ent.get("name_zh") == ref or ent.get("name") == ref:
            return ent
        if ent.get("trigger") == ref:
            return ent
    return {}


def resolve_prop_key(prop_id: str, props_kb: dict) -> str:
    """The registry KEY a prop reference denotes, or "".

    Two references that name the same object -- `monitor` in one shot and
    `computer_monitor` in the next -- have to collapse to one identity before
    anything can relate them to each other. The greybox needs exactly that to
    put a prop on the surface of another prop: the host is written once in the
    registry, and the panel spells both it and the prop however the enrichment
    happened to that shot.
    """
    if not prop_id or not props_kb:
        return ""
    reg = props_kb.get("props", props_kb) or {}
    ref = str(prop_id).strip()
    if isinstance(reg.get(ref), dict):
        return ref
    for key, ent in reg.items():
        if key.startswith("_") or not isinstance(ent, dict):
            continue
        if ref in (ent.get("aliases") or []) or ref in (ent.get("name"),
                                                        ent.get("name_zh"),
                                                        ent.get("id")):
            return key
    return ""


def resolve_prop(prop_id: str, props_kb: dict) -> dict:
    """The registry entry a prop reference denotes, or {}.

    The character lookup has read aliases since it was written; this one was
    a bare `dict.get`, so a reference the enrichment spelled differently from
    the registry resolved to nothing and the prop's agreed appearance was
    replaced by a noun phrase built from the identifier. On another production that is 14
    of 21 references: `u_pan` for the USB drive at the turning point of the
    film, which compiled as "a u pan in frame".
    """
    if not prop_id or not props_kb:
        return {}
    reg = props_kb.get("props", props_kb) or {}
    ref = str(prop_id).strip()
    entry = reg.get(ref)
    if isinstance(entry, dict):
        return entry
    for key, ent in reg.items():
        if key.startswith("_") or not isinstance(ent, dict):
            continue
        if ref in (ent.get("aliases") or []):
            return ent
        if ref in (ent.get("name"), ent.get("name_zh"), ent.get("id")):
            return ent
    return {}


def unresolved_characters(names, characters_kb: dict) -> list[str]:
    """Which references the registry cannot account for, in order."""
    out: list[str] = []
    for n in names or []:
        if not isinstance(n, str) or not n.strip():
            continue
        if not resolve_character(n, characters_kb):
            out.append(n)
    return out


def reference_images_for(characters, age_states: dict, characters_kb: dict,
                         per_character: int = 1,
                         limit: int | None = None) -> list[str]:
    """Appearance references for every declared subject, not just the lead.

    A scene with two people needs both faces held. Age state selects a
    dedicated reference set when the registry declares one, since the
    same character at a different life stage is a different identity.
    Duplicates are collapsed rather than stacked: sending one image twice
    spends a reference slot without adding information.
    """
    out: list[str] = []
    seen: set[str] = set()
    for c in characters or []:
        ent = resolve_character(c, characters_kb)
        if not ent:
            continue
        refs = None
        age = (age_states or {}).get(c) or (age_states or {}).get(
            c.partition("@")[0])
        by_age = ent.get("reference_images_by_age_state") or {}
        if age and isinstance(by_age, dict) and by_age.get(age):
            refs = by_age[age]
        if refs is None:
            refs = ent.get("reference_images")
        if isinstance(refs, str):
            refs = [refs]
        for r in (refs or [])[:max(1, per_character)]:
            if isinstance(r, str) and r and r not in seen:
                seen.add(r)
                out.append(r)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def lora_trigger_for(characters: list[str], age_states: dict[str, str],
                     characters_kb: dict) -> str | None:
    """LoRA trigger word for the panel's primary character, or None."""
    if not characters or not characters_kb:
        return None
    first = characters[0]
    char_kb = resolve_character(first, characters_kb)
    lora = char_kb.get("lora") or {}
    if "by_age_state" in lora:
        age = age_states.get(first)
        if age and age in lora["by_age_state"]:
            return lora["by_age_state"][age].get("trigger_word")
    if "trigger_word" in lora:
        return lora["trigger_word"]
    return char_kb.get("trigger") or None


# ── PAI 1.1 readers that were missing from the original compilers ────
# Pure value extractors. Each compiler renders them in its own prose
# style (compile_flux2 flowing, compile_gpt_image_2 labeled, compile_
# nano_banana sentence). Skip silently when fields are absent — the
# caller decides empty-vs-default semantics.

def emotion_clauses_of(shot: dict, panel: dict | None = None) -> list[str]:
    """Visible/implied emotion cues. Reads events.emotions[*], preferring
    .explicit when authored, falling back to .implicit. Skips entries
    where both are None (placeholder emotion records).

    A panel's `events_override.emotions` wins over the shot's, the same rule
    as `action0_of`, and an empty list there means the panel carries no cue.
    A shot split into one close-up per subject still has one emotion list:
    without the override the reaction panel inherits the other subject's
    trembling hands."""
    out: list[str] = []
    panel_emos = dig(panel or {}, "events_override", "emotions")
    emos = panel_emos if panel_emos is not None else dig(shot, "events", "emotions")
    for em in (emos or []):
        if not isinstance(em, dict):
            continue
        s = (em.get("explicit") or em.get("implicit") or "").strip()
        if s:
            out.append(s)
    return out


def props_of(shot: dict) -> list[dict]:
    """Setup.props[] entries with at least one non-empty descriptive
    field. Filters out placeholder/empty props so the compiler doesn't
    emit "with  in frame"."""
    out: list[dict] = []
    for p in (dig(shot, "setup", "props") or []):
        if not isinstance(p, dict):
            continue
        if any((p.get(k) or "") for k in ("prop_id", "material", "pattern",
                                          "state", "color", "size")):
            out.append(p)
    return out


_ARTICLES = ("a ", "an ", "the ")


def _with_count(text: str, n) -> str:
    """State a registered prop's cardinality as a numeral.

    EXPERIMENT. A count of one was left implicit on the reasoning that the
    description carries its own article; an article is not a quantity, and
    a panel with one registered device rendered two. Saying it in words
    ("a single ...") made that worse. This tests the numeral form, which
    is what worked for cast count.
    """
    body, low = text, text.lower()
    started_with_article = False
    for art in _ARTICLES:
        if low.startswith(art):
            body = text[len(art):]
            started_with_article = True
            break
    # An anchor with no article is describing a plural or mass fixture --
    # "wraparound interior screen panels lining the cabin walls". Forcing the
    # numeral onto it produces "exactly 1 ... panels", which is both wrong and
    # a quantity claim the production never made. The numeral experiment is
    # about countable objects that duplicated; it does not apply here.
    if not started_with_article and not isinstance(n, int):
        return text
    k = n if isinstance(n, int) and n > 0 else 1
    return f"exactly {k} {body}"


def prop_phrase(p: dict, props_kb: dict | None = None) -> str:
    """Render one Prop as a noun phrase: "a cracked handheld tablet",
    "a stack of polished wooden tablets". Order: count → state → color
    → material → prop_id-stem.

    When the project's prop registry is supplied and the entry carries a
    written description, that description is used instead of a phrase
    assembled from the identifier. The registry holds the appearance a
    production has agreed on ("a futuristic handheld gaming device with a
    sleek design"); building "a game device" from the id discards it and
    leaves the look of the object to the generator, which is the same
    failure as naming a character by reference instead of describing them.
    """
    # A prop's DECLARED STATE outranks its registry appearance.
    #
    # This branch used to return the registry text unchanged, and `state` was
    # consulted only in the identifier fallback below -- so for any prop with
    # a registry entry, which in practice is all of them, the state field was
    # read and could not take effect. That is how a beat asserting the car has
    # lost all power was compiled alongside "glowing with interface graphics"
    # into one paragraph that contradicts itself.
    #
    # Applied by substitution, not by adding a denial: every Flux workflow
    # here runs cfg=1.0, so the negative prompt is never evaluated and a thing
    # can only be removed by not writing it.
    from pace_core.breakdown.world_state import rewrite_for_state, states_for_prop
    states = states_for_prop(p)

    pid_key = (p.get("prop_id") or "").strip()
    if props_kb and pid_key:
        entry = resolve_prop(pid_key, props_kb) or None
        if isinstance(entry, dict):
            for field in ("anchor", "generic_anchor", "description", "appearance"):
                text = entry.get(field)
                if isinstance(text, str) and text.strip():
                    body = text.strip()
                    if states:
                        body, _ = rewrite_for_state(body, states)
                    return _with_count(body, p.get("count"))
    bits: list[str] = []
    count = p.get("count")
    if isinstance(count, int) and count > 1:
        bits.append(f"{count}")
    elif count == 1 or count is None:
        bits.append("a")
    state = p.get("state")
    if state:
        bits.append(state.replace("_", " "))
    color = p.get("color")
    if color:
        bits.append(color.replace("_", " "))
    material = p.get("material")
    if material:
        bits.append(material.replace("_", " "))
    pid = (p.get("prop_id") or "").replace("_", " ").strip()
    if pid:
        bits.append(pid)
    return " ".join(bits) if bits else "a prop"


# Where a prop declares itself in the frame, in words -- only for the zones that
# put it away from the cast. A prop described with no place is attached by the
# sampler to whatever in frame matches its words: one scene's "mechanical
# robotic arms emerging from the wreckage", declared in the background, came
# back as the hands of the man in the foreground. A prop held in the hands or
# set in the foreground is left as written; its place is the cast's.
_PROP_ZONE_WORDS = {
    "background": "in the background",
    "background_midground": "in the background",
    "midground": "in the middle distance",
}


def prop_placement(p: dict) -> str:
    """A prop's declared screen zone as a trailing phrase, or "" if it has none
    that separates it from the cast."""
    zone = ((p.get("screen_position") or {}).get("zone") or "").strip()
    phrase = _PROP_ZONE_WORDS.get(zone)
    return f" {phrase}" if phrase else ""


# The anchors a prop's registry `placement` may name for the greybox to stage
# it. The greybox builder reads this list; the compiler reads it too, because
# it has to know what the control image will hold, and two copies of the list
# is how the two would come to disagree.
FIXTURE_ANCHORS = ("shell_front", "shell_rear", "shell_center", "floor",
                   "per_seat", "shell_sides", "ground", "on_surface")

# How long a prop has to be before it is a thing in the scene rather than a
# thing on a person.
_LARGE_PROP_M = 1.0


def prop_left_unstaged_away_from_cast(p: dict, props_kb: dict | None = None) -> bool:
    """True for a large prop, declared away from the cast, that the greybox
    does not stage -- one the prompt should not name.

    Such a prop has nothing in the control image to be drawn on, so the
    sampler draws it on what is there. On one scene's second beat, "mechanical
    robotic arms emerging from the wreckage" (1.5 m, declared in the
    background, no placement) came back as the hands of the one man in frame,
    with or without its place written after it; the same render without the
    words gave him human hands. A prop held or worn is not affected: its place
    is the cast's, and it is drawn on the cast by design.
    """
    if not prop_placement(p):
        return False                      # not declared away from the cast
    entry = resolve_prop((p.get("prop_id") or "").strip(), props_kb) if props_kb else None
    entry = entry if isinstance(entry, dict) else {}
    if (entry.get("placement") or {}).get("anchor") in FIXTURE_ANCHORS:
        return False                      # the greybox stages it
    size = (entry.get("physical_attributes") or {}).get("size_m") or []
    longest = max((float(s) for s in size if isinstance(s, (int, float))), default=0.0)
    return longest >= _LARGE_PROP_M or "vehicle" in str(p.get("size") or "")


def props_out_of_frame(greyboxes_dir, panel_id: str) -> frozenset[str]:
    """Props the panel's greybox staged where its camera does not see them.

    Staged is not the same as in frame. The cabin camera stands at the front
    looking back, so the console anchored at the front of the cabin is built
    behind the lens; named in the prompt anyway, it had nowhere in the
    control image to be drawn, and one render put it on the back wall while
    the same seed with the console dark put a pane of glass across the cast.
    The build writes what its camera sees beside the frame; with no record
    (no greybox yet, or an older build) nothing is left out.
    """
    if not greyboxes_dir or not panel_id:
        return frozenset()
    import json
    from pathlib import Path
    try:
        rec = json.loads((Path(greyboxes_dir) / f"{panel_id}_90001_.png.props.json").read_text())
    except (OSError, ValueError):
        return frozenset()
    return frozenset(k for k, v in rec.items() if isinstance(v, dict) and v.get("in_frame") is False)


# What a panel declares about whether a subject or prop is in the picture
# (types_v1.InFrame). A blocking sheet lists what enters the frame apart from
# where things stand, and marks what the source does not say as to be
# confirmed; the build's record (props_out_of_frame) checks the declaration
# and stands in for it where there is none.
IN_FRAME_VALUES = ("yes", "partial", "no", "tbd")


def declared_in_frame(entry: dict) -> str | None:
    """The entry's in-frame declaration, or None when it makes none."""
    v = str((entry or {}).get("in_frame") or "").strip().lower()
    return v if v in IN_FRAME_VALUES else None


def prop_frame_status(p: dict, staged_out: frozenset[str] | set[str] = frozenset()) -> str:
    """"yes", "partial" or "no" for one prop in one panel.

    The declaration decides when it gives one. Without one -- or with "tbd",
    which says the source is silent -- the build's record decides, and a prop
    it has no record of is taken to be in frame, as before.
    """
    d = declared_in_frame(p)
    if d in ("yes", "partial", "no"):
        return d
    return "no" if (p.get("prop_id") or "") in staged_out else "yes"


def props_not_in_frame(shot: dict, staged_out: frozenset[str] | set[str] = frozenset()) -> frozenset[str]:
    """Every prop this panel must not name as visible: declared out of frame,
    or recorded out of frame by the build and not declared in."""
    status = {(p.get("prop_id") or ""): prop_frame_status(p, staged_out) for p in props_of(shot)}
    out = {k for k, v in status.items() if v == "no"}
    out |= {k for k in staged_out if k not in status}
    return frozenset(k for k in out if k)


def in_frame_extent_phrase(p: dict) -> str:
    """How much of a prop declared partly in frame the frame shows."""
    if declared_in_frame(p) != "partial":
        return ""
    extent = str(p.get("in_frame_extent") or "").strip()
    return f" (only {extent} visible)" if extent else " (partly cut off by the frame edge)"


def wardrobe_of(props_kb: dict | None, subject: dict | None) -> dict | None:
    """The library entry a shot named through `setup.subjects[].costume_id`.

    Garments are props: they are sourced, they have states, and they have to
    survive a cut. Before they had ids, a costume was a sentence on a
    character record, and two panels could not be compared for wearing the
    same thing.
    """
    cid = (subject or {}).get("costume_id")
    if not isinstance(cid, str) or not cid.strip():
        return None
    props = props_kb or {}
    if isinstance(props, dict):
        props = props.get("props", props) or {}
    if isinstance(props, list):
        props = {p.get("id") or p.get("prop_id"): p
                 for p in props if isinstance(p, dict)}
    e = props.get(cid.strip())
    return e if isinstance(e, dict) else None


def costume_text(entry: dict | None, age_state: str | None = None,
                 *, wardrobe: dict | None = None) -> str | None:
    """What a character wears here: the shot's wardrobe entry, or the default.

    One reading for every consumer: the prompt compiler writes it into the
    character's description, and the greybox tones the proxy's garments from
    it, so the words and the control image cannot name two different outfits.

    A `wardrobe` entry wins over the registry, because a shot that names a
    `costume_id` has decided; the character's `costumes` map is only what they
    wear when no shot says otherwise.
    """
    anchor = (wardrobe or {}).get("anchor")
    if isinstance(anchor, str) and anchor.strip():
        return anchor.strip()
    costumes = (entry or {}).get("costumes")
    outfit = None
    if isinstance(costumes, dict):
        outfit = (costumes.get(age_state) if age_state else None) or costumes.get("default")
    elif isinstance(costumes, str):
        outfit = costumes
    return outfit.strip() if isinstance(outfit, str) and outfit.strip() else None


def secondary_subjects_of(shot: dict) -> list[str]:
    """Setup.secondary_subjects[] — character refs visible but not focal."""
    out: list[str] = []
    for ref in (dig(shot, "setup", "secondary_subjects") or []):
        if isinstance(ref, str) and ref.strip():
            out.append(ref.strip())
    return out


def framing_of(shot: dict) -> str | None:
    """Camera.creative_intent.framing (e.g. "two_shot", "over_shoulder",
    "dutch_tilt"). None when unset or "empty"/"single" (already implied)."""
    f = dig(shot, "camera", "creative_intent", "framing")
    if not isinstance(f, str):
        return None
    f = f.strip()
    if f in ("", "empty", "single"):
        return None
    return f


def aspect_ratio_of(shot: dict) -> str | None:
    """Camera.creative_intent.aspect_ratio — "2.35:1", "16:9", "1:1", "9:16"."""
    ar = dig(shot, "camera", "creative_intent", "aspect_ratio")
    return ar if isinstance(ar, str) and ar.strip() else None


def setting_kind_of(shot: dict) -> str | None:
    """Setup.backdrop.setting — "int" | "ext". Returns the spelled-out
    form ("interior" / "exterior") or None."""
    s = dig(shot, "setup", "backdrop", "setting")
    return {"int": "interior", "ext": "exterior"}.get(s)


def weather_of(shot: dict) -> str | None:
    """Setup.backdrop.weather — open-set string (rain, snow, fog, …)."""
    w = dig(shot, "setup", "backdrop", "weather")
    return w if isinstance(w, str) and w.strip() else None


def season_of(shot: dict) -> str | None:
    """Setup.backdrop.season — open-set string (spring, autumn, …)."""
    s = dig(shot, "setup", "backdrop", "season")
    return s if isinstance(s, str) and s.strip() else None


def change_in_environment_of(shot: dict) -> str | None:
    """Events.change_in_environment — atmospheric change mid-shot
    ("rain starting to fall", "candle flame guttering")."""
    c = dig(shot, "events", "change_in_environment")
    return c if isinstance(c, str) and c.strip() else None
