"""Script-to-scene splitter.

Takes a literary script (prose or screenplay format) and breaks it into discrete
scenes — each scene being a continuous unit of action in one location at one
time-of-day. Output is one PAI 1.1 file per scene at
`kb/on_scene/projects/<project>/scene_NN.json`, with narrative_meta,
shot_defaults, initial shots, and one starter panel per shot filled in.

Splitting is LLM-driven: a single LLM call returns structured JSON (scene
boundaries + per-scene era/region/culture), so it works on any prose or
screenplay and infers period from the script's own content — no per-film
heuristics. Uses the assets/models.json registry ($MODELS_FILE); default model =
claude-sonnet-5.

Usage:
  python3 pipeline/split_script.py path/to/script.md
  python3 pipeline/split_script.py path/to/script.md --model claude-sonnet-5
  python3 pipeline/split_script.py - < /dev/stdin   # read from stdin
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, # 设置级别为 INFO（低于INFO的DEBUG日志将被忽略）
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'), # 写入日志文件
        logging.StreamHandler()                            # 同时输出到控制台
    ]
)


# ───────────────────────  shot scaffolding helpers  ────────────────────────

# Longest-first within each language: "大特写" must be tested before "特写",
# and "extreme close" before "close". The table was Chinese-only, so an
# English screenplay could not match any pattern and every panel fell to the
# "medium" default, which is how a 27-panel corpus came to author one shot
# size throughout and lost the ability to test anything that varies with it.
_SHOT_SIZE_PATTERNS = [
    ("extreme close", "extreme_close_up"),
    ("establishing", "establishing"),
    ("long shot", "wide"),
    ("wide shot", "wide"),
    ("full shot", "wide"),
    ("two shot", "medium"),
    ("two-shot", "medium"),
    ("medium close", "medium_close_up"),
    ("close on", "close_up"),
    ("tight on", "close_up"),
    ("close-up", "close_up"),
    ("closeup", "close_up"),
    ("close up", "close_up"),
    ("insert", "extreme_close_up"),
    ("aerial", "wide"),
    ("wide", "wide"),
    ("medium", "medium"),
    ("大特写", "extreme_close_up"),
    ("大远景", "establishing"),
    ("特写", "close_up"),
    ("中近景", "medium_close_up"),
    ("中景", "medium"),
    ("近景", "close_up"),
    ("全景", "wide"),
    ("远景", "wide"),
    ("主观镜头", "medium"),
    ("主观视角", "medium"),
    ("双人中景", "medium"),
]

_ANGLE_PATTERNS = [
    ("俯拍", "high"),
    ("仰拍", "low"),
    ("航拍", "aerial"),
    ("低机位", "low"),
    ("顶摄", "overhead"),
    ("鸟瞰", "overhead"),
]

_MOVEMENT_2D_PATTERNS = [
    ("从左向右摇", "pan_right"),
    ("从右向左摇", "pan_left"),
    ("摇过", "pan_right"),
    ("上摇", "tilt_up"),
    ("下摇", "tilt_down"),
    ("拉近", "zoom_in"),
    ("变焦推近", "zoom_in"),
    ("变焦拉远", "zoom_out"),
]

_MOVEMENT_3D_PATTERNS = [
    ("缓慢推近", "push_in"),
    ("缓慢推进", "push_in"),
    ("推近", "push_in"),
    ("推进", "push_in"),
    ("推入", "push_in"),
    ("拉远", "pull_out"),
    ("跟拍", "tracking"),
    ("横移", "trucking"),
    ("升起", "crane"),
    ("下降", "crane"),
    ("旋转", "arc"),
    ("环视", "arc"),
    ("绕", "arc"),
]

_GEAR_PATTERNS = [
    ("斯坦尼康", "steadicam"),
    ("手持", "handheld"),
    ("固定机位", "tripod"),
    ("固定", "tripod"),
    ("航拍", "drones"),
    ("摇臂", "cranes"),
    ("轨道", "dolly"),
]

_BLACK_MARKERS = ["黑屏", "纯黑", "5 秒纯黑", "切黑屏", "BLACK"]
_TITLE_MARKERS = ["TITLE", "出片名", "落版字幕", "字幕", "片名"]
_TIME_RE = re.compile(r"[（(]\s*(\d+(?:\.\d+)?)\s*(?:秒|sec|s)\s*[)）]")

# culture → Stage B style family. Best-effort: the LLM returns a free-form
# `culture` per scene; known values map here, everything else falls back to
# the scene's stage_b_style or the inkwash default in _build_shot_defaults.
_CULTURE_TO_STYLE = {
    "kuchean_buddhist": "inkwash_bw",
    "later_qin_chang_an_buddhist": "inkwash_bw",
    "former_qin_frontier": "sketch_bw",
    "later_liang_frontier": "sketch_bw",
    "modern_urban_chinese": "photoreal",
    "time_bridge_modern_historical": "chiaroscuro",
    "time_bridge_dream": "chiaroscuro",
}


def _first_match(text: str | None, patterns: list[tuple[str, str]]) -> str | None:
    """First matching pattern, case-insensitively.

    Screenplay camera terms are conventionally uppercase (CLOSE ON, WIDE
    SHOT), so a case-sensitive test against lowercase patterns matches
    nothing and every panel falls to its default. CJK patterns are
    unaffected, since they have no case.
    """
    lowered = (text or "").lower()
    for needle, value in patterns:
        if needle.lower() in lowered:
            return value
    return None


def _all_matches(text: str | None, patterns: list[tuple[str, str]]) -> list[str]:
    text = text or ""
    out: list[str] = []
    for needle, value in patterns:
        if needle in text and value not in out:
            out.append(value)
    return out


def _duration_hint_s(text: str | None) -> float | None:
    m = _TIME_RE.search(text or "")
    return float(m.group(1)) if m else None


def _clean_beat(text: str | None) -> str:
    return _TIME_RE.sub("", text or "").strip().rstrip("，,。.").strip()


def _normalize_time_of_day(value: str | None) -> str | None:
    if not value:
        return None
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "pre_dawn": "dawn",
        "night_to_dawn": "dawn",
        "indeterminate": None,
        "mixed": None,
        "noon": "midday",
        "noonday": "midday",
        "golden_hour": "sunset",
    }
    if s in aliases:
        return aliases[s]
    allowed = {
        "day", "night", "morning", "evening", "dawn", "dusk",
        "late_night", "midday", "sunrise", "sunset", "afternoon",
    }
    return s if s in allowed else None


def _normalize_setting(value: str | None) -> str | None:
    if not value:
        return None
    s = str(value).upper()
    if s.startswith("INT"):
        return "int"
    if s.startswith("EXT"):
        return "ext"
    return None


def _natural_light_for(setting: str | None, tod: str | None, culture: str | None) -> list[str]:
    if setting == "ext":
        return ["moonlight"] if tod in ("night", "late_night") else ["sunlight"]
    if setting == "int":
        if tod in ("night", "late_night") and culture and "buddhist" in culture:
            return ["firelight"]
        if tod in ("day", "morning", "midday", "afternoon", "dawn", "dusk", "sunrise", "sunset"):
            return ["sunlight"]
    return []


def _lighting_condition(setting: str | None, tod: str | None, natural: list[str]) -> str | None:
    if "firelight" in natural:
        return "candlelight"
    if setting == "ext" and tod in ("dawn", "dusk", "sunrise", "sunset", "evening"):
        return "golden_hour"
    if setting == "ext" and tod in ("day", "morning", "midday", "afternoon"):
        return "clear_daylight"
    return None


def _classify_kind(text: str | None, explicit_nonvisual: bool = False) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("运镜：") or stripped.startswith("运镜:"):
        return "meta"
    if any(m in stripped for m in _TITLE_MARKERS):
        return "title_card"
    if any(m in stripped for m in _BLACK_MARKERS):
        return "black"
    return "black" if explicit_nonvisual else "visual"


def _shot_size_for(action: str | None, declared: str | None, kind: str) -> str:
    if kind != "visual":
        return "wide"
    if declared:
        mapped = _first_match(declared, _SHOT_SIZE_PATTERNS)
        if mapped:
            return mapped
    parsed = _first_match(action, _SHOT_SIZE_PATTERNS)
    return parsed or "medium"


def _framing_for(action: str | None, kind: str, subject_count: int) -> str:
    text = action or ""
    if kind in ("black", "title_card", "meta"):
        return "empty"
    if "主观" in text or "POV" in text.upper():
        return "pov"
    if "过肩" in text or "OTS" in text.upper():
        return "ots"
    if "双人" in text or subject_count == 2:
        return "two_shot"
    if subject_count >= 3:
        return "crowd"
    if subject_count == 0:
        return "empty"
    return "single"


def _subjects_for(scene: dict, kind: str) -> list[dict]:
    if kind in ("black", "title_card", "meta"):
        return []
    ages = scene.get("character_age_states") or {}
    subjects = []
    for idx, cid in enumerate((scene.get("characters_present") or [])[:3]):
        subjects.append({
            "character_id": cid,
            "age_state": ages.get(cid),
            "gaze": None,
            "screen_position": {
                "zone": "center" if idx == 0 else ("center_left" if idx == 1 else "center_right"),
                "x": None,
                "y": None,
                "depth": "foreground" if idx == 0 else "midground",
            },
        })
    return subjects


def _primary_focus_for(subjects: list[dict], kind: str, scene: dict) -> dict:
    if subjects:
        return {
            "type": "character",
            "ref": subjects[0].get("character_id") or "",
            "of_character": None,
            "coverage_pct": 55,
        }
    return {
        "type": "environment",
        "ref": scene.get("location_ref") or scene.get("location_raw") or "scene",
        "of_character": None,
        "coverage_pct": 70 if kind == "visual" else None,
    }


def _beat_by_language(text: str) -> dict:
    """{description_zh, description_en} with `text` in whichever is right.

    A beat is one short sentence, so the test is simply whether it contains
    CJK: mixed text is treated as Chinese, because the translate node handles
    a Chinese sentence containing English fragments and not the reverse.
    """
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
    return {"description_zh": text if has_cjk else "",
            "description_en": "" if has_cjk else text}


def _action_for(beat: str, action_text: str | None) -> dict:
    return {
        "standalone": beat or None,
        "interactive": None,
        "temporal": "atomic",
        "foreground": "focal",
        "background": False,
        "uncertainty": None,
        # Route the beat by the language it is actually in. Writing it to
        # description_zh unconditionally put English beats in the Chinese
        # field, where compile_flux2's fallback hands them to the
        # FluxZhEnTranslate node -- an English sentence run through a
        # Chinese-to-English translator on the way to the prompt.
        **_beat_by_language(beat or action_text or ""),
        "beat_features": [],
        "intensity": "medium",
        "duration_hint_s": _duration_hint_s(action_text),
    }


def _dialogues_for(scene: dict, shot_index: int, total_shots: int) -> list[dict]:
    lines = scene.get("on_screen_dialogue") or []
    if not lines:
        return []
    picked = lines if total_shots <= 1 else ([lines[shot_index]] if shot_index < len(lines) else [])
    out = []
    for line in picked:
        if isinstance(line, str):
            out.append({"type_of_delivery": None, "foreground": "focal", "speaker": None, "text": line})
        elif isinstance(line, dict):
            out.append({
                "type_of_delivery": None,
                "foreground": "focal",
                "speaker": line.get("speaker") or line.get("character"),
                "text": line.get("text") or line.get("line") or "",
            })
    return out


def _build_shot_defaults(scene: dict) -> dict:
    setting = _normalize_setting(scene.get("interior_exterior"))
    tod = _normalize_time_of_day(scene.get("time_of_day"))
    # Period context (era / region / culture) is inferred by the LLM split
    # and carried on the scene dict — no hardcoded per-film rules.
    era     = scene.get("era") or None
    region  = scene.get("region") or None
    culture = scene.get("culture") or None
    natural = _natural_light_for(setting, tod, culture)
    condition = _lighting_condition(setting, tod, natural)
    style = _CULTURE_TO_STYLE.get(culture, scene.get("stage_b_style") or "inkwash_bw")
    color_temperature = "warm" if condition in ("candlelight", "golden_hour") else None
    return {
        "camera": {"creative_intent": {"aspect_ratio": "2.35:1"}},
        "setup": {
            "backdrop": {
                "setting": setting,
                "time_of_day": tod,
                "location": scene.get("location_ref") or scene.get("location_raw"),
                "era": era,
                "region": region,
                "culture": culture,
            },
            "environment": {"style": style, "mood": None, "scale": None, "elements": []},
        },
        "lighting": {
            "natural": natural,
            "condition": condition,
            "color_temperature": color_temperature,
            "position": "key_light" if setting else None,
        },
    }


def _build_initial_shots(scene: dict) -> tuple[list[dict], list[dict]]:
    actions = list(scene.get("key_actions") or [])
    if not actions and scene.get("titles"):
        actions = ["TITLE: " + " / ".join(scene.get("titles") or [])]
    if not actions:
        actions = [scene.get("summary") or scene.get("story_text") or scene.get("heading") or "Scene beat"]

    has_declared = scene.get("panel_shot_sizes") is not None
    declared = list(scene.get("panel_shot_sizes") or [])
    if len(declared) < len(actions):
        declared.extend([None] * (len(actions) - len(declared)))

    shots = []
    hints = []
    for i, action_text in enumerate(actions, start=1):
        kind = _classify_kind(action_text, explicit_nonvisual=has_declared and declared[i - 1] is None)
        beat = _clean_beat(action_text)
        shot_id = f"shot_{i:02d}"
        panel_id = f"{scene['scene_id']}_{shot_id}_panel_{i:04d}"
        subjects = _subjects_for(scene, kind)
        movements_2d = _all_matches(action_text, _MOVEMENT_2D_PATTERNS)
        movements_3d = _all_matches(action_text, _MOVEMENT_3D_PATTERNS)
        gear = _first_match(action_text, _GEAR_PATTERNS)
        static = not movements_2d and not movements_3d
        shot_size = _shot_size_for(action_text, declared[i - 1], kind)
        framing = _framing_for(action_text, kind, len(subjects))
        primary_focus = _primary_focus_for(subjects, kind, scene)
        scale = "intimate" if shot_size in ("extreme_close_up", "close_up", "medium_close_up", "medium") else "expansive"

        shot = {
            "shot_id": shot_id,
            "camera": {
                "intrinsics": {},
                "extrinsics": {"angle": _first_match(action_text, _ANGLE_PATTERNS) or "eye_level", "position": "front"},
                "trajectory": {
                    "static": static,
                    "movement_2d": movements_2d,
                    "movement_3d": movements_3d,
                    "gear": gear or ("tripod" if static else None),
                    "easing": "ease_out" if movements_2d or movements_3d else "linear",
                },
                "creative_intent": {"shot_size": shot_size, "framing": framing},
            },
            "setup": {
                "texture": {},
                "geometry": {},
                "space": {},
                "backdrop": {},
                "environment": {"scale": scale, "elements": []},
                "props": [],
                "subjects": subjects,
                "primary_focus": primary_focus,
                "secondary_subjects": [],
                "excluded": [],
                "text_generation": [],
            },
            "lighting": {},
            "events": {
                "actions": [_action_for(beat, action_text)],
                "emotions": [],
                "dialogues": _dialogues_for(scene, i - 1, len(actions)),
                "change_in_environment": None,
                "advanced": {"story_structure": None, "pace": "slow", "regularity": "regular"},
            },
            "panels": [{
                "id": panel_id,
                "panel_number": 1,
                "scene_id": scene["scene_id"],
                "shot_id": shot_id,
                "camera_override": None,
                "setup_override": None,
                "lighting_override": None,
                "events_override": None,
                "primary_focus": primary_focus,
                "notes": None if kind == "visual" else kind,
            }],
        }
        shots.append(shot)
        hints.append({"panel_id": panel_id, "flux": {}, "gpt_image_2": {}, "wan_i2v": {}})
    return shots, hints


def _lift_dialogue(items, kind: str) -> list[dict]:
    """Normalize bare-string dialogue (legacy) or dict (already DialogueLine) → DialogueLine dict.
    Mirrors migrate_kb_to_scine.py to keep the two writers in sync.
    """
    out = []
    for it in items or []:
        if isinstance(it, str):
            out.append({"character": None, "text": it, "start_s": None, "kind": kind})
        elif isinstance(it, dict):
            out.append({
                "character": it.get("character"),
                "text":      it.get("text", ""),
                "start_s":   it.get("start_s"),
                "kind":      it.get("kind", kind),
            })
    return out


def _to_pai_1_1(sc: dict, source_script: str) -> dict:
    """Wrap a heuristic/LLM scene dict into the PAI 1.1 unified shape.

    The splitter produces a complete first-pass structure: scene narrative,
    scene-wide shot_defaults, initial shots, and one starter panel per shot.
    """
    sc = dict(sc)
    sc.setdefault("scene_id", f"scene_{int(sc.get('scene_number') or 1):02d}")
    narrative_meta = {
        "location_ref":         sc.get("location_ref"),
        "location_raw":         sc.get("location_raw"),
        "time_of_day":          _normalize_time_of_day(sc.get("time_of_day")) or sc.get("time_of_day"),
        "interior_exterior":    sc.get("interior_exterior"),
        "summary":              sc.get("summary"),
        "story_beat":           sc.get("story_beat"),
        "key_actions":          sc.get("key_actions", []),
        "vo_lines":             _lift_dialogue(sc.get("vo_lines"), "vo"),
        "on_screen_dialogue":   _lift_dialogue(sc.get("on_screen_dialogue"), "on_screen_text"),
        "titles":               sc.get("titles", []),
        "sfx_notes":            sc.get("sfx_notes", []),
        "characters_present":   sc.get("characters_present", []),
        "character_age_states": sc.get("character_age_states", {}),
        "implied_shot_count":   sc.get("implied_shot_count"),
        "stage_b_style":        sc.get("stage_b_style"),
    }
    shots, compile_hints = _build_initial_shots(sc)
    return {
        "_schema_version": "pai-1.1",
        "scene_id":        sc.get("scene_id"),
        "scene_number":    sc.get("scene_number"),
        "scene_heading":   sc.get("heading"),
        "act":             sc.get("act"),
        "narrative_meta":  narrative_meta,
        "shot_defaults":   _build_shot_defaults(sc),
        "physical_layout": None,
        "shots":           shots,
        "compile_hints":   compile_hints,
        "_max_shot_n":     len(shots),
        "_max_panel_n":    len(shots),
        "_source":         source_script,
        "_generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


PIPELINE_DIR = Path(__file__).resolve().parent


def _models_file_default() -> str:
    """The registry path, for --help. Imported lazily so importing this module
    never depends on the paths package being importable."""
    try:
        from pace_core.paths import MODELS_FILE
        return str(MODELS_FILE)
    except Exception:                                          # noqa: BLE001
        return "assets/models.json"

# Multi-tenant: no module-level path defaults. main() resolves them per
# --project via paths_for() at call time; library callers (studio_server's
# /api/split-script handler) build paths themselves and pass them in.
# Keys exist with None sentinels for back-compat with existing code that
# checks `dest or DEFAULTS[k]` — None falls through to require an explicit arg.
DEFAULTS: dict = {"locations_dir": None, "out_dir": None}


def _load_locations(locations_dir: Path | None, legacy_scenes_file: Path | None = None) -> dict:
    """Load location anchors so we can match raw script locations to kb refs."""
    locs: dict = {}
    if locations_dir and locations_dir.is_dir():
        for f in locations_dir.glob("*.bible.json"):
            key = f.stem.replace(".bible", "")
            try:
                locs[key] = json.loads(f.read_text())
            except Exception:
                pass
    if legacy_scenes_file and legacy_scenes_file.exists():
        for k, v in json.loads(legacy_scenes_file.read_text()).items():
            locs.setdefault(k, v)
    return locs


def _match_location(raw: str, locations: dict) -> str | None:
    """Best-effort match raw location string to a kb/on_scene/locations key."""
    raw_low = raw.lower()
    raw_norm = re.sub(r"[^a-z0-9]+", "_", raw_low).strip("_")
    if raw_norm in locations:
        return raw_norm
    # Substring match against kb keys
    for key in locations:
        if key in raw_norm or raw_norm in key:
            return key
    # Token overlap fallback (e.g. "RAINY STREET" -> "street")
    raw_tokens = set(re.findall(r"[a-z]+", raw_low))
    for key in locations:
        key_tokens = set(re.findall(r"[a-z]+", key))
        if raw_tokens & key_tokens:
            return key
    return None


# ─────────────────────────  LLM-backed split  ──────────────────────────────


# How much script goes into one call.
#
# The gateway sits behind Cloudflare, which returns 524 when an origin holds a
# connection for ~100s without sending anything, and it BUFFERS: measured on a
# 30s generation, the first SSE frame arrived at 28.1s. So streaming does not
# keep the connection alive here and the ~100s ceiling is real for any single
# call. Splitting is the call most likely to cross it, because the prompt asks
# for `story_text` -- the verbatim passages -- which makes the output about as
# large as the input. A feature screenplay cannot be echoed back inside the
# window at any temperature or timeout.
#
# So the script is cut into pieces that each finish well inside it. 12000
# characters -- about 3k tokens in and a similar amount out -- measured well
# under a minute on Sonnet, which is what the chain ran on when that number was
# chosen.
#
# It is not a size limit, it is a TIME limit, and the same characters take
# longer on a slower model. Splitting a 5236-character Chinese short on Opus 5
# went over the 100s wall in a single chunk: 12000 could not help, because the
# whole script was one chunk already. 2500 gives that script three pieces of
# roughly 2100, each finishing with room to spare, and leaves headroom for the
# slowest model anyone is likely to select. Raise it with PAI_SPLIT_CHUNK_CHARS
# on a faster model; the merge that rejoins scenes across a boundary is the
# same either way.
#: One generator default for the whole breakdown chain.
from pace_core.breakdown.extract_script_ir import (  # noqa: E402
    GENERATOR_DEFAULT,
)

CHUNK_CHARS = int(os.environ.get("PAI_SPLIT_CHUNK_CHARS") or 2500)


def _chunk_script(text: str, target: int = CHUNK_CHARS) -> list[str]:
    """Cut the script into pieces of roughly `target` characters.

    Cuts fall on blank lines, which in a screenplay are between elements
    rather than inside one, so a chunk never begins halfway through a line of
    dialogue. It can still begin halfway through a SCENE -- that is what the
    merge below is for.
    """
    if len(text) <= target:
        return [text]
    out, buf = [], []
    size = 0
    for block in text.split("\n\n"):
        if size and size + len(block) > target:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(block)
        size += len(block) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out


_UNKNOWN = {"", "unknown", "unknown location", "n/a", "none", "unspecified", "-"}


def _is_continuation(sc: dict) -> bool:
    """Whether this scene is the tail of one that began in the previous chunk.

    A chunk that starts mid-scene gives the model no heading to read, and it
    says so rather than inventing one: measured on this splitter, the first
    scene of every continuation chunk came back with location_raw "unknown" or
    empty and a heading of "INT. UNKNOWN - UNKNOWN". That placeholder is the
    signal, and it is a better one than comparing locations -- the half with no
    heading cannot be compared on the field it is missing.
    """
    loc = str((sc or {}).get("location_raw") or "").strip().lower()
    head = str((sc or {}).get("heading") or "").strip().lower()
    return loc in _UNKNOWN or "unknown" in head


def _same_scene(a: dict, b: dict) -> bool:
    """Whether two scenes either side of a cut are really one scene.

    A chunk boundary lands wherever the character budget ran out, so the last
    scene of one chunk and the first of the next are frequently two halves of
    the same scene. Either the second half says it could not identify itself,
    or both halves agree on the three things that DEFINE a scene change --
    location, time and interior/exterior -- which is the rule the prompt states.
    """
    if _is_continuation(b):
        return True

    def k(x, f):
        return str((x or {}).get(f) or "").strip().lower()
    return (k(a, "location_raw") == k(b, "location_raw")
            and k(a, "time_of_day") == k(b, "time_of_day")
            and k(a, "interior_exterior") == k(b, "interior_exterior"))


def _join_scenes(a: dict, b: dict) -> dict:
    """Fuse the two halves, keeping the first's identity and both's content."""
    out = dict(a)
    out["story_text"] = ((a.get("story_text") or "").rstrip() + "\n\n"
                         + (b.get("story_text") or "").lstrip()).strip()
    for f in ("characters_present", "key_actions"):
        seen, merged = set(), []
        for v in (a.get(f) or []) + (b.get(f) or []):
            if v not in seen:
                seen.add(v); merged.append(v)
        out[f] = merged
    out["implied_shot_count"] = ((a.get("implied_shot_count") or 0)
                                 + (b.get("implied_shot_count") or 0)) or None
    if not (out.get("summary") or "").strip():
        out["summary"] = b.get("summary")
    return out


SYSTEM_PROMPT = """You are a script supervisor. Break the script into individual scenes.

A scene = continuous action in ONE location at ONE time-of-day with ONE set of characters.
A scene change occurs whenever location, time, OR all characters leave / enter.

For each scene, return JSON with these fields:
  scene_number       (int, 1-indexed)
  heading            (str, "INT./EXT. LOCATION - TIME OF DAY", uppercase)
  interior_exterior  ("INT" | "EXT")
  location_raw       (str, location as it appears in the text)
  time_of_day        (str, e.g. "night", "morning", "dusk", "afternoon")
  characters_present (list of character names, lowercase)
  summary            (str, one sentence)
  story_beat         (str, 1-3 word arc tag like "opening" / "first connection" / "climax")
  story_text         (str, the verbatim text passages that fall in this scene)
  implied_shot_count (int, rough estimate of shots needed to cover the scene)
  key_actions        (list of strings, the discrete physical actions that drive the scene)
  era                (str|null, time period — e.g. "公元 413 年" / "5th c. CE" / "modern day" / "2099"; null if unknowable)
  region             (str|null, place/setting — e.g. "长安·逍遥园译场" / "kucha, western regions" / "modern Shanghai"; null if unknowable)
  culture            (str|null, short snake_case cultural/visual context tag — e.g. "later_qin_chang_an_buddhist" / "modern_urban_chinese" / "kuchean_buddhist"; null if unknowable)

Infer era / region / culture from the scene's own content (period cues,
place names, characters, props) — do NOT assume any specific film.

Return ONE JSON object: {"scenes": [...]}. No prose, no markdown fences."""


def _build_user_prompt(script_text: str) -> str:
    return f"Split the following script into scenes. Return JSON only.\n\n---\n\n{script_text}"


def split_llm(text: str, model_key: str = GENERATOR_DEFAULT,
              locations_dir: Path | None = None,
              legacy_scenes_file: Path | None = None,
              models_file: Path | None = None) -> list[dict]:
    """Use an LLM to split prose/screenplay into scene chunks.

    Reuses the dispatchers from llm_client.py so model routing matches the
    rest of the pipeline (default model registry: paths.MODELS_FILE).
    """
    from pace_core.llm_client import call_model, strip_fences  # type: ignore

    # One registry, resolved the way every other path in this codebase is.
    #
    # This used to try five hardcoded relative locations in turn, four of
    # which never existed (PIPELINE_DIR is this module's own directory), so
    # in practice it fell through to `production/kb/behind_scene/models.json`
    # and worked only when the process happened to be running from the repo
    # root. That was a second registry as well as a hardcoded path: the
    # studio's model picker lists `paths.MODELS_FILE`, so the models offered
    # in the UI and the models this splitter could resolve were two different
    # sets, and the three RouterBase entries the breakdown actually defaults
    # to were unofferable.
    from pace_core.paths import MODELS_FILE
    mf = Path(models_file) if models_file else MODELS_FILE
    if not mf.exists():
        raise FileNotFoundError(
            f"LLM models registry not found at {mf}. Pass `models_file`, or "
            f"set $MODELS_FILE."
        )

    models = json.loads(mf.read_text())
    print(f"[split_llm] using models registry: {mf} with keys: {list(models.keys())}", file=sys.stderr)
    if model_key not in models:
        raise ValueError(f"unknown model '{model_key}'. options: {list(models.keys())}")
    cfg = models[model_key]

    chunks = _chunk_script(text)
    print(f"[split_llm] calling model: {model_key} over {len(chunks)} chunk(s) "
          f"of {len(text)} chars", file=sys.stderr)

    scenes_acc: list[dict] = []
    cost = 0.0
    for i, chunk in enumerate(chunks):
        head = ("This is part {} of {} of a longer script. It may begin or end "
                "part-way through a scene; describe what is here and do not "
                "invent a boundary at the edges.\n\n").format(i + 1, len(chunks)) \
               if len(chunks) > 1 else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": head + _build_user_prompt(chunk)},
        ]
        try:
            raw, c = call_model(cfg, messages)
        except SystemExit as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise ValueError(
                f"chunk {i + 1}/{len(chunks)} ({len(chunk)} chars): {e}") from e
        cost += c
        try:
            part = json.loads(strip_fences(raw)).get("scenes") or []
        except json.JSONDecodeError as e:
            raise ValueError(
                f"chunk {i + 1}/{len(chunks)} returned unparseable JSON: {e}") from e
        # A scene split across the cut is one scene, not two.
        if scenes_acc and part and _same_scene(scenes_acc[-1], part[0]):
            scenes_acc[-1] = _join_scenes(scenes_acc[-1], part[0])
            part = part[1:]
        scenes_acc.extend(part)
        print(f"  [llm] chunk {i + 1}/{len(chunks)}: {len(part)} scene(s) "
              f"(running total {len(scenes_acc)})  ${c:.4f}", file=sys.stderr)

    for n, sc in enumerate(scenes_acc, 1):
        sc["scene_number"] = n
    raw = json.dumps({"scenes": scenes_acc}, ensure_ascii=False)

    print(f"  [llm] model={model_key} cost=${cost:.4f}", file=sys.stderr)
    raw = strip_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nfirst 500 chars: {raw[:500]}")

    scenes = data["scenes"] if isinstance(data, dict) else data
    if not isinstance(scenes, list):
        raise ValueError(f"expected list of scenes, got {type(scenes)}")

    locations = _load_locations(locations_dir or DEFAULTS["locations_dir"], legacy_scenes_file)
    for i, sc in enumerate(scenes):
        sc.setdefault("scene_number", i + 1)
        sc["scene_id"] = f"scene_{sc['scene_number']:02d}"
        sc["location_ref"] = _match_location(sc.get("location_raw", ""), locations)
    return scenes


# ─────────────────────────  public API  ──────────────────────────────


def script_bytes_to_text(raw: bytes, filename: str = "script.md") -> tuple[str, str]:
    """Decode uploaded script bytes into text.

    Supports the same source kinds as the Studio upload path: .md, .txt, .pdf.
    Returns (text, source_label). This is intentionally library-level so API
    servers do not need to duplicate byte/PDF decoding before splitting.
    """
    ext = Path(filename or "script.md").suffix.lower()
    source_label = filename or f"<bytes:{len(raw)}>"
    if raw.lstrip().startswith(b"%PDF"):
        ext = ".pdf"
    if not ext:
        ext = ".md"
    if ext == ".pdf":
        import tempfile
        from pypdf import PdfReader
        tmp = Path(tempfile.gettempdir()) / f"split_script_{os.getpid()}_{datetime.now(timezone.utc).timestamp()}.pdf"
        try:
            tmp.write_bytes(raw)
            reader = PdfReader(str(tmp))
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        if not text:
            raise ValueError("PDF contained no extractable text (scanned image?)")
        return text, source_label
    if ext not in (".md", ".txt"):
        raise ValueError(f"unsupported script type {ext!r}; accepted: .pdf .md .txt")
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("script text is empty")
    return text, source_label


def split_text_to_pai(text: str, *, source_label: str = "<memory>",
                      model: str = "claude-sonnet-5-rb",
                      output_dir: Path | str | None = None,
                      locations_dir: Path | str | None = None,
                      models_file: Path | str | None = None,
                      legacy_scenes_file: Path | str | None = None,
                      dry_run: bool = False) -> dict:
    """Split script text (via LLM) and optionally write scene_NN.json files.

    This is the non-CLI API used by studio_server background jobs. It mirrors
    main() but returns structured metadata instead of exiting the process.
    """
    out_dir = Path(output_dir) if output_dir else DEFAULTS["out_dir"]
    loc_dir = Path(locations_dir) if locations_dir else DEFAULTS["locations_dir"]
    mf = Path(models_file) if models_file else None
    legacy = Path(legacy_scenes_file) if legacy_scenes_file else None

    logging.info('[split_script] LLM split (model=%s)', model)
    scenes = split_llm(text, model_key=model, locations_dir=loc_dir,
                       legacy_scenes_file=legacy, models_file=mf)

    if not scenes:
        raise ValueError("split produced no scenes")

    unified_docs = [_to_pai_1_1(sc, source_script=source_label) for sc in scenes]

    out = {
        "_meta": {
            "source_script":   source_label,
            "schema_version":  "pai-1.1",
            "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_used":      model,
            "scene_count":     len(scenes),
        },
        "scenes": unified_docs,
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for unified in unified_docs:
            sid = unified.get("scene_id") or f"scene_{unified.get('scene_number', '?')}"
            path = out_dir / f"{sid}.json"
            path.write_text(json.dumps(unified, indent=2, ensure_ascii=False))
    return out


# ─────────────────────────  CLI  ──────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("script_path", help="path to script.md / .txt; use '-' for stdin")
    ap.add_argument("--model", default="claude-sonnet-5-rb",
                    help="model key from the models registry")
    ap.add_argument("--output-dir", "--output", default=None,
                    help=f"dir to write per-scene JSONs. Default: {DEFAULTS['out_dir']}")
    ap.add_argument("--locations-dir", default=None,
                    help=f"dir of location bibles to match scene locations against. "
                         f"Default: {DEFAULTS['locations_dir']}")
    ap.add_argument("--models-file", default=None,
                    help=f"LLM model registry JSON. "
                         f"Default: $MODELS_FILE ({_models_file_default()})")
    ap.add_argument("--legacy-scenes-file", default=None,
                    help="optional legacy kb/on_scene/scenes.json for fallback location matching")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write files; print summary to stderr + JSON to stdout if --print")
    ap.add_argument("--print", action="store_true", help="also print the JSON to stdout")
    args = ap.parse_args()

    out_dir       = Path(args.output_dir)        if args.output_dir       else DEFAULTS["out_dir"]
    locations_dir = Path(args.locations_dir)     if args.locations_dir    else DEFAULTS["locations_dir"]
    models_file   = Path(args.models_file)       if args.models_file      else None
    legacy_scenes = Path(args.legacy_scenes_file) if args.legacy_scenes_file else None

    if args.script_path == "-":
        text = sys.stdin.read()
        src_label = "<stdin>"
    else:
        text = Path(args.script_path).read_text()
        src_label = str(Path(args.script_path).resolve())

    scenes = split_llm(text, model_key=args.model,
                       locations_dir=locations_dir,
                       legacy_scenes_file=legacy_scenes,
                       models_file=models_file)

    unified_docs = [_to_pai_1_1(sc, source_script=src_label) for sc in scenes]

    out = {
        "_meta": {
            "source_script":   src_label,
            "schema_version":  "pai-1.1",
            "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_used":      args.model,
            "scene_count":     len(scenes),
        },
        "scenes": unified_docs,
    }

    if args.dry_run:
        print(f"[dry-run] would write {len(unified_docs)} scene file(s) → {out_dir}/",
              file=sys.stderr)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for unified in unified_docs:
            sid  = unified.get("scene_id") or f"scene_{unified.get('scene_number', '?')}"
            path = out_dir / f"{sid}.json"
            path.write_text(json.dumps(unified, indent=2, ensure_ascii=False))
        print(f"wrote {len(unified_docs)} PAI 1.1 scene file(s) → {out_dir}/",
              file=sys.stderr)

    if args.print:
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
