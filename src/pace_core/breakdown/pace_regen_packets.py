#!/usr/bin/env python3
"""Prepare PACE-bound regeneration packets for a PAI project.

This script does not render images or solve SMPL-X. It derives the reviewable
handoff artifacts needed before those expensive stages:

1. anchors for characters, props, and locations from the project library;
2. panel-by-panel storyboard prompt packets compiled through the real compiler;
3. coarse SMPL-X vector requests tied to PACE subjects/actions;
4. camera work items tied to composition targets and anchor readiness.
"""
from __future__ import annotations

import argparse
import json
import re
import re as _re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from pace_core.compilers.compile_common import (  # noqa: E402
    CompileContext, film_of,
)
from pace_core.compilers.compile_flux2 import compile_flux_for_base  # noqa: E402
from pace_core.setup.composition_solver import target_xy  # noqa: E402
from pace_core.camera.camera_planner import plan_camera_track  # noqa: E402
from pace_core.pai_compat import angle_of, movement_of, primary_focus_of, shot_size_of, dig  # noqa: E402
from pace_core.paths import iter_canonical_scene_files, paths_for  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MODEL_SUFFIXES = {".glb", ".gltf", ".blend", ".fbx", ".obj"}
MAX_REFS = 12

BODY_JOINTS = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]

SMPLX_PRESETS = {
    "standing_or_still": {"left_shoulder": [0.0, 0.0, -0.75], "right_shoulder": [0.0, 0.0, 0.75]},
    "walking_or_stepping": {"left_hip": [0.35, 0.0, 0.0], "right_hip": [-0.25, 0.0, 0.0], "left_knee": [0.2, 0.0, 0.0], "right_knee": [0.45, 0.0, 0.0], "left_shoulder": [0.0, 0.0, -0.65], "right_shoulder": [0.0, 0.0, 0.65]},
    "upper_body_hand_action": {"spine2": [0.12, 0.0, 0.0], "left_shoulder": [-0.35, 0.0, -0.45], "right_shoulder": [-0.45, 0.0, 0.25], "left_elbow": [0.45, 0.0, 0.0], "right_elbow": [0.85, 0.0, 0.0]},
    "seated": {"left_hip": [-1.15, 0.0, 0.0], "right_hip": [-1.15, 0.0, 0.0], "left_knee": [1.25, 0.0, 0.0], "right_knee": [1.25, 0.0, 0.0], "left_shoulder": [0.0, 0.0, -0.65], "right_shoulder": [0.0, 0.0, 0.65]},
    "kneeling": {"left_hip": [-0.9, 0.0, 0.0], "right_hip": [-0.9, 0.0, 0.0], "left_knee": [1.7, 0.0, 0.0], "right_knee": [1.7, 0.0, 0.0], "left_ankle": [-0.45, 0.0, 0.0], "right_ankle": [-0.45, 0.0, 0.0]},
    "lying": {"spine1": [0.0, 0.0, 0.0], "left_shoulder": [0.0, 0.0, -0.85], "right_shoulder": [0.0, 0.0, 0.85]},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def local_exists(project_root: Path, maybe_path: Any) -> bool:
    if not maybe_path:
        return False
    if isinstance(maybe_path, dict):
        maybe_path = maybe_path.get("path") or maybe_path.get("file") or maybe_path.get("filename")
    if not isinstance(maybe_path, str):
        return False
    p = Path(maybe_path)
    return p.exists() if p.is_absolute() else (project_root / p).exists()


def rel_project_path(project_root: Path, maybe_path: Any) -> str | None:
    if not maybe_path:
        return None
    if isinstance(maybe_path, dict):
        maybe_path = maybe_path.get("path") or maybe_path.get("file") or maybe_path.get("filename")
    if not isinstance(maybe_path, str):
        return None
    p = Path(maybe_path)
    if p.is_absolute():
        return str(p)
    # A registered path may already be written from the repo root rather than
    # from the project. Joining it to the project root again produced
    # production/projects/X/production/projects/X/... -- a path that resolves
    # to nothing, carried on every panel that referenced the prop.
    if str(p).startswith(str(project_root)):
        return rel(p)
    return rel(project_root / p)


def tokenize(value: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", value.lower()) if len(t) >= 3]


def scan_files(project_root: Path) -> tuple[list[Path], list[Path]]:
    roots = [project_root / "06_reference", project_root / "meshes"]
    images: list[Path] = []
    models: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                images.append(path)
            elif suffix in MODEL_SUFFIXES:
                models.append(path)
    return images, models


def matching_paths(paths: list[Path], tokens: list[str], *, exclude_storyboards: bool = False, limit: int = MAX_REFS) -> list[str]:
    if not tokens:
        return []
    scored: list[tuple[int, str, Path]] = []
    for path in paths:
        hay = str(path).lower()
        if exclude_storyboards and "/storyboards/" in hay:
            continue
        score = sum(1 for token in tokens if token in hay)
        if score:
            scored.append((score, str(path), path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [rel(path) for _, _, path in scored[:limit]]


def scene_location_ref(scene: dict[str, Any]) -> str | None:
    nm = scene.get("narrative_meta") or {}
    return nm.get("location_ref") or scene.get("location_ref")


def shots_of(scene: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in scene.get("shots") or [] if isinstance(s, dict)]


def panels_of(shot: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in shot.get("panels") or [] if isinstance(p, dict)]


def subjects_of(shot: dict[str, Any], panel: dict[str, Any]) -> list[dict[str, Any]]:
    # A panel narrows its cast through `setup_override` (a shot split into
    # one single per subject); reading only `setup` fell through to the
    # shot's whole cast and counted every split single as the full group.
    for setup in ((panel.get("setup_override") or {}), (panel.get("setup") or {}),
                  (shot.get("setup") or {})):
        subs = setup.get("subjects") or []
        if isinstance(subs, list) and subs:
            return [s for s in subs if isinstance(s, dict)]
    return []


def props_of(shot: dict[str, Any], panel: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obj in (shot, panel):
        out.extend(p for p in ((obj.get("setup") or {}).get("props") or []) if isinstance(p, dict))
    return out


def panel_id_for(scene_id: str, shot: dict[str, Any], panel: dict[str, Any], ordinal: int) -> str:
    return str(panel.get("id") or f"{scene_id}_{shot.get('shot_id', 'shot')}_panel_{panel.get('panel_number') or ordinal:04d}")


def subject_ref(subject: dict[str, Any]) -> str | None:
    cid = subject.get("character_id") or subject.get("ref")
    if not cid:
        return None
    age = subject.get("age_state")
    return f"{cid}@{age}" if age else str(cid)


def focus_subject(subjects: list[dict[str, Any]], shot: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any] | None:
    pf = primary_focus_of(panel, shot) or {}
    wanted = pf.get("of_character") or pf.get("ref")
    if wanted:
        base = str(wanted).split("@", 1)[0]
        for sub in subjects:
            if subject_ref(sub) == wanted or sub.get("character_id") == base:
                return sub
    if subjects:
        return subjects[0]
    if pf.get("type") in {"environment", "location", "vehicle", "prop"}:
        return {
            "ref": wanted or pf.get("type"),
            "screen_position": pf.get("screen_position") or {"zone": "center", "x": 0.5, "y": 0.52, "depth": "midground"},
            "_focus_kind": pf.get("type"),
        }
    return None


def lora_records(entry: dict[str, Any]) -> dict[str, Any]:
    lora = entry.get("lora") if isinstance(entry, dict) else None
    if not isinstance(lora, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(lora.get("canonical"), dict):
        out["canonical"] = lora["canonical"]
    if isinstance(lora.get("by_age_state"), dict):
        out["by_age_state"] = lora["by_age_state"]
    if lora.get("path"):
        out["default"] = {k: v for k, v in lora.items() if k in {"path", "trigger_word", "weight_single"}}
    return out


def build_character_anchors(project_root: Path, chars: dict[str, Any], images: list[Path]) -> dict[str, Any]:
    anchors: dict[str, Any] = {}
    for cid, entry in chars.items():
        if cid.startswith("_") or not isinstance(entry, dict):
            continue
        age_states = entry.get("age_states") if isinstance(entry.get("age_states"), dict) else {}
        age_keys = sorted(age_states.keys()) or [None]
        age_refs: dict[str, Any] = {}
        registered_refs = entry.get("reference_images_by_age_state") if isinstance(entry.get("reference_images_by_age_state"), dict) else {}
        for age in age_keys:
            age_key = age or "default"
            tokens = tokenize(" ".join([cid, age or "", entry.get("trigger") or "", (entry.get("lora") or {}).get("trigger_word") or ""]))
            refs = []
            for registered in registered_refs.get(age or "", []) if isinstance(registered_refs, dict) else []:
                rp = rel_project_path(project_root, registered)
                if rp:
                    refs.append(rp)
            # Token overlap alone is not identity. "alice@adult_30" tokenizes
            # to {alice, adult, 30}, and "bob_adult_30_identity_plate.png"
            # matches two of those three — so a character's reference set
            # quietly acquired other people's identity plates, and the redux
            # channel conditioned a three-person panel on four adults from
            # two different families. The character's own name must appear.
            own = {cid.lower()}
            for alias in (entry.get("aliases") or []):
                if isinstance(alias, str) and alias.strip():
                    own.add(alias.strip().lower())
            for key in ("trigger", "name", "name_en"):
                v = entry.get(key)
                if isinstance(v, str) and v.strip():
                    own.add(v.strip().lower())
            refs.extend(x for x in matching_paths(images, tokens, exclude_storyboards=True)
                        if x not in refs and any(o in str(x).lower() for o in own))
            lora = lora_records(entry)
            lora_for_age = None
            if age and isinstance(lora.get("by_age_state"), dict):
                lora_for_age = lora["by_age_state"].get(age)
            if not lora_for_age:
                lora_for_age = lora.get("default") or lora.get("canonical")
            body_proxy = entry.get("body_proxy_3d") if isinstance(entry.get("body_proxy_3d"), dict) else None
            body_proxy_path = rel_project_path(project_root, body_proxy.get("mesh_file")) if body_proxy else None
            text_anchor = None
            generic = entry.get("generic_anchors") if isinstance(entry.get("generic_anchors"), dict) else {}
            if age and generic.get(age):
                text_anchor = generic[age]
            else:
                text_anchor = entry.get("generic_anchor") or entry.get("anchor") or (age_states.get(age) if age else None)
            ready_bits = [bool(text_anchor), bool(refs or lora_for_age), bool(body_proxy_path and local_exists(project_root, body_proxy.get("mesh_file") if body_proxy else None))]
            age_refs[age_key] = {
                "text_anchor": text_anchor,
                "reference_images": refs[:MAX_REFS],
                "lora": lora_for_age,
                "body_proxy_3d": body_proxy_path,
                "anchor_status": "ready" if all(ready_bits) else ("partial" if any(ready_bits) else "missing"),
            }
        anchors[cid] = {
            "tier": entry.get("tier"),
            "trigger": entry.get("trigger"),
            "age_states": age_refs,
        }
    return anchors


def build_prop_anchors(project_root: Path, props: dict[str, Any], images: list[Path]) -> dict[str, Any]:
    entries = props.get("props") if isinstance(props.get("props"), dict) else props
    anchors: dict[str, Any] = {}
    for pid, entry in entries.items():
        if str(pid).startswith("_") or not isinstance(entry, dict):
            continue
        refs = []
        for registered in entry.get("reference_images") or []:
            rp = rel_project_path(project_root, registered)
            if rp:
                refs.append(rp)
        tokens = tokenize(" ".join([pid, entry.get("name") or "", entry.get("anchor") or ""]))
        # As with characters, token overlap is not identity: the anchor prose
        # for a handheld device shares words with the car it is used in, so
        # "game_device" collected the location plates of two different cars.
        # A prop's reference must name the prop.
        own = {str(pid).lower()}
        nm = entry.get("name")
        if isinstance(nm, str) and nm.strip():
            own.add(nm.strip().lower().replace(" ", "_"))
        seen = {str(r) for r in refs}
        for x in matching_paths(images, tokens, exclude_storyboards=True):
            # str(): refs holds resolved strings while matching_paths yields
            # Paths, so `x not in refs` never fired and a registered plate was
            # re-added by token match.
            if str(x) not in seen and any(o in str(x).lower() for o in own):
                refs.append(x)
                seen.add(str(x))
        model = entry.get("model_3d") if isinstance(entry.get("model_3d"), dict) else None
        model_path = rel_project_path(project_root, model.get("mesh_file")) if model else None
        anchors[pid] = {
            "name": entry.get("name"),
            "anchor": entry.get("anchor"),
            "reference_images": refs[:MAX_REFS],
            "model_3d": model_path,
            "anchor_status": "ready" if (entry.get("anchor") and (refs or model_path)) else ("partial" if entry.get("anchor") or refs or model_path else "missing"),
        }
    return anchors


def build_location_anchors(project_root: Path, loc_stubs: dict[str, Any], images: list[Path], scene_to_loc: dict[str, str]) -> dict[str, Any]:
    storyboard_images = [p for p in images if "/storyboards/" in str(p)]
    # Location anchors must be actual environment/location images. Character
    # sheets, LoRA portraits, and prop texture maps are not valid location
    # anchors even when words like a region name or "lamp" happen to match.
    location_images = [
        p for p in images
        if any(part in str(p).lower() for part in (
            "/06_reference/locations/",
            "/06_reference/location/",
            "/06_reference/environments/",
            "/06_reference/environment/",
            "/06_reference/plates/",
        ))
    ]
    anchors: dict[str, Any] = {}
    loc_to_scenes: dict[str, list[str]] = defaultdict(list)
    for scene_id, loc in scene_to_loc.items():
        loc_to_scenes[loc].append(scene_id)
    for loc_id, stub in loc_stubs.items():
        tokens = tokenize(" ".join([loc_id, str(stub)]))
        # Restricting to the locations directory is not enough: every plate in
        # it shares the tokens "car", "location" and "plate", so one interior
        # collected the plates of three unrelated environments and the panels
        # of a single scene were each conditioned on a different room. A
        # location's reference must name that location.
        refs = [x for x in matching_paths(location_images, tokens, exclude_storyboards=True)
                if str(loc_id).lower() in str(x).lower()]
        candidates: list[str] = []
        for scene_id in loc_to_scenes.get(loc_id, [])[:8]:
            scene_matches = sorted(
                p for p in storyboard_images
                if p.name.startswith(f"{scene_id}_") or p.name.startswith(f"{scene_id}__")
            )
            for path in scene_matches[:3]:
                item = rel(path)
                if item not in candidates:
                    candidates.append(item)
        mesh = stub.get("location_mesh") if isinstance(stub.get("location_mesh"), dict) else None
        mesh_path = rel_project_path(project_root, mesh.get("mesh_file")) if mesh else None
        anchors[loc_id] = {
            "text_anchor": stub,
            "reference_images": refs[:MAX_REFS],
            "panel_visual_candidates": candidates[:MAX_REFS],
            "scene_ids": sorted(loc_to_scenes.get(loc_id, [])),
            "location_mesh": mesh_path,
            "anchor_status": "ready" if refs else "textual_only",
            "note": None if refs else "No explicit location/environment reference image was found in the library; panel_visual_candidates are generated storyboard evidence only and need human approval before reuse.",
        }
    return anchors


# Free-text fields an action can be described in. Serialising the whole
# events object instead would put schema keys and enum values into the text
# the pose classifier reads, and one of them collides: "intensity" contains
# "sit", which classified every subject with an intensity as seated.
_ACTION_TEXT_FIELDS = ("standalone", "description_en", "description_zh",
                       "description", "text", "label", "beat", "summary")


def action_text(shot: dict[str, Any], panel: dict[str, Any]) -> str:
    """Human-readable action description only, never the serialised structure."""
    bits: list[str] = []

    def harvest(val):
        if isinstance(val, str):
            bits.append(val)
        elif isinstance(val, dict):
            for k in _ACTION_TEXT_FIELDS:
                if isinstance(val.get(k), str):
                    bits.append(val[k])
        elif isinstance(val, list):
            for item in val:
                harvest(item)

    for obj in (shot.get("events") or {}, panel.get("events") or {}):
        for key in ("actions", "emotional_beats", "changes"):
            harvest(obj.get(key))
    return " ".join(bits)


def _hit(text: str, words: list[str]) -> bool:
    """Whole-word match for latin terms, substring for CJK (which has no
    word boundaries). A bare substring test is what let "intensity" match
    "sit"; a cue has to be a word to be evidence."""
    for w in words:
        if re.search(r"[a-z]", w):
            if re.search(rf"\b{re.escape(w)}\w*\b", text):
                return True
        elif w in text:
            return True
    return False


# Nouns that can carry a pose verb without having a body. "his device lying
# face-down on the tray" is not a person lying down.
_INANIMATE = ("device", "console", "tray", "screen", "panel", "phone", "tablet",
              "controller", "scroll", "door", "window", "lamp", "candle", "cup",
              "seat", "bag", "book", "sign", "light", "camera")

_POSE_CUES = ("kneel", "lie", "lying", "lies", "walk", "step", "approach", "enter",
              "ride", "run", "reach", "hold", "touch", "write", "carve", "hand",
              "play", "activate", "sit", "seated", "sitting", "stand", "leaning",
              "bent", "twisted")


def _mask_cues_governed_by_objects(text: str) -> str:
    """Blank pose cues whose nearest preceding noun is a thing, not a body.

    A pose cue is evidence about a body only if a body is what is doing it.
    Scoring the raw beat let a prop's verb become the cast's posture: the beat
    "his device lying face-down on the tray" labelled all three subjects as
    lying down, and the SMPL-X seed laid them on their backs in a moving car.
    Clause-level filtering is not enough, because the possessive keeps a person
    in the clause ("his device") while the noun governing the verb is the
    device. So the window is the cue's own neighbourhood.

    Same shape as the earlier defect where the classifier read the serialised
    event record and matched "sit" inside "intensity": a cue lifted out of the
    context that gave it meaning.
    """
    words = re.findall(r"\w+|\W+", text)
    # Prefix match, because _hit() matches cues as \bcue\w*\b. Exact-token
    # matching here let "the console that runs low along the window line" past
    # the mask -- "runs" is not "run" -- and the console's motion verb became
    # the cast's posture: three seated people labelled walking_or_stepping.
    idx = [i for i, w in enumerate(words)
           if any(w.strip().lower().startswith(c) for c in _POSE_CUES if w.strip())]
    for i in idx:
        window = [w.strip().lower() for w in words[max(0, i - 6):i] if w.strip()]
        if any(w in _INANIMATE for w in window[-3:]):
            words[i] = " "
    return "".join(words)


def pose_label_from_action(text: str) -> str:
    lo = _mask_cues_governed_by_objects(text).lower()
    if _hit(lo, ["kneel", "跪"]):
        return "kneeling"
    if _hit(lo, ["lie", "lying", "lies", "躺", "倒"]):
        return "lying"
    if _hit(lo, ["walk", "step", "approach", "enter", "ride", "run", "走", "进", "骑", "跑"]):
        return "walking_or_stepping"
    if _hit(lo, ["reach", "hold", "touch", "write", "carve", "hand", "play", "activate",
                 "palm", "伸", "握", "写", "刻"]):
        return "upper_body_hand_action"
    if _hit(lo, ["sit", "seated", "sitting", "seat", "坐"]):
        return "seated"
    return "standing_or_still"


def body_pose_for_hint(pose_hint: str) -> list[list[float]]:
    body = [[0.0, 0.0, 0.0] for _ in BODY_JOINTS]
    preset = SMPLX_PRESETS.get(pose_hint) or SMPLX_PRESETS["standing_or_still"]
    for joint, axis_angle in preset.items():
        if joint in BODY_JOINTS:
            body[BODY_JOINTS.index(joint)] = [float(x) for x in axis_angle]
    return body


def flatten(values: list[list[float]]) -> list[float]:
    return [float(x) for row in values for x in row]


def motionx_322_frame(body: list[list[float]], *, transl: list[float] | None = None) -> list[float]:
    """Build one Motion-X smplx_322-compatible frame.

    Motion-X stores 322-D SMPL-X parameters as root_orient(3), pose_body(63),
    pose_hand(90), pose_jaw(3), face_expr(50), face_shape(100), trans(3),
    betas(10).
    """
    frame = []
    frame.extend([0.0, 0.0, 0.0])
    frame.extend(flatten(body))
    frame.extend([0.0] * 90)
    frame.extend([0.0, 0.0, 0.0])
    frame.extend([0.0] * 50)
    frame.extend([0.0] * 100)
    frame.extend(transl or [0.0, 0.0, 0.0])
    frame.extend([0.0] * 10)
    assert len(frame) == 322
    return frame


# Postures a location implies regardless of what the action text says. A
# screenplay writes "family banter", never "the family sits", because being
# seated is entailed by being in a car; inferring posture from the action
# alone therefore stands everyone up inside a moving vehicle.
_LOCATION_POSTURE = (
    (("car", "vehicle", "cockpit", "cabin", "seat", "驾驶", "车内", "车"), "seated"),
    (("bed", "床"), "lying"),
)


def posture_from_location(location_ref: str | None) -> str | None:
    if not location_ref:
        return None
    lo = str(location_ref).lower()
    for cues, posture in _LOCATION_POSTURE:
        if any(c in lo for c in cues):
            return posture
    return None



# SMPL-X beta 0 is overall size and beta 1 overall build; the rest are finer
# shape axes a coarse stage has no basis to set. Deriving the first two from
# the character's own age state and build makes a body specific to a person
# rather than a neutral adult, and keeps it constant wherever that person
# appears, which is what a shape parameter is for.
_BUILD_BETA1 = {
    "slender": -0.9, "slim": -0.9, "small": -0.6, "petite": -0.9,
    "athletic": 0.6, "muscular": 1.1, "stocky": 1.0, "heavy": 1.4,
    "average": 0.0, "lean": -0.7, "gaunt": -1.2,
}


# Where beta[4] has to sit for the chest to come out flat, as a function of
# build. A fixed offset is calibrated at one build and drifts everywhere else:
# at -2.0 a zero-build body lands at -0.4 mm of bust protrusion, but the
# corpus's slim eighteen-year-old kept +21.5 mm, because build and sex act on
# the same tissue. Solving for the zero crossing across build gives a straight
# line, residuals within 0.3 over beta[1] in [-1, 1]:
#
#     build b1   -1.00  -0.50   0.00   0.50   1.00
#     b4 at 0    -6.95  -3.55  -2.30  -1.05  -0.65
#
# Measured on what the bake WRITES, not on the raw model: the proxy is posed
# with its arms down, transformed to Z-up and scaled to a target stature, and
# stature scaling alone rescales every absolute millimetre. Fitting on the
# rest-pose mesh and checking on the baked one is how an earlier version of
# this left the slim character at +14 mm while claiming to zero him.
#
# Clamped at 0 because a male body should never have chest ADDED, and at -4
# because past there the shape leaves the range the model was fit over. The
# clamp binds below about build -0.6, where the true crossing runs away from
# the line; for a heavy-build male proxy -4 lands at +0.25 mm, but a
# slimmer one would keep some protrusion rather than reach zero.
_SEX_B4_SLOPE, _SEX_B4_INTERCEPT = 2.723, -2.706


def betas_for_subject(ref: str, characters_kb: dict) -> list[float]:
    """Per-character body shape, constant across every panel they appear in.

    Leaving betas at zero gives an eight-year-old the same body as an adult,
    so a shape parameter that exists precisely to distinguish people
    distinguishes nobody.
    """
    from pace_core.pai_compat import resolve_character
    name, _, age_state = str(ref or "").partition("@")
    entry = resolve_character(name, characters_kb or {}) or {}
    betas = [0.0] * 10

    yrs = None
    m = _re.search(r"(\d{1,3})", age_state or "")
    if m:
        yrs = int(m.group(1))
    if yrs is None:
        yrs = (entry.get("_extracted") or {}).get("age_estimate")
    if isinstance(yrs, (int, float)):
        # Height-like axis: a child is markedly smaller, an elder slightly so.
        if yrs <= 12:
            betas[0] = -2.2 + (yrs / 12.0) * 0.8
        elif yrs <= 19:
            betas[0] = -0.8
        elif yrs >= 65:
            betas[0] = -0.3
        else:
            betas[0] = 0.0

    text = " ".join(str(entry.get(k) or "") for k in ("anchor", "generic_anchor")).lower()
    for word, val in _BUILD_BETA1.items():
        if _re.search(rf"\b{word}\b", text):
            betas[1] = val
            break

    # Sex-typical shape, on the axis that carries it WHEN THE BODY DOES NOT.
    #
    # This is a fallback, not the mechanism. With the gendered SMPL-X models
    # installed the template is already male -- -9.2 mm at the chest before any
    # coefficient -- and a consumer that has them should drop this value rather
    # than add it, which drives a body past male instead of to it. The bake
    # does exactly that. What is left here is for consumers with only the
    # neutral model, the motion handoff among them.
    #
    # The gendered SMPL-X models are licence-gated separately and are often not
    # installed, so `gender="neutral"` is what a body gets built from. That is
    # not the same as having no sex: the neutral shape space is learned over
    # both, and the dimorphism is in there as an unlabelled direction. Sweeping
    # each coefficient and measuring bust protrusion (chest depth at bust height
    # minus at underbust) and the shoulder-to-hip width ratio finds it at
    # beta[4]: at 0 the mean body carries a 9 mm bust and a 2.61 ratio, and at
    # -2.0 that is 0 mm and 3.05, with stature moving 2 mm, so it does not
    # fight the stature normalisation the proxies are baked with. beta[3],
    # beta[5] and beta[6] flatten the chest too but leave the shoulders alone;
    # beta[0] and beta[1] are taken here by age and build.
    #
    # Female is left at zero rather than pushed positive: the neutral mean
    # already reads female at the chest, which is exactly why every character
    # in a corpus staged from it looked like one person.
    if str(entry.get("sex") or "").lower() == "male":
        betas[4] = max(-4.0, min(0.0, _SEX_B4_SLOPE * betas[1] + _SEX_B4_INTERCEPT))
    return [round(b, 3) for b in betas]


_PERFORMER_FIELDS = ("subject_ref", "character_ref", "of_character", "performer",
                     "actor", "who", "subject")


def _names_subject(entry: dict, cid: str, aliases: set) -> bool:
    """Does this action name that person, either by field or in its prose?"""
    for k in _PERFORMER_FIELDS:
        v = entry.get(k)
        if isinstance(v, str) and v.partition("@")[0].strip().lower() == cid:
            return True
        if isinstance(v, list) and any(
                isinstance(x, str) and x.partition("@")[0].strip().lower() == cid
                for x in v):
            return True
    prose = " ".join(str(entry.get(k) or "") for k in _ACTION_TEXT_FIELDS).lower()
    return any(_re.search(rf"\b{_re.escape(a)}\b", prose) for a in aliases if a)


def action_text_for_subject(subject: dict, shot: dict, panel: dict,
                            characters_kb: dict | None = None) -> str:
    """The actions this person performs, not every action in the shot.

    Pose was resolved once per panel, so every declared subject received the
    same body: a beat in which the parents reach for a device put the child's
    arms out too. An action belongs to whoever it names. When no action in
    the shot names anyone, there is nothing to attribute and the shot-wide
    text stands, which is the behaviour this replaces.
    """
    from pace_core.pai_compat import resolve_character
    ref = str(subject.get("character_id") or subject.get("character_ref") or "")
    cid = ref.partition("@")[0].strip().lower()
    if not cid:
        return action_text(shot, panel)
    entry = resolve_character(cid, characters_kb or {}) or {}
    aliases = {cid}
    for key in ("name", "name_en", "name_zh", "trigger"):
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            aliases.add(v.strip().lower())
    for a in (entry.get("aliases") or []):
        if isinstance(a, str) and a.strip():
            aliases.add(a.strip().lower())

    entries, mine = [], []
    for obj in (shot.get("events") or {}, panel.get("events") or {}):
        for key in ("actions", "emotional_beats", "changes"):
            for item in (obj.get(key) or []):
                if isinstance(item, dict):
                    entries.append(item)
    for e in entries:
        if _names_subject(e, cid, aliases):
            mine.append(" ".join(str(e.get(k) or "") for k in _ACTION_TEXT_FIELDS))
    if mine:
        return " ".join(mine)
    # No action names this person. If some other action names someone, the
    # attribution is real and this subject simply is not performing; only
    # when nothing is attributable does the shot-wide text apply.
    attributed = any(
        any(e.get(k) for k in _PERFORMER_FIELDS) for e in entries)
    return "" if attributed else action_text(shot, panel)


def smplx_motionx_vector(subject: dict[str, Any], shot: dict[str, Any], panel: dict[str, Any], *, frames: int = 30, location_ref: str | None = None, characters_kb: dict | None = None) -> dict[str, Any]:
    # Posture and gesture are orthogonal and must compose. A single label
    # makes them exclusive, so "parents activate devices" replaced "seated"
    # and stood everyone up inside a moving car: the reaching was recorded
    # and the sitting was lost. The location fixes the lower body, the
    # action fixes the upper body, and the gesture's joints override the
    # posture's where they overlap.
    action_hint = pose_label_from_action(
        action_text_for_subject(subject, shot, panel, characters_kb))
    posture = posture_from_location(location_ref)
    pose_hint = action_hint
    if posture and action_hint in ("standing_or_still", "upper_body_hand_action"):
        merged = dict(SMPLX_PRESETS.get(posture) or {})
        if action_hint == "upper_body_hand_action":
            merged.update(SMPLX_PRESETS["upper_body_hand_action"])
            pose_hint = f"{posture}+hand_action"
        else:
            pose_hint = posture
        body = [list(merged.get(j, [0.0, 0.0, 0.0])) for j in BODY_JOINTS]
    else:
        body = body_pose_for_hint(pose_hint)
    # Small deterministic root translation for locomotion hints, so Motion-X
    # consumers receive a temporal sequence rather than a duplicated still pose.
    sequence = []
    for i in range(frames):
        t = i / max(1, frames - 1)
        step = 0.35 * t if pose_hint == "walking_or_stepping" else 0.0
        sequence.append(motionx_322_frame(body, transl=[step, 0.0, 0.0]))
    return {
        "subject_ref": subject_ref(subject),
        "pose_hint": pose_hint,
        "status": "generated_coarse_smplx_motionx_sequence",
        "needs_performance_fit_for_final": True,
        "fusion_policy": {
            "smplx_role": "explicit body, hand, jaw, expression, shape, and translation parameter target",
            "motionx_role": "30fps temporal whole-body sequence contract and semantic motion prior",
            "pace_role": "panel action text, character identity, screen position, and human approval gates constrain the motion",
        },
        "global_orient_axis_angle": [0.0, 0.0, 0.0],
        "transl_m": sequence[-1][309:312],
        "body_pose_axis_angle_21x3": body,
        "left_hand_pose_axis_angle_15x3": [[0.0, 0.0, 0.0] for _ in range(15)],
        "right_hand_pose_axis_angle_15x3": [[0.0, 0.0, 0.0] for _ in range(15)],
        "jaw_pose_axis_angle": [0.0, 0.0, 0.0],
        "expression_50": [0.0 for _ in range(50)],
        "face_shape_100": [0.0 for _ in range(100)],
        "betas_10": betas_for_subject(
            subject_ref(subject) if isinstance(subject, dict) else subject,
            characters_kb or {}),
        "motionx_smplx_322_fps": 30,
        "motionx_smplx_322_frames": sequence,
        "source": "PACE action text mapped to a coarse SMPL-X/Motion-X seed. Replace with Motion-X retrieval/generation or fitted performance for final animation.",
    }


def anchor_for_subject(character_anchors: dict[str, Any], subject: dict[str, Any]) -> dict[str, Any] | None:
    cid = subject.get("character_id") or subject.get("ref")
    if not cid or cid not in character_anchors:
        return None
    age = subject.get("age_state") or "default"
    ages = character_anchors[cid].get("age_states") or {}
    return ages.get(age) or ages.get("default") or next(iter(ages.values()), None)


def atmosphere_of(shot: dict[str, Any]) -> str | None:
    """The mood this shot renders under, or None.

    Read off the RESOLVED shot, so a value declared once in a scene's
    shot_defaults counts for every panel under it. Nothing in the corpus
    currently uses that rung -- every mood is repeated per shot -- which is
    why omitting it on one shot silently leaves that panel with no atmosphere
    at all rather than falling back to its scene's.
    """
    env = (shot.get("setup") or {}).get("environment") or {}
    return (env.get("mood") or "").strip() or None


def p_principles(subjects: list[dict[str, Any]], props_here: list[dict[str, Any]], loc_ref: str | None, shot: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    focus = focus_subject(subjects, shot, panel)
    return {
        "p1_event_delta_present": bool(action_text(shot, panel) or panel.get("description") or panel.get("narrative")),
        "p2_spec_bindings_present": bool((subjects or focus) and loc_ref and shot_size_of(shot)),
        "p3_read_point_resolved": bool(focus and target_xy(focus.get("screen_position"))),
        # Atmosphere is compiled into the prompt by every backend
        # ("with a {mood} atmosphere"), so a panel that resolves none is not
        # neutral -- it hands the sampler the one thing the panel was supposed
        # to say about how the shot should feel. It is reported per panel
        # because it is a per-panel property: the field inherits down
        # scene defaults -> shot -> panel, and a shot that omits it after a
        # sibling set it produces a scene whose panels disagree about the
        # atmosphere while every other check passes.
        "p4_atmosphere_resolved": bool(atmosphere_of(shot)),
        "bindings": {
            "cast": [subject_ref(s) for s in subjects],
            "props": [p.get("prop_id") or p.get("ref") for p in props_here],
            "location_ref": loc_ref,
            "shot_size": shot_size_of(shot),
            "atmosphere": atmosphere_of(shot),
        },
    }


def planner_shot_for_camera(scene_id: str, loc_ref: str | None, shot: dict[str, Any], focus: dict[str, Any] | None) -> dict[str, Any] | None:
    xy = target_xy((focus or {}).get("screen_position")) if focus else None
    if not xy:
        return None
    cam = shot.get("camera") or {}
    ci = cam.get("creative_intent") or {}
    lens = dig(cam, "lens", "focal_length_mm") or dig(cam, "optics", "lens_mm") or dig(cam, "technical", "lens_mm")
    planner = {
        "scene_ref": loc_ref or scene_id,
        "camera": {
            "shot_size": shot_size_of(shot) or ci.get("shot_size") or "medium",
            "angle": angle_of(shot) or "eye_level",
            "lens_mm": lens or 50,
            "aperture": dig(cam, "lens", "aperture") or "",
        },
        "frame": {
            "movement": movement_of(shot),
            "movement_easing": dig(cam, "trajectory", "easing") or "linear",
        },
        "composition": {
            "subject_xy": [0.0, 0.0],
            "target_x": xy[0],
            "target_y": xy[1],
        },
    }
    return planner


def camera_track_for_panel(scene_id: str, loc_ref: str | None, shot: dict[str, Any], focus: dict[str, Any] | None) -> dict[str, Any] | None:
    planner = planner_shot_for_camera(scene_id, loc_ref, shot, focus)
    if planner is None:
        return None
    track = plan_camera_track(planner, n_frames=24)
    return {
        "planner_input": planner,
        "fps": 24,
        "frames": track,
        "start": track[0] if track else None,
        "end": track[-1] if track else None,
    }


DEFAULT_BASE_MODEL = "flux2_dev_fp8mixed.safetensors"


def build_packets(project: str, base_model: str | None) -> dict[str, Any]:
    resolved_base_model = base_model or DEFAULT_BASE_MODEL
    paths = paths_for(project)
    scenes = [read_json(x) for x in iter_canonical_scene_files(paths.scenes_dir)]
    chars = read_json(paths.chars_file) if paths.chars_file.exists() else {}
    props = read_json(paths.props_file) if paths.props_file.exists() else {}
    loc_doc = read_json(paths.loc_stubs_file) if paths.loc_stubs_file.exists() else {}
    loc_stubs = loc_doc.get("stubs") or {}
    images, _models = scan_files(paths.storage)
    scene_to_loc = {str(s.get("scene_id")): scene_location_ref(s) for s in scenes if scene_location_ref(s)}

    character_anchors = build_character_anchors(paths.storage, chars, images)
    prop_anchors = build_prop_anchors(paths.storage, props, images)
    location_anchors = build_location_anchors(paths.storage, loc_stubs, images, scene_to_loc)
    ctx = CompileContext(characters_kb=chars, props_kb=props, location_stubs=loc_stubs,
                         film=film_of(paths.film_file), greyboxes_dir=paths.greyboxes_dir,
                         target_width=1280, target_height=544)

    regen_packets: list[dict[str, Any]] = []
    smplx_plan: list[dict[str, Any]] = []
    camera_worklist: list[dict[str, Any]] = []
    counts = Counter()

    for scene in scenes:
        scene_id = str(scene.get("scene_id"))
        loc_ref = scene_location_ref(scene)
        for shot in shots_of(scene):
            shot_id = str(shot.get("shot_id") or "shot")
            for ordinal, panel in enumerate(panels_of(shot), 1):
                panel_id = panel_id_for(scene_id, shot, panel, ordinal)
                counts["panels"] += 1
                subjects = subjects_of(shot, panel)
                props_here = props_of(shot, panel)
                focus = focus_subject(subjects, shot, panel)
                positive = negative = None
                prompt_error = None
                try:
                    positive, negative = compile_flux_for_base(resolved_base_model, scene, shot, panel, ctx)
                    # The same posture that goes into the SMPL-X vector has to
                    # go into the prompt that draws the panel. Emitting it only
                    # into the body parameters lets the two disagree: two panels
                    # of one scene carried an identical "seated" seed while one
                    # rendered the family standing inside a moving car. The
                    # renderer cannot read the vector, so a posture it is never
                    # told is a posture it is free to invent.
                    posture = posture_from_location(loc_ref)
                    if posture and posture not in (positive or "").lower():
                        positive = f"{positive.rstrip('. ')}. All subjects are {posture}."
                        counts["posture_injected"] += 1
                    counts["prompt_compiled"] += 1
                except Exception as exc:
                    prompt_error = f"{type(exc).__name__}: {exc}"
                    counts["prompt_failed"] += 1
                subject_anchors = []
                for subject in subjects:
                    anchor = anchor_for_subject(character_anchors, subject)
                    subject_anchors.append({
                        "subject_ref": subject_ref(subject),
                        "screen_position": subject.get("screen_position"),
                        "anchor_status": (anchor or {}).get("anchor_status"),
                        "reference_images": (anchor or {}).get("reference_images", [])[:4],
                        "body_proxy_3d": (anchor or {}).get("body_proxy_3d"),
                        "lora": (anchor or {}).get("lora"),
                    })
                prop_refs = []
                for prop in props_here:
                    pid = prop.get("prop_id") or prop.get("ref")
                    if pid:
                        prop_refs.append({"prop_id": pid, **(prop_anchors.get(pid) or {})})
                loc_anchor = location_anchors.get(loc_ref or "", {})
                principle = p_principles(subjects, props_here, loc_ref, shot, panel)
                packet = {
                    "panel_id": panel_id,
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "panel_number": panel.get("panel_number"),
                    "human_in_loop_gates": [
                        "approve_asset_anchors_before_prompt_regen",
                        "approve_storyboard_panel_before_smplx_and_camera",
                        "approve_smplx_proxy_before_camera_finalize",
                        "approve_camera_blocking_before_render",
                    ],
                    "pace_principles": principle,
                    "compiled_storyboard_prompt": {
                        "base_model": resolved_base_model,
                        "positive": positive,
                        "negative": negative,
                        "error": prompt_error,
                    },
                    "anchors": {
                        "characters": subject_anchors,
                        "props": prop_refs,
                        "location": {
                            "location_ref": loc_ref,
                            "anchor_status": loc_anchor.get("anchor_status"),
                            "text_anchor": loc_anchor.get("text_anchor"),
                            "reference_images": loc_anchor.get("reference_images", [])[:4],
                            "panel_visual_candidates": loc_anchor.get("panel_visual_candidates", [])[:4],
                        },
                    },
                }
                regen_packets.append(packet)

                for subject in subjects:
                    smplx_plan.append({
                        "panel_id": panel_id,
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        **smplx_motionx_vector(subject, shot, panel,
                                               location_ref=loc_ref,
                                               characters_kb=chars),
                    })

                focus_anchor = anchor_for_subject(character_anchors, focus or {}) if focus else None
                xy = target_xy((focus or {}).get("screen_position")) if focus else None
                blockers = []
                if not xy:
                    blockers.append("missing_primary_screen_position")
                focus_kind = (focus or {}).get("_focus_kind")
                if focus_kind in {"environment", "location", "vehicle", "prop"}:
                    if not loc_anchor.get("location_mesh"):
                        blockers.append("missing_location_proxy_3d")
                elif not focus_anchor or not focus_anchor.get("body_proxy_3d"):
                    blockers.append("missing_focus_body_proxy_3d")
                if loc_anchor.get("anchor_status") == "textual_only":
                    blockers.append("location_image_anchor_missing")
                camera_track = camera_track_for_panel(scene_id, loc_ref, shot, focus) if not blockers or xy else None
                camera_worklist.append({
                    "panel_id": panel_id,
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "focus_subject": subject_ref(focus or {}) if focus else None,
                    "target_xy": list(xy) if xy else None,
                    "shot_size": shot_size_of(shot),
                    "camera_spec": shot.get("camera") or {},
                    "camera_track": camera_track,
                    "readiness": "ready_for_camera_solve" if not blockers else "blocked",
                    "blockers": blockers,
                })

    return {
        "project": project,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"project_root": rel(paths.storage)},
        "asset_anchor_manifest": {
            "characters": character_anchors,
            "props": prop_anchors,
            "locations": location_anchors,
        },
        "regen_packets": regen_packets,
        "smplx_vector_plan": smplx_plan,
        "camera_worklist": camera_worklist,
        "summary": summarize(character_anchors, prop_anchors, location_anchors, regen_packets, smplx_plan, camera_worklist, counts),
    }


def summarize(chars: dict[str, Any], props: dict[str, Any], locs: dict[str, Any], packets: list[dict[str, Any]], smplx: list[dict[str, Any]], camera: list[dict[str, Any]], counts: Counter) -> dict[str, Any]:
    char_ages = [age for c in chars.values() for age in (c.get("age_states") or {}).values()]
    return {
        "characters": {
            "entries": len(chars),
            "age_variants": len(char_ages),
            "ready_age_variants": sum(1 for a in char_ages if a.get("anchor_status") == "ready"),
            "partial_age_variants": sum(1 for a in char_ages if a.get("anchor_status") == "partial"),
        },
        "props": {
            "entries": len(props),
            "ready_or_partial": sum(1 for p in props.values() if p.get("anchor_status") in {"ready", "partial"}),
            "with_3d_model": sum(1 for p in props.values() if p.get("model_3d")),
        },
        "locations": {
            "entries": len(locs),
            "with_explicit_reference_images": sum(1 for l in locs.values() if l.get("reference_images")),
            "textual_only": sum(1 for l in locs.values() if l.get("anchor_status") == "textual_only"),
        },
        "panels": {
            "total": counts["panels"],
            "prompt_compiled": counts["prompt_compiled"],
            "prompt_failed": counts["prompt_failed"],
            "p1_event_delta_present": sum(1 for p in packets if p["pace_principles"]["p1_event_delta_present"]),
            "p2_spec_bindings_present": sum(1 for p in packets if p["pace_principles"]["p2_spec_bindings_present"]),
            "p3_read_point_resolved": sum(1 for p in packets if p["pace_principles"]["p3_read_point_resolved"]),
            "p4_atmosphere_resolved": sum(1 for p in packets if p["pace_principles"]["p4_atmosphere_resolved"]),
        },
        "smplx": {
            "subject_vectors_requested": len(smplx),
            "coarse_smplx_motionx_sequences": len(smplx),
            "fitted_performance_sequences": 0,
            "note": "These are generated coarse SMPL-X/Motion-X seed sequences, not fitted final performance capture.",
        },
        "camera": {
            "work_items": len(camera),
            "ready_for_camera_solve": sum(1 for c in camera if c.get("readiness") == "ready_for_camera_solve"),
            "blocked": sum(1 for c in camera if c.get("readiness") == "blocked"),
            "blocker_counts": dict(Counter(b for c in camera for b in c.get("blockers", [])).most_common()),
        },
    }


def write_outputs(bundle: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(bundle.get("project") or "project")
    files = {
        f"{prefix}_anchor_manifest.json": bundle["asset_anchor_manifest"],
        f"{prefix}_regen_packets.json": bundle["regen_packets"],
        f"{prefix}_smplx_vector_plan.json": bundle["smplx_vector_plan"],
        f"{prefix}_camera_worklist.json": bundle["camera_worklist"],
        f"{prefix}_pace_regen_bundle.json": bundle,
    }
    for name, payload in files.items():
        (out_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"{prefix}_pace_regen_summary.md").write_text(render_summary(bundle), encoding="utf-8")


def render_summary(bundle: dict[str, Any]) -> str:
    s = bundle["summary"]
    project = bundle.get("project") or "Project"
    lines = [
        f"# {project} PACE Regeneration Prep",
        "",
        f"Generated: `{bundle['generated_at']}`",
        f"Project root: `{bundle['source']['project_root']}`",
        "",
        "## Anchors",
        "",
        f"- Character entries: {s['characters']['entries']}",
        f"- Character age variants ready: {s['characters']['ready_age_variants']} / {s['characters']['age_variants']}",
        f"- Character age variants partial: {s['characters']['partial_age_variants']} / {s['characters']['age_variants']}",
        f"- Props with 3D models: {s['props']['with_3d_model']} / {s['props']['entries']}",
        f"- Locations with explicit reference images: {s['locations']['with_explicit_reference_images']} / {s['locations']['entries']}",
        f"- Locations textual-only: {s['locations']['textual_only']} / {s['locations']['entries']}",
        "",
        "## Panel Packets",
        "",
        f"- Panels prepared: {s['panels']['total']}",
        f"- Storyboard prompts compiled: {s['panels']['prompt_compiled']}",
        f"- P1 event delta present: {s['panels']['p1_event_delta_present']}",
        f"- P2 spec bindings present: {s['panels']['p2_spec_bindings_present']}",
        f"- P3 read point resolved: {s['panels']['p3_read_point_resolved']}",
        f"- P4 atmosphere resolved: {s['panels']['p4_atmosphere_resolved']}",
        "",
        "## SMPL-X And Camera",
        "",
        f"- SMPL-X subject vectors requested: {s['smplx']['subject_vectors_requested']}",
        f"- Coarse SMPL-X/Motion-X sequences generated: {s['smplx']['coarse_smplx_motionx_sequences']}",
        f"- Fitted performance sequences: {s['smplx']['fitted_performance_sequences']}",
        f"- Camera items ready: {s['camera']['ready_for_camera_solve']} / {s['camera']['work_items']}",
        f"- Camera items blocked: {s['camera']['blocked']} / {s['camera']['work_items']}",
        "",
        "## Camera Blockers",
        "",
    ]
    if s["camera"]["blocker_counts"]:
        lines.extend(f"- {k}: {v}" for k, v in s["camera"]["blocker_counts"].items())
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Production Meaning",
        "",
        "This is the phase-gated handoff: approve anchors first, regenerate storyboard panels from the compiled prompts, fit or solve SMPL-X vectors from the approved panels, then finalize camera blocking against the resolved PACE composition target.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    paths = paths_for(args.project)
    out_dir = Path(args.out_dir) if args.out_dir else paths.docs_dir / "pace_experiments"
    bundle = build_packets(args.project, args.base_model)
    write_outputs(bundle, out_dir)
    json.dump(bundle["summary"], sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
