"""Camera planner — single source of truth for per-shot camera geometry.

Pure functions, no I/O. Consumed by:
  - blender_render.py  (sets cam.location, cam.rotation, cam.data.lens)
  - studio_server.py /api/shot/<s>/<sh>/trajectory (UI debug + Wan VACE control)
  - Wan motion-prompt compilers (read the plan to inform motion text)

Design rule: the planner owns ALL the geometry math that used to live in
blender_render._position_camera_for_shot. blender_render becomes a consumer.

Output shapes
─────────────

  static plan        ─ for movement = [] or "static"
                       {position, rotation_deg, lens_mm, aperture,
                        target_pos, subject_distance_m}

  movement plan      ─ for movement ∈ {push_in, dolly_*, pan_*, crane_*, ...}
                       above + {movement: {kinds, start_position,
                                           end_position, start_lens_mm,
                                           end_lens_mm, easing}}

  N-keyframe track   ─ plan_camera_track(shot, n_frames, easing=...)
                       returns a list of N dicts with per-frame
                       {frame_index, t, position, rotation_deg, lens_mm}
                       — Blender's depth-pass renderer + Wan VACE
                       control_video both consume this.

PAI spec alignment
──────────────────

The PAI 0.3 schema (pai_lang/types.py) puts movement on Frame.movement
as `list[Movement]` — composite moves like ["pan_left", "zoom_in"] are
intentional, not a fluke. This planner accepts BOTH:

  - shot.frame.movement  (canonical PAI 0.3 path; list)
  - shot.camera.movement (legacy single-string path; auto-promoted to list)

Composite moves compose ADDITIVELY: each Movement contributes its delta
to the end keyframe. Contradictory pairs (e.g. push_in + pull_out) just
cancel out mathematically — caller's responsibility to declare a
coherent set. We don't reject them; we just warn.

The Movement Literal vocabulary splits three ways:

  geometric    push_in, push_in_slow, pull_out, dolly_{in,out,left,right},
               pan_{left,right,lr,rl}, tilt_{up,down}, crane_{up,down},
               zoom_{in,out}, tracking
               → contribute deltas to MOVEMENT_DELTAS, drive Blender.

  stylistic    handheld, steadicam, rapid
               → no geometric delta. Get a motion-blur / shake prompt
               hint for Flux + Wan but don't move the camera.

  Wan-only     orbit, from_behind, reverse_shot
               → planner emits static keyframes; the directive flows to
               Wan VACE's text prompt and (for orbit) is intended to
               drive the VACE control_video via a future arc sampler.

Easing
──────

  linear        t                              (default — back-compat)
  ease_in       t²                             slow start → fast end
  ease_out      1 - (1-t)²                     fast start → slow end
  ease_in_out   t<0.5 ? 2t² : 1-2(1-t)²        slow-fast-slow
"""

from __future__ import annotations
import math
from typing import Optional, Sequence


# ──────────────────────────────────────────────────────────────────────
#  Lookup tables — moved verbatim from blender_render.py to centralize
# ──────────────────────────────────────────────────────────────────────

SHOT_SIZE_DISTANCE_M = {
    # The ShotSize Literal, tightest to widest. A miss here does not raise, it
    # takes the 2.0 m default and moves the camera silently, so every declared
    # size needs an entry (pinned by tests/test_shot_size_tables_cover_the_schema.py).
    "extreme_close_up":  0.4,   # 大特写
    "close_up":          0.7,   # 特写
    "medium_close_up":   1.2,   # 中近景
    "medium":            2.0,   # 中景
    "medium_full":       2.0,   # 中全景 — worth its own distance, but that is a
                                # staging decision; 2.0 is what it resolved to before.
    "full":              3.0,   # 全景
    "master":            3.0,   # 主镜头
    "wide":              5.0,   # 远景
    "establishing":     10.0,   # 大远景 / 定场镜头
    # Not ShotSize values: planner-side vocabulary that reaches this table from
    # free-text camera language rather than from a PACE document.
    "medium_wide":       3.0,
    "extreme_wide":     10.0,
    "aerial":           50.0,
    "extreme close-up":  0.4,
    "close-up":          0.7,
    "medium close-up":   1.2,
    "medium wide":       3.0,
    "extreme wide":     10.0,
}

# (tilt_deg, vertical_height_offset_m).
# tilt > 0 = camera tilts down (high angle); tilt < 0 = tilts up (low angle).
ANGLE_OFFSETS = {
    "eye_level":          (0,    0),
    "low_angle":          (-25, -0.5),
    "high_angle":         ( 25,  0.8),
    "top_down":           ( 80,  3.0),
    "overhead":           ( 75,  3.0),
    "over_the_shoulder":  (0,    0),
    # space-separated legacy variants
    "eye level":          (0,    0),
    "low angle":          (-25, -0.5),
    "high angle":         ( 25,  0.8),
    # PAI Angle aliases (Panel.frame.angle)
    "high":               ( 25,  0.8),    # same as high_angle
    "low":                (-25, -0.5),    # same as low_angle
    "aerial":             ( 80,  3.0),    # same as top_down
    "dutch":              (0,    0),      # tilt is roll, not pitch — out of scope for this planner
    "low_position":       (-15, -0.7),    # below eye, slight upward tilt
}

# Per-location subject-eye / mid-body anchors. Used as fallback when a
# bible doesn't define camera_planning.eye_level_m / mid_body_m.
DEFAULT_EYE_LEVEL_M = {
    "street":         1.65,  # standing adult
    "cafe_interior":  1.2,   # seated
    "cafe_flashback": 1.2,
    "apartment":      1.65,
}
DEFAULT_MID_BODY_M = {
    "street":         0.9,
    "cafe_interior":  0.7,
    "cafe_flashback": 0.7,
    "apartment":      0.9,
}

DEFAULT_LENS_MM = 50

# Movement vocabulary recognised by the planner. Anything outside this set
# is logged as "static" to avoid silently miscoded movement.
MOVEMENT_KINDS = {
    "static",
    "push_in",
    "pull_out",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "crane_up",
    "crane_down",
    "zoom_in",
    "zoom_out",
    "handheld",
    # PAI Movement Literal additions
    "push_in_slow",   # treated like push_in but smaller delta (see MOVEMENT_DELTAS)
    "tracking",       # parallel-track — alias-mapped to dolly_left in DELTAS for now
    "pan_lr",         # left→right pan
    "pan_rl",         # right→left pan
    "orbit",          # circular arc — degraded to static for now (planner can't keyframe arc)
    "steadicam",      # treat as handheld jitter
    "from_behind",    # rear-position camera — degraded to static
    "reverse_shot",   # 180° flip — degraded to static (caller swaps subjects)
    "rapid",          # fast cuts / shaky — handled by Wan, planner stays static
}

# Per-movement displacement (meters / degrees) over the full clip duration.
# Conservative defaults — keeps the depth-pass interpretation stable.
MOVEMENT_DELTAS = {
    "push_in":     {"distance_delta_m": -0.8, "lens_zoom_pct": 0},
    "pull_out":    {"distance_delta_m":  0.8, "lens_zoom_pct": 0},
    "dolly_in":    {"distance_delta_m": -0.8, "lens_zoom_pct": 0},
    "dolly_out":   {"distance_delta_m":  0.8, "lens_zoom_pct": 0},
    "dolly_left":  {"y_delta_m": -0.5},
    "dolly_right": {"y_delta_m":  0.5},
    "pan_left":    {"yaw_delta_deg":  -10},
    "pan_right":   {"yaw_delta_deg":   10},
    "tilt_up":     {"pitch_delta_deg": -5},
    "tilt_down":   {"pitch_delta_deg":  5},
    "crane_up":    {"z_delta_m":       0.5},
    "crane_down":  {"z_delta_m":      -0.5},
    "zoom_in":     {"lens_zoom_pct":  40},   # 50mm -> 70mm
    "zoom_out":    {"lens_zoom_pct": -40},
    "handheld":    {"_note": "subtle jitter, no fixed delta — handled by Wan I2V"},
    # PAI Movement deltas
    "push_in_slow":{"distance_delta_m": -0.4, "lens_zoom_pct": 0},   # gentler push
    "tracking":    {"y_delta_m": -0.5},                              # mapped to dolly_left
    "pan_lr":      {"yaw_delta_deg":   10},                          # left→right
    "pan_rl":      {"yaw_delta_deg":  -10},                          # right→left
    "orbit":       {"_note": "circular — planner emits static; Wan VACE handles arc"},
    "steadicam":   {"_note": "treated as handheld; no fixed delta"},
    "from_behind": {"_note": "rear-position — caller must rotate scene 180° or set negative subject offset"},
    "reverse_shot":{"_note": "180° flip — caller swaps Subject_A/B before planning"},
    "rapid":       {"_note": "fast jitter — Wan handles; planner stays static"},
}

# Three-way classification of the Movement vocabulary — drives downstream
# routing. See module docstring for the full taxonomy.
GEOMETRIC_KINDS = {
    "push_in", "push_in_slow", "pull_out",
    "dolly_in", "dolly_out", "dolly_left", "dolly_right", "tracking",
    "pan_left", "pan_right", "pan_lr", "pan_rl",
    "tilt_up", "tilt_down",
    "crane_up", "crane_down",
    "zoom_in", "zoom_out",
}
STYLISTIC_KINDS = {"handheld", "steadicam", "rapid"}
WAN_ONLY_KINDS  = {"orbit", "from_behind", "reverse_shot"}

EASING_KINDS = ("linear", "ease_in", "ease_out", "ease_in_out")


def _ease(t: float, kind: str) -> float:
    """Map normalized time t∈[0,1] through an easing curve. Out-of-range
    inputs are clamped — caller can pass any float and get a sane result."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        u = 1.0 - t
        return 1.0 - u * u
    if kind == "ease_in_out":
        return 2.0 * t * t if t < 0.5 else 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
    return t   # linear (default + unknown fallback)


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────


def _shot_size_to_distance_m(shot_size: str) -> float:
    return SHOT_SIZE_DISTANCE_M.get(shot_size, 2.0)


def _angle_to_tilt_height(angle: str) -> tuple[float, float]:
    return ANGLE_OFFSETS.get(angle, (0, 0))


def _bible_eye_level_m(bible: dict, location_id: str) -> float:
    cp = (bible or {}).get("camera_planning") or {}
    if "eye_level_m" in cp:
        return float(cp["eye_level_m"])
    return DEFAULT_EYE_LEVEL_M.get(location_id, 1.65)


def _bible_mid_body_m(bible: dict, location_id: str) -> float:
    cp = (bible or {}).get("camera_planning") or {}
    if "mid_body_m" in cp:
        return float(cp["mid_body_m"])
    return DEFAULT_MID_BODY_M.get(location_id, 0.9)


def _resolve_lens_mm(camera_cfg: dict) -> int:
    """Focal length for this shot, or the default.

    Two document shapes reach here. The legacy one puts the lens flat on
    `shot.camera`; PACE nests it under `camera.intrinsics`, alongside the
    sensor and fov class. Reading only the flat key meant every PACE-shaped
    scene fell through to the default -- a shot declaring `lens_mm: 35` was
    planned, framed and rendered at 50 mm, with nothing reporting a
    substitution. The declared value is not a hint; it is the shot.
    """
    lens = camera_cfg.get("lens_mm")
    if lens is None:
        intr = camera_cfg.get("intrinsics") or {}
        # Both key vintages, as in movement_io.shot_to_planner_dict: the schema
        # name and the one the scene documents on disk actually carry.
        lens = intr.get("focal_length_mm") or intr.get("lens_mm")
    if lens is None:
        return DEFAULT_LENS_MM
    if isinstance(lens, str):
        lens = lens.replace("mm", "").strip()
    try:
        return int(float(lens))
    except (TypeError, ValueError):
        return DEFAULT_LENS_MM


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────


def plan_cameras(shot: dict, bible: Optional[dict] = None) -> list[dict]:
    """Multi-camera form. Returns a list of camera plans for one shot.

    Two shot shapes accepted:
      single (legacy): shot.camera = {shot_size, angle, lens_mm, ...}
                        → returns [plan with cam_id='A']
      multi:           shot.cameras = [{id: 'A', shot_size, ...},
                                        {id: 'B', ...}, ...]
                        → returns one plan per entry, cam_id from .id field
                        (defaults to alphabetical 'A', 'B', 'C', ... if id absent)

    Each plan dict carries a 'cam_id' field for downstream routing
    (file naming: <shot_id>_<cam_id>.png when multi, plain <shot_id>.png
     when single-cam for backward compatibility).
    """
    cams = shot.get("cameras")
    if cams:
        plans = []
        for i, cam in enumerate(cams):
            cam_id = cam.get("id") or chr(ord("A") + i)
            sub_shot = {**shot, "camera": cam}
            sub_shot.pop("cameras", None)
            plan = plan_camera(sub_shot, bible)
            plan["cam_id"] = cam_id
            plans.append(plan)
        return plans

    # Single-cam: wrap plan_camera in a list with cam_id='A'
    plan = plan_camera(shot, bible)
    plan["cam_id"] = "A"
    return [plan]


def _read_movements(shot: dict) -> list[str]:
    """Extract the Movement list from a *planner-input* shot dict.

    Planner input shape is the output of `movement_io.shot_to_planner_dict`:
    `{"frame": {"movement": [...]}, "camera": {...}, "scene_ref": ...}`. It
    deliberately differs from the on-disk pai-1.0 shape so the planner can
    stay schema-agnostic — `movement_io` is the single bridge.

    Tolerates two legacy shapes still seen in fixtures and tests:
      - shot.camera.movement = "push_in" (single string)
      - shot.camera.movement = ["push_in", "pan_lr"] (list)
    Names lower-cased + hyphens to underscores; empties skipped.
    """
    frame = shot.get("frame") or {}
    cam   = shot.get("camera") or {}
    raw = frame.get("movement") if frame.get("movement") else cam.get("movement")
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.lower().replace("-", "_")] if raw else []
    if isinstance(raw, list):
        return [str(r).lower().replace("-", "_") for r in raw if r]
    return []


def plan_camera(shot: dict, bible: Optional[dict] = None) -> dict:
    """Return a fully resolved camera plan for one shot.

    Mirrors blender_render._position_camera_for_shot semantics for the static
    path (so refactoring blender_render to consume this is a no-op for the
    existing 24-shot film) and adds explicit lens, target, and movement fields.

    Movement resolution: reads `shot.frame.movement` (PAI 0.3 list[Movement])
    first, falls back to `shot.camera.movement` (legacy single string OR list).
    Multiple movements compose additively. The returned `movement` block
    contains the union of kinds, the composed end keyframe, and the easing.
    Unknown kinds are dropped with a `_movement_warning`.

    Composition: `shot.composition = {"subject_xy": [x, y], "target_x": tx,
    "target_y": ty}` (optional; subject height comes from this same
    function's own eye_level/mid_body anchors, so the caller only needs to
    know the subject's lateral placement, e.g. from a mannequin slot). When
    present, the default dead-center aim below is replaced by
    composition_solver.solve_rotation_for_screen_position, so the subject
    projects to (tx, ty) instead of frame center — grounding PAI's
    screen_position field in the same explicit scene the trajectory is
    compiled into, rather than leaving it as authored-but-unconsumed
    metadata. Which point on the subject is driven to (tx, ty) depends on
    whether the eye_level/mid_body span fits inside the frame at this
    magnification: loose enough and it is the MIDPOINT of the two anchors'
    independent projections (an approximation of the silhouette centroid);
    tighter than that and it stays the single base_aim_z anchor, because a
    close-up whose whole-body centroid sits on the target is a shot with
    the subject's head tilted out of frame. `composition_solved.anchor`
    records which rule fired. Absent (the default for every existing
    shot), behavior is unchanged from before this field existed.
    """
    cam_cfg  = shot.get("camera") or {}
    location = shot.get("scene_ref") or shot.get("location_ref") or ""
    bible    = bible or {}

    distance      = _shot_size_to_distance_m(cam_cfg.get("shot_size", "medium"))
    tilt, h_off   = _angle_to_tilt_height(cam_cfg.get("angle", "eye_level"))
    eye_level     = _bible_eye_level_m(bible, location)
    mid_body      = _bible_mid_body_m(bible, location)
    base_aim_z    = mid_body if distance >= 4.0 else eye_level

    cam_x         = distance
    cam_y         = 0.0
    cam_z         = base_aim_z + h_off
    # MINUS, not plus. Blender's rx is 90 at level and DECREASES as the camera
    # tilts down: a person standing at the origin fills 61% of the frame at
    # rx=65, 1% at rx=90, and 0% at rx=115. ANGLE_OFFSETS states tilt > 0 =
    # tilts down, so a high angle has to subtract.
    #
    # Adding it aimed every high-angle shot at the sky, and top_down (tilt 80)
    # at rx=170 — very nearly straight up, the exact opposite of overhead. With
    # a subject in view a mesh still filled some of the frame, so the mistake
    # survived; on a scene render with only ground it produced an all-black
    # depth map, which is what surfaced it.
    rot_x_deg     = 90 - tilt
    rot_z_deg     = 90        # camera looks along -X toward origin

    lens_mm       = _resolve_lens_mm(cam_cfg)
    aperture      = cam_cfg.get("aperture", "")

    plan: dict = {
        "position":           [cam_x, cam_y, cam_z],
        "rotation_deg":       [rot_x_deg, 0.0, rot_z_deg],
        "lens_mm":            lens_mm,
        "aperture":           aperture,
        "target_pos":         [0.0, 0.0, base_aim_z],
        "subject_distance_m": distance,
    }

    comp = shot.get("composition") or {}
    subject_xy = comp.get("subject_xy")
    target_x, target_y = comp.get("target_x"), comp.get("target_y")
    if subject_xy is not None and target_x is not None and target_y is not None:
        from pace_core.setup.composition_solver import (
            SENSOR_WIDTH_MM, solve_rotation_for_screen_position)
        sx, sy = float(subject_xy[0]), float(subject_xy[1])
        z_low, z_high = min(eye_level, mid_body), max(eye_level, mid_body)
        # Which anchor the shot is composed on depends on whether the
        # subject's body actually fits the frame, which is a question about
        # MAGNIFICATION, not distance — so it is computed here rather than
        # read off the distance >= 4.0 threshold base_aim_z uses.
        #
        # Loose enough to show the figure: target the MIDPOINT of the
        # z_low/z_high anchors' independent projections, an approximation
        # of the silhouette centroid, which is what "where the subject sits
        # in the frame" means when you can see the whole subject.
        #
        # Too tight for that: keep the single-anchor aim. Driving a
        # whole-body centroid to the target in a close-up is not a better
        # composition, it is a worse one — the camera has to tilt down far
        # enough to push the head out of frame to get there (measured: on
        # scene_08's three panels the two-point objective cut the
        # mask-centroid offset to ~1-3% of frame height while filling the
        # frame with the subject's torso and losing the face entirely).
        # The composition-precision measurement that produced these
        # numbers lived in scripts/ and was removed with it; the numbers
        # it reported are quoted in the PACE paper's composition section.
        cam_to_subject = math.dist(plan["position"], [sx, sy, (z_low + z_high) / 2.0])
        half_v_tan = ((SENSOR_WIDTH_MM / 2.0) / lens_mm) / (16 / 9)
        frame_h_m = 2.0 * half_v_tan * cam_to_subject
        span_fits = (z_high - z_low) <= 0.8 * frame_h_m

        aim_low = [sx, sy, z_low]
        aim_high = [sx, sy, z_high]
        solved = solve_rotation_for_screen_position(
            plan["position"], plan["rotation_deg"], lens_mm,
            aim_low if span_fits else [sx, sy, base_aim_z],
            float(target_x), float(target_y),
            subject_head_pos=aim_high if span_fits else None,
        )
        plan["rotation_deg"] = solved["rotation_deg"]
        plan["target_pos"] = [sx, sy, (z_low + z_high) / 2.0 if span_fits else base_aim_z]
        plan["composition_solved"] = {
            "target": [float(target_x), float(target_y)],
            "subject_pos": list(plan["target_pos"]),
            "subject_span_z": [z_low, z_high] if span_fits else [base_aim_z, base_aim_z],
            "anchor": "silhouette_midpoint" if span_fits else "single_height",
            "frame_height_m": frame_h_m,
            "iterations": solved["iterations"],
            "residual": solved["residual"],
            "converged": solved["converged"],
        }

    movements = _read_movements(shot)
    valid:   list[str] = []
    unknown: list[str] = []
    stylistic_only = True
    for m in movements:
        if m == "static" or not m:
            continue
        if m not in MOVEMENT_KINDS:
            unknown.append(m)
            continue
        valid.append(m)
        if m in GEOMETRIC_KINDS:
            stylistic_only = False
    if unknown:
        plan["_movement_warning"] = (
            f"unknown movement(s) {unknown!r}; dropped"
        )
    # Easing comes from shot.frame.movement_easing if provided, else default.
    easing = ((shot.get("frame") or {}).get("movement_easing")
              or cam_cfg.get("movement_easing")
              or "linear")
    if easing not in EASING_KINDS:
        easing = "linear"

    has_geometric_or_stylistic = bool(valid)
    if has_geometric_or_stylistic and not stylistic_only:
        plan["movement"] = _build_movement_block(
            kinds         = valid,
            start_position= plan["position"],
            start_lens_mm = lens_mm,
            start_rotation= plan["rotation_deg"],
            easing        = easing,
        )
    elif has_geometric_or_stylistic:
        # All movements are stylistic-only (handheld / steadicam / rapid).
        # No geometric deltas, but record them so downstream prompts know.
        plan["movement"] = {
            "kind":              valid[0],   # back-compat
            "kinds":             valid,
            "start_position":    plan["position"],
            "end_position":      list(plan["position"]),
            "start_rotation_deg": plan["rotation_deg"],
            "end_rotation_deg":   list(plan["rotation_deg"]),
            "start_lens_mm":     lens_mm,
            "end_lens_mm":       lens_mm,
            "easing":            easing,
            "_stylistic_only":   True,
        }

    return plan


def _build_movement_block(*, kinds: list[str], start_position: list[float],
                          start_lens_mm: int, start_rotation: list[float],
                          easing: str = "linear") -> dict:
    """Compose end_position / end_rotation / end_lens by summing every
    geometric Movement kind's deltas. Stylistic kinds in the list are kept
    in `.kinds` but contribute nothing to the geometry."""
    end_pos = list(start_position)
    end_rot = list(start_rotation)
    end_lens = float(start_lens_mm)

    for kind in kinds:
        if kind not in GEOMETRIC_KINDS:
            continue
        delta = MOVEMENT_DELTAS.get(kind, {})
        if "distance_delta_m" in delta:
            end_pos[0] += delta["distance_delta_m"]
        if "y_delta_m" in delta:
            end_pos[1] += delta["y_delta_m"]
        if "z_delta_m" in delta:
            end_pos[2] += delta["z_delta_m"]
        if "yaw_delta_deg" in delta:
            end_rot[2] += delta["yaw_delta_deg"]
        if "pitch_delta_deg" in delta:
            end_rot[0] += delta["pitch_delta_deg"]
        if delta.get("lens_zoom_pct"):
            end_lens = end_lens * (1 + delta["lens_zoom_pct"] / 100)

    # `kind` (singular) = the dominant/first geometric kind, kept for
    # back-compat with consumers that pre-date composite movement
    # (blender_render.py, older tests).
    geometric_first = next((k for k in kinds if k in GEOMETRIC_KINDS), kinds[0])
    return {
        "kind":               geometric_first,   # back-compat — first geometric
        "kinds":              list(kinds),
        "start_position":     start_position,
        "end_position":       end_pos,
        "start_rotation_deg": start_rotation,
        "end_rotation_deg":   end_rot,
        "start_lens_mm":      start_lens_mm,
        "end_lens_mm":        max(8, int(round(end_lens))),
        "easing":             easing,
    }


# ──────────────────────────────────────────────────────────────────────
#  N-keyframe sampling — for Blender depth-pass + Wan VACE control_video
# ──────────────────────────────────────────────────────────────────────


def plan_camera_track(shot: dict, n_frames: int = 16,
                      easing: Optional[str] = None,
                      bible: Optional[dict] = None) -> list[dict]:
    """Sample a camera move into N keyframes using the configured easing curve.

    The returned list always has `n_frames` entries. For a static plan (no
    movement block) every entry is the same start keyframe — callers can
    still iterate uniformly without special-casing.

    Args
    ----
    shot      same dict accepted by plan_camera()
    n_frames  number of keyframes (>= 1). 16 maps cleanly to 1s @ 16fps
              for Wan I2V; 24 maps to 1s @ 24fps for standard cinema.
    easing    override the shot's frame.movement_easing. Useful for
              quick A/B testing in the studio UI. None = use shot's value.
    bible     same as plan_camera()

    Returns
    -------
    List of dicts:
      [{frame_index: int, t: float,
        position: [x,y,z], rotation_deg: [rx,ry,rz], lens_mm: int}, ...]
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    plan = plan_camera(shot, bible=bible)
    mv = plan.get("movement")
    use_easing = easing if (easing in EASING_KINDS) else (
        (mv or {}).get("easing", "linear")
    )

    if not mv:
        # Static: identical keyframe N times. Callers (Blender) can elide
        # the dedup, but the uniform shape simplifies the iteration code.
        frame = {
            "position":     list(plan["position"]),
            "rotation_deg": list(plan["rotation_deg"]),
            "lens_mm":      int(plan["lens_mm"]),
        }
        return [{"frame_index": i, "t": (i / max(1, n_frames - 1)),
                 **{k: list(v) if isinstance(v, list) else v for k, v in frame.items()}}
                for i in range(n_frames)]

    sp, ep = mv["start_position"], mv["end_position"]
    sr, er = mv["start_rotation_deg"], mv["end_rotation_deg"]
    sl, el = float(mv["start_lens_mm"]), float(mv["end_lens_mm"])
    track: list[dict] = []
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        e = _ease(t, use_easing)
        track.append({
            "frame_index":  i,
            "t":            t,
            "position":     [sp[k] + (ep[k] - sp[k]) * e for k in range(3)],
            "rotation_deg": [sr[k] + (er[k] - sr[k]) * e for k in range(3)],
            "lens_mm":      int(round(sl + (el - sl) * e)),
        })
    return track


# ──────────────────────────────────────────────────────────────────────
#  World-track escape hatch — direct meter-scale input
# ──────────────────────────────────────────────────────────────────────


def plan_camera_track_from_world(
    *,
    keyframes: list[dict],
    n_frames: int,
    easing: str = "linear",
) -> list[dict]:
    """Build an N-frame track from explicit meter-scale keyframes — the
    low-level entry point used by callers (LAMP DSL bridge, manual
    overrides, blender-bake importers) that don't go through a PAI shot
    dict.

    `keyframes` is a list of ≥2 dicts each with
        {position: [x,y,z], rotation_deg: [rx,ry,rz], lens_mm: int}
    spaced UNIFORMLY across the timeline. The first keyframe lands at
    frame_index 0, the last at frame_index n_frames-1, intermediates
    spread evenly between.

    Pairs of adjacent keyframes are interpolated with the given easing.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if len(keyframes) < 2:
        raise ValueError(f"need at least 2 keyframes, got {len(keyframes)}")

    n_kf = len(keyframes)
    span_per_segment = (n_frames - 1) / (n_kf - 1) if n_frames > 1 else 0.0

    # Snap each keyframe to a timeline frame_index. Last keyframe lands
    # exactly at n_frames-1 so the camera ends precisely on the last pose.
    kf_at_frame: list[int] = [
        round(i * span_per_segment) for i in range(n_kf)
    ]
    kf_at_frame[-1] = n_frames - 1

    track: list[dict] = []
    for fi in range(n_frames):
        # Find which segment (k_lo → k_hi) this frame falls into.
        k_hi = next((i for i, f in enumerate(kf_at_frame) if f >= fi), n_kf - 1)
        k_hi = max(1, k_hi)
        k_lo = k_hi - 1
        f_lo, f_hi = kf_at_frame[k_lo], kf_at_frame[k_hi]
        denom = max(1, f_hi - f_lo)
        t = (fi - f_lo) / denom
        e = _ease(t, easing)

        sp = keyframes[k_lo]["position"]
        ep = keyframes[k_hi]["position"]
        sr = keyframes[k_lo]["rotation_deg"]
        er = keyframes[k_hi]["rotation_deg"]
        sl = float(keyframes[k_lo].get("lens_mm", 35))
        el = float(keyframes[k_hi].get("lens_mm", sl))

        track.append({
            "frame_index":  fi,
            "t":            fi / max(1, n_frames - 1),
            "position":     [sp[k] + (ep[k] - sp[k]) * e for k in range(3)],
            "rotation_deg": [sr[k] + (er[k] - sr[k]) * e for k in range(3)],
            "lens_mm":      int(round(sl + (el - sl) * e)),
        })
    return track


# ──────────────────────────────────────────────────────────────────────
#  Object motion + follow-shot composition (world-space tracks)
#
#  The depth renderer (render_depth_vace.py) is a dumb playback engine: it
#  plays a per-frame camera track and per-frame OBJECT tracks. It computes
#  no motion. Anything that moves — the camera, a car, a pedestrian — is
#  driven by an explicit per-frame transform list produced HERE.
#
#  `make_linear_motion` is the speed→positions integrator that used to live
#  (wrongly) inside the Blender script as `--car-speed-mps`. `plan_follow_shot`
#  composes a camera rig expressed in a moving subject's LOCAL frame into a
#  world-space camera track, so a chase cam stays glued to the subject.
#
#  Rotation convention matches the rest of this module and Blender's default
#  XYZ Euler: rotation_deg = [pitch(x), roll(y), yaw(z)] and the rotation
#  matrix is M = Rz @ Ry @ Rx  (verified against mathutils.Euler.to_matrix).
# ──────────────────────────────────────────────────────────────────────


def _euler_xyz_to_matrix(rot_deg: Sequence[float]) -> list[list[float]]:
    """[rx, ry, rz] degrees → 3×3 rotation matrix, M = Rz @ Ry @ Rx."""
    rx, ry, rz = (math.radians(a) for a in rot_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Closed form of Rz @ Ry @ Rx (derived + verified against Blender).
    return [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy],
    ]


def _matrix_to_euler_xyz(m: list[list[float]]) -> list[float]:
    """3×3 rotation matrix → [rx, ry, rz] degrees, inverse of the above."""
    sy = max(-1.0, min(1.0, -m[2][0]))
    cy = math.sqrt(1.0 - sy * sy)
    if cy > 1e-6:
        rx = math.atan2(m[2][1], m[2][2])
        ry = math.asin(sy)
        rz = math.atan2(m[1][0], m[0][0])
    else:                              # gimbal lock (pitch ±90°)
        rx = math.atan2(-m[1][2], m[1][1])
        ry = math.asin(sy)
        rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _matvec3(m: list[list[float]], v: Sequence[float]) -> list[float]:
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def make_linear_motion(
    *,
    n_frames: int,
    speed_mps: float,
    fps: int = 16,
    axis: str = "x",
    start: Sequence[float] = (0.0, 0.0, 0.0),
    rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[dict]:
    """Constant-velocity straight-line path → per-frame world track.

    Returns [{frame_index, position, rotation_deg}], one entry per frame.
    `axis` ∈ {x, y, z}; +axis is the direction of travel. This is the
    speed→positions integrator — keep it OUT of the Blender renderer.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    ai = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if ai is None:
        raise ValueError(f"axis must be x|y|z, got {axis!r}")
    dt = speed_mps / fps           # metres advanced per frame
    track: list[dict] = []
    for i in range(n_frames):
        pos = list(start)
        pos[ai] += i * dt
        track.append({
            "frame_index":  i,
            "position":     pos,
            "rotation_deg": list(rotation_deg),
        })
    return track


def plan_follow_shot(
    *,
    subject_track: list[dict],
    relative_camera_track: list[dict],
) -> dict:
    """Glue a camera rig (expressed in the subject's LOCAL frame) onto a
    moving subject, producing world-space tracks the renderer plays verbatim.

    Per frame i:
        world_cam_pos = subject_pos_i + R(subject_rot_i) @ rel_cam_pos_i
        world_cam_rot = euler( R(subject_rot_i) @ R(rel_cam_rot_i) )

    For straight-line motion (subject rotation = 0) this degenerates to a
    plain offset add, so a chase cam keeps its framing as the subject drives
    forward. When the subject turns, the camera banks with it.

    `subject_track` and `relative_camera_track` must be the same length
    (one entry per rendered frame). Lens is carried from the camera track.

    Returns {"subject_track": [...world...], "camera_track": [...world...]}.
    """
    if len(subject_track) != len(relative_camera_track):
        raise ValueError(
            f"subject_track ({len(subject_track)}) and relative_camera_track "
            f"({len(relative_camera_track)}) must be the same length")

    cam_world: list[dict] = []
    subj_world: list[dict] = []
    for i, (s, c) in enumerate(zip(subject_track, relative_camera_track)):
        s_pos = list(s["position"])
        s_rot = list(s.get("rotation_deg", [0.0, 0.0, 0.0]))
        rs = _euler_xyz_to_matrix(s_rot)

        rel_pos = list(c["position"])
        rel_rot = list(c.get("rotation_deg", [0.0, 0.0, 0.0]))

        w_pos = [s_pos[k] + _matvec3(rs, rel_pos)[k] for k in range(3)]
        w_rot = _matrix_to_euler_xyz(
            _matmul3(rs, _euler_xyz_to_matrix(rel_rot)))

        cam_world.append({
            "frame_index":  i,
            "position":     w_pos,
            "rotation_deg": w_rot,
            "lens_mm":      int(c.get("lens_mm", 35)),
        })
        subj_world.append({
            "frame_index":  i,
            "position":     s_pos,
            "rotation_deg": s_rot,
        })
    return {"subject_track": subj_world, "camera_track": cam_world}


# ──────────────────────────────────────────────────────────────────────
#  LAMP DSL bridge — DSL → meter-scale track via camera_planner
# ──────────────────────────────────────────────────────────────────────

# LAMP-units → meters. The LAMP base distance is 64 units per segment
# (see lamp_compile._BASE_DISTANCE), inside a 512³ cube. We map 1 LAMP
# unit ≈ 4 cm (default), so a `far_in` push (64 units) ≈ 2.56 m forward
# — a useful "large camera push" for typical interior shots. Override
# per-call when the scene needs a different scale.
LAMP_DEFAULT_UNITS_PER_METER = 25.0


# A camera with no rotation is not a camera pointing forward. Blender aims an
# unrotated camera down -Z -- at the floor -- while every anchor_position we
# default to puts it a few metres back on -Y at eye height, which only makes
# sense looking level toward the origin. The two defaults contradicted each
# other, and the rotation won: 304 of 311 playground jobs ran with [0, 0, 0].
#
# With a subject mesh that still produced something (the camera stared down at
# the mesh right below it) but a depth map with 2 distinct values instead of
# 26 -- a silhouette carrying no shape. With no mesh, a scene render over the
# deliberately-flat centre lane of the ground contour, it produced a blank
# control: 4 distinct values across the frame. VACE was then bound to that
# featureless target at control_strength 0.7 and invented a large orange mass
# to fill it.
#
# +90 deg about X takes -Z to +Y: level, looking toward the origin from -Y.
LEVEL_ANCHOR_ROTATION_DEG = (90.0, 0.0, 0.0)


def plan_camera_track_from_dsl(
    dsl: str,
    *,
    anchor_position: list[float],
    anchor_rotation_deg: list[float] | None = None,
    lens_mm: int = 35,
    n_frames: int = 81,
    units_per_meter: float = LAMP_DEFAULT_UNITS_PER_METER,
    easing: str = "linear",
) -> list[dict]:
    """Compile a LAMP DSL line into a meter-scale N-keyframe track.

    The DSL grammar (24 tokens = 4 segments × 6 tokens) is parsed by
    `pace_core.camera.lamp_dsl.parse_dsl`. Each segment contributes a
    cumulative truck/boom/dolly (mx/my/mz) translation plus yaw/pitch/roll
    delta. Segments compose ADDITIVELY in the camera's CURRENT local
    frame at the start of each segment — same semantics as
    `lamp_compile._move_to_world`, just rescaled to meters and anchored
    to `anchor_position` instead of LAMP's (256,256,256) cube centre.

    Output matches `plan_camera_track` so vace_pipeline, blender_render,
    and VACE motion-prompts can consume both shapes identically.
    """
    # Lazy import — these modules transitively import LLM clients.
    from pace_core.camera.lamp_dsl import parse_dsl, N_SEGMENTS
    from pace_core.camera.lamp_compile import _move_to_world, _BASE_DISTANCE

    if anchor_rotation_deg is None:
        anchor_rotation_deg = list(LEVEL_ANCHOR_ROTATION_DEG)

    segments = parse_dsl(dsl)
    if len(segments) != N_SEGMENTS:
        raise ValueError(f"parse_dsl returned {len(segments)} segments")

    # 5 keyframes: 1 anchor at t=0, plus 1 at the end of each of 4 segments.
    keyframes: list[dict] = [{
        "position":     list(anchor_position),
        "rotation_deg": list(anchor_rotation_deg),
        "lens_mm":      int(lens_mm),
    }]
    cur_pos = list(anchor_position)
    cur_rot = list(anchor_rotation_deg)
    # `rotation_deg` on a track is read straight into Blender's camera, where
    # level is +90 about X. _move_to_world steers in LAMP's frame, where level
    # is 0. The same number was being used as both, which is why raising the
    # anchor default to level sent a `far_in` push into the floor instead of
    # forward: the mover read 90 as "pitched 90 down".
    #
    # So the two frames are kept apart. Steering happens in LAMP's, emitted
    # keyframes are in Blender's, and this offset is the only bridge.
    lamp_pitch = cur_rot[0] - LEVEL_ANCHOR_ROTATION_DEG[0]

    for seg in segments:
        # Apply translation in the camera's CURRENT orientation (matches
        # lamp_compile semantics — _move_to_world rotates the delta by
        # the camera's yaw/pitch/roll BEFORE adding to position).
        dx_u, dy_u, dz_u = _move_to_world(
            seg.mx, seg.my, seg.mz,
            strength=1.0,
            yaw=cur_rot[2], pitch=lamp_pitch, roll=cur_rot[1],
            base_distance=_BASE_DISTANCE,
        )
        # Axis remap: LAMP uses (X=right, Y=up, Z=fwd with `in` = -Z).
        # Blender / camera_planner use (X=right, Y=forward into scene,
        # Z=up). So:  world_x = LAMP_x,  world_y = -LAMP_z  (LAMP `in`
        # = -Z = +Y forward in Blender),  world_z = LAMP_y.
        scale = 1.0 / units_per_meter
        cur_pos = [
            cur_pos[0] + dx_u * scale,
            cur_pos[1] + (-dz_u) * scale,
            cur_pos[2] + dy_u * scale,
        ]
        # Rotation: LAMP uses (yaw, tilt, roll) per-segment integer degrees.
        # Convention: rotation_deg = [pitch (x), roll (y), yaw (z)].
        cur_rot = [
            cur_rot[0] + float(seg.tilt),
            cur_rot[1] + float(seg.roll),
            cur_rot[2] + float(seg.yaw),
        ]
        lamp_pitch += float(seg.tilt)        # the same tilt, in the other frame
        keyframes.append({
            "position":     list(cur_pos),
            "rotation_deg": list(cur_rot),
            "lens_mm":      int(lens_mm),
        })

    return plan_camera_track_from_world(
        keyframes=keyframes, n_frames=n_frames, easing=easing,
    )
