"""Project stage-frame world coordinates into per-shot screen positions.

PAI 1.1 prototype helper. Given a scene with `physical_layout` filled in,
walks each shot's `camera_setups` entry and computes a `screen_position`
({zone, x, y, depth}) for every subject + prop the camera can see.

Output is non-destructive: returns a dict of {shot_id: {entity_ref: ScreenPosition}}
that you can apply to the scene file with --apply, or just preview.

Coordinate model — stage_top_view:
  - x grows east, y grows north
  - camera looks_at a point; subjects to the LEFT of look_at line appear
    on the LEFT of frame; in FRONT (closer to camera) = foreground
  - lens_mm controls horizontal FOV (35mm full-frame sensor assumed)
  - subjects behind the camera or beyond ~30m get screen_position = None
    (off-frame / unreasonable)

Usage:
    python3 pipeline/derive_screen_position.py --project main --scene scene_05
    python3 pipeline/derive_screen_position.py --project main --scene scene_05 --apply
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent

from pace_core.paths import paths_for


# 35mm full-frame sensor width = 36mm.  fov_h = 2 * atan((sensor_w/2) / focal_mm)
_SENSOR_W_MM = 36.0


def _fov_h_deg(focal_mm: float) -> float:
    return math.degrees(2 * math.atan((_SENSOR_W_MM / 2) / focal_mm))


def _project_entity(cam_xy: list[float], cam_z: float, look_at_xy: list[float],
                    lens_mm: float, entity_xy: list[float], entity_z: float) -> dict | None:
    """Returns ScreenPosition dict or None if entity is behind camera /
    out of frame. Top-down 2D model — vertical y is derived from entity_z
    only (no full 3D camera tilt yet)."""
    cx, cy = cam_xy
    lx, ly = look_at_xy
    ex, ey = entity_xy

    # Camera-forward unit vector
    fx, fy = lx - cx, ly - cy
    fmag = math.hypot(fx, fy)
    if fmag < 1e-6:
        return None
    fx, fy = fx / fmag, fy / fmag
    # Camera-right (perpendicular, right-handed: rotate forward by -90°)
    rx, ry = fy, -fx

    # Entity vector from camera
    dx, dy = ex - cx, ey - cy

    # Decompose into forward + right components
    forward = dx * fx + dy * fy      # depth into frame
    right   = dx * rx + dy * ry      # horizontal offset from frame center

    if forward <= 0.1:               # behind camera (or essentially on top of it)
        return None

    # Horizontal angle of entity from camera-forward
    angle_h_deg = math.degrees(math.atan2(right, forward))
    half_fov = _fov_h_deg(lens_mm) / 2
    if abs(angle_h_deg) > half_fov:
        return None                  # out of frame

    # Normalized x: -half_fov → 0 (left edge), 0 → 0.5 (center), +half_fov → 1 (right edge)
    x = 0.5 + (angle_h_deg / (2 * half_fov))

    # Vertical: model the entity's z relative to camera z, mapped linearly.
    # entity_z 0 (ground) at cam_z 1.65 → below center; entity_z 1.65 same → center.
    # Use a simple proportional mapping: every meter ≈ 0.2 normalized.
    y = 0.5 + (cam_z - entity_z - 1.0) * 0.2
    y = max(0.0, min(1.0, y))

    # Depth bucket from forward distance
    if forward <= 3.0:    depth = "foreground"
    elif forward <= 8.0:  depth = "midground"
    else:                 depth = "background"

    # Coarse zone (rule-of-thirds)
    col = "left" if x < 0.33 else ("right" if x > 0.66 else "center")
    row_below = y > 0.66
    row_above = y < 0.33
    if   row_above and col == "center": zone = "upper"
    elif row_above:                     zone = f"upper_{col}"
    elif row_below and col == "center": zone = "lower"
    elif row_below:                     zone = f"lower_{col}"
    elif col == "left":                 zone = "center_left"
    elif col == "right":                zone = "center_right"
    else:                               zone = "center"

    return {
        "zone":  zone,
        "x":     round(x, 3),
        "y":     round(y, 3),
        "depth": depth,
    }


def derive(scene: dict) -> dict[str, dict[str, dict]]:
    """Returns {shot_id: {entity_ref: screen_position_dict}}."""
    pl = scene.get("physical_layout")
    if not pl:
        return {}

    by_ref = {}
    for e in (pl.get("subjects") or []) + (pl.get("props") or []):
        by_ref[e["ref"]] = e

    out: dict[str, dict[str, dict]] = {}
    for cs in (pl.get("camera_setups") or []):
        shot_id = cs["shot_id"]
        cam_xy = cs.get("world_xy") or [0, 0]
        cam_z  = cs.get("z", 1.65)
        look   = cs.get("looking_at_xy") or [cam_xy[0] + 1, cam_xy[1]]
        lens   = cs.get("lens_mm") or 50.0

        out[shot_id] = {}
        for ref, e in by_ref.items():
            sp = _project_entity(cam_xy, cam_z, look, lens,
                                 e.get("world_xy") or [0, 0], e.get("z", 0))
            if sp:
                out[shot_id][ref] = sp
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", default="main")
    ap.add_argument("--scene", required=True, help="e.g. scene_05")
    ap.add_argument("--apply", action="store_true",
                    help="Write derived screen_positions back onto subjects[] / props[]")
    args = ap.parse_args()

    path = paths_for(args.project).scenes_dir / f"{args.scene}.json"
    if not path.exists():
        raise SystemExit(f"scene not found: {path}")

    scene = json.loads(path.read_text())
    derived = derive(scene)
    if not derived:
        print("(no physical_layout in this scene — nothing to project)")
        return

    if not args.apply:
        print(json.dumps(derived, ensure_ascii=False, indent=2))
        return

    # Apply: write each derived screen_position onto matching subjects[]/props[]
    n_written = 0
    for shot in scene.get("shots") or []:
        shot_derived = derived.get(shot["shot_id"]) or {}
        if not shot_derived:
            continue
        for sub in (shot.get("setup") or {}).get("subjects") or []:
            cid = sub.get("character_id")
            age = sub.get("age_state")
            ref = f"{cid}@{age}" if age else cid
            if ref in shot_derived:
                sub["screen_position"] = shot_derived[ref]
                n_written += 1
        for prop in (shot.get("setup") or {}).get("props") or []:
            pid = prop.get("prop_id")
            if pid and pid in shot_derived:
                prop["screen_position"] = shot_derived[pid]
                n_written += 1
    path.write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {n_written} screen_position(s) back to {path.name}")


if __name__ == "__main__":
    main()
