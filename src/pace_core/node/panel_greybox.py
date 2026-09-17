"""panel_greybox — a panel's control geometry: N bodies, in seats, under its own camera.

Text cannot hold a subject count. Measured on one corpus scene: an
enumerated three-person cast rendered four people, and adding one prop clause
to a prompt that had just rendered three brought the fourth back at the same
seed. Nothing in the sampler enforces cardinality and cfg=1.0 leaves no
negative prompt to push back with, so any phrasing that works is working by
luck at that seed. So the count stops being a request and becomes geometry:
this renders exactly the subjects a panel declares, seated where the location
says seats are, seen through the camera the panel specifies. The frame feeds
`structure_flux2` as its init image, where it decides composition while the
prompt decides appearance.

A panel is already a scene description. Everything here is read, not invented:

    Setup.subjects[]              how many bodies, and which mesh (age_state)
    subjects[].screen_position.x  which seat, left to right
    backdrop.location -> stub     cabin size, from scale_meters
    camera.intrinsics.lens_mm     the lens
    camera.extrinsics.angle       eye level / high / low
    camera.extrinsics.position    which side the lens stands on

The cabin is built here rather than through add_location_proxy because
LocationShape covers chamber/subway/office/outdoor and a vehicle interior is
none of those. Seat layout follows the location's own words -- "two front
seats side by side facing the windscreen and one rear seat directly behind
them, all occupants facing forward" -- which every text-only render so far has
ignored in favour of a four-seat club cabin.

Nothing about the camera is hand-tuned. Every corner of every body is measured
in world space, which gives the cast's true horizontal and vertical extent;
the lens and sensor give the angles that extent has to fit into; distance
follows from the two. The focus lands in the middle of the frame because it is
the centre of what was measured, not a constant someone chose. Three guessed
camera positions failed before this -- inside a body, aimed at a wall, and
under a roof beam that cut every head off -- each differently, which is what a
constant does when the scene changes underneath it. Bodies are seated the same
way, on the seat contact measured off each mesh -- not on its lowest point,
which is a foot, and resting a foot on the cushion floats the whole body. The
seat's own height is measured too, from the tallest occupant's seated hips,
and everything a seated body touches is placed off that: a declared fixture's
placement, the generic seat back. Heights fixed in absolute metres were only ever right for the stature
they were tuned against.

The shell is not hidden and not stood outside of: it is built after the camera
is solved, extended past it, and dressed. Both halves of that matter. Standing
inside the cabin nothing sits between lens and cast, and every pixel of the
frame carries structure -- where it carried grey emptiness instead, the
generator read "nothing specified here" and returned a white border with an
invented fourth person in it. Where it carried a flat grey wall, it returned
the picture in a narrow centre box with blank bands down both sides. So the
walls get what a cabin has: pillars between window bays, sill and cant rails,
door panels, headliner ribs, glass in panes across the rear. All of it is
sized off the solved shell, and the floor is left open because the location
says it is.

Dual-mode module. Imported in the venv it is the host-side spec builder
(`build_spec` reads the KB; `make_panel_greybox` drives BlenderBox). Run by
Blender --

    blender --background --python panel_greybox.py -- --kernel greybox --spec S

-- it is the greybox KERNEL (the bpy code at the bottom).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

try:
    import bpy
    import mathutils
except ImportError:                       # venv host — no Blender
    bpy = None
    mathutils = None
else:
    # Blender's Python ignores PYTHONPATH; BlenderBox passes this package's location.
    sys.path.insert(0, os.environ.get("PACE_CORE_PATH") or str(Path(__file__).resolve().parents[2]))

from pace_core.breakdown.world_state import states_for_prop
from pace_core.pai_compat import FIXTURE_ANCHORS, resolve_prop, resolve_prop_key
from pace_core.setup.location_shape import SHAPES, shape_from_stub, shape_maps_from_doc
from pace_core.paths import paths_for

DEFAULT_RES = (1280, 544)
# The shell this builds is a cabin: a floor, a roof, two side walls with
# window bays, and no front wall to look in through. That is right for a
# vehicle interior and wrong for everything else -- on the evaluation corpus, 22 of
# 36 panels are vehicle_interior and the other 14 are a highway and a crash
# site, where a cabin would be an invention rather than a proxy. A batch has
# to know the difference, so eligibility is a question that can be asked
# without building anything.
# The shell this builds is a cabin, so the panel has to be in one. That is a
# question about the location's SHAPE, and the KB already answers it -- read it
# through the shared derivation rather than keeping a second opinion here.
# `subway` is the shape the proxy library uses for seated rows inside a moving
# vehicle, which is what a cabin is.
# Which shapes THIS builder has a branch for. Deliberately its own set rather
# than an alias of the vocabulary: a shape can exist in the KB before anything
# here can build it, and eligibility has to say so. The test asserts it stays a
# subset of SHAPES, which is what stops the two drifting through a typo.
BUILDABLE_SHAPES = {"subway", "outdoor", "chamber", "office"}
assert BUILDABLE_SHAPES <= set(SHAPES)
# How far apart people stand on open ground: a body's width, not the
# location's.
SUBJECT_SPACING_M = 0.95

# Where a seat sits, as a fraction of cabin width either side of centre. The
# seats therefore span 2x this, and the rest of the floor is bare.
#
# It was 0.24, so seats covered 48% of the width and left a quarter of the
# cabin empty on each side. Bare floor in a control image is an invitation:
# one shot came back with a chair and a wraparound console invented
# into it. Staging the seats a vehicle actually has (`_unoccupied_slots`)
# closed that for cars carrying fewer people than seats, and did nothing for
# the case where the box itself is simply too wide for its seating.
#
# 0.375 puts the seats across 75% of the width. The cabins were narrowed by
# the reciprocal (x0.64) in the same change, so every seat stays at the exact
# metre position it already occupied and only the shell moved.
#
# Both numbers are downstream of a defect neither of them fixes. That change
# kept "every seat at the exact metre position it already occupied", so the
# outer hips are still 1538 mm apart. SAE J1100 hip room W5 for a WHOLE front
# row, trimmed wall to trimmed wall, is 1321 mm on a Corvette, 1379 on a
# Civic, 1452 on an R8 -- two of our occupants sit further apart than a real
# car's entire cabin is wide. 0.24 is the right number and it cannot be used
# yet: it is only right in a cabin built at the width the KB declares, and
# the shell is not (see the shell-sizing comment in the kernel, and `shell`
# in the build result, which reports the two side by side).
SEAT_SPREAD_FRAC = 0.375

# Front row to second row, SgRP to SgRP -- SAE J1100 calls it L50, the couple
# distance. Metric and taken from cars, not a fraction of whatever box the KB
# declared: the rows are where a car's rows are, and a longer cabin gets a
# boot, not more legroom.
#
# It was `-cabin[1] * 0.14`, which on a 2.1 m cabin puts the rows 504 mm
# apart. Two rows half a metre apart do not read as two rows -- three
# occupants staged that way merge into a single bench, which is what they
# looked like.
COUPLE_DISTANCE_M = 0.85


def row_seat_xs(n: int, fx: float) -> list[float]:
    """Where n occupants sit across ONE row, screen-left first.

    Screen-left is world +X and `subs` arrives ascending by declared screen x,
    so this descends. Three across is a rear bench, which is real; the
    front-centre position that used to appear is not a seat in any car and is
    unreachable from here, because a row's occupants are laid out together
    rather than each given an x by one rule and a y by another.
    """
    if n <= 0:
        return []
    if n == 1:
        return [fx]
    if n == 2:
        return [fx, -fx]
    if n == 3:
        return [fx, 0.0, -fx]
    step = (2 * fx) / (n - 1)
    return [fx - step * i for i in range(n)]
# 景别 → how much of a body is in frame, as a fraction of its own measured
# height taken down from the top of the head, and how much air to leave around
# what is framed. Anatomy, not taste: a close-up is head-and-shoulders because
# that is where a head and shoulders end, and `wide` keeps the 1.3 the solved
# framing was verified at so the shots already rendered do not move.
#
# Ignoring shot_size framed a declared close-up exactly like a declared wide --
# the same class of mistake as the azimuth constant, and invisible until you
# put the two frames side by side.
SHOT_SIZE_FRAMING = {
    "establishing":    (1.00, 2.20),
    "master":          (1.00, 1.70),
    "wide":            (1.00, 1.30),
    "full":            (1.00, 1.20),
    "medium_full":     (0.75, 1.18),
    "medium":          (0.55, 1.15),
    "medium_close_up": (0.40, 1.12),
    "close_up":        (0.26, 1.10),
    "extreme_close_up": (0.15, 1.06),
}
DEFAULT_FRAMING = SHOT_SIZE_FRAMING["wide"]
# How the aim arbitrates between subjects whose declared positions cannot all
# be held. A frame has one horizontal degree of freedom and a two-shot declares
# two positions, so somebody's declared position is going to be wrong; this
# says whose. At 0 the aim minimises the SUM of the errors, which sounds
# neutral and is not: a sum cannot tell "everyone slightly off" from "one
# subject exact and one ruined", and it reliably picks the second -- measured
# on this corpus, minimising the sum reproduces the old focus-centred aim
# almost exactly (7.66% vs 7.78% mean error) and loses the same 7 subjects off
# frame. At 1 the aim equalises the errors and singles nobody out.
#
# Measured over the 29 multi-subject panels, moving from 0 to 0.30 costs 0.70
# points of mean placement error and buys back all 7 lost subjects, drops the
# worst-served subject from 13.9% to 10.0%, and cuts the gap between best- and
# worst-served from 13.8% to 4.9%. Declared left-to-right order holds at 29/29
# throughout and delivered body height moves under 0.2%, so the trade is paid
# for in absolute placement only. The frontier is flat above 0.30.
#
# It is a constant here because this corpus has one answer, but it is the kind
# of decision a production would want to set per panel, and it is deliberately
# a named, documented number rather than a rule buried in the solve.
AIM_ARBITRATION_W = 0.5
# Half a metre either side of the fit's own aim, at 1 cm resolution. Wider than
# any offset the solve actually chooses on this corpus.
_AIM_SEARCH_M, _AIM_STEP_M = 1.6, 0.01

# ── seeing past the set ──────────────────────────────────────────────────
#
# How far the lens may move to see a head the set hides, per declared angle.
# The class is what the panel declared: raised from 5 to 12 degrees an
# eye-level shot is still eye level, raised to 20 it is a high angle nobody
# asked for. Degrees of elevation above the horizontal through the aim.
ELEVATION_RANGE_DEG = {"low_angle": (-14.0, -4.0), "high_angle": (10.0, 24.0)}
DEFAULT_ELEVATION_RANGE_DEG = (2.0, 12.0)
# Either side of the declared azimuth: a front shot stays a front shot.
AZIMUTH_TOLERANCE_DEG = 12.0
# A "dutch" angle that gives no degrees: a conventional cant, clockwise.
DUTCH_ROLL_DEG = 15.0


def _roll_of(extr: dict) -> float:
    """The cant about the optical axis a panel asks for, in degrees.

    `roll_deg` when the panel gives one (+ turns the frame clockwise), so a
    high or low camera can be canted without losing its elevation; otherwise
    DUTCH_ROLL_DEG for the vocabulary's "dutch" angle, and none.
    """
    r = (extr or {}).get("roll_deg")
    if isinstance(r, (int, float)) and not isinstance(r, bool):
        return float(r)
    return DUTCH_ROLL_DEG if (extr or {}).get("angle") == "dutch" else 0.0


def _angle_class(extr: dict) -> str | None:
    """The elevation class a panel declares, in the elevation table's terms.

    The type vocabulary spells the raised and lowered classes `high` and
    `low`; the corpus writes `high_angle` and `low_angle`, and only those
    reached the table -- a pilot shot declared `high` was built at the
    eye-level default.
    """
    a = (extr or {}).get("angle")
    return {"high": "high_angle", "low": "low_angle"}.get(a, a)
# Below this share of a head's sample points reaching the lens the kernel
# looks for another pose, and after the search it removes the furniture in
# the way. The gate's floor (greybox_gate.MIN_READ_POINT_VISIBLE) is lower:
# a head half behind a desk passes the gate and is still worth clearing.
HEAD_CLEAR_SHARE = 0.9
# Set pieces too large to be furniture, which the occlusion step never
# removes: a slab this long is a wall, a floor or a ceiling of the location.
_MAX_REMOVABLE_SPAN_M = 6.0
# Shell and ground pieces, by the names the kernel gives them. They are what
# a frame is built of, not what stands in front of it.
_STRUCTURAL_PREFIXES = ("floor", "roof", "wall_", "glass_", "sill_", "cant_",
                        "door_", "kick_", "pillar_", "rib_", "headliner",
                        "rear_", "parcel", "ground", "mass_", "seat_",
                        "cam_target", "body_")


def camera_search_order(elev0: float, yaw0: float, elev_range: tuple[float, float],
                        yaw_tol: float, *, elev_step: float = 2.0,
                        yaw_step: float = 3.0) -> list[tuple[float, float]]:
    """Camera poses to try, as (elevation, azimuth) in degrees, least change first.

    The solved pose comes first, then every pose on a grid inside the declared
    angle class and azimuth tolerance, ordered by how far it moves the lens --
    so the first pose that clears every head is also the smallest departure
    from what the panel asked for.
    """
    lo, hi = elev_range
    n_e = int(round((hi - lo) / elev_step))
    n_y = int(yaw_tol // yaw_step)
    grid = [(lo + i * elev_step, yaw0 + k * yaw_step)
            for i in range(n_e + 1) for k in range(-n_y, n_y + 1)]
    grid.sort(key=lambda p: (abs(p[0] - elev0) + abs(p[1] - yaw0),
                             abs(p[1] - yaw0), p[0]))
    return [(elev0, yaw0)] + [p for p in grid if p != (elev0, yaw0)]


def inside_shell(x: float, y: float, z: float, bounds: dict | None,
                 clearance: float = 0.15) -> bool:
    """Can the lens stand here? Inside the built shell, or above open ground."""
    if bounds is None:
        return z > 0.3
    return (z < bounds["top_z"] - clearance
            and abs(x) < bounds["half_w"] - clearance
            and y < bounds["front_y"] - clearance)


# Where a limb's joints sit along it, as a share of the limb's reach from its
# root (shoulder, hip). Adult proportions: the elbow is about 0.42 of the way
# from shoulder to fingertip and the wrist 0.76; the knee 0.46 of hip to sole
# and the ankle 0.92. Reach is straight-line, so a bent limb lands its joints
# a little early -- close enough to name which joint a frame edge is at.
_JOINT_BANDS = {"arm": (("elbow", 0.38, 0.46), ("wrist", 0.72, 0.79)),
                "leg": (("knee", 0.42, 0.50), ("ankle", 0.88, 0.95))}


def joint_points(parts: dict) -> dict:
    """Joint positions from an SMPL-X proxy's six part groups.

    `parts` maps head / torso / arm_l / arm_r / leg_l / leg_r to point lists.
    The proxy carries no skeleton, but its parts meet where the joints are:
    the neck is where the head meets the torso, a shoulder or hip where a limb
    does, and the joints along a limb fall at fixed shares of its reach.
    """
    torso = parts.get("torso") or []
    if not torso:
        return {}

    def mean(pts):
        n = len(pts)
        return tuple(sum(p[i] for p in pts) / n for i in range(3))

    def nearest(pts, to, share=0.1):
        k = max(1, int(math.ceil(share * len(pts))))
        return sorted(pts, key=lambda p: math.dist(p, to))[:k]

    centre = mean(torso)
    out = {}
    if parts.get("head"):
        out["neck"] = mean(nearest(parts["head"], centre))
    for key in ("arm_l", "arm_r", "leg_l", "leg_r"):
        pts = parts.get(key) or []
        if not pts:
            continue
        limb, side = key.split("_")
        root = mean(nearest(pts, centre))
        out[("shoulder" if limb == "arm" else "hip") + "_" + side] = root
        d = [math.dist(p, root) for p in pts]
        reach = max(d) or 1.0
        for name, a, b in _JOINT_BANDS[limb]:
            band = [p for p, di in zip(pts, d) if a * reach <= di <= b * reach]
            if band:
                out[f"{name}_{side}"] = mean(band)
    return out

# Composition patterns that put more than the focus in frame. `single` is the
# only one where a coverage target for one subject is unambiguous: asking for
# one body at 55% of a frame that must also hold two others over-constrains it,
# and on this project coverage_pct is 55 on every panel including the crowd
# wides, so read literally it would turn every wide into a close-up.
MULTI_SUBJECT_FRAMINGS = {"two_shot", "crowd"}
# The lens stands outside the windscreen and the shell is stretched forward to
# enclose it, which only works along the axis the cabin opens on.
BUILDABLE_POSITIONS = {"front": 0.0, "three_quarter": 25.0}
# Over-the-shoulder, in metres off the near subject's own head. Not an
# azimuth, which is why it is not in the table above: the lens is placed
# from the pair (see the kernel), and these are the offsets that make a
# shoulder read as a shoulder rather than as an obstruction. Behind, so the
# near head is between lens and far face. Below the crown, so the frame holds
# the back of the head, the ears and the neck-to-shoulder line: at 0.34 m
# behind and 6 cm above it, the foreground was a cropped skull, and every
# delivered frame repainted it as upholstery or a dark panel; at 0.75 m the
# near head still filled the frame down to the neck, with no shoulder line,
# and was repainted as a seat headrest. At 1.00 m the shoulder line is in
# frame and the back of the head is delivered as one. Across is a
# floor; the kernel raises it until the near head keeps OTS_CLEARANCE_M off
# the line of sight to the far face (roughly a head's radius plus half a
# face). Face drop aims below the crown, where the eyes are.
OTS_BEHIND_M, OTS_ACROSS_M, OTS_RISE_M, OTS_FACE_DROP_M = 1.00, 0.30, -0.15, 0.12
OTS_CLEARANCE_M = 0.23
# How far behind the near subject's crown the lens stands, per declared shot
# size. An over-the-shoulder's camera is placed from the PAIR -- the near
# shoulder and the far face -- so the fit-to-cast distance solve that every
# other position runs is overwritten, and `creative_intent.shot_size` reached
# the build result and nothing else: a panel switched from medium to close_up
# reported the new size and rendered subjects the same size to four decimals.
# The standoff is what a size means for this position, so the ladder sets it.
# Starting values on the same footing as the gate's thresholds, read off one
# staged pair (the far subject's head went 0.261 to 0.184 of frame height
# across 0.80 to 1.70 m), not a corpus measurement.
OTS_BEHIND_BY_SIZE = {
    "extreme_close_up": 0.70, "close_up": 0.85, "medium_close_up": 1.05,
    "medium": 1.30, "medium_full": 1.55, "full": 1.70, "wide": 1.85,
    "master": 2.00, "establishing": 2.00,
}
# `extrinsics.angle` was inert here for the same reason the size was: the spec
# resolved it to elevation_deg (-8 / 5 / 14) and the OTS placement never read
# it, so `overhead` and `low_angle` built byte-identical cameras. The lens
# height is solved from it in the kernel, where the face being aimed at is
# known -- an over-the-shoulder that only approximated the angle missed the
# declared elevation by six degrees, which is the one thing every other
# position reproduces exactly.
#
# A lens above the near crown sees the top of a skull and no shoulder, which is
# not an over-the-shoulder however the angle is declared, so the solved height
# is clamped to this band off the crown and the clamp is reported on the build.
OTS_RISE_RANGE_M = (-0.35, -0.05)


OTS_EYE_LEVEL_DEG = 5.0


def _elevation_deg(extr: dict) -> float:
    """The elevation an angle class declares. One table, two readers: the
    spec's own `elevation_deg` and the over-the-shoulder rise map."""
    return {"low_angle": -8.0, "high_angle": 14.0}.get(_angle_class(extr),
                                                       OTS_EYE_LEVEL_DEG)


def _ots_offsets(shot_size: str | None) -> dict:
    """The standoff an over-the-shoulder's declared size implies.

    The height is not here: it is solved in the kernel against the face the
    lens aims at, which this side of the build has no heights for.
    """
    return {"behind_m": OTS_BEHIND_BY_SIZE.get(
        shot_size or "", OTS_BEHIND_BY_SIZE["medium"])}
# Fallback when no proxy mesh resolves, so a spec can still be built and the
# caller can report the missing mesh rather than dying inside Blender.
DEFAULT_SEAT_TOP = 0.45


# ── host side: read the panel, measure the proxies, build the spec ────────

def _mesh_for(age_state: str, meshes_dir: Path, pose: str = "sitting",
              character: str = "") -> str:
    """Character, stature and pose -> which SMPL-X mesh.

    A per-character mesh wins when one exists, so a cast of five is staged as
    five bodies rather than one. Without it the lookup falls back to stature:
    the project ships 125 cm and 175 cm variants of each pose, and an adult and
    an 18-year-old both read as the taller one. That fallback is what every
    panel used before per-character proxies were baked, and it is why one body
    stood in for the whole cast in every greybox: this function was never given
    a character to look up.

    A pose with no mesh on disk degrades along `_POSE_FALLBACK` rather than
    failing, because a project generated before the vocabulary grew has only
    the original four files. `lying` resolves to a STANDING mesh on purpose:
    SMPL-X body_pose is relative to the pelvis, so lying is orientation, and
    the assembler lays the body down itself.
    """
    young = any(t in (age_state or "") for t in ("child", "7", "kid"))
    tag = "125" if young else "175"
    # Per-character first, walking the same pose-fallback chain as the stature
    # proxies: a character who has a standing mesh but no kneeling one should
    # land on their own standing body, not on somebody else's kneel.
    if character:
        seen, want = set(), pose or "standing"
        while want and want not in seen:
            seen.add(want)
            cand = meshes_dir / f"smplx_{want}_{character}.obj"
            if cand.is_file():
                return str(cand)
            want = _POSE_FALLBACK.get(want)
    seen, want = set(), pose or "standing"
    last = want
    while want and want not in seen:
        seen.add(want)
        cand = meshes_dir / f"smplx_{want}_{tag}.obj"
        if cand.is_file():
            return str(cand)
        last, want = want, _POSE_FALLBACK.get(want)
    # Nothing on disk. Name the END of the fallback chain, not the pose that
    # was asked for: that is the file the panel would actually have loaded, so
    # it is the one the caller's "missing proxy mesh" error should name.
    # Naming `lying` would tell an operator to generate smplx_lying_*.obj,
    # which this project deliberately does not have -- a lying body is a
    # standing proxy the assembler lays down.
    return str(meshes_dir / f"smplx_{last}_{tag}.obj")


# The KB writes `pose` as prose ("kneeling or crouched grief pose", "lying
# motionless on the road"), because it is authored for a reader. The project
# ships four proxies -- sitting and standing at two statures -- so the job here
# is to land that prose on the nearest one that exists, and to say when it
# cannot.
#
# `sitting` doubles as the kneeling/crouching proxy on open ground: the hips
# and knees are already bent 1.5 rad, and _mesh_for's own docstring notes that
# a seated proxy in a street "reads as someone kneeling in traffic". `lying` is
# a standing proxy laid down by the kernel, not a different mesh -- SMPL-X
# body_pose is relative to the pelvis, so lying down is orientation, not joints.
_POSE_WORDS = (
    # Order is precedence, most specific first. `kneeling` used to live inside
    # the sitting bucket -- the comment said a seated proxy "reads as someone
    # kneeling in traffic" -- so every authored kneel in the corpus staged as
    # somebody sitting down. It has its own mesh now, so it is tested first.
    # "motionless" was here and describes STILLNESS, not posture: a body can
    # be motionless standing, sitting or lying. It only ever looked right
    # because the corpus that taught this table wrote "lying motionless",
    # where "lying" already matches. On another production it put 10 subjects on the floor,
    # including "standing tall, motionless, watching building".
    ("lying",    ("lying", "lies", "lay", "prone", "supine",
                  "collapsed", "sprawled", "unconscious", "on the ground")),
    ("kneeling", ("kneeling", "kneels", "kneel", "crouched", "crouching",
                  "crouch", "squat", "squatting", "hunched over")),
    ("sitting",  ("sitting", "sits", "seated", "seat")),
    ("walking",  ("walking", "walks", "running", "runs", "stepping", "steps",
                  "limps", "limping", "approaches", "approaching", "strides")),
    # An upper-body action performed on your feet: the arms are what reads,
    # and a plain standing proxy gives the silhouette of someone waiting.
    ("reaching", ("pushes", "pushing", "push", "reaches", "reaching", "grabs",
                  "grabbing", "braces", "bracing", "shoves", "shoving",
                  "presses", "pressing", "holds out", "bangs", "banging")),
    ("standing", ("standing", "stands", "upright")),
)

# What to stage when a pose has no mesh in this project. A project generated
# before the vocabulary grew has four files, not ten, and must keep working:
# the nearest silhouette is better than a hard failure, and far better than
# silently substituting something that reads as a different action.
_POSE_FALLBACK = {"kneeling": "sitting", "walking": "standing",
                  "reaching": "standing", "lying": "standing"}


# Compiled once, word-bounded. Phrases ("on the ground", "holds out") work
# the same way -- the boundary is on the phrase's outer edges.
_POSE_PATTERNS = tuple(
    (key, re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"))
    for key, words in _POSE_WORDS
)


def pose_key_for(pose_text: str | None, default: str = "standing") -> str:
    """Which shipped proxy a subject's authored pose asks for.

    Matched on word boundaries. As a substring test this read "lay" out of
    "player", "layer" and "overlay" and "lies"/"lying" out of "flies"/
    "flying", which put all eight of one corpus's matches on the floor --
    among them a shot whose subject walks up to a wall, since `lying` is
    tested before `walking` and the first bucket to match wins."""
    s = (pose_text or "").strip().lower()
    if not s:
        return default
    for key, rx in _POSE_PATTERNS:
        if rx.search(s):
            return key
    return default


def _seat_contact_z(mesh: str) -> float:
    """Height at which this proxy meets whatever it sits or stands on.

    Not its lowest point: a sitting proxy is authored standing on the floor
    with its feet forward and its hips already at seat height, so its lowest
    point is a foot and resting THAT on the cushion floats the whole body.
    The contact is the underside of the buttocks, which is the lowest point of
    the body's rear slab -- the feet are in the front slab, so whichever slab
    sits higher is the one bearing weight. A standing proxy answers ~0 to the
    same question, which is also correct, so this needs no per-pose branch.
    """
    vs = [[float(n) for n in ln.split()[1:4]]
          for ln in Path(mesh).read_text().splitlines() if ln.startswith("v ")]
    if not vs:
        raise ValueError(f"no vertices in {mesh!r}")
    ys = [v[1] for v in vs]
    lo, hi, span = min(ys), max(ys), max(ys) - min(ys)
    front = min(v[2] for v in vs if v[1] <= lo + 0.2 * span)
    back = min(v[2] for v in vs if v[1] >= hi - 0.2 * span)
    return max(front, back)


def _proxy_length_m(mesh: str) -> float:
    """Head-to-foot length of a proxy (+Z up), which is what it spans lying."""
    zs = [float(ln.split()[3])
          for ln in Path(mesh).read_text().splitlines() if ln.startswith("v ")]
    if not zs:
        raise ValueError(f"no vertices in {mesh!r}")
    return max(zs) - min(zs)


# Where a fixture sits, in units of the shell it sits in. Fractions rather
# than metres so one placement survives a cabin of different proportions --
# a console spans the width of whatever cabin it is in.
# One list, kept in pai_compat because the prompt compiler reads it too: a
# prop the greybox will not stage is one the compiler must not name away from
# the cast (prop_left_unstaged_away_from_cast).
_FIXTURE_ANCHORS = FIXTURE_ANCHORS

# When a staged prop counts as in frame (see the prop-visibility pass in the
# kernel): its projected bounding box covers PROP_IN_FRAME_MIN of the frame,
# and either at least PROP_SHOWN_MIN of that box falls inside the frame or
# the box fills PROP_FILLS_FRAME of it. The second decides the cabin console:
# the camera stands ahead of the windscreen, the console runs across the
# cabin just below the frame, and two stubs of it at the bottom corners are
# not the full-width panel the prompt would describe. The third keeps a prop
# too close to be framed whole -- the wreck beside the cast in scene 7, most
# of whose box lies outside a frame it still fills.
PROP_IN_FRAME_MIN = 0.005
PROP_SHOWN_MIN = 0.25
PROP_FILLS_FRAME = 0.25

# `on_surface` is measured against another fixture rather than against the
# shell, because that is how the object is actually described: a monitor is on
# a desk, and it is on that desk wherever in the room the desk is. Every other
# anchor answers to the shell, which is why this corpus could stage a console
# spanning a cabin but not a keyboard -- and why an office film had fifteen
# props and no staged furniture at all.
#
# The host is named by prop_id and resolved through the same aliases the
# prompt uses. A fixture whose host is not in THIS panel is not built: a
# monitor with no desk under it is a floating box, and the shell coordinates
# that would put it somewhere plausible are exactly the guess this module
# refuses to make elsewhere. The skip is reported, not silent.
_SURFACE_ANCHOR = "on_surface"

# How far behind the cast a set object may still reach and be kept. A desk the
# cast stands at ends within centimetres of them; a row of desks between them
# and the lens does not.
_SET_CULL_MARGIN_M = 0.15

# Below this, an object is ground rather than furniture: a floor, a ceiling
# slab, a ground seam. Everyone stands inside the floor's footprint, so the
# containment test has to know the difference.
_SET_CULL_MIN_H_M = 0.25

# How a mesh is fitted into the span its placement declares. The span answers
# WHERE a fixture goes; the mesh answers WHAT SHAPE it is; and when the two
# disagree only the author knows which one was the claim.
#
#   stretch  scale each axis to fill the box. The span wins outright, and the
#            mesh's own proportions are discarded.
#   uniform  one scale, the largest that fits inside the box. The mesh wins,
#            and the fixture ends up smaller than its slot on every axis but
#            the tightest.
#   x|y|z    one scale, taken from the named axis. For the common case where
#            the span states one real dimension -- a console spans the cabin
#            width -- and the other two were written to look reasonable.
_FIT_POLICIES = ("stretch", "uniform", "x", "y", "z")
_FIT_DEFAULT = "stretch"


def fit_scale(ext, size, policy=_FIT_DEFAULT):
    """Per-axis scales to put a mesh of extent `ext` into a box of `size`.

    Returns `(scales, anisotropy)`. `anisotropy` is always the ratio between
    the largest and smallest PER-AXIS factor, whatever policy is chosen: it
    measures how far the mesh's proportions are from the declared box, which
    is a fact about the disagreement and not about how we settled it. A
    fixture reported at 14x was not modelled for the slot it is standing in.

    Measured in-build on the evaluation corpus's cabin under the old unconditional
    per-axis fit: car_console 2.29x, cabin_panels 14.60x -- the wall lining
    squeezed to 3.9% of its own width to reach a 7 cm door-card slot.

    Read it in-build and not from the file. Blender's glTF importer applies
    the Y-up to Z-up conversion, so measuring the same GLB's raw vertices
    outside Blender returns a PERMUTED extent and a different, wrong
    anisotropy -- 3.90x for cabin_panels rather than 14.60x.
    """
    per = [(size[i] / ext[i]) if ext[i] > 1e-6 else 1.0 for i in range(3)]
    lo, hi = min(per), max(per)
    anis = (hi / lo) if lo > 1e-9 else float("inf")
    # An unrecognised policy is a typo in the KB, and it resolves the same way
    # here as it does in `_panel_fixtures` -- the two must not disagree, or a
    # fixture is reported under one policy and built under another.
    if policy not in _FIT_POLICIES:
        policy = _FIT_DEFAULT
    if policy == "stretch":
        return per, anis
    if policy in ("x", "y", "z"):
        u = per["xyz".index(policy)]
    else:
        u = lo
    return [u, u, u], anis


def _record_surface(surfaces: dict, fx: dict, pos, size) -> None:
    """Remember where a fixture was built, so another can be put on top of it.

    Keyed by registry key rather than by the panel's reference, because the
    prop that hosts and the prop that sits on it are written in the registry
    once and spelled by the panel however that shot's enrichment spelled them.

    The DECLARED box, not the object's bounding box. Under the default
    `stretch` fit the two are the same, and reading the object back instead
    would mean forcing a depsgraph update per fixture to get a matrix that is
    not stale. The cost is that a fixture fitted `uniform` sits smaller than
    its slot and its guest floats by the difference -- which is the same gap
    `fit_scale` already reports as anisotropy, and is a disagreement between
    mesh and span rather than a placement error.
    """
    key = fx.get("key") or fx.get("id")
    if key:
        surfaces[key] = (tuple(pos), tuple(size))


# A registered location mesh is deliberately NOT used as the shell, and the
# two functions that used to resolve one were removed rather than finished.
# They read well and were never read: `_shell_mesh` picked a location's GLB out
# of the KB registry, `build_spec` put it in the spec as "shell_mesh", and
# nothing anywhere consumed it. The shell has always been the procedural one
# below.
#
# Three reasons it should stay that way:
#
#   The proportions already arrive without it. `cabin` comes from the stub's
#   scale_meters, declared in the KB and metric, which is what a registered
#   mesh would have been wanted for. A generated mesh is not metric.
#
#   What is left is surface detail, and that is the one thing that must not go
#   there. The shell is built AFTER the camera is solved and extended past it
#   so nothing sits between lens and cast (see the module docstring, which
#   records what happened when that was violated: a white border with an
#   invented fourth person, and the picture squeezed into a narrow centre box).
#   An arbitrary closed mesh at arbitrary scale with no open camera side is
#   that failure by construction -- reproduced with cabin_panels, whose
#   wraparound became a tub between lens and cast.
#
#   And what image-to-3D returns for a location is not a room. All four
#   the evaluation corpus's locations were run through it: two collapsed (a hairline,
#   a blob), and the two that worked came back as car EXTERIORS -- the tool
#   reconstructs a silhouette, the inverse of a space that contains a camera.
#
# The deleted `_mesh_is_substantial` also carried a warning worth keeping: every
# location in this KB declares a `location_mesh` pointing at an 8-vertex box
# written by the library proxy step, and modelling family_car from its PACE
# anchor card produced a flat pictogram. Any future attempt to feed a mesh in
# here has to prove it is an interior, not merely that it is geometry.


def _panel_fixtures(paths, setup: dict) -> list[dict]:
    """Fixed props this panel declares, with where they go and what they are.

    A prop is furniture when its record says where it stands. Nothing here
    knows what a console is: the KB says a prop is anchored to the front of
    the shell and how much of the width it takes, and the builder places
    whatever it is told about. That is the difference between a pipeline and
    one film's set dressing compiled into it -- the shared builder had
    `box("console", ...)` in it, which is one production's car living in
    code every other project also runs.

    A `model_3d.mesh_file` that exists on disk is used in place of the
    primitive, which is how a prop's design stops being re-invented per panel:
    a box is a different console every render, a mesh is the same one.
    """
    try:
        props_kb = json.loads(Path(paths.props_file).read_text())
    except Exception:                                          # noqa: BLE001
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for entry in (setup.get("props") or []):
        pid = (entry or {}).get("prop_id") if isinstance(entry, dict) else None
        rec = resolve_prop(pid, props_kb) if pid else {}
        if not rec:
            continue                      # resolve_prop returns {} on a miss
        place = rec.get("placement")
        if not isinstance(place, dict) or place.get("anchor") not in _FIXTURE_ANCHORS:
            continue                      # a prop with no declared place is not furniture
        ident = resolve_prop_key(pid, props_kb) or pid
        if ident in seen:
            continue      # two spellings of one object are one piece of furniture
        seen.add(ident)
        mesh = ((rec.get("model_3d") or {}).get("mesh_file") or "")
        mesh_p = Path(mesh) if mesh and mesh not in ("None", "null") else None
        if mesh_p is not None and not mesh_p.is_absolute():
            mesh_p = Path(paths.storage) / mesh_p
        out.append({
            "id": pid,
            "anchor": place["anchor"],
            # Which way it faces. A generated mesh has whatever orientation the
            # reconstructor gave it, and a wraparound fixture placed backwards
            # shows the camera its outer wall and buries the cast behind it.
            "yaw_deg": float(place.get("yaw_deg") or 0.0),
            # Fractions of the shell, and of the seat pitch for per_seat.
            "span": [float(v) for v in (place.get("span") or [0.9, 0.1, 0.1])],
            "offset": [float(v) for v in (place.get("offset") or [0.0, 0.0, 0.0])],
            "mesh": str(mesh_p) if (mesh_p and mesh_p.is_file()) else None,
        })
        # A beat that leaves this prop lying on someone says where it is, and
        # that outranks where the registry parks it. Written only when set,
        # for the anchor-field reason below.
        if entry.get("rests_on"):
            out[-1]["on_subject"] = str(entry["rests_on"])
        # A screen the panel declares powered off is dark in the frame, so the
        # control image has to carry it: the prompt alone reached the words
        # and not the pixels the sampler starts from. Written only when the
        # state is declared, for the anchor-field reason below, so a panel
        # that declares nothing keeps the anchor it already has.
        if any(a in ("power", "display") and v == "off"
               for a, v in states_for_prop(entry)):
            out[-1]["dark"] = True
        # `key` and `host` are written only when they carry information, for
        # the same reason `fit` is: `fixtures` is an anchor field, and a key
        # present on every fixture restamps anchor_version across every
        # project for geometry that did not move.
        key = resolve_prop_key(pid, props_kb)
        if key and key != pid:
            out[-1]["key"] = key
        if place["anchor"] == _SURFACE_ANCHOR:
            out[-1]["host"] = resolve_prop_key(place.get("host") or "", props_kb)

    # A prop that declares a host declares that the host is present. The
    # enrichment names the object a shot is ABOUT -- four of another production's scenes
    # name a monitor and only two name any desk -- so requiring the panel to
    # list both would leave the monitor unbuilt in most of the film it appears
    # in. This is not the builder inventing set dressing: the relation is
    # written in this film's registry, and honouring it is the same as
    # honouring the span beside it. Marked `implied` so the spec says which
    # furniture the panel asked for and which its props brought with them.
    for fx in list(out):
        host = fx.get("host")
        if not host or host in seen:
            continue
        rec = (props_kb.get("props", props_kb) or {}).get(host) or {}
        place = rec.get("placement") or {}
        if place.get("anchor") not in _FIXTURE_ANCHORS or place["anchor"] == _SURFACE_ANCHOR:
            continue          # only a self-standing host can be implied, one level
        seen.add(host)
        out.append({
            "id": host, "anchor": place["anchor"], "implied": True,
            "yaw_deg": float(place.get("yaw_deg") or 0.0),
            "span": [float(v) for v in (place.get("span") or [0.9, 0.1, 0.1])],
            "offset": [float(v) for v in (place.get("offset") or [0.0, 0.0, 0.0])],
            "mesh": None,
        })
        # How the mesh is reconciled with the span above. Written only when it
        # is not the default, because `fixtures` is an anchor field: a key
        # present on every fixture would change every anchor_version in every
        # project for geometry that did not move, and the gate would read a
        # whole corpus as stale. Unrecognised values are dropped rather than
        # failing the build -- a typo in the KB should cost a fixture its fit
        # policy, not the panel its control geometry.
        if place.get("fit") in _FIT_POLICIES and place["fit"] != _FIT_DEFAULT:
            out[-1]["fit"] = place["fit"]
    return out


def _cabin_slots(n: int, cabin: list) -> list:
    """Where a vehicle's seats are, occupied or not, in cabin coordinates.

    The same fx/fy/ry the occupied placement uses, so an empty seat sits on
    the grid its neighbours do. Rows fill front-first, which is how a car is
    used and how the location stubs describe these vehicles ("front seats").
    """
    fx = cabin[0] * SEAT_SPREAD_FRAC
    fy, ry = cabin[1] * 0.10, -cabin[1] * 0.22
    if n <= 2:
        return [(fx, fy), (-fx, fy)][:max(n, 0)]
    if n == 3:
        return [(fx, fy), (-fx, fy), (0.0, ry)]
    return [(fx, fy), (-fx, fy), (fx, ry), (-fx, ry)][:n]


def _unoccupied_slots(seat_count: int, occupied: list, cabin: list) -> list:
    """The vehicle's seats that nobody in this shot is sitting in.

    Seats were staged one per SUBJECT, so a two-person shot in a four-seat car
    put two chairs on the floor and left the rest of it bare — and bare floor
    in a control image is an invitation rather than a constraint. Measured on
    one shot: the greybox staged two seats, and the delivered panel
    came back with a third chair and a wraparound console invented into the
    empty half of the cabin. How many seats a car has is a fact about the set,
    and the set is what the greybox exists to state.

    Each occupied position claims the slot nearest it, so the empties are
    whatever the cast did not take, wherever the tuned placement put them.
    """
    if not seat_count or seat_count <= len(occupied):
        return []
    slots = _cabin_slots(int(seat_count), cabin)
    free = list(range(len(slots)))
    for ox, oy in occupied:
        if not free:
            break
        near = min(free, key=lambda i: (slots[i][0] - ox) ** 2 + (slots[i][1] - oy) ** 2)
        free.remove(near)
    return [slots[i] for i in free]


def _facing_from_gaze(subs: list, places: list, i: int) -> float:
    """Turn a body toward the subject its declared gaze names.

    `facing_deg` is an offset from the location's own facing -- the assembler
    applies `180 + facing_deg` -- so 0.0 keeps every body square to the
    windscreen, which is the staging every panel had before this existed.

    Derived only for a gaze that names another subject IN THIS PANEL. A gaze
    at a prop or an off-frame direction leaves the seat alone: the head turns,
    the body does not, and rotating the whole proxy for a glance would move a
    silhouette the framing solve is fitting. Without this the eye-line clause
    could not fail, because two subjects declared looking at each other were
    staged facing the same way (Gaze's own docstring names eyeline-match
    continuity as what it is for).
    """
    g = subs[i].get("gaze") or {}
    if g.get("target_type") != "character":
        return 0.0
    j = next((k for k, s in enumerate(subs)
              if k != i and s.get("character_id") == g.get("target_ref")), None)
    if j is None:
        return 0.0
    dx, dy = places[j][0] - places[i][0], places[j][1] - places[i][1]
    if not (dx or dy):
        return 0.0
    # The default body faces +Y and atan2 measures from +X, so the quarter
    # turn between the two axes comes off the derived bearing.
    return math.degrees(math.atan2(dy, dx)) - 90.0


# Garment tones for the proxy, read off the registered costume. The greybox is
# one flat grey, and denoised from it at 0.55 the sampler decides a garment
# from the body's shape alone -- so the same declared shirt came back a
# different shirt in every panel of one scene, with the character's own
# plate as a reference and without. A tone in the control image is a decision
# the sampler starts from rather than one it makes. Values are the grey the
# garment should read as, on the 0..1 scale of the greybox's own 0.55.
_GARMENT_TOPS = ("t-shirt", "tshirt", "tee", "tank", "vest", "shirt", "blouse",
                 "sweater", "jumper", "hoodie", "jacket", "coat", "top", "dress")
_SHORT_SLEEVED = ("t-shirt", "tshirt", "tee", "tank", "vest")
_GARMENT_BOTTOMS = ("jeans", "trousers", "pants", "slacks", "shorts", "skirt", "leggings")
_GARMENT_TONES = (("black", 0.12), ("charcoal", 0.22), ("navy", 0.25), ("dark", 0.28),
                  ("grey", 0.45), ("gray", 0.45), ("khaki", 0.60), ("beige", 0.68),
                  ("faded", 0.72), ("light", 0.72), ("pale", 0.75), ("cream", 0.82),
                  ("white", 0.88))


def garment_tones(costume: str | None) -> dict:
    """{"top": {"tone", "sleeve"}, "bottom": {"tone"}} for the garments a
    costume names with a colour; a garment named without one is left grey."""
    out: dict = {}
    for phrase in re.split(r",| and | with ", (costume or "").lower()):
        words = re.findall(r"[a-z-]+", phrase)
        tone = next((t for w in words for name, t in _GARMENT_TONES if w == name), None)
        if tone is None:
            continue
        if "top" not in out and any(g in words for g in _GARMENT_TOPS):
            out["top"] = {"tone": tone,
                          "sleeve": "short" if any(g in words for g in _SHORT_SLEEVED) else "long"}
        elif "bottom" not in out and any(g in words for g in _GARMENT_BOTTOMS):
            out["bottom"] = {"tone": tone}
    return out


def hair_style(descriptor: str | None) -> str | None:
    """The HAIR_STYLES key a character's own description asks for, or None.

    Only hair the proxy can carry as a close shell: short or curly. Longer
    hair is left bald rather than capped wrong, and a description that names
    no hair gets none.
    """
    from pace_core.compilers.compile_common import _hair_phrase

    words = set((_hair_phrase(descriptor or "") or "").lower().replace("-", " ").split())
    if words & {"curly", "coily", "afro"}:
        return "curly"
    if words & {"short", "cropped", "buzzed", "close", "crew"}:
        return "short"
    return None


def build_spec(project: str, scene_id: str, panel_id: str,
               out_png: str | Path, res: tuple[int, int] = DEFAULT_RES, *,
               scene: dict | None = None, dress: bool = False) -> dict:
    """Everything the kernel needs, read off the panel and the location stub.

    `scene` builds from a scene document that is not installed in the
    project (the paper's beat pilot); `dress` tones each proxy's garments from
    its registered costume (see garment_tones).
    """
    p = paths_for(project)
    if scene is None:
        scene = json.loads((Path(p.scenes_dir) / f"{scene_id}.json").read_text())
    shot = panel = None
    for sh in scene.get("shots") or []:
        for pl in sh.get("panels") or []:
            if pl.get("id") == panel_id:
                shot, panel = sh, pl
    if shot is None:
        raise ValueError(f"panel {panel_id!r} not in {scene_id}")
    # Same inheritance the compilers apply: scene defaults, then this panel's
    # own overrides. Reading shot["camera"] raw skipped both, so a scene-wide
    # camera default never reached the staged camera and a panel could not
    # move it at all — which is the whole reason a moving shot's two panels
    # were staged from one pose.
    from pace_core.pai_compat import primary_focus_of as _primary_focus_of, resolve_shot
    shot = resolve_shot(scene, shot, panel)

    def _declared_x(s: dict) -> float:
        x = (s.get("screen_position") or {}).get("x")
        return 0.5 if x is None else float(x)

    def _declared_foreground(s: dict) -> bool:
        """Whether this subject asked to be nearer the camera than the rest.

        `screen_position.depth` is a declared field with three legal values,
        and the seat layout used to ignore it completely: distance from camera
        came from which slot a subject's screen x sorted it into. A cabin puts
        its middle seat at the BACK, so a subject declared foreground was
        staged furthest away whenever the blocking put them centre -- measured
        on this corpus, the declared-foreground subject projected larger in
        only 23 of 43 mixed-depth pairs, which is chance, and 8 of the 22
        shots declaring mixed depth staged both subjects at exactly the same
        distance.
        """
        return ((s.get("screen_position") or {}).get("depth") or "") == "foreground"

    setup = shot.get("setup") or {}
    subs = list(setup.get("subjects") or [])
    # Left to right on screen decides who sits where, so the geometry agrees
    # with the panel's own declared staging instead of an arbitrary order.
    subs.sort(key=_declared_x)

    fixtures = _panel_fixtures(p, setup)

    _stub_doc = json.loads(Path(p.loc_stubs_file).read_text()) or {}
    stubs = _stub_doc.get("stubs") or {}
    stub = stubs.get((setup.get("backdrop") or {}).get("location") or "") or {}
    sm = stub.get("scale_meters") or [2.0, 3.2, 1.5]
    cabin = [float(sm[0]), float(sm[1]), float(sm[2])]

    _env_shape, _prefix_shape = shape_maps_from_doc(_stub_doc)
    shape = shape_from_stub(stub, env_shape=_env_shape, prefix_shape=_prefix_shape)
    pose = "sitting" if shape == "subway" else "standing"

    if shape == "subway":
        # Two front seats side by side, one rear seat behind them. The cabin
        # camera sits in front of the cast looking back, which mirrors world
        # X onto screen X (+X renders screen-left, -X renders screen-right —
        # confirmed by measuring rendered body mattes against declared
        # screen_position for one panel). Seats are built
        # in screen-left-to-right order so they can be indexed directly by
        # `subs`, which is sorted ascending by declared screen x.
        fx = cabin[0] * SEAT_SPREAD_FRAC
        # Row separation. It was 0.22 of cabin depth behind a front row at
        # 0.10, and at a three-quarter azimuth that put a rear subject
        # directly behind the centre-front one: scene 2 rendered 67% of
        # one subject's silhouette under another's in every one of its four panels,
        # and the greybox gate had to grow a clause to notice.
        #
        # Widening the rear row is the other lever and this cabin no longer
        # has room for it -- at 2.05 m across, a rear subject is already
        # 0.26 m from the wall. Closing the rows costs no width: the two
        # rows subtend a wider angle from the same camera, so the rear pair
        # clears the front centre without anyone moving sideways.
        fy = cabin[1] * 0.10
        ry = fy - COUPLE_DISTANCE_M
        if len(subs) == 3:
            # The middle (declared-center) subject belongs in the actual
            # center seat, not a front-edge seat next to a flanking subject.
            places = [(fx, fy), (0.0, ry), (-fx, fy)]
        elif len(subs) == 2:
            # No third slot exists here to hand a declared-center subject an
            # actual center seat the way n==3 does, so the old fixed +-fx
            # binary pick placed BOTH subjects at the same hard offset no
            # matter what x they declared -- a subject declared exactly
            # centered (x=0.5, the common case: one foreground/hero subject
            # plus one midground secondary) still landed a full seat-width
            # off center. Interpolating world x continuously from each
            # subject's own declared x (screen-left..screen-right ==
            # world +fx..-fx, same sign convention as above) instead means
            # x=0.5 truly lands at world x=0, and only a genuinely
            # off-center declaration moves off it. Measured against the
            # declared target on the rendered head mattes for five
            # corpus scenes (17-33% off center
            # under the old binary pick, <1% after this change).
            #
            # SUBJECT_SPACING_M is a hard floor here for exactly the reason it
            # is on open ground: two subjects whose declared x happen to sit
            # close together are still two people, and interpolating them onto
            # the same patch of bench renders them interpenetrating. The
            # interpolation above, shipped without this floor, did that to
            # EVERY two-subject seated shot in the corpus -- all eight declare
            # 0.38/0.50, which lands them 0.18--0.20 m apart inside a 0.95 m
            # body. The binary pick it replaced never collided, because +-fx is
            # always a seat apart; the centring it bought had quietly been paid
            # for with overlap.
            #
            # Anchor on whichever subject declares nearest true centre -- the
            # one the camera is aimed at, so the one whose position the
            # composition solve is actually holding -- and push the other out
            # to a full body's width. Direction comes from declared order
            # (`subs` ascends by screen x, and screen-left is world +X), not
            # from the interpolated values, so it stays well defined when two
            # subjects declare the same x.
            #
            # The floor is not free, and the cost lands on the pushed subject's
            # declared position: enforcing it moved this corpus's two-subject
            # placement error from 5.7% of frame width to 9.2% (max 29.9% to
            # 32.2%). That is the correct trade and worth stating plainly --
            # a cabin narrower than two declared positions cannot satisfy both
            # the declaration and the fact that two people do not occupy one
            # seat, and of the two, only one is negotiable.
            #
            # Since distance from camera follows the declared depth (below),
            # all eight of those shots declare one subject foreground and sit
            # the pair in two rows, so on this corpus the floor no longer
            # binds; it still decides a pair the declaration puts in one row.
            xs = [fx - _declared_x(s) * 2 * fx for s in subs]
            if abs(xs[0] - xs[1]) < SUBJECT_SPACING_M:
                keep = min(range(2), key=lambda i: abs(_declared_x(subs[i]) - 0.5))
                push = 1 - keep
                xs[push] = xs[keep] + (-SUBJECT_SPACING_M if push > keep
                                       else SUBJECT_SPACING_M)
            places = [(x, fy) for x in xs]
        else:
            places = [(fx, fy), (-fx, fy), (0.0, ry)][: max(1, len(subs))]
            if len(subs) > 3:
                places += [(fx, ry), (-fx, ry)][: len(subs) - 3]

        # Distance from camera is the declared field's to set, not the slot's.
        # The rows above pick y by which seat a subject's screen x sorted them
        # into, which puts the middle of three at the BACK -- so a subject
        # declared foreground and blocked centre was staged furthest away, the
        # exact inversion the depth measurement found. Only subjects that
        # declare a depth are moved; one that declares none keeps the seat its
        # slot gave it, because an unstated depth is not a request for the
        # front row.
        #
        # This re-lays out the ROW, rather than overwriting y and keeping the
        # x the slot list happened to give. Keeping x is how a subject ended
        # up at (0, front): the centre x belongs to the three-across rear
        # bench, and moved forward it is a seat no car has. A person sits in
        # a seat, so both coordinates have to be decided by the same step.
        if any((s.get("screen_position") or {}).get("depth") for s in subs):
            rows = ([i for i, s in enumerate(subs) if _declared_foreground(s)],
                    [i for i, s in enumerate(subs) if not _declared_foreground(s)])
            places = list(places)
            for idx, row_y in zip(rows, (fy, ry)):
                xs = row_seat_xs(len(idx), fx)
                # A row of one has a free choice of x, and it must be the one
                # the panel declared: horizontal order across the WHOLE cast
                # is a relation that holds on 29/29 panels, and a lone front
                # occupant snapped to a fixed side breaks it against the rear
                # pair. Interpolating the way the two-subject branch does puts
                # a declared-centre subject on the centre of the row, which on
                # a bench is a seat.
                if len(idx) == 1:
                    xs = [fx - _declared_x(subs[idx[0]]) * 2 * fx]
                for k, i in enumerate(idx):
                    places[i] = (xs[k], row_y)
    else:
        # Open ground has no furniture to seat anyone on, so the only thing
        # deciding where they stand is the panel's own left-to-right order.
        # Spacing is a body's width, not the location's: a crash site is 35 m
        # across, and three subjects spread over that are three separate shots
        # rather than one panel. The slight depth stagger keeps them from
        # reading as a line-up and gives the solve some depth to clear.
        #
        # `subs` is sorted ascending by declared screen x (above), so i=0 is
        # the declared-leftmost subject and should get the MOST POSITIVE
        # world x under this file's own sign convention (+X renders
        # screen-left, confirmed against rendered body mattes -- see the
        # subway branch above); the original `-span/2 + i*SPACING` did the
        # opposite, putting the declared-leftmost subject screen-right and
        # vice versa.
        #
        # Fixing only that sign is not enough on its own: an evenly-spaced
        # line still has no reason to put a declared-centre subject at
        # world x=0 rather than at whichever slot their rank happens to
        # land on -- measured on one scene (a subject declared centre, x=0.5)
        # landing at a rendered read-point 27%+ off target either way the
        # sign runs. SUBJECT_SPACING_M has to stay a hard floor (this is a
        # crash site, not a car bench: crowding two subjects together
        # because their declared x values happen to be close is the
        # opposite of what the spacing exists for), so what shifts is which
        # subject the evenly-spaced line is centred ON: whichever subject's
        # declared x is nearest true centre anchors world x=0, and every
        # other subject keeps its full spacing from there in the correct
        # direction.
        # `max(1, ...)` made range(n) == [0] for an empty cast, so the anchor
        # lookup below indexed subs[0] of an empty list and died with a bare
        # IndexError. `eligible_panels` screens these out upstream, but a
        # caller reaching build_spec directly deserves the reason rather than
        # a stack trace -- an environment beat with no cast (vehicles hovering,
        # a panel closing) is a legitimate panel this stage simply cannot
        # stage, because the camera is solved by fitting the cast.
        # An ENVIRONMENT beat declares no cast -- the waiting vehicles hovering
        # motionless, the road panel closing over the wreck. It is a real beat
        # and a real image; there is simply nobody to seat. `max(1, len(subs))`
        # used to force range(n) == [0] here and index subs[0] of an empty
        # list, which died with a bare IndexError.
        #
        # A body's width is what an UPRIGHT subject occupies across the frame.
        # A lying one occupies its length, laid across the road, and spacing it
        # like a standing one put its neighbour 0.95 m from its centre -- on
        # top of it. Each neighbour is spaced by what both of them occupy, so a
        # cast with nobody lying is spaced exactly as before.
        _chars = Path(p.meshes_dir) / "characters"

        def _footprint_m(s: dict) -> float:
            if pose_key_for(s.get("pose"), pose) != "lying":
                return SUBJECT_SPACING_M
            mesh = Path(_mesh_for(s.get("age_state") or "", _chars, "lying",
                                  s.get("character_id") or ""))
            return _proxy_length_m(str(mesh)) if mesh.is_file() else 1.75

        n = len(subs)
        if n:
            widths = [_footprint_m(s) for s in subs]
            raw = [0.0]
            for a, b in zip(widths, widths[1:]):
                raw.append(raw[-1] - (a + b) / 2)
            anchor_i = min(range(n), key=lambda i: abs(_declared_x(subs[i]) - 0.5))
            offset = raw[anchor_i]
            places = [(r - offset, (0.35 if i % 2 else 0.0)) for i, r in enumerate(raw)]
        else:
            places = []

    cam = shot.get("camera") or {}
    intr = cam.get("intrinsics") or {}
    extr = cam.get("extrinsics") or {}
    meshes_dir = Path(p.meshes_dir) / "characters"

    # Where the camera stands is the panel's to say, the same as its lens and
    # its angle -- these are the schema's own literals (types_v1
    # CameraExtrinsics.position). A constant here would silently render every
    # panel from the same side no matter what it declared, which is the
    # mistake the distance solve already stopped making.
    # An unstated position is not the same statement as "front", and it was
    # being treated as one. One scene declares three_quarter on three of its
    # four shots and leaves shot_03 empty, so that shot alone swung 25 degrees
    # onto the axis mid-scene and the cast's spread jumped from 0.255 of frame
    # width to 0.400 — a continuity break nothing reported, produced by an
    # absent field rather than a directorial choice. Recorded on the spec the
    # same way coverage_dropped is, so a frame shot from a default can say so.
    declared_position = extr.get("position")
    position_defaulted = None if declared_position else (
        "camera.extrinsics.position is unset; staged from the 'front' default")
    azimuth = BUILDABLE_POSITIONS.get(declared_position or "front")
    if azimuth is None and declared_position != "ots":
        raise ValueError(
            f"camera position {extr.get('position')!r} is not buildable here yet. "
            "A 35 mm lens cannot cover three seats from inside a 2.1 m cabin, so "
            "the lens stands outside the windscreen and the shell is stretched "
            "forward to enclose it -- which only works along the axis the cabin "
            "opens on. front and three_quarter do; profile and behind need "
            "the shell to open on the side the lens is on.")
    if declared_position == "ots":
        # Over-the-shoulder is the one position in that list the shell does
        # not have to open for, because an OTS camera never stands outside
        # the cabin: it sits at one subject's shoulder and looks at another,
        # so it is close by construction and the reason the other three need
        # a stretched shell -- covering the whole cast from far enough back
        # -- does not arise. It needs a PAIR instead of an angle, so the
        # azimuth an orbit would use is not what places it (see the kernel).
        azimuth = 0.0

    ci = cam.get("creative_intent") or {}
    shot_size = ci.get("shot_size") or "wide"
    pattern = ci.get("framing")
    band, margin = SHOT_SIZE_FRAMING.get(shot_size, DEFAULT_FRAMING)
    if shot_size == "close_up" and pattern == "single":
        # The shared 0.26 band reaches shoulders and upper chest (a body's
        # head alone is ~13% of standing height, per face_masks.py's own
        # measurement, so 0.26 is roughly twice head height) -- fine for a
        # close_up that shares the frame with someone else, where the band
        # also has to clear whatever else is being fitted, but wrong for
        # the schema's own definition of close_up as head only, no
        # shoulders (compile_flux2.py's framing text already says exactly
        # that). Tightened only for `single`-pattern close_ups, where
        # nothing else being framed depends on the looser band: a
        # multi-subject close_up (two_shot, crowd) keeps 0.26, since this
        # project already carries several of those as a known, deliberately
        # unresolved schema tension (a close_up sharing a frame with other
        # subjects) that a band change here would silently re-litigate.
        band = 0.15
    # The panel's own focus wins over the shot's, the way `primary_focus_of`
    # has always resolved it for the regen packets and the restage pass. This
    # read went to the shot alone, so `Panel.primary_focus` -- a declared
    # field with a helper of its own -- reached those two consumers and never
    # the geometry: a panel that named a different subject was staged, aimed
    # and framed on the shot's, silently.
    focus = _primary_focus_of(panel, shot)
    # Which shoulder, and whose face beyond it. The panel already names the
    # face -- primary_focus is the subject the aim solve prefers -- so the
    # shoulder is the nearest other declared subject, taken in declared screen
    # order so a three-hander picks the neighbour rather than the far side of
    # the frame. A panel that declares one subject, or none to focus on, has
    # no pair and is refused rather than staged from a guess.
    ots_pair = None
    ots_focus_conflict = None
    if declared_position == "ots":
        ids = [s.get("character_id") for s in subs]
        if len(ids) < 2:
            raise ValueError(
                "camera position 'ots' needs two declared subjects, one to "
                f"shoot past and one to look at; this panel declares {ids}")
        # Which shoulder is decided by the seats, not by the panel's focus.
        # These cabins seat everyone facing the same way, so a lens behind the
        # rear passenger sees the back of the front one's head -- the only
        # over-the-shoulder a forward-facing cabin admits looks the other way:
        # in past the shoulder of whoever sits nearest the windscreen, back at
        # the face of whoever sits behind them. The near subject is therefore
        # the one seated closest to the camera side, and the far one is what
        # the lens can actually see a face of.
        depth = {s.get("character_id"): sy for s, (sx, sy) in zip(subs, places)}
        near = max(ids, key=lambda cid: depth.get(cid, 0.0))
        far = min(ids, key=lambda cid: depth.get(cid, 0.0))
        ots_pair = {"aim_subject": far, "near_subject": near}
        # What the panel's own size and angle mean for this position. Both
        # were declared and neither reached the placement; the kernel has read
        # these two keys as per-panel overrides all along and nothing wrote
        # them, so filling them here is what connects the declarations.
        ots_pair.update(_ots_offsets(shot_size))
        # And that can contradict the panel. `primary_focus` names the subject
        # the aim solve prefers; an OTS can only frame the one seated deeper,
        # so when the panel asks for the near subject the two declarations are
        # not jointly satisfiable. Recorded rather than silently resolved, the
        # same treatment coverage and an unstated position already get.
        if focus.get("ref") and focus.get("ref") != far:
            ots_focus_conflict = (
                f"primary_focus={focus.get('ref')!r} but position='ots' can only "
                f"frame {far!r}, who sits behind {near!r}; the seat order decides")
    # A coverage target is honoured where it can be met without contradicting
    # who else the panel says is in frame; where it cannot, it is dropped WITH
    # the reason on the spec rather than silently, so a frame that ignored a
    # declared number can say which number and why.
    coverage = focus.get("coverage_pct") if focus.get("type") == "character" else None
    coverage_dropped = None
    if coverage is not None and pattern in MULTI_SUBJECT_FRAMINGS:
        coverage_dropped = (f"coverage_pct={coverage} names one subject's share, but "
                            f"framing={pattern!r} puts {len(subs)} in frame; the "
                            f"framing pattern decides the extent and coverage follows")
        coverage = None

    # Pose per subject, not per location. The shot's own beat is what differs
    # between them -- "the man kneels beside the woman lying motionless" is two
    # poses in one shot -- and staging both from the location's default put
    # three identical crouching bodies in a row where one should have been on
    # the ground.
    subjects = [{"character_id": s.get("character_id") or f"subj{i}",
                 "mesh": _mesh_for(s.get("age_state") or "", meshes_dir,
                                   pose_key_for(s.get("pose"), pose),
                                   s.get("character_id") or ""),
                 "pose": pose_key_for(s.get("pose"), pose),
                 "facing_deg": _facing_from_gaze(subs, places, i),
                 # The aim solve needs to know what each subject asked for,
                 # not just where its seat ended up.
                 "declared_x": _declared_x(s)}
                for i, s in enumerate(subs)]
    for sj in subjects:
        if Path(sj["mesh"]).is_file():
            sj["seat_contact"] = _seat_contact_z(sj["mesh"])
    if dress:
        from pace_core.pai_compat import costume_text, wardrobe_of
        _kb = json.loads(Path(p.chars_file).read_text()) if Path(p.chars_file).is_file() else {}
        _kb = _kb.get("characters", _kb) if isinstance(_kb, dict) else {}
        _wd = (json.loads(Path(p.props_file).read_text())
               if Path(p.props_file).is_file() else {})
        for sj, s in zip(subjects, subs):
            tones = garment_tones(costume_text(
                _kb.get(sj["character_id"]), s.get("age_state"),
                wardrobe=wardrobe_of(_wd, s)))
            if tones:
                sj["costume"] = tones
            # The proxy's scalp is a smooth dome, and a lens behind a head
            # delivers that dome: a bald back of the head under a description
            # that names short hair.
            entry = _kb.get(sj["character_id"]) or {}
            style = hair_style((entry.get("age_states") or {}).get(s.get("age_state"))
                               or entry.get("anchor"))
            if style:
                sj["hair"] = style

    # The set `build_locations` already built for this scene, if it is there.
    # Per scene rather than per location because that is how build_locations
    # names them; scenes sharing a location share its geometry only insofar
    # as their bibles agreed, which design_check reports on separately.
    _dir = getattr(p, "blender_scenes_dir", None)
    _blend = Path(_dir) / f"{scene.get('scene_id')}.blend" if _dir else None
    location_blend = str(_blend) if _blend and _blend.is_file() else None

    return {
        "shape": shape,
        "cabin": cabin,
        # Only when present: `fixtures` and the rest of `_ANCHOR_FIELDS` decide
        # the anchor hash, and a key that is always there but usually null
        # would change every anchor in every project that has no set built.
        **({"location_blend": location_blend} if location_blend else {}),
        "fixtures": fixtures,
        "seats": [list(s) for s in places],
        "subjects": subjects,
        # The seats nobody in this shot is in. Staged as set dressing so the
        # cabin's own capacity is in the control image rather than left as
        # bare floor for the sampler to furnish.
        "empty_seats": _unoccupied_slots(stub.get("seat_count") or 0,
                                         places, cabin) if shape == "subway" else [],
        "position_defaulted": position_defaulted,
        "aim_arbitration_w": AIM_ARBITRATION_W,
        "lens_mm": float(intr.get("lens_mm") or 35),
        "sensor_width_mm": float(intr.get("sensor_width_mm") or 36),
        # Ahead of the windscreen looking back in. A 35 mm lens cannot cover
        # three seats from inside a 2.1 m cabin -- placed behind the rear seat
        # it sits inside the body occupying it, which is the interior-bounds
        # problem camera_planner has in the other direction. The shell is
        # deliberately built without a front wall so this view reads straight
        # through where the windscreen is.
        # No camera position here: it is solved in-scene from the cast's
        # measured extents (see the framing block). What the panel gets to say
        # is the side and the angle it asked for, not the distance.
        #
        # How high the seat is, from the proxies that sit on it: the tallest
        # occupant's seated hips. A shorter occupant then sits at the same
        # height with its feet off the floor, which is what happens in a car.
        # In a cabin the surface is the seat, measured from the tallest
        # occupant's hips. On open ground it is the ground, and the same
        # contact measurement answers ~0 for a standing proxy, so the bodies
        # land on it without a second rule.
        "seat_top": (max((sj["seat_contact"] for sj in subjects
                          if "seat_contact" in sj), default=DEFAULT_SEAT_TOP)
                     if shape == "subway" else 0.0),
        "framing": {
            # Which slice of a body is in frame, and around what.
            "shot_size": shot_size,
            "band": band,
            "pattern": pattern,
            # Frame the focus alone only when the panel says it is alone.
            "focus_id": (focus.get("ref") or None) if pattern == "single" else None,
            # Unlike focus_id, not gated to `single`: which body the camera's
            # HORIZONTAL aim targets, even when others share the frame. See
            # the kernel's camera-solve comment for why this has to be a
            # second field rather than reusing focus_id.
            "aim_focus_id": focus.get("ref") or None,
            "coverage": (float(coverage) / 100.0) if coverage else None,
            "coverage_dropped": coverage_dropped,
        },
        "fit_margin": margin,
        **({"ots": ots_pair} if ots_pair else {}),
        # Present only when the panel contradicts itself, so a panel that
        # does not keeps its anchor_version; left out of _ANCHOR_FIELDS
        # because a note about a declaration is not geometry.
        **({"ots_focus_conflict": ots_focus_conflict} if ots_focus_conflict else {}),
        "azimuth_deg": azimuth,
        "elevation_deg": _elevation_deg(extr),
        # How far the kernel may move the lens to see a head past the set,
        # without leaving the declared angle class.
        "elevation_range_deg": list(ELEVATION_RANGE_DEG.get(
            _angle_class(extr), DEFAULT_ELEVATION_RANGE_DEG)),
        "azimuth_tolerance_deg": AZIMUTH_TOLERANCE_DEG,
        # Present only when non-zero, so every upright panel keeps its
        # anchor_version (see there).
        **({"roll_deg": _roll_of(extr)} if _roll_of(extr) else {}),
        "res": list(res),
        "out": str(out_png),
    }


# Fields that decide the GEOMETRY. Everything a control signal is projected
# from must come from one anchor, so the anchor needs an identity: two
# controls that disagree about the camera are worse than one control, and
# without a version nobody can tell.
#
# Deliberately excludes the output path and anything about when it ran. Two
# builds of the same panel into different files are the same anchor; a build
# after the lens changed is not.
_ANCHOR_FIELDS = ("cabin", "seats", "shape", "lens_mm", "sensor_width_mm",
                  "azimuth_deg", "elevation_deg", "res", "framing", "ots",
                  "elevation_range_deg", "azimuth_tolerance_deg",
                  "fit_margin", "aim_arbitration_w", "seat_top",
                  "empty_seats", "fixtures", "position_defaulted",
                  # The location's built set: appending it changes what the
                  # frame contains, so a build with one is not the same anchor
                  # as a build without. `anchor_version` reads a missing key
                  # as None, so a project with no set keeps its hash.
                  "location_blend")


def anchor_version(spec: dict) -> str:
    """A stable id for the geometry this spec describes.

    Subjects are reduced to what moves the camera -- who, which mesh, which
    pose, which declared position -- so a spec that differs only in a field
    the geometry never reads keeps its anchor.
    """
    import hashlib
    payload = {k: spec.get(k) for k in _ANCHOR_FIELDS}
    # A canted camera is different geometry. Added only when there is one:
    # listed in _ANCHOR_FIELDS it would enter every payload as None and
    # restamp every upright panel in every project.
    if spec.get("roll_deg"):
        payload["roll_deg"] = spec["roll_deg"]
    payload["subjects"] = [
        {"id": s.get("character_id"), "mesh": Path(s.get("mesh") or "").name,
         "pose": s.get("pose"), "facing_deg": s.get("facing_deg"),
         "declared_x": s.get("declared_x"), "seat_contact": s.get("seat_contact")}
        for s in (spec.get("subjects") or [])]
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def missing_meshes(spec: dict) -> list[str]:
    """Proxy meshes the spec names that are not on disk."""
    return [s["mesh"] for s in spec["subjects"] if not Path(s["mesh"]).is_file()]


def eligible_panels(project: str, scene_id: str | None = None) -> list[dict]:
    """Every panel in scope, with whether a greybox can be built for it.

    Answers without building, so a batch can report what it skipped and why
    instead of either failing at the first exterior panel or quietly filling a
    crash site with cabins. `reason` is None exactly when `eligible` is True.
    """
    p = paths_for(project)
    _stub_doc = json.loads(Path(p.loc_stubs_file).read_text()) or {}
    stubs = _stub_doc.get("stubs") or {}
    _env_shape, _prefix_shape = shape_maps_from_doc(_stub_doc)
    meshes_dir = Path(p.meshes_dir) / "characters"
    files = ([Path(p.scenes_dir) / f"{scene_id}.json"] if scene_id
             else sorted(Path(p.scenes_dir).glob("scene_*.json")))

    out: list[dict] = []
    for sf in files:
        if not sf.is_file():
            raise FileNotFoundError(f"scene document not found: {sf}")
        doc = json.loads(sf.read_text())
        sid = doc.get("scene_id") or sf.stem
        for sh in doc.get("shots") or []:
            # Resolved, not raw. The builder below calls `resolve_shot`, so a
            # scan that reads the shot alone disagrees with it: a project
            # whose backdrop is declared once in `shot_defaults` -- which is
            # what `split_script` produces -- reported every panel skipped
            # for `location ''` while the builder would have found it.
            from pace_core.pai_compat import resolve_shot as _resolve
            setup = (_resolve(doc, sh) or {}).get("setup") or {}
            subs = setup.get("subjects") or []
            loc = (setup.get("backdrop") or {}).get("location") or ""
            stub = stubs.get(loc) or {}
            env = stub.get("environment_type")
            shape = shape_from_stub(stub, env_shape=_env_shape,
                                    prefix_shape=_prefix_shape)
            pos = (((sh.get("camera") or {}).get("extrinsics") or {})
                   .get("position") or "front")
            reason = None
            # A cast-less panel is an ENVIRONMENT beat, not a mistake: the
            # waiting vehicles hovering, the road panel closing over the wreck.
            # It stages the space and its fixtures, with the camera fitted to
            # the location's own declared extent rather than to a cast. It
            # still needs a buildable location, so the checks below apply.
            if shape is None:
                reason = (f"location {loc!r} declares neither `shape` nor "
                          f"`environment_type`, so there is nothing to build from — "
                          f"set one on its stub")
            elif shape not in BUILDABLE_SHAPES:
                reason = (f"location {loc!r} is {env or 'unclassified'} ({shape}); "
                          f"this builds a cabin and no other shell yet")
            elif pos not in BUILDABLE_POSITIONS:
                reason = (f"camera position {pos!r} needs the shell to open on the "
                          f"side the lens is on")
            else:
                # Same default the build itself uses (line ~922): the
                # location decides what an unposed subject does, and the
                # precheck has to resolve the file the build would load or it
                # names a mesh nobody was going to open.
                loc_pose = "sitting" if shape == "subway" else "standing"
                missing = [m for m in
                           {_mesh_for(s.get("age_state") or "", meshes_dir,
                                      pose_key_for(s.get("pose"), loc_pose),
                                      s.get("character_id") or "")
                            for s in subs}
                           if not Path(m).is_file()]
                if missing:
                    reason = f"missing proxy mesh: {', '.join(Path(m).name for m in missing)}"
            for pl in sh.get("panels") or []:
                out.append({
                    "scene_id": sid, "shot_id": sh.get("shot_id"),
                    "panel_id": pl.get("id"),
                    "panel_n": str(pl.get("id") or "").rsplit("_", 1)[-1],
                    "subjects": len(subs),
                    # The set belongs to the location, not the panel, so a
                    # caller exporting one needs to know which panels share it.
                    "location": loc,
                    "eligible": reason is None, "reason": reason,
                })
    return out


def make_panel_greybox(project: str, scene_id: str, panel_id: str,
                       out_png: str | Path,
                       res: tuple[int, int] = DEFAULT_RES,
                       export_glb: str | None = None,
                       export_glb_include_bodies: bool = False,
                       camera_delta: dict | None = None,
                       scene: dict | None = None,
                       dress: bool = False) -> dict:
    """Build one panel's control frame. Returns the kernel's result dict.

    `scene`: build from this scene document rather than the installed one --
    how a restage pass tries a blocking before it writes it.

    `dress`: tone each proxy's garments from its registered costume. Only
    `build_spec` took this, so anything going through here built undressed
    bodies, and a caller that wanted the dressed ones had to reach past this
    function -- which is how the paper's beat pilot came to be the one set of
    greyboxes nothing could rebuild.

    Blender is reached through BlenderBox, which is the only host-side caller
    allowed to spawn the binary -- imported here rather than at module scope
    so the kernel half of this file does not pull it in under Blender.

    `camera_delta`: optional `{"height_delta_m": ..., "pull_out_scale": ...}`,
    applied after the panel's own camera solve rather than in place of it —
    a crane-and-pull-out offset from the solved pose, for rendering a moving
    shot's later frame off the same staged cast (Section 4.2's trajectory
    compiler names the move; this is what running it looks like on pixels).
    """
    from pace_core.node.blender_box import BlenderBox, PanelGreyboxSpec

    spec = build_spec(project, scene_id, panel_id, out_png, res=res, scene=scene,
                      dress=dress)
    if camera_delta:
        spec["camera_delta"] = camera_delta
    missing = missing_meshes(spec)
    if missing:
        return {"ok": False, "_missing_meshes": missing,
                "error": f"missing proxy mesh(es): {missing} — generate them "
                         f"with an SMPL-X proxy builder"}
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    out = BlenderBox().render_panel_greybox(
        PanelGreyboxSpec(spec=spec, out_png=str(out_png), export_glb=export_glb,
                          export_glb_include_bodies=export_glb_include_bodies))
    # Stamp the anchor on the result. Every control signal a caller derives
    # from this build -- depth, mattes, the init frame -- comes from this
    # camera, and two controls that disagree about the camera are worse than
    # one control. Without an id on the build nobody can tell that they did.
    if isinstance(out, dict):
        out["anchor_version"] = anchor_version(spec)
    out.setdefault("bodies", len(spec["subjects"]))
    out.setdefault("seats", spec["seats"])
    return out


# ── kernel (runs INSIDE Blender) ──────────────────────────────────────────



# Interior proportions, as fractions of the shell's own height. A cabin and a
# room are the same shell with different numbers: both are a box the camera
# stands inside, with glass in bays between uprights, a rail below it and one
# above, a panel underneath and something across the ceiling. What differs is
# where those land -- a car's waist is low and its glass deep; a room's dado
# sits higher, its windows are shorter and its uprights fall further apart.
#
# scene_proxies._build_box_room was the obvious thing to reuse and does not
# fit: it builds a closed room centred on the origin at a fixed size, so it
# cannot enclose a camera the solver placed. Standing the camera outside and
# hiding the wall it looks through is the exact arrangement that left a grey
# void with an invented fourth person in it.
INTERIOR_STYLES = {
    #            glass_z glass_h sill_z cant_z panel_z panel_h bay_m
    # "vehicle"'s glass was only a little larger than "room"'s (0.32 vs 0.24
    # glass_h, 0.44 vs 0.52 sill_z) -- not enough contrast for a cabin this
    # small to read as a car rather than a boxy room once it lost its double-
    # drawn seat clutter. Widened the gap: a car's glazing runs low and tall,
    # closer to a single expansive pane per bay than a room's punched window.
    "vehicle":  (0.60,   0.42,   0.36,  0.80,  0.23,   0.34,   1.1),
    "room":     (0.66,   0.24,   0.52,  0.88,  0.24,   0.46,   1.7),
}


def _build_interior(box, *, style, half_w, mid_y, depth, top_z, back_y,
                    pillar_limit):
    """Floor, ceiling, walls and their dressing, around an enclosed camera.

    Every dimension arrives already solved: this places geometry, it does not
    decide where the shell goes.
    """
    glass_z, glass_h, sill_z, cant_z, panel_z, panel_h, bay_m = INTERIOR_STYLES[style]
    box("floor",     (0, mid_y, 0),               (half_w * 2, depth, 0.05))
    box("roof",      (0, mid_y, top_z),           (half_w * 2, depth, 0.05))
    box("wall_l",    (-half_w, mid_y, top_z / 2), (0.05, depth, top_z))
    box("wall_r",    ( half_w, mid_y, top_z / 2), (0.05, depth, top_z))
    box("wall_rear", (0, back_y, top_z / 2),      (half_w * 2, 0.05, top_z))

    # ── dressing: every surface gets something to read ──
    #
    # Flat grey is not "a wall" to the generator, it is "nothing specified
    # here". A window band alone left the rest of the shell blank and the
    # picture came back in a narrow centre box with grey bands down both sides
    # -- the milder form of the emptiness that once returned as a white border.
    # So the shell carries what a cabin has, sized off the shell being dressed
    # rather than off constants: pillars dividing the glass into bays, a sill
    # and a cant rail framing it, a door panel and kick plate below, ribs
    # across the headliner, glass in panes across the rear. What hugs a surface
    # runs the full depth, including the stretch alongside and above the lens,
    # which is where the last bare band was; only the pillars, which stand
    # proud enough to occlude, stop short of the lens. The floor gets seams and
    # nothing else -- the location calls it open and uncluttered, so it may
    # have joins but not objects.
    bays = max(2, round(depth / bay_m))
    edges = [back_y + i * depth / bays for i in range(bays + 1)]

    for sx in (-half_w, half_w):
        side = "l" if sx < 0 else "r"
        sgn = 1 if sx < 0 else -1            # +t is always into the cabin

        def at(t, _sx=sx, _sgn=sgn):
            return _sx + _sgn * t

        box(f"glass_{side}", (at(0.03), mid_y, top_z * glass_z), (0.02, depth, top_z * glass_h))
        box(f"sill_{side}",  (at(0.07), mid_y, top_z * sill_z), (0.14, depth, 0.07))
        box(f"cant_{side}",  (at(0.05), mid_y, top_z * cant_z), (0.10, depth, 0.05))
        box(f"door_{side}",  (at(0.04), mid_y, top_z * panel_z), (0.08, depth * 0.98, top_z * panel_h))
        box(f"kick_{side}",  (at(0.09), mid_y, 0.05),         (0.18, depth, 0.10))
        for i, ey in enumerate(edges):
            if ey <= pillar_limit:
                box(f"pillar_{side}{i}", (at(0.06), ey, top_z / 2), (0.12, 0.13, top_z))

    for i, ey in enumerate(edges):
        box(f"rib_{i}", (0, ey, top_z - 0.05), (half_w * 2, 0.12, 0.07))
    box("headliner", (0, mid_y, top_z - 0.045), (0.30, depth, 0.05))
    for sx in (-half_w * 0.52, half_w * 0.52):
        box(f"floor_seam_{'l' if sx < 0 else 'r'}", (sx, mid_y, 0.02),
            (0.05, depth, 0.03))
    # The rear wall is the backdrop the whole cast is read against, so it
    # speaks the same vocabulary as the sides -- glass in panes between rails
    # -- instead of one wide pane, which at this size reads no differently
    # from grey.
    ry = back_y + 0.05
    box("rear_cant", (0, ry + 0.02, top_z * cant_z), (half_w * 2, 0.10, 0.05))
    box("rear_sill", (0, ry + 0.03, top_z * 0.50), (half_w * 2, 0.14, 0.07))
    box("parcel",    (0, ry + 0.09, top_z * 0.46), (half_w * 1.6, 0.24, 0.07))
    panes = 3
    pane_w = half_w * 2 * 0.88 / panes
    for i in range(panes):
        box(f"rear_pane_{i}",
            (-half_w * 0.88 + pane_w * (i + 0.5), ry, top_z * 0.655),
            (pane_w * 0.88, 0.02, top_z * 0.26))


def _render_depth_twin(sc, spec: dict) -> str:
    """Render the current scene's Z pass to `<out>.depth.png`, near=white.

    Mirrors vace_pipeline's depth setup — MapRange from near to far, inverted so
    near reads white — because the two produce the same kind of image for the
    same consumer, and a control that means one thing in a still and another in
    a sequence would be worse than no control.

    The window is sized from the cabin itself rather than a constant: an
    interior is 2-4 m deep, and the 0.5-20 m fallback that suits an exterior
    puts the whole room inside 15% of the range, which is how a scene render
    produced a control with four distinct values in it.
    """
    W, D, H = spec["cabin"]
    reach = float(max(W, D, H))
    near, far = max(0.05, reach * 0.05), reach * 1.6

    out_png = spec["out"] + ".depth.png"
    prev_engine = sc.render.engine
    prev_color_mode = sc.render.image_settings.color_mode
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.use_nodes = True
    nt = sc.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new(type="CompositorNodeRLayers")
    mp = nt.nodes.new(type="CompositorNodeMapRange")
    mp.inputs["From Min"].default_value = near
    mp.inputs["From Max"].default_value = far
    mp.inputs["To Min"].default_value = 1.0        # near = white
    mp.inputs["To Max"].default_value = 0.0
    fo = nt.nodes.new(type="CompositorNodeOutputFile")
    fo.base_path = str(Path(out_png).parent)
    fo.file_slots[0].path = Path(out_png).stem + "_"
    bpy.context.view_layer.use_pass_z = True
    nt.links.new(rl.outputs["Depth"], mp.inputs["Value"])
    nt.links.new(mp.outputs["Value"], fo.inputs[0])

    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = "BW"
    bpy.ops.render.render(write_still=True)

    # CompositorNodeOutputFile appends the frame number; give the caller one
    # stable name instead of whatever frame the scene happened to sit on.
    produced = sorted(Path(out_png).parent.glob(Path(out_png).stem + "_*.png"))
    if not produced:
        raise RuntimeError("depth pass wrote no file")
    produced[-1].replace(out_png)
    for stray in produced[:-1]:
        stray.unlink(missing_ok=True)
    sc.render.engine = prev_engine
    sc.use_nodes = False
    # Put back what this pass changed. BW has no alpha channel, and leaving it
    # set made every later matte render opaque — three per-body mattes came out
    # identical, each covering the whole frame, because "the body" was the
    # entire image.
    sc.render.image_settings.color_mode = prev_color_mode
    return out_png


# Hair on the bald SMPL-X proxy, per style, in metres: shell thickness, grain
# size, grain depth. Opt-in per spec subject (`hair`), so a panel that names
# none renders exactly as before -- the corpus's own panels included.
HAIR_STYLES = {
    "short": {"thickness": 0.010, "grain": 0.010, "depth": 0.004},
    "curly": {"thickness": 0.022, "grain": 0.014, "depth": 0.012},
}


def _hair_cap(head_src, cid: str, style: dict):
    """A hair shell cut from the proxy's own scalp, raised and grained.

    The proxy is bald, which costs nothing from the front, where the face
    carries the cues, and everything from behind: flattened to twelve grey
    levels a hairless crown is a smooth dome, and on the one over-the-shoulder
    panel tried, the sampler repainted it as seat upholstery. The greybox
    renders a single flat colour, so hair can only read through shape -- a
    grained surface for the cavity pass to catch, and the step where it ends.

    Imported proxy frame: +Z up, face toward -Y (the rotation below turns it
    to the windscreen). A scalp face is kept when every corner sits above a
    hairline that falls from the forehead to the nape.
    """
    import bmesh
    me = head_src.data.copy()
    bm = bmesh.new()
    bm.from_mesh(me)
    zs = [v.co.z for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    z0, z1, y0, y1 = min(zs), max(zs), min(ys), max(ys)

    def on_scalp(v):
        height = (v.co.z - z0) / ((z1 - z0) or 1.0)   # chin 0 -> crown 1
        back = (v.co.y - y0) / ((y1 - y0) or 1.0)     # face 0 -> occiput 1
        return height > 0.84 - 0.56 * back

    bmesh.ops.delete(bm, geom=[f for f in bm.faces
                               if not all(on_scalp(v) for v in f.verts)],
                     context="FACES")
    # Strays the cut leaves behind would still count toward the body's bounds.
    # Normals are left as the proxy authored them (outward): recomputing them
    # on an open cap can flip the whole shell into the skull.
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces],
                     context="VERTS")
    bm.to_mesh(me)
    bm.free()
    hair = bpy.data.objects.new(f"hair_{cid}", me)
    bpy.context.collection.objects.link(hair)
    hair.matrix_world = head_src.matrix_world.copy()
    sub = hair.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 2
    tex = bpy.data.textures.new(f"hair_grain_{cid}", type="CLOUDS")
    tex.noise_scale = style["grain"]
    grain = hair.modifiers.new("grain", "DISPLACE")
    grain.texture = tex
    grain.strength = style["depth"]
    # Below zero so every point lifts at least a quarter of the depth: a shell
    # whose inner face coincided with the scalp would z-fight it.
    grain.mid_level = -0.25
    shell = hair.modifiers.new("shell", "SOLIDIFY")
    shell.thickness = style["thickness"]
    shell.offset = 1.0
    # Not even-offset: it divides by the angle between neighbouring faces, and
    # on the ragged hairline that threw spikes up to three metres long. Clamped
    # to local edge length for the same reason.
    shell.use_even_offset = False
    shell.thickness_clamp = 1.0
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = hair
    hair.select_set(True)
    for m in list(hair.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
    # Smooth-shaded, so the grain reads as hair. Flat-shaded, every facet of
    # the displaced shell caught the cavity pass as a separate tile, and
    # flattened to twelve grey levels the back of the head became quilted
    # upholstery -- the sampler painted a seat headrest where the head was.
    for p in hair.data.polygons:
        p.use_smooth = True
    return hair


# Garment tones ride on a colour attribute, and the beauty pass switches to
# reading it only when some subject is dressed -- every other greybox renders
# exactly as it did.
_TONE_ATTR = "pace_tone"
_GREY = (0.55, 0.55, 0.56, 1.0)
# What a surface declared powered off renders as: dark enough to read as off
# through the init's 12-level posterisation, not black, which would read as a
# hole in the geometry rather than an unlit panel.
_DARK = (0.07, 0.07, 0.075, 1.0)


def _tone_mesh(ob, rgba):
    """Give a mesh the tone attribute, filled with one colour."""
    ca = ob.data.color_attributes
    attr = ca.get(_TONE_ATTR) or ca.new(_TONE_ATTR, "FLOAT_COLOR", "CORNER")
    attr.data.foreach_set("color", list(rgba) * len(attr.data))
    return attr


def _use_tone(ob):
    """Make the tone attribute the one this mesh renders with."""
    ca = ob.data.color_attributes
    for field, value in (("active_color_name", _TONE_ATTR), ("default_color_name", _TONE_ATTR),
                         ("render_color_index", ca.find(_TONE_ATTR))):
        try:
            setattr(ca, field, value)
        except (AttributeError, TypeError):
            pass


def _dress(parts, costume: dict):
    """Paint a proxy's declared garments onto its own part groups.

    The proxy OBJ carries head, torso, arm_l, arm_r, leg_l and leg_r as named
    groups, identical in vertex order across every pose, so a garment follows
    the body part rather than a guessed height: the shirt is the torso above
    the waist and, for a short sleeve, the third of each arm nearest the
    shoulder; the trousers are the rest of the torso and both legs. Mesh
    coordinates are the proxy's own (+Z up), before the kernel poses or lays
    the body down, so the split holds for every pose.
    """
    import mathutils
    by = {}
    for o in parts:
        by.setdefault(o.name.split(".")[0], o)
        _tone_mesh(o, _GREY)
    top, bottom = costume.get("top"), costume.get("bottom")

    def paint(o, inside, tone):
        me = o.data
        attr = me.color_attributes[_TONE_ATTR]
        cols = [0.0] * (len(attr.data) * 4)
        attr.data.foreach_get("color", cols)
        rgba = (tone, tone, tone, 1.0)
        for poly in me.polygons:
            if all(inside(me.vertices[v].co) for v in poly.vertices):
                for li in poly.loop_indices:
                    cols[li * 4:li * 4 + 4] = rgba
        attr.data.foreach_set("color", cols)

    torso = by.get("torso")
    if torso is None or not torso.data.vertices:
        return
    zs = [v.co.z for v in torso.data.vertices]
    waist = min(zs) + 0.22 * (max(zs) - min(zs))
    centre = sum((v.co for v in torso.data.vertices), mathutils.Vector()) / len(torso.data.vertices)
    if top:
        paint(torso, lambda c: c.z >= waist, top["tone"])
        reach_frac = 0.30 if top.get("sleeve") == "short" else 0.85
        for side in ("arm_l", "arm_r"):
            arm = by.get(side)
            if arm is None or not arm.data.vertices:
                continue
            pts = [v.co.copy() for v in arm.data.vertices]
            shoulder = min(pts, key=lambda c: (c - centre).length)
            reach = max((c - shoulder).length for c in pts) or 1.0
            paint(arm, lambda c, s=shoulder, r=reach: (c - s).length <= reach_frac * r, top["tone"])
    if bottom:
        paint(torso, lambda c: c.z < waist, bottom["tone"])
        for side in ("leg_l", "leg_r"):
            leg = by.get(side)
            if leg is not None:
                paint(leg, lambda c: True, bottom["tone"])


def _removable(ob, set_names: list[str]) -> bool:
    """May the occlusion step take this object out of the frame?"""
    if ob is None or ob.type != "MESH" or ob.name.startswith(_STRUCTURAL_PREFIXES):
        return False
    c = [ob.matrix_world @ mathutils.Vector(v) for v in ob.bound_box]
    dx = max(p.x for p in c) - min(p.x for p in c)
    dy = max(p.y for p in c) - min(p.y for p in c)
    dz = max(p.z for p in c) - min(p.z for p in c)
    if ob.name in set_names and (dz < _SET_CULL_MIN_H_M
                                 or max(dx, dy) > _MAX_REMOVABLE_SPAN_M):
        return False
    return True


def _see_past_set(*, cam, focus, bodies, heads, set_names, spec,
                  shell_bounds) -> dict:
    """Give every head a line of sight to the lens, moving the lens first.

    Kernel-only. Casts a ray from the lens to sample points on each head; a
    ray stopped by anything but a body is a head the set hides. When one is
    hidden it tries the poses camera_search_order lists -- the same
    horizontal distance to the aim, so the shot size holds -- and keeps the
    first that clears every head, or else the one that leaves the worst head
    most visible. A head still hidden after that has the furniture in front
    of it removed, the way set pieces between lens and cast already are, and
    never the structural shell. Returns what it saw and what it did.
    """
    report = {"searched": False, "moved": None, "removed": [], "visible": {}}
    if spec.get("ots") or not heads:
        return report
    sc = bpy.context.scene
    cast = set(bodies) | set(heads.values())
    state = {"deps": bpy.context.evaluated_depsgraph_get()}

    def samples(h):
        vs = h.data.vertices
        step = max(1, len(vs) // 60)
        return [h.matrix_world @ vs[i].co for i in range(0, len(vs), step)]

    head_pts = {cid: samples(h) for cid, h in heads.items()}

    def look(eye):
        seen = {}
        for cid, pts in head_pts.items():
            clear, hits = 0, []
            for p in pts:
                d = p - eye
                if d.length < 1e-6:
                    clear += 1
                    continue
                ok, _l, _n, _i, ob, _m = sc.ray_cast(
                    state["deps"], eye, d.normalized(), distance=d.length - 1e-3)
                src = getattr(ob, "original", ob)
                if not ok or src is None or src in cast:
                    clear += 1
                else:
                    hits.append(src.name)
            seen[cid] = (clear / max(len(pts), 1), hits)
        return seen

    eye0 = cam.location.copy()
    seen = look(eye0)
    worst = min(v[0] for v in seen.values())
    if worst < HEAD_CLEAR_SHARE:
        report["searched"] = True
        off = eye0 - focus
        dh = math.hypot(off.x, off.y)
        yaw0 = math.degrees(math.atan2(off.x, off.y))
        elev0 = math.degrees(math.atan2(off.z, dh))
        best = (worst, eye0, seen, elev0, yaw0)
        for e, y in camera_search_order(
                elev0, yaw0,
                tuple(spec.get("elevation_range_deg") or DEFAULT_ELEVATION_RANGE_DEG),
                float(spec.get("azimuth_tolerance_deg", AZIMUTH_TOLERANCE_DEG)))[1:]:
            eye = mathutils.Vector((focus.x + dh * math.sin(math.radians(y)),
                                    focus.y + dh * math.cos(math.radians(y)),
                                    focus.z + dh * math.tan(math.radians(e))))
            if not inside_shell(eye.x, eye.y, eye.z, shell_bounds):
                continue
            s = look(eye)
            w = min(v[0] for v in s.values())
            if w >= HEAD_CLEAR_SHARE or w > best[0] + 0.05:
                best = (w, eye, s, e, y)
            if w >= HEAD_CLEAR_SHARE:
                break
        worst, eye, seen, e, y = best
        if eye is not eye0:
            cam.location = eye
            report["moved"] = {"elevation_deg": [round(elev0, 2), round(e, 2)],
                               "azimuth_deg": [round(yaw0, 2), round(y, 2)]}
    # Still hidden after the search: take out the furniture in the way.
    dropped: set[str] = set()
    for cid, (share, hits) in seen.items():
        if share >= HEAD_CLEAR_SHARE:
            continue
        counts: dict[str, int] = {}
        for n in hits:
            counts[n] = counts.get(n, 0) + 1
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            ob = bpy.data.objects.get(name)
            if name in dropped or not _removable(ob, set_names):
                continue
            bpy.data.objects.remove(ob, do_unlink=True)
            dropped.add(name)
            report["removed"].append({"name": name, "for": cid, "rays": n})
    if dropped:
        bpy.context.view_layer.update()
        state["deps"] = bpy.context.evaluated_depsgraph_get()
        seen = look(cam.location.copy())
    report["visible"] = {cid: round(v[0], 3) for cid, v in seen.items()}
    # What still stands in the way, so a head left hidden names its cause.
    report["blocked_by"] = {}
    for cid, (share, hits) in seen.items():
        if share < HEAD_CLEAR_SHARE and hits:
            counts: dict[str, int] = {}
            for n in hits:
                counts[n] = counts.get(n, 0) + 1
            report["blocked_by"][cid] = sorted(counts, key=lambda n: -counts[n])[:3]
    return report


def _kernel_greybox(spec: dict) -> dict:
    W, D, H = spec["cabin"]          # metres: width, length, height
    # What each mesh fixture had to be scaled by to reach its declared span.
    # Collected during the build rather than derived after it, because the
    # extents that matter are the YAWED ones -- measuring a side panel in its
    # own axes reports a disagreement it does not have.
    fit_report: list = []
    shell_built: dict = {}
    # Fixtures that asked for a surface this panel does not contain. Reported
    # rather than dropped: "the monitor is not in the greybox" is a fact about
    # the panel's prop list, and a silent skip is how a fixture goes missing
    # for a whole corpus without anyone noticing.
    skipped_fixtures: list[dict] = []

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # The location's own set, when one was built. `build_locations` already
    # renders each bible's `primitives_spec` into
    # `blender_scenes/<scene>.blend` -- another production's scene holds a floor, a
    # dropped ceiling, two walls, a window curtain wall, three desk row banks,
    # a hero desk and a cubicle partition -- and nothing read it. The greybox
    # built its own empty box beside it, so a scene about a hundred people at
    # workstations staged in a room with no workstations.
    #
    # Appended rather than linked: a linked object is not editable in the file
    # the render runs from, and the fixtures below are placed against these.
    #
    # Not for a vehicle cabin. One corpus scene's .blend holds
    # car_floor/car_ceiling/left_wall/right_wall as four 1x1 unit planes at
    # fixed local positions, never scaled to the panel's own solved shell --
    # subway's shell is already sized per panel (fit to camera and cast, W/D/H
    # a few metres each way), so a unit plane appended verbatim lands wherever
    # a metre-scale coordinate happens to put it, which was between the lens
    # and both subjects' heads on its first shot. A vehicle cabin also
    # already builds its own complete shell (pillars, sills, panels, glass)
    # below, so nothing about a room's bible was missing here the way it was
    # for another production's empty office box -- this append exists for shapes that
    # start from nothing, not for the one shape that never does.
    set_objects = 0
    set_names: list[str] = []
    blend = spec.get("location_blend") if spec.get("shape") != "subway" else None
    if blend and Path(blend).is_file():
        try:
            with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
                dst.objects = list(src.objects)
            for ob in dst.objects:
                if ob is not None:
                    bpy.context.collection.objects.link(ob)
                    set_names.append(ob.name)
                    set_objects += 1
        except Exception as e:                        # noqa: BLE001
            print(f"[greybox] location set {blend} failed to append: {e}")
            set_objects = 0
    shell_built["set_objects"] = set_objects
    shell_built["location_blend"] = str(blend) if set_objects else None

    def box(name, pos, size):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        o = bpy.context.active_object
        o.name, o.scale = name, size
        return o

    def seat_geometry(name, cx, cy, seat_z, sx, sy, back_h=None):
        """A seat built as a seat, around the height its occupant sits at.

        Anchored on `seat_z` -- the same seat_top the bodies were placed off,
        measured from the tallest occupant's seated hips -- rather than on the
        centre of the KB's span box. Built around the span box instead, the
        cushion lands under the hips and the backrest never clears the torso,
        so the whole thing reads as a bench.

        The generic fallback used to be two boxes, a cube and a slab, which
        reads as a plinth. The alternative was whatever mesh the KB carried,
        and the evaluation corpus's `swivel_seat.glb` is a single-view Hunyuan3D
        generation whose own vertex profile is an egg: 0.24 x 0.09 at the
        floor, widest 0.78 x 0.82 at mid-height, tapering to a point. Fitted
        into the seat slot it stages a pod, and the delivered panel drew a pod,
        because the control image is the sampler's starting latents.

        Six boxes are not a beautiful chair, but the silhouette is
        unambiguously a chair: a seat plane at hip height, a back rising above
        it, arms either side, on a pedestal. That is the shape the generator
        needs in order to render a chair rather than invent one.
        """
        back_h = back_h if back_h is not None else max(0.45, seat_z * 0.95)
        # Pedestal: a floor disc and a column up to the seat plane.
        box(f"{name}_base",   (cx, cy, 0.03),          (sx * 0.55, sy * 0.55, 0.06))
        box(f"{name}_column", (cx, cy, seat_z * 0.5),  (sx * 0.20, sy * 0.20, seat_z))
        # Seat plane at hip height, its top flush with seat_z.
        box(f"{name}_cushion", (cx, cy, seat_z - 0.04), (sx, sy, 0.08))
        # Back panel behind the occupant, rising clear of the shoulders.
        box(f"{name}_back", (cx, cy - sy * 0.42, seat_z + back_h * 0.5),
            (sx, sy * 0.16, back_h))
        # Arms either side at forearm height.
        for s, side in ((-1.0, "l"), (1.0, "r")):
            box(f"{name}_arm_{side}", (cx + s * sx * 0.46, cy + sy * 0.04, seat_z + 0.20),
                (sx * 0.12, sy * 0.66, 0.06))

    def fitted_mesh(name, mesh_path, pos, size, yaw_deg=0.0, fit=_FIT_DEFAULT):
        """Import a mesh and sit it in the space a box would have taken.

        `fit` decides who wins when the mesh's proportions and the declared
        span disagree; see `_FIT_POLICIES`. It defaults to `stretch`, which
        scales each axis to fill the box, because the span in the KB is a
        statement about the object -- a console spans the width of the cabin
        -- and a generated mesh has whatever proportions the generator gave
        it. Fitting uniformly by default honours the mesh and breaks the
        placement: the console came back a third of the cabin wide, because
        uniform scale takes the tightest ratio and a console slot is thin.

        What was missing was not a better rule but a reading. Whichever
        policy is chosen, the disagreement is measured and reported in
        `fit_report`, so a fixture being crushed to a third of its depth is
        a number the author can see rather than something the render quietly
        absorbs. Returns None if the file yields no geometry, and the caller
        falls back to the box -- a missing mesh should cost a fixture its
        detail, not the panel its control geometry.
        """
        before = set(bpy.data.objects)
        # Dispatch on the extension: the proxies on disk are .obj, and every
        # modeling engine in the registry outputs .glb, so a fixture mesh
        # arrives as either depending on whether it was authored or generated.
        suffix = mesh_path.rsplit(".", 1)[-1].lower()
        try:
            if suffix in ("glb", "gltf"):
                bpy.ops.import_scene.gltf(filepath=mesh_path)
            elif suffix == "fbx":
                bpy.ops.import_scene.fbx(filepath=mesh_path)
            else:
                bpy.ops.wm.obj_import(filepath=mesh_path)
        except Exception:                                      # noqa: BLE001
            return None
        made = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
        if not made:
            return None
        bpy.ops.object.select_all(action="DESELECT")
        for o in made:
            o.select_set(True)
        bpy.context.view_layer.objects.active = made[0]
        if len(made) > 1:
            bpy.ops.object.join()
        o = bpy.context.active_object
        o.name = name
        # The glTF importer leaves objects in QUATERNION mode, where writing
        # rotation_euler does nothing: every .glb fixture was staged unturned.
        # The fit report showed it -- a car yawed 90 degrees measured its
        # length still along Y, was fitted width-for-length into the span and
        # came out 1.85 m long. Switching the mode keeps the current rotation.
        o.rotation_mode = "XYZ"
        o.rotation_euler = (o.rotation_euler[0], o.rotation_euler[1],
                            o.rotation_euler[2] + math.radians(yaw_deg))
        bpy.context.view_layer.update()
        # Bake the turn into the mesh before measuring and scaling. The span is
        # written in world axes and `o.scale` acts in the object's own: once a
        # fixture actually turns, a 90-degree yaw swaps the two, and the cabin's
        # door cards -- 7 cm thin against the wall, 0.95 m along it -- came out
        # 0.95 m across the cabin, a slab in front of the outer passengers.
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()
        cs = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        ext = [max(v[i] for v in cs) - min(v[i] for v in cs) for i in range(3)]
        k, anis = fit_scale(ext, size, fit)
        fit_report.append({"id": name, "fit": fit,
                           "extents": [round(v, 4) for v in ext],
                           "target": [round(v, 4) for v in size],
                           "scale": [round(v, 4) for v in k],
                           "anisotropy": round(anis, 3)})
        o.scale = (o.scale[0] * k[0], o.scale[1] * k[1], o.scale[2] * k[2])
        bpy.context.view_layer.update()
        cs = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        mid = [(max(v[i] for v in cs) + min(v[i] for v in cs)) / 2 for i in range(3)]
        o.location = (o.location[0] + pos[0] - mid[0],
                      o.location[1] + pos[1] - mid[1],
                      o.location[2] + pos[2] - mid[2])
        return o

    def fixture(name, pos, size, mesh_path=None, yaw_deg=0.0,
                fit=_FIT_DEFAULT):
        """A fixture is its mesh when it has one, and a box when it does not.

        The box is a different console in every panel, because a featureless
        slab is a shape the generator gets to invent. The mesh is the same
        console every time, which is the whole reason a prop carries one.
        """
        if mesh_path:
            o = fitted_mesh(name, mesh_path, pos, size, yaw_deg=yaw_deg, fit=fit)
            if o is not None:
                return o
        return box(name, pos, size)

    def _on_subject_at(fx, size):
        """Centre of a prop the beat leaves lying on a subject, or None.

        It covers the body from its middle toward the FEET and leaves the head
        and chest out, so the person pinned is still recognisably the person
        the previous panels showed: covering the head end hid the face and the
        shirt that carry him from panel to panel, and the generator repainted
        him in the open in a different one. The feet end is opposite the
        proxy's own up axis, laid down with it. The underside is at the body's
        mid-thickness, so it pins the body rather than floating over it.
        """
        body = bpy.data.objects.get(f"body_{fx['on_subject']}")
        if body is None:
            return None
        ws = [body.matrix_world @ v.co for v in body.data.vertices]
        cx = (min(w.x for w in ws) + max(w.x for w in ws)) / 2
        cy = (min(w.y for w in ws) + max(w.y for w in ws)) / 2
        zmid = (min(w.z for w in ws) + max(w.z for w in ws)) / 2
        m3 = body.matrix_world.to_3x3()
        ux, uy = m3[0][2], m3[1][2]
        n = math.hypot(ux, uy)
        ux, uy = (ux / n, uy / n) if n > 1e-6 else (0.0, 0.0)
        reach = abs(ux) * size[0] + abs(uy) * size[1]
        # `covers: "chest"` puts the head end under instead and leaves the
        # legs out. Written only by a caller that asks for it.
        sgn = 1.0 if fx.get("covers") == "chest" else -1.0
        return (cx + sgn * ux * reach / 2, cy + sgn * uy * reach / 2,
                zmid + size[2] / 2)

    def place_fixtures(*, W, D, H, base_z, seats=(), origin=(0.0, 0.0),
                       span_scale=None, center=(0.0, 0.0)):
        """Place every declared fixture, each at its own anchor, in shell units.

        Every fixture, not one anchor per call: the caller used to name the
        anchors it wanted, so adding `shell_center` to the vocabulary placed
        nothing until a matching call was added too. The fixture was read,
        reported, and silently never built -- a greybox that rendered
        pixel-identical to one without it.

        Spans and offsets are fractions of the shell, not metres, so one
        placement authored once survives a cabin of different proportions --
        which is what makes this the KB's business rather than the builder's.
        """
        # Hosts before their guests. `on_surface` is the one anchor measured
        # against another fixture, so the desk has to exist -- and be built,
        # with whatever the mesh fit did to its real height -- before the
        # monitor can be put on top of it.
        surfaces: dict[str, tuple] = {}
        fixtures = list(spec.get("fixtures") or [])
        # WHERE a fixture goes and HOW BIG it is are scaled separately, because
        # in a room they answer to different things. `shell_rear` means the
        # back wall, which is a fact about the room, while a desk is 1.6 m wide
        # in any room. In a cabin the two coincide -- a console IS the width of
        # the car -- so the vehicle branch passes nothing and both stay the
        # shell. Collapsing them put a wall totem 50 cm behind the cast.
        SW, SD, SH = span_scale if span_scale else (W, D, H)
        for fx in ([f for f in fixtures if f.get("anchor") != _SURFACE_ANCHOR]
                   + [f for f in fixtures if f.get("anchor") == _SURFACE_ANCHOR]):
            anchor = fx.get("anchor")
            sw, sd, sh = fx["span"]
            ox, oy, oz = fx["offset"]
            size = (sw * SW, sd * SD, sh * SH)
            if fx.get("on_subject"):
                # A prop the beat leaves lying ON someone -- "the car falls on
                # top of him" -- goes where they are, not where the registry
                # parks it.
                place_at = _on_subject_at(fx, size)
                if place_at is None:
                    skipped_fixtures.append({"id": fx["id"],
                                             "want_subject": fx["on_subject"]})
                    continue
                fixture(fx["id"], place_at, size, fx.get("mesh"),
                        fx.get("yaw_deg", 0.0), fit=fx.get("fit", _FIT_DEFAULT))
                _record_surface(surfaces, fx, place_at, size)
                continue
            if anchor == _SURFACE_ANCHOR:
                host = surfaces.get(fx.get("host") or "")
                if host is None:
                    # No desk in this panel, so no desktop. Placing it at the
                    # shell coordinates that would put it "about right" is the
                    # guess this module refuses to make everywhere else, and a
                    # monitor floating at chest height in open air is a worse
                    # control image than no monitor at all.
                    skipped_fixtures.append({"id": fx["id"],
                                             "want_host": fx.get("host") or ""})
                    continue
                (hx, hy, hz), (hw, hd, hh) = host
                # Spans are fractions of the HOST, not of the shell: a monitor
                # takes a third of the desk it stands on, whatever room the
                # desk is in. Read any other way, a keyboard written as 0.35
                # of a 9 m office wall is 3 m across.
                size = (sw * hw, sd * hd, sh * hh)
                place_at = (hx + ox * hw, hy + oy * hd,
                            hz + hh / 2 + size[2] / 2 + oz * hh)
                fixture(fx["id"], place_at, size, fx.get("mesh"),
                        fx.get("yaw_deg", 0.0),
                        fit=fx.get("fit", _FIT_DEFAULT))
                _record_surface(surfaces, fx, place_at, size)
                continue
            if anchor == "per_seat":
                for i, sxy in enumerate(seats):
                    place_at = (sxy[0] + ox * W, sxy[1] + oy * D, base_z + oz * H)
                    if fx.get("mesh"):
                        fixture(f"{fx['id']}_{i}", place_at, size,
                                fx["mesh"], fx.get("yaw_deg", 0.0),
                                fit=fx.get("fit", _FIT_DEFAULT))
                    else:
                        # A seat with no usable mesh is drawn as a seat. The
                        # generic `box` fallback puts a single cube in the slot,
                        # which is a plinth, and a plinth is a shape the
                        # generator has to guess its way out of.
                        seat_geometry(f"{fx['id']}_{i}",
                                      sxy[0] + ox * W, sxy[1] + oy * D,
                                      base_z, size[0], size[1])
                continue
            # Two instances, one per side wall, mirrored in x and each turned
            # to face inward. A wraparound lining the walls is not one object
            # at the middle of the cabin: placed that way it sits between the
            # lens and the cast and buries them, which is what shell_center
            # does with a fixture this size however the span is written.
            #
            # `span` is read in world axes here, not the mesh's own, because
            # fitted_mesh yaws first and then scales the ROTATED bounding box
            # to match. So a side panel is written the way it reads — thin in
            # x, long in y, tall in z — and the yaw only decides which face
            # points into the cabin.
            # Open ground has no shell to be a fraction of, so an outdoor
            # caller passes W=D=H=1 and the span is read in metres. It also has
            # no origin at the world centre — the cast stands wherever the
            # solve put them — so the offset is measured from `origin`, which
            # outdoors is the focus the camera is aimed at.
            if anchor == "ground":
                place_at = (origin[0] + ox, origin[1] + oy, size[2] / 2 + oz)
                fixture(fx["id"], place_at, size, fx.get("mesh"),
                        fx.get("yaw_deg", 0.0),
                        fit=fx.get("fit", _FIT_DEFAULT))
                _record_surface(surfaces, fx, place_at, size)
                continue
            if anchor == "shell_sides":
                for i, sign in enumerate((-1.0, 1.0)):
                    x = sign * (W / 2 - size[0] / 2) + ox * W
                    place_at = (x, oy * D, base_z + oz * H)
                    fixture(f"{fx['id']}_{i}", place_at, size, fx.get("mesh"),
                            fx.get("yaw_deg", 0.0) + (90.0 * sign),
                            fit=fx.get("fit", _FIT_DEFAULT))
                continue
            # shell_center is for a fixture that surrounds the space rather
            # than standing at one end of it -- screen panels lining the walls
            # around the seats are not at the front or the back, they are
            # around.
            #
            # `floor` is the other metric anchor: a desk stands 1.45 m behind
            # the cast, not a quarter of the way down whatever room it is in.
            # A shell anchor is the opposite -- `shell_rear` IS the back wall
            # -- and it is measured against the shell that was BUILT, centred
            # on `center`, because the built one is the room as far as the
            # frame is concerned. A room's declared depth is 14 m and the shell
            # that holds the camera is 6, so a totem written to the declared
            # wall lands 4 m outside the wall the picture has.
            if anchor == "floor":
                pos = (ox, oy, size[2] / 2 + oz)
            else:
                y = {"shell_front": D / 2 - size[1] / 2,
                     "shell_rear": -D / 2 + size[1] / 2,
                     "shell_center": 0.0}.get(anchor, 0.0) + oy * D + center[1]
                pos = (ox * W + center[0], y, base_z + oz * H)
            fixture(fx["id"], pos, size, fx.get("mesh"),
                    fx.get("yaw_deg", 0.0), fit=fx.get("fit", _FIT_DEFAULT))
            _record_surface(surfaces, fx, pos, size)

    # The shell is built AFTER the camera is solved, so it can enclose it --
    # see the framing block. Built up front it could only ever be a box the
    # camera stands outside of, and hiding the walls that occluded left a grey
    # void the generator read as "no picture" and rendered as a white border.

    # Everything a seated body touches is placed off the seat surface, which is
    # itself measured from the proxy that sits on it (see seat_top). Absolute
    # heights here would only be right for the stature they were tuned against.
    seat_top = spec["seat_top"]

    seats = spec["seats"]

    # ── bodies: exactly the declared subjects, one per seat ──
    bodies = []
    heads = {}   # character_id -> a standalone copy of just the head sub-mesh
    joints_world = {}   # character_id -> {joint: world point}; see joint_points
    for i, subj in enumerate(spec["subjects"]):
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=subj["mesh"])
        imported = [o for o in set(bpy.data.objects) - before if o.type == "MESH"]
        if not imported:
            continue
        # smplx_proxy.py's OBJ carries the head as its own named `o head`
        # group specifically so a mask can be taken of it alone; the join
        # below immediately erases that split for framing purposes (the
        # camera solve needs one solid body volume, head included), so the
        # split is preserved here on a throwaway COPY instead, before the
        # join, rather than by trying to recover it after.
        head_src = next((o for o in imported
                         if o.name == "head" or o.name.startswith("head.")), None)
        if head_src is not None:
            head_copy = head_src.copy()
            head_copy.data = head_src.data.copy()
            bpy.context.collection.objects.link(head_copy)
            # Coincident with the joined body's own head geometry, so left
            # visible it would z-fight the beauty/depth render and (since
            # export_glb below selects by object, not by hide_render) get
            # exported into the location's set GLB as a stray duplicate.
            # Unhidden only for its own matte pass, further down.
            head_copy.hide_render = True
            heads[subj["character_id"]] = head_copy
        # Joints, while the six part groups are still apart: the join below
        # erases where one part meets the next, which is where they are.
        parts = {}
        for o in imported:
            base = o.name.split(".")[0]
            if base in ("head", "torso", "arm_l", "arm_r", "leg_l", "leg_r"):
                vs = o.data.vertices
                parts[base] = [tuple(o.matrix_world @ vs[k].co)
                               for k in range(0, len(vs), 4)]
        joints_local = joint_points(parts)
        # Hair joins the body, not the head copy: the head matte is what the
        # read point is measured against, and a cap would lift its centroid.
        style = HAIR_STYLES.get(subj.get("hair") or "")
        if style is not None and head_src is not None:
            imported.append(_hair_cap(head_src, subj["character_id"], style))
        # Garments before the join, while the part groups are still apart.
        if subj.get("costume"):
            _dress(imported, subj["costume"])
        bpy.ops.object.select_all(action="DESELECT")
        for o in imported:
            o.select_set(True)
        bpy.context.view_layer.objects.active = imported[0]
        if len(imported) > 1:
            bpy.ops.object.join()
        body = bpy.context.active_object
        body.name = f"body_{subj['character_id']}"
        m_join = body.matrix_world.copy()
        # The sitting mesh is authored facing -Y (its back reads to a camera at
        # +Y), so occupants need turning to face the windscreen the location
        # says they all face. Measured by rendering and looking at three backs.
        xform_rot = (0.0, 0.0, math.radians(180.0 + subj.get("facing_deg", 0.0)))
        # A subject the beat puts on the ground is laid down, not stood up.
        # SMPL-X body_pose is relative to the pelvis, so lying is a rotation
        # rather than a different mesh: tip the standing proxy back 90 degrees
        # about X and it rests along the floor, feet toward the camera's right.
        # Without this, "the woman lying motionless" staged as a third standing
        # body in a row, and the panel showed three people in identical
        # crouches because the geometry never said otherwise.
        lying = subj.get("pose") == "lying"
        if lying:
            # Tipping about X alone lays the body along the DEPTH axis, so its
            # 1.75 m runs away from the lens instead of across the frame. The
            # cast's measured extents are what the framing solve fits, and one
            # body stretched through depth moved the camera far enough that two
            # of one scene's three subjects left the frame entirely. Adding the
            # quarter turn lays it across the road, which is both what the beat
            # describes and an extent the solve can frame.
            xform_rot = (math.radians(-90.0), 0.0,
                         math.radians(90.0 + subj.get("facing_deg", 0.0)))
        body.rotation_euler = xform_rot
        sx, sy = seats[i][0], seats[i][1]
        # Sit the mesh ON the seat by the point that actually rests on one. Its
        # lowest point is its feet -- a sitting proxy is authored standing on
        # the floor with its hips already at seat height -- so lifting min(z)
        # to the seat lifted the whole body 45 cm and left three people
        # hovering with their feet on the cushions and the dashboard across
        # their chests. subj["seat_contact"] is that height, measured off the
        # mesh itself, so a child on an adult-height seat correctly ends up
        # with dangling feet.
        # A lying body rests on the floor, not on the seat plane its standing
        # contact height was measured for; laid flat, its own depth is what
        # holds it off the ground.
        # It is also laid down about its MIDDLE. The proxy's origin is its
        # feet, so a rotation about it left the feet on the seat and ran the
        # whole 1.75 m out from there -- through whoever stood beside it: the
        # pilot's climax staged one subject's feet on another's chest, and another scene
        # laid a subject under the kneeling one. The seat is where the body is,
        # so the body's centre is what goes on it.
        if lying:
            body.location = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
            ws = [body.matrix_world @ v.co for v in body.data.vertices]
            cx = (min(w.x for w in ws) + max(w.x for w in ws)) / 2
            cy = (min(w.y for w in ws) + max(w.y for w in ws)) / 2
            body.location = (sx - cx, sy - cy, -min(w.z for w in ws))
        else:
            body.location = (sx, sy, seat_top - subj["seat_contact"])
        head_copy = heads.get(subj["character_id"])
        if head_copy is not None:
            # The copy was taken pre-transform, in the same coordinate frame
            # as the body it was split from, so the identical rotation +
            # location keeps it seated on the same head it came off of.
            head_copy.rotation_euler = xform_rot
            head_copy.location = body.location
        bpy.context.view_layer.update()
        if joints_local:
            # The joints were taken in the frame the parts were imported in;
            # whatever turned, laid down and seated the body moves them too.
            to_world = body.matrix_world @ m_join.inverted()
            joints_world[subj["character_id"]] = {
                k: to_world @ mathutils.Vector(v) for k, v in joints_local.items()}
        bodies.append(body)
    # No bodies is a failure only when the panel DECLARED some. A panel that
    # declares none is an environment beat and stages the space alone; a panel
    # that declares three and imported none has lost its proxy meshes, which
    # is a real error and must stay one.
    if not bodies and spec["subjects"]:
        return {"ok": False, "error": "no body meshes imported"}

    # ── camera: the panel's own lens, placed INSIDE the shell ──
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_width = spec["sensor_width_mm"]
    cam_data.lens = spec["lens_mm"]
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    # Frame from what is actually there. Every corner of every body, in world
    # space, gives the group's true horizontal and vertical extent; the lens
    # and sensor give the angles it has to fit into. Solving distance from
    # those two puts the whole cast in frame by construction and leaves nothing
    # for a hand-tuned camera position to get wrong -- which it did, 3 times.
    fr = spec.get("framing") or {}
    # Who the frame is set by. `single` means the focus is alone, so the frame
    # is its own; every other pattern says someone else is in it too, and a
    # frame set to one body would crop the others out of a shot that declared
    # them.
    framed = [b for b in bodies
              if fr.get("focus_id") and b.name == f"body_{fr['focus_id']}"] or bodies
    # With no cast, the frame is set by the SPACE. An environment beat is a
    # statement about the place -- vehicles holding station, a panel closing --
    # so the extent to fit is the location's own declared box, and the shot
    # size's band does not apply to it: a "close up" of a room is not the top
    # quarter of the room.
    environment_only = not framed
    if environment_only:
        W, D, H = spec["cabin"]
        min_x, max_x = -W / 2, W / 2
        min_y, max_y = -D / 2, D / 2
        min_z, max_z = 0.0, H
    # And how much of them. A close-up is not a wide taken from nearer: it is
    # the top quarter of a body filling the frame, so the extent being fitted
    # is the band the shot size names, measured off the mesh in that band --
    # a head is far narrower than the hips its bounding box also contains.
    band = float(fr.get("band") or 1.0)
    corners = [b.matrix_world @ mathutils.Vector(c) for b in framed for c in b.bound_box]
    # Per body, not across the group. The band says how much OF A BODY is in
    # frame, and cutting the group's combined range assumes every subject
    # stands at the same height. Once poses differ it stops being true: with
    # one subject lying and one standing, a close-up's 0.26 band cut at 1.30 m
    # — above every vertex the lying and the kneeling subject own — so the
    # frame fitted the standing body alone and the other two left the shot.
    # Cutting each body's own top quarter keeps all of them in the extent.
    pts = []
    for b in framed:                                           # empty when environment-only
        bc = [b.matrix_world @ mathutils.Vector(c) for c in b.bound_box]
        # For one body in frame, measured along its own head-to-foot axis (the
        # proxy is +Z up), not world height. Laid down, a close-up's top
        # quarter is the head end; cutting world height kept the whole length,
        # and a declared close-up of a man under a car staged him full length.
        # Upright, the two axes are the same and nothing moves. With several
        # bodies of different poses the frame is decided by the group -- from
        # a head on the ground to a standing head -- and cutting a lying body
        # along its own axis only widened scene 6's mixed-pose close-ups, so
        # there the cut stays in world height, as before.
        up = ((b.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()
              if len(framed) == 1 else mathutils.Vector((0.0, 0.0, 1.0)))
        vs = [b.matrix_world @ v.co for v in b.data.vertices]
        proj = [v.dot(up) for v in vs]
        top, bottom = max(proj), min(proj)
        cut = top - band * (top - bottom)
        kept = [v for v, p in zip(vs, proj) if p >= cut]
        pts.extend(kept or bc)
    pts = pts or corners
    # A prop the beat leaves lying on a subject is part of what the frame is
    # about: "the car falls on top of him" framed to the man alone cropped the
    # car to a slab across the top edge. Its box joins the fitted extent --
    # outdoors, where a span is already metres, in world axes, before any
    # shell exists to scale it. Only in a shot wide enough to hold a body
    # whole: a close-up cuts anything the size of a car at the frame edge,
    # and fitting its box made a declared close-up of the pinned man a wide.
    if spec.get("shape") == "outdoor" and band >= SHOT_SIZE_FRAMING["medium_full"][0]:
        for fx in spec.get("fixtures") or []:
            at = _on_subject_at(fx, fx["span"]) if fx.get("on_subject") else None
            if at is None:
                continue
            hx, hy, hz = (v / 2 for v in fx["span"])
            pts += [mathutils.Vector((at[0] + ex, at[1] + ey, at[2] + ez))
                    for ex in (-hx, hx) for ey in (-hy, hy) for ez in (-hz, hz)]
    if not environment_only:
        min_x = min(v.x for v in pts); max_x = max(v.x for v in pts)
        min_y = min(v.y for v in pts); max_y = max(v.y for v in pts)
        min_z = min(v.z for v in pts); max_z = max(v.z for v in pts)
    # Horizontal aim: the declared focus subject's own position when one is
    # named, not the framed group's bounding-box midpoint -- the two only
    # coincide when the group happens to be symmetric around that subject
    # (three seats with the focus in the real centre one), which is what
    # let an asymmetric two-person "two_shot" go unnoticed: centering on
    # the pair's midpoint instead of the declared-centre subject put their
    # rendered read point a full half of their separation off target
    # (measured 27% on one scene). aim_focus_id is deliberately not gated to
    # the `single` pattern the way focus_id/`framed` above is -- fitting
    # the frame's WIDTH still has to include everyone the panel puts in
    # it, only WHERE the lens points should prefer the one the panel names.
    aim_name = f"body_{fr['aim_focus_id']}" if fr.get("aim_focus_id") else None
    aim_body = next((b for b in bodies if b.name == aim_name), None) if aim_name else None
    if aim_body is not None:
        aim_corners = [aim_body.matrix_world @ mathutils.Vector(c) for c in aim_body.bound_box]
        aim_x = (min(v.x for v in aim_corners) + max(v.x for v in aim_corners)) / 2
    else:
        aim_x = (min_x + max_x) / 2
    focus = mathutils.Vector((aim_x, (min_y + max_y) / 2, (min_z + max_z) / 2))

    margin = spec.get("fit_margin", 1.25)
    span_h = (max_x - min_x) * margin
    span_v = (max_z - min_z) * margin
    res_x, res_y = spec["res"]
    sensor_h = spec["sensor_width_mm"]
    sensor_v = sensor_h * res_y / res_x
    hfov = 2 * math.atan(sensor_h / (2 * spec["lens_mm"]))
    vfov = 2 * math.atan(sensor_v / (2 * spec["lens_mm"]))
    # Whichever axis needs more room decides the distance; the other then fits.
    d_h = span_h / 2 / math.tan(hfov / 2)
    d_v = span_v / 2 / math.tan(vfov / 2)
    dist = max(d_h, d_v)
    # Which axis won is the difference between a shot size that was honoured
    # and one that was merely resolved. The band the shot size names crops the
    # framed extent VERTICALLY; when the pattern puts several bodies in frame,
    # the group's WIDTH is what the camera has to clear, and the band stops
    # binding at all. That is why a declared extreme close-up on a three-person
    # crowd stages looser than a declared wide on a two-shot -- measured across
    # this corpus, median staged body height did not order by declared size.
    #
    # It is not fixable by pulling in: the panel also declares who is in frame,
    # and honouring the tighter size would crop out cast the prompt enumerates.
    # An over-constrained panel is reported rather than silently resolved one
    # way, the same treatment coverage already gets above.
    shot_size_bound_by = "height" if d_v >= d_h else "width"
    dist += (max_y - min_y) / 2          # clear the depth of the group itself
    # A kept coverage target is a stronger statement than "fit it with air
    # around it": it names the share of the frame the focus occupies. Frame
    # area at distance d is 4 d^2 tan(h/2) tan(v/2), so the d that puts a
    # w x h subject at `coverage` of it falls straight out. Never closer than
    # the fit distance, or the thing being covered is cropped out of frame.
    # First order: the band's bounding box, not its silhouette, so a body that
    # does not fill its own box lands somewhat under the number.
    cov = fr.get("coverage")
    if cov:
        area = max((max_x - min_x) * (max_z - min_z), 1e-6)
        d_cov = math.sqrt(area / (4 * float(cov)
                                  * math.tan(hfov / 2) * math.tan(vfov / 2)))
        dist = max(dist, d_cov)

    yaw = math.radians(spec.get("azimuth_deg", 25.0))    # three-quarter, from the front
    elev = math.radians(spec.get("elevation_deg", 6.0))

    def _cam_at(cam_focus):
        return mathutils.Vector((cam_focus.x + dist * math.sin(yaw),
                                 cam_focus.y + dist * math.cos(yaw),
                                 cam_focus.z + dist * math.tan(elev)))

    def _screen_x(cam_focus, pts):
        """Horizontal screen position of a point cloud, 0 at frame left.

        Computed from the pose rather than read back from the render, because
        the camera is aimed by a TRACK_TO constraint that is not added until
        the scene is finished -- and because the search below needs hundreds
        of evaluations. The projection is the same one the render performs:
        look direction from camera to focus, right vector from that and world
        up, divide by depth and the horizontal field of view. Points behind
        the lens are dropped, and a subject wholly behind it has no position.
        """
        eye = _cam_at(cam_focus)
        fwd = (cam_focus - eye)
        if fwd.length < 1e-9:
            return None
        fwd.normalize()
        right = fwd.cross(mathutils.Vector((0.0, 0.0, 1.0)))
        if right.length < 1e-9:
            return None
        right.normalize()
        tan_h = math.tan(hfov / 2)
        num = den = 0.0
        for v in pts:
            rel = v - eye
            z = rel.dot(fwd)
            if z <= 1e-6:
                continue
            num += 0.5 + (rel.dot(right) / (z * tan_h)) / 2
            den += 1
        return (num / den) if den else None

    def _screen_x_area(cam_focus, pts, grid=(160, 68)):
        """Horizontal centre of the screen cells a point cloud covers.

        The silhouette's centre rather than the cloud's mean: in-frame cells
        only, each counted once however many vertices land in it -- the same
        quantity the object-index matte is measured at, to the grid's
        resolution.
        """
        eye = _cam_at(cam_focus)
        fwd = (cam_focus - eye)
        if fwd.length < 1e-9:
            return None
        fwd.normalize()
        right = fwd.cross(mathutils.Vector((0.0, 0.0, 1.0)))
        if right.length < 1e-9:
            return None
        right.normalize()
        up = right.cross(fwd)
        tan_h, tan_v = math.tan(hfov / 2), math.tan(vfov / 2)
        cells = set()
        for v in pts:
            rel = v - eye
            z = rel.dot(fwd)
            if z <= 1e-6:
                continue
            sx = 0.5 + (rel.dot(right) / (z * tan_h)) / 2
            sy = 0.5 - (rel.dot(up) / (z * tan_v)) / 2
            if 0.0 <= sx < 1.0 and 0.0 <= sy < 1.0:
                cells.add((int(sx * grid[0]), int(sy * grid[1])))
        if not cells:
            return None
        return (sum(c[0] for c in cells) / len(cells) + 0.5) / grid[0]

    # ── whose declared position gives way ────────────────────────────────
    #
    # Aiming at the named focus subject centres that subject and hands the
    # entire disagreement to everyone else, because the frame has one
    # horizontal degree of freedom and the panel declared several positions.
    # That is an arbitration, and leaving it implicit is what put 7 of this
    # corpus's 76 staged subjects off the side of the frame while the subject
    # beside them sat exactly on its mark.
    #
    # Moving the aim is a lateral dolly, not a pan: the camera is placed
    # relative to `focus` and looks at it, so shifting focus.x slides the
    # camera sideways and keeps its direction. That means the fit distance,
    # and therefore the declared shot size, is unaffected by this search --
    # which is why the trade is paid for in placement alone.
    # Every 20th vertex: the search runs hundreds of probes and a body proxy
    # carries thousands of vertices, while the quantity being estimated is a
    # centroid. Checked against the full cloud, the stride moves the answer by
    # well under a millimetre of aim.
    # Strided by index, not by slice: bpy_prop_collection does not accept a
    # step slice, so `vertices[::20]` raises TypeError the moment this
    # comprehension evaluates its value expression. It never did, because the
    # `if` below filtered every subject out until panel overrides started
    # resolving into the spec — a latent break, one guard away, for as long as
    # no subject carried a declared_x.
    declared = [([b.matrix_world @ b.data.vertices[i].co
                  for i in range(0, len(b.data.vertices), 20)],
                 sj["declared_x"])
                for b, sj in zip(bodies, spec["subjects"])
                if sj.get("declared_x") is not None]   # empty with no cast
    single_area = len(declared) == 1
    if single_area:
        # One subject: aim the centre of its silhouette, which is what the
        # matte is measured at. Neither cheaper stand-in is it: a proxy's
        # vertices crowd into the head and hands, and a seated body's box
        # runs out toward the lens, so both land several percent of the frame
        # off -- and with one subject there is no arbitration for that to
        # vanish into. The screen cells any vertex falls in stand in for the
        # silhouette, so area weighs and vertex density does not.
        b0 = next(b for b, sj in zip(bodies, spec["subjects"])
                  if sj.get("declared_x") is not None)
        declared = [([b0.matrix_world @ b0.data.vertices[i].co
                      for i in range(0, len(b0.data.vertices), 3)],
                     declared[0][1])]
    w = float(spec.get("aim_arbitration_w", AIM_ARBITRATION_W))
    # One declared subject is searched too. With one position there is
    # nothing to arbitrate and the search simply puts it on its mark; skipping
    # it centred every single, which cost nothing while every single in the
    # corpus declared 0.5 and put a shot / reverse-shot pair split from a
    # two-shot (0.39 and 0.61) both in the middle of the frame.
    if declared:
        base_x = focus.x
        best, best_cost = 0.0, None
        steps = int(_AIM_SEARCH_M / _AIM_STEP_M)
        for i in range(-steps, steps + 1):
            probe = mathutils.Vector((base_x + i * _AIM_STEP_M, focus.y, focus.z))
            errs = []
            for pts, want in declared:
                sx = _screen_x_area(probe, pts) if single_area else _screen_x(probe, pts)
                errs.append(1.0 if sx is None else abs(sx - want))
            cost = (1 - w) * (sum(errs) / len(errs)) + w * max(errs)
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = i * _AIM_STEP_M, cost
        focus.x = base_x + best

    cam.location = _cam_at(focus)

    # Over-the-shoulder, which is not an azimuth. `front` and `three_quarter`
    # orbit the cast's own centre at whatever distance fits everyone, so one
    # number places them. An OTS shot is defined by two bodies instead: the
    # lens sits behind one subject's shoulder and looks at the other, so its
    # distance falls out of where those two people are rather than out of a
    # fit, and its shot size is an outcome the same way Section 6.4.2 reports
    # shot size to be. It runs here rather than in the spec because the
    # shoulder it has to clear is a rendered mesh, not a declared number.
    ots = spec.get("ots") or {}
    if ots:
        def _crown(o):
            cs = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
            return mathutils.Vector((
                (min(v.x for v in cs) + max(v.x for v in cs)) / 2,
                (min(v.y for v in cs) + max(v.y for v in cs)) / 2,
                max(v.z for v in cs)))

        near_body = next((b for b in bodies
                          if b.name == f"body_{ots.get('near_subject')}"), None)
        far_body = next((b for b in bodies
                         if b.name == f"body_{ots.get('aim_subject')}"), None)
        if near_body is not None and far_body is not None:
            near_head, far_head = _crown(near_body), _crown(far_body)
            # From the far subject toward the near one, which is the direction
            # the lens has to stand PAST: the camera ends up in front of the
            # near shoulder looking back, not behind it looking forward. The
            # second is what a cabin's seating makes tempting and it delivers
            # the back of both heads.
            look = near_head - far_head
            look.z = 0.0
            if look.length > 1e-6:
                look.normalize()
                # Sideways off the pair's own axis, so the lens clears the
                # near head instead of being buried inside it. Either shoulder
                # is a legal OTS; this takes the one that moves the lens
                # toward the cabin's centre line, because a cabin is narrow
                # and the other shoulder puts the camera in the wall.
                # Per-panel overrides, defaulting to the module's offsets: how
                # much shoulder an OTS shows is a framing choice, not a fact of
                # the geometry, and a lens a skull's width behind the crown
                # shows skull and no shoulder.
                behind = float(ots.get("behind_m", OTS_BEHIND_M))
                # The near head clears the line of sight to the far face by
                # across * D / (behind + D), D the pair's horizontal distance,
                # so a fixed sideways offset that clears it close in buries the
                # far face once the lens pulls back: at 0.34 m behind 0.30 m
                # across cleared it, at 0.6-1.0 m it covered 100% of the face.
                # Solve for the offset that keeps OTS_CLEARANCE_M between them.
                pair = near_head - far_head
                pair.z = 0.0
                d = max(pair.length, 1e-6)
                across = float(ots.get("across_m", max(
                    OTS_ACROSS_M, OTS_CLEARANCE_M * (behind + d) / d)))
                side = mathutils.Vector((-look.y, look.x, 0.0))
                # Which shoulder decides which side of frame the near subject
                # lands on, so it is the declared screen order's choice when
                # there is one: taking the cabin-centre shoulder regardless
                # crossed the line on scene_11's cut into its OTS.
                near_x = next((s.get("declared_x") for s in spec.get("subjects") or []
                               if s.get("character_id") == ots.get("near_subject")), None)
                if near_x is not None and abs(float(near_x) - 0.5) > 1e-6:
                    view = far_head - (near_head + look * behind + side * across)
                    right = mathutils.Vector((view.y, -view.x, 0.0))
                    lands_right = (near_head - (near_head + look * behind
                                                + side * across)).dot(right) > 0
                    if lands_right != (float(near_x) > 0.5):
                        side = -side
                elif abs(near_head.x + side.x * across) > \
                        abs(near_head.x - side.x * across):
                    side = -side
                cam.location = near_head + look * behind + side * across
                # Aim at the face, not the crown: the top of a head's box is
                # hair, and a lens that tracks it puts the eyeline low.
                focus = mathutils.Vector((far_head.x, far_head.y,
                                          far_head.z - OTS_FACE_DROP_M))
                # The lens height is the declared angle, solved rather than
                # approximated: the camera aims at `focus`, so putting it
                # tan(elevation) x the horizontal run above that face makes the
                # view direction leave at exactly the declared elevation, which
                # is what every other position's solve already delivers and
                # this one used to miss by six degrees. The horizontal run does
                # not depend on the height, so there is nothing to iterate.
                run = math.hypot(cam.location.x - focus.x, cam.location.y - focus.y)
                want = focus.z + run * math.tan(math.radians(
                    float(spec.get("elevation_deg", OTS_EYE_LEVEL_DEG))))
                # Except where holding the angle would stop holding a shoulder.
                # Above the near crown the picture is the top of a skull, and an
                # over-the-shoulder that shows no shoulder is not one however
                # the angle reads; the band is named off the crown, and a panel
                # that states `rise_m` outright is taken at its word.
                lo, hi = (near_head.z + OTS_RISE_RANGE_M[0],
                          near_head.z + OTS_RISE_RANGE_M[1])
                cam.location.z = (near_head.z + float(ots["rise_m"])
                                  if "rise_m" in ots else min(max(want, lo), hi))
                shell_built["ots"] = {
                    "near": [round(v, 3) for v in near_head],
                    "far": [round(v, 3) for v in far_head],
                    "cam": [round(v, 3) for v in cam.location],
                    "focus": [round(v, 3) for v in focus],
                    "elevation_held": abs(cam.location.z - want) < 1e-6,
                }

    # A moving-camera demonstration frame: the fit-to-cast solve above still
    # runs unchanged (it is what makes `focus` and the starting pose real),
    # and this applies a crane-and-pull-out delta on top of it, in the same
    # focus-relative terms camera_movement_kb's "reveal" entry names (crane
    # height + pull_out) -- not a second, independent camera placement.
    # TRACK_TO is added below and keeps aiming at `focus` regardless, so the
    # move stays a crane, not a pan.
    delta = spec.get("camera_delta") or {}
    if delta:
        off = cam.location - focus
        scale = float(delta.get("pull_out_scale", 1.0))
        cam.location = focus + mathutils.Vector(
            (off.x * scale, off.y * scale, off.z + float(delta.get("height_delta_m", 0.0))))

    # ── the set is authored blind to where the camera will stand ──
    #
    # A location .blend is built once from the location bible; the camera is
    # solved per panel, from the shot size and the cast. So an open-plan floor
    # of desk rows is correct as a room and wrong as a frame: on one scene the
    # solve put the lens at y=4.6 and `desk_row_bank_mid` spans y=2.75..4.45,
    # a 15 m slab 16 cm in front of it. It hid two of the three subjects and
    # filled two thirds of the picture -- proxy_coverage_pct 14.1 -- and the
    # panel still reported ok, because nothing measures the set against the
    # solve.
    #
    # The rule is the one the shell already follows: nothing between lens and
    # cast. STRICTLY between -- an object reaching back into the cast is the
    # desk they are standing at, and one reaching past the lens is behind the
    # camera and costs nothing. Only the slab wholly in the gap is removed,
    # and which ones went is reported.
    #
    # And the second way the two disagree: nobody stands inside furniture. The
    # cast is placed from declared screen positions, which this corpus solves
    # to within 1.8% and must not be moved to dodge a desk, so it is the desk
    # that goes. Footprint containment, not bounding-box overlap: a subject
    # standing AT a desk overlaps it -- that is what standing at a desk is --
    # while a subject whose feet are inside its footprint is buried in it.
    # one scene staged three at one workstation and the middle one came out a
    # head resting on a desktop.
    # The horizontal centre of each body's box, not its object origin: a
    # proxy's origin is wherever the generator put it, and the question here
    # is where the person is standing.
    feet = []
    for b in bodies:
        c = [b.matrix_world @ mathutils.Vector(v) for v in b.bound_box]
        feet.append(((min(p.x for p in c) + max(p.x for p in c)) / 2,
                     (min(p.y for p in c) + max(p.y for p in c)) / 2))
    set_culled: list[str] = []
    for name in set_names:
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != "MESH" or not len(ob.data.vertices):
            continue
        pts = [ob.matrix_world @ v.co for v in ob.data.vertices]
        xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
        why = ""
        if min(ys) > max_y + _SET_CULL_MARGIN_M and max(ys) < cam.location.y:
            why = "between lens and cast"
        elif (max(zs) - min(zs)) > _SET_CULL_MIN_H_M and min(zs) < _SET_CULL_MIN_H_M \
                and any(min(xs) < fx < max(xs) and min(ys) < fy < max(ys)
                        for fx, fy in feet):
            # Floors and ceilings are excluded by the height test: everyone
            # stands inside the floor's footprint, and removing it is how a
            # greybox becomes the void this module's docstring is about.
            why = "a subject is standing inside it"
        if why:
            bpy.data.objects.remove(ob, do_unlink=True)
            set_culled.append({"name": name, "why": why})
    shell_built["set_culled"] = set_culled
    shell_bounds = None      # where the occlusion search may move the lens

    if spec.get("shape") == "subway":
        # ── shell, sized to contain the camera as well as the cast ──
        #
        # A control image has to fill the frame. Hiding the walls that occluded
        # left grey emptiness around the subjects, and the generator read that as
        # "nothing specified here" and returned it as a white border -- then
        # invented a fourth person in the space it was left to fill. So instead of
        # removing the shell, extend it past the camera: standing inside the cabin,
        # the walls surround the view, nothing is between lens and cast, and every
        # pixel carries structure.
        cam_y = cam.location.y
        back_y = min(min_y, focus.y) - 0.9                 # beyond the far seat
        front_y = cam_y + 0.6                              # behind the lens
        # NOT the KB's cabin: stretched until it contains the camera and the
        # cast plus 1.4 m of clearance. family_car declares 1.45 m across --
        # the Civic's SAE shoulder room W3 of 1448 mm -- and is built here at
        # 5.7 m, so no vehicle in any project is ever the width it declares,
        # which is also why narrowing scale_meters changes nothing visible.
        #
        # Building it at the declared size instead was tried and is worse, for
        # a reason that is about the CAMERA and not the cabin: a wide three-
        # shot cannot be framed from inside a 2 m car, so the solve stands the
        # lens outside, and a car-sized shell then leaves the cast a diorama
        # in black -- the emptiness this module's docstring records as the
        # failure that came back with an invented fourth person. Closing that
        # needs an exterior around the cabin, not a smaller cabin.
        #
        # `shell` in the result reports what was actually built against what
        # was declared, so the gap is a number rather than an inference.
        half_w = max(W, (max_x - min_x) + 1.4) / 2
        half_w = max(half_w, abs(cam.location.x) + 0.5)
        top_z = max(H, cam.location.z + 0.5, max_z + 0.35)
        shell_bounds = {"half_w": half_w, "top_z": top_z, "front_y": front_y}
        mid_y = (front_y + back_y) / 2
        depth = front_y - back_y

        # Furniture last, for the same reason as the shell: it is placed where the
        # panel says it is, and a tight shot solves to a camera that can sit ahead
        # of it. A console between lens and face is a dashboard occluding the
        # subject the panel is about, so it is dropped from THAT frame rather than
        # moved -- moving it would put the cabin somewhere the location does not
        # say it is.
        # Furniture after the camera, for the same reason as the shell: a tight
        # shot solves to a camera that can end up ahead of the dashboard, and
        # whether that matters is a question about the frame, not about the cabin.
        # Nothing is dropped: the console occluding a lap is what a car interior
        # looks like from the front, and it is what the location declares.
        # No console here, and no other named piece of one film's set. What
        # stands in this cabin is whatever the panel's props declare a
        # placement for; see _panel_fixtures.
        place_fixtures(W=W, D=D, H=H, base_z=seat_top, seats=seats)
        # The vehicle's own empty seats, drawn with the same geometry the
        # occupied ones get. Without them a two-person shot in a four-seat car
        # showed two chairs and bare floor, and the sampler filled that floor
        # with furniture of its own invention — a different chair every render.
        for j, (ex, ey) in enumerate(spec.get("empty_seats") or []):
            seat_geometry(f"seat_empty_{j}", ex, ey, seat_top, 0.62, 0.62)

        # A per_seat fixture (e.g. swivel_seat) already IS the seat -- drawing
        # the generic box on top of it is not a fallback, it is a second,
        # conflicting chair occupying the same space. The KB's own mesh is
        # what "the chair should look regular" is asking for; the box beneath
        # it (visible as a mismatched pedestal in every subway-shape cabin)
        # was never conditional on the fixture existing at all.
        has_seat_fixture = any(fx.get("anchor") == "per_seat" for fx in spec["fixtures"])
        for i, sxy in enumerate(seats):
            if not has_seat_fixture:
                # Same chair the meshless fixture path draws, so a cabin that
                # declares no seat prop and one that declares a seat without a
                # mesh do not stage two different pieces of furniture.
                seat_geometry(f"seat_{i}", sxy[0], sxy[1], seat_top, 0.62, 0.62)
        # No tray here. A slab was drawn across every lap in every cabin,
        # unconditionally and outside the has_seat_fixture guard, so a panel
        # that staged its own chair mesh got the mesh AND a hardcoded table
        # fused to it — which is what made the chair read wrong in the
        # delivered frame. Nothing in any KB declares a tray; it was one
        # film's set dressing living in code every other project also runs,
        # which is the thing the comment above this block objects to. A film
        # that wants tray tables declares a prop with a `per_seat` placement,
        # the same way swivel_seat is staged.

        shell_built.update(style="vehicle", width=round(half_w * 2, 4),
                           depth=round(depth, 4), height=round(top_z, 4),
                           kb_cabin=[W, D, H],
                           cast_x_extent=round(max_x - min_x, 4),
                           camera_x=round(cam.location.x, 4))
        _build_interior(box, style="vehicle", half_w=half_w, mid_y=mid_y,
                        depth=depth, top_z=top_z, back_y=back_y,
                        pillar_limit=cam_y - 0.8)
    elif spec.get("shape") in ("chamber", "office"):
        # ── a room: the same shell as the cabin, at a room's proportions ──
        #
        # Sized from what the location declares, then extended until it
        # contains the camera. A room the solver has stood outside of is a
        # room whose near wall occludes the cast, and hiding that wall is what
        # produced the void the generator filled with an invented person.
        # There is deliberately no front wall: the camera looks in through
        # where it would be, the way the cabin looks in through the windscreen.
        cam_y = cam.location.y
        back_y = min(min_y, focus.y) - max(1.2, D * 0.25)
        front_y = cam_y + 0.6
        half_w = max(W, (max_x - min_x) + 2.0) / 2
        half_w = max(half_w, abs(cam.location.x) + 0.6)
        top_z = max(H, cam.location.z + 0.6, max_z + 0.6)
        shell_bounds = {"half_w": half_w, "top_z": top_z, "front_y": front_y}
        mid_y = (front_y + back_y) / 2
        depth = front_y - back_y
        _build_interior(box, style="room", half_w=half_w, mid_y=mid_y,
                        depth=depth, top_z=top_z, back_y=back_y,
                        pillar_limit=cam_y - 0.8)
        # A room's furniture, staged like the cabin's. This branch was added
        # without this call, so an office prop was read by _panel_fixtures,
        # reported in the spec, and silently never built -- the exact failure
        # place_fixtures' own docstring records for shell_center. Declared
        # dimensions and floor level rather than the built shell: the shell is
        # stretched until it contains the camera, and measuring a desk against
        # THAT would slide it to a wall that is only there to fill the frame.
        # Spans in metres, positions in the room. A cabin fixture spans the
        # cabin -- a console IS the width of whatever car it is in -- which is
        # what makes fractions right there. A desk is 1.6 m across in a 6 m
        # office and in an 18 m one, and another production's rooms run 6 m to 18 m wide, so
        # one authored span read fractionally would be three different desks.
        # Where it stands is the opposite: `shell_rear` is the back wall, and
        # that IS a fact about the room.
        place_fixtures(W=half_w * 2, D=depth, H=top_z, base_z=0.0,
                       origin=(focus.x, focus.y), span_scale=(1.0, 1.0, 1.0),
                       center=(0.0, mid_y))

    else:
        # ── open ground: no shell, because a street has no ceiling ──
        #
        # The cabin's problem was emptiness reading as "nothing specified here".
        # Outdoors that is only half true: sky IS featureless, and inventing
        # geometry in it would be inventing the shot. What must not be empty is
        # the GROUND, which runs to the horizon and would otherwise be one flat
        # field with three figures on it.
        #
        # So: ground under and past the cast, a horizon line at a distance the
        # solve knows, and massing on it. The massing is not decoration -- this
        # location's own words are "overturned futuristic vehicle, scattered
        # debris, responders", and blocks at that scale are the greybox of them.
        # All of it sized off the camera distance, so a wide and a close-up get a
        # horizon in the same place relative to the frame rather than in metres
        # someone picked.
        reach = max(dist * 3.0, 12.0)
        box("ground", (focus.x, focus.y, -0.02), (reach * 2, reach * 2, 0.04))

        # The set dressing an exterior declares, staged like any other fixture.
        # place_fixtures used to be called only from the cabin branch, so a
        # prop outdoors could not be staged however its placement was written:
        # one scene declares robot_arms "emerging from the wreckage" and the
        # greybox held two bodies and open ground, so the model invented both
        # the wreck and the arms and put them where the composition had room —
        # across the chest of the subject in the middle. Metres, not fractions
        # of a shell, because out here there is no shell to take a fraction of.
        place_fixtures(W=1.0, D=1.0, H=1.0, base_z=0.0,
                       origin=(focus.x, focus.y))

        # Ground seams, the same trick the cabin floor uses: a plane at a grazing
        # angle needs an edge to read as a plane at all.
        for i in range(-3, 4):
            box(f"ground_seam_{i + 3}", (focus.x + i * reach * 0.22, focus.y, 0.005),
                (0.06, reach * 2, 0.02))

        # Horizon massing: a band of blocks at the far edge, varied in width and
        # height so it reads as a built horizon rather than a wall. Heights stay
        # under a fifth of the distance, which keeps them below the eyeline of any
        # camera the solver picks.
        far_y = focus.y - reach * 0.85
        n_mass = 9
        for i in range(n_mass):
            t = (i / (n_mass - 1)) - 0.5
            w = reach * (0.16 + 0.05 * ((i * 7) % 3))
            h = reach * (0.06 + 0.03 * ((i * 5) % 4))
            box(f"mass_{i}", (focus.x + t * reach * 1.7, far_y - w * 0.2, h / 2),
                (w, w * 0.8, h))

        # Nearer debris, at the cast's own scale, so the middle distance is not a
        # jump from bodies straight to the horizon.
        for i, (dx, dy, sz) in enumerate((( 2.4, -1.8, 0.9), (-2.9, -2.6, 1.3),
                                          ( 3.6, -3.4, 0.7), (-3.9, -1.2, 0.5))):
            box(f"debris_{i}", (focus.x + dx, focus.y + dy, sz * 0.35),
                (sz, sz * 1.6, sz * 0.7))

    # Aim with a constraint, not with hand-tuned Euler angles: the target is
    # the centre of the occupied seats, so the framing follows the cast rather
    # than a guess that has to be re-guessed whenever the cabin or seat layout
    # changes.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=focus)
    target = bpy.context.active_object
    target.name = "cam_target"
    trk = cam.constraints.new(type="TRACK_TO")
    trk.target = target
    trk.track_axis, trk.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    bpy.context.scene.camera = cam

    # ── see past the set: a head the lens cannot reach is not a read point ──
    #
    # The set is placed where the location says and the camera where the fit
    # says, and nothing checked that one leaves a line of sight to the other:
    # ten of another production's 78 panels put a desk, a partition or a building between
    # the lens and the face the panel is about, and the gate only reported it.
    shell_built["occlusion"] = _see_past_set(
        cam=cam, focus=focus, bodies=bodies, heads=heads, set_names=set_names,
        spec=spec, shell_bounds=shell_bounds)

    # ── a canted horizon (dutch angle), about the optical axis ──
    #
    # Applied last, once the aim and the occlusion search have settled the
    # pose: the aim constraint would undo a roll, so the evaluated pose is
    # baked, the constraint dropped, and the camera turned about its own
    # viewing axis. The frame rotates about its centre, so the read point the
    # aim put there stays there.
    roll = float(spec.get("roll_deg") or 0.0)
    if roll:
        bpy.context.view_layer.update()
        mw = cam.matrix_world.copy()
        for c in list(cam.constraints):
            cam.constraints.remove(c)
        cam.matrix_world = mw @ mathutils.Matrix.Rotation(math.radians(roll), 4, "Z")
        bpy.context.view_layer.update()

    # ── flat greybox render: shape only, no material or lighting cues ──
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.55, 0.55, 0.56)
    # A dressed cast carries its garment tones as a colour attribute, and a
    # fixture the panel declares powered off carries its own; every other mesh
    # gets the same attribute at the greybox's own grey, so the frame is
    # unchanged wherever nothing was dressed and nothing was switched off.
    dark_ids = [fx["id"] for fx in (spec.get("fixtures") or []) if fx.get("dark")]
    if any(s.get("costume") for s in spec["subjects"]) or dark_ids:
        for o in bpy.data.objects:
            if o.type == "MESH":
                if _TONE_ATTR not in o.data.color_attributes:
                    _tone_mesh(o, _GREY)
                _use_tone(o)
        # The dark tone goes on after the fill above, or the fill would erase
        # it. The fixture's own mesh keeps its shape, so a screen that is off
        # is the same screen: only what it renders as changes.
        for fid in dark_ids:
            # A fixture on a repeated anchor is built once per place, named
            # `<id>_0`, `<id>_1`, so the side panels of a cabin are two objects
            # under one declaration; matching the id alone left them lit.
            roots = [o for o in bpy.data.objects
                     if o.name == fid or o.name.startswith(fid + "_")]
            for root in roots:
                for o in [m for m in [root, *root.children_recursive]
                          if m.type == "MESH"]:
                    _tone_mesh(o, _DARK)
                    _use_tone(o)
        sh.color_type = "VERTEX"
    sh.show_cavity = True
    sc.render.resolution_x, sc.render.resolution_y = spec["res"]
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = spec["out"]
    bpy.ops.render.render(write_still=True)

    # ── the set alone, for what stands behind a head ──
    #
    # A vertical edge that runs into a head from above reads as something
    # growing out of it -- the lamp post behind the anchor. The beauty frame
    # hides the stretch of the edge the head covers, so the gate reads it off
    # the set rendered without the cast: same camera, same shading.
    # ── which staged props the camera sees ──
    #
    # A fixture is built wherever its anchor puts it, not where the camera
    # looks. The prompt compiler leaves out what the frame does not hold
    # (pai_compat.props_out_of_frame): a prop behind the lens, one the frame
    # crops to a sliver (less than PROP_SHOWN_MIN of its projected box inside
    # the frame, or less than PROP_IN_FRAME_MIN of the frame), or one the
    # occlusion search hid.
    props_out = None
    try:
        from bpy_extras.object_utils import world_to_camera_view
        bpy.context.view_layer.update()
        cam_ev = cam.evaluated_get(bpy.context.evaluated_depsgraph_get())
        seen_props = {}
        for fx in spec.get("fixtures") or []:
            o = bpy.data.objects.get(fx["id"])
            if o is None:
                continue
            parts = [m for m in [o, *o.children_recursive] if m.type == "MESH"]
            pts = [world_to_camera_view(sc, cam_ev, m.matrix_world @ mathutils.Vector(c))
                   for m in parts for c in m.bound_box]
            front = [q for q in pts if q.z > 0]
            share = shown = 0.0
            if front and not o.hide_render:
                ux0, ux1 = min(q.x for q in front), max(q.x for q in front)
                uy0, uy1 = min(q.y for q in front), max(q.y for q in front)
                share = (max(0.0, min(1.0, ux1) - max(0.0, ux0))
                         * max(0.0, min(1.0, uy1) - max(0.0, uy0)))
                whole = (ux1 - ux0) * (uy1 - uy0)
                shown = share / whole if whole > 0 else 0.0
            rec = {"in_frame": share >= PROP_IN_FRAME_MIN
                               and (shown >= PROP_SHOWN_MIN or share >= PROP_FILLS_FRAME),
                   "frame_share": round(share, 4), "shown": round(shown, 3)}
            for k in {fx["id"], fx.get("key")} - {None}:
                seen_props[k] = rec
        props_out = spec["out"] + ".props.json"
        with open(props_out, "w") as fh:
            json.dump(seen_props, fh, ensure_ascii=False)
    except Exception as e:                                   # noqa: BLE001
        print(f"prop visibility not measured: {e}")
        props_out = None

    set_out = None
    try:
        was = {o.name: o.hide_render for o in bodies}
        for o in bodies:
            o.hide_render = True
        set_out = spec["out"] + ".set.png"
        sc.render.filepath = set_out
        bpy.ops.render.render(write_still=True)
        for o in bodies:
            o.hide_render = was[o.name]
    except Exception as e:                                   # noqa: BLE001
        print(f"set pass not rendered: {e}")
        set_out = None
    sc.render.filepath = spec["out"]

    # ── where each body's joints land in the frame ──
    #
    # A frame edge through a knee or an elbow reads as an amputation. The
    # gate needs the joints in screen space to say so, and only the kernel
    # has the camera: x and y as shares of the frame, y from the top, and
    # depth, which is negative behind the lens.
    joints_out = None
    try:
        from bpy_extras.object_utils import world_to_camera_view
        bpy.context.view_layer.update()
        cam_ev = cam.evaluated_get(bpy.context.evaluated_depsgraph_get())
        projected = {}
        for cid, js in joints_world.items():
            projected[cid] = {}
            for k, p in js.items():
                co = world_to_camera_view(sc, cam_ev, p)
                projected[cid][k] = [round(co.x, 4), round(1.0 - co.y, 4), round(co.z, 4)]
        joints_out = spec["out"] + ".joints.json"
        with open(joints_out, "w") as fh:
            json.dump(projected, fh, ensure_ascii=False)
    except Exception as e:                                   # noqa: BLE001
        print(f"joints not projected: {e}")
        joints_out = None

    # ── the same geometry as DEPTH, for use as a video control ──
    # The clay render above is a beauty pass: Workbench shading, studio light,
    # cavity. Fed to VACE as control_video it carries that appearance along with
    # the structure, and the clip comes back as clay mannequins at a strength
    # high enough to hold the cast, or loses the cast entirely at a strength low
    # enough to look photoreal. Measured on one corpus shot: cs 0.50 kept three
    # bodies and rendered them in clay; cs 0.30 dissolved them; cs 0.15 produced
    # a different scene.
    #
    # Depth has no appearance to leak. Same camera, same frame, so it drops
    # straight in where the clay render was being used.
    depth_out = None
    try:
        depth_out = _render_depth_twin(sc, spec)
    except Exception as e:                                   # noqa: BLE001
        # A missing depth twin costs the video path its control, not the panel
        # its greybox — the still render above is already written.
        print(f"depth twin not rendered: {e}")
    # What share of the frame the bodies actually occupy, counted in render
    # space rather than derived from a bounding box. This is the quantity the
    # coverage measurement watches: past some share, a crude proxy stops
    # staging a shot and starts being rendered as literal grey objects, and
    # reporting it lets a batch name the panels sitting near that line instead
    # of it surfacing in a delivered frame. A box would answer backwards here
    # -- the thin band an extreme close-up fits is a small box around bodies
    # that fill the frame -- so this is a second pass with everything but the
    # cast hidden, alpha counted, at a fraction of the resolution.
    coverage_pct = None
    try:
        hidden = [o for o in bpy.data.objects
                  if o.type == "MESH" and o not in bodies]
        for o in hidden:
            o.hide_render = True
        sc.render.film_transparent = True
        sc.render.resolution_x = max(160, spec["res"][0] // 4)
        sc.render.resolution_y = max(68, spec["res"][1] // 4)
        mask_path = spec["out"] + ".proxy_mask.png"
        sc.render.filepath = mask_path
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(mask_path)
        alpha = list(img.pixels)[3::4]
        coverage_pct = round(100.0 * sum(1 for a in alpha if a > 0.5) / max(len(alpha), 1), 1)
        bpy.data.images.remove(img)
        os.remove(mask_path)
    except Exception as e:                                   # noqa: BLE001
        print(f"proxy coverage not measured: {e}")

    # ── the set as a reusable asset, when asked ──
    # Everything above assembles the location: a shell at the KB's
    # scale_meters, seats where the layout puts them, and each fixture scaled
    # into the span the KB gives it. That assembly is rebuilt from scratch for
    # every panel and thrown away, which is why the cabin is dimensionally
    # identical across panels and looks different in each one — geometry is
    # re-derived, appearance is re-invented.
    #
    # Exported once it is an asset: the same object every panel renders from,
    # a real mesh for the video path instead of scene_mode's ground contour,
    # and something that can carry materials later. Bodies are excluded by
    # default — the cast is per-panel and the set is not — EXCEPT for the
    # video path's mask_from_subject: that needs a subject mesh in the GLB to
    # render a matte from at all, so export_glb_include_bodies opts them in.
    set_glb = spec.get("export_glb")
    include_bodies = bool(spec.get("export_glb_include_bodies"))
    if set_glb:
        try:
            for b in bodies:
                b.hide_render = True
                b.select_set(False)
            bpy.ops.object.select_all(action="DESELECT")
            head_copies = set(heads.values())
            for o in bpy.data.objects:
                if o.type == "MESH" and o not in head_copies and (include_bodies or o not in bodies):
                    o.select_set(True)
            Path(set_glb).parent.mkdir(parents=True, exist_ok=True)
            bpy.ops.export_scene.gltf(filepath=set_glb, export_format="GLB",
                                      use_selection=True, export_animations=False)
            for b in bodies:
                b.hide_render = False
        except Exception as e:                               # noqa: BLE001
            print(f"set GLB not exported: {e}")
            set_glb = None

    # ── one matte per body, for masked per-face identity ──
    # Whole-frame reference conditioning cannot carry a group panel: measured on
    # one shot, eight reference plates rendered four people where three
    # were declared, and so did five, and so did two plates of inanimate props.
    # inpaint_ref_flux2 solves that by bounding every edited pixel to a mask —
    # but nothing was producing the masks. The geometry already knows exactly
    # where each declared body is, so it can say.
    #
    # Alpha only; the host turns each matte into a head mask (see
    # face_mask_from_matte). Keyed by character_id so a mask can be paired with
    # the right person's reference without matching anything up by position.
    body_mattes: dict = {}
    head_mattes: dict = {}
    visible_mattes: dict = {}
    try:
        prev_hidden = {o.name: o.hide_render for o in bpy.data.objects if o.type == "MESH"}
        sc.render.film_transparent = True
        # RGBA explicitly: a matte IS its alpha channel, and inheriting a BW or
        # RGB setting from an earlier pass makes every body read as opaque.
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.resolution_x, sc.render.resolution_y = spec["res"]
        for b in bodies:
            cid = b.name[5:] if b.name.startswith("body_") else b.name
            for o in bpy.data.objects:
                if o.type == "MESH":
                    o.hide_render = (o is not b)
            path = spec["out"] + f".body_{cid}.png"
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            body_mattes[cid] = path
            # A separate matte of just the head sub-mesh (see heads{} above):
            # a true head-region alpha rather than a fraction of the whole
            # body's silhouette, for read-point measurement against the
            # declared screen_position independent of the fixed-percentage
            # heuristic node/face_masks.py uses for reference-image bounding.
            head_obj = heads.get(cid)
            if head_obj is not None:
                for o in bpy.data.objects:
                    if o.type == "MESH":
                        o.hide_render = (o is not head_obj)
                head_path = spec["out"] + f".head_{cid}.png"
                sc.render.filepath = head_path
                bpy.ops.render.render(write_still=True)
                head_mattes[cid] = head_path
        # ── the same bodies, but with the SET left standing ──
        #
        # The mattes above hide every other mesh, so each is the silhouette a
        # body WOULD have alone. That is the right input for screen position
        # and for body-vs-body occlusion, and it is blind to the set: a wall
        # panel standing between camera and cast costs a subject nothing in a
        # measurement taken with the wall hidden. Scene 2 rendered a subject
        # whose head was entirely behind a side panel and every clause passed.
        #
        # Rendering the body against the set with a flat two-tone shading
        # settles it in the z-buffer instead of in a compositor: everything
        # is black, the one body is white, and what comes back white is what
        # the camera can actually see of it. Workbench has no holdout or
        # cryptomatte, which is why this is object colour rather than a mask
        # pass. Other BODIES stay hidden -- body-vs-body is already measured,
        # and mixing the two would make one number that cannot say which.
        sh = sc.display.shading
        prev_light, prev_ctype = sh.light, sh.color_type
        prev_colors = {o.name: tuple(o.color) for o in bpy.data.objects if o.type == "MESH"}
        sh.light, sh.color_type = "FLAT", "OBJECT"
        sc.render.film_transparent = False
        sc.render.image_settings.color_mode = "BW"
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.color = (0.0, 0.0, 0.0, 1.0)
        body_set = set(bodies) | set(heads.values())
        for b in bodies:
            cid = b.name[5:] if b.name.startswith("body_") else b.name
            for o in bpy.data.objects:
                if o.type == "MESH":
                    o.hide_render = (o in body_set and o is not b)
            b.color = (1.0, 1.0, 1.0, 1.0)
            path = spec["out"] + f".visible_{cid}.png"
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            visible_mattes[cid] = path
            b.color = (0.0, 0.0, 0.0, 1.0)
        sh.light, sh.color_type = prev_light, prev_ctype
        for o in bpy.data.objects:
            if o.type == "MESH" and o.name in prev_colors:
                o.color = prev_colors[o.name]

        for o in bpy.data.objects:
            if o.type == "MESH" and o.name in prev_hidden:
                o.hide_render = prev_hidden[o.name]
    except Exception as e:                                   # noqa: BLE001
        print(f"body mattes not rendered: {e}")

    # Evaluated world rotation (the TRACK_TO constraint above only resolves
    # at depsgraph-evaluation time) — a caller wiring this panel's exported
    # GLB into a video job's trajectory_track needs both halves of the
    # camera pose the same solve produced, not just its location.
    deps = bpy.context.evaluated_depsgraph_get()
    cam_eval = cam.evaluated_get(deps)
    cam_rotation_deg = [round(math.degrees(v), 3) for v in cam_eval.matrix_world.to_euler()]

    return {"ok": True, "out": spec["out"], "depth_out": depth_out,
            "set_out": set_out, "joints_out": joints_out, "props_out": props_out,
            "set_glb": set_glb,
            "body_mattes": body_mattes,
            "head_mattes": head_mattes,
            "visible_mattes": visible_mattes,
            "fixture_fit": fit_report,
            # What the shell actually came out at, which is not what the KB
            # asked for whenever the clearance around the cast is the binding
            # constraint. Reported so that is visible rather than inferred.
            "shell": shell_built,
            "fixtures_skipped": skipped_fixtures,
            "bodies": len(bodies),
            "seat_top": round(seat_top, 4),
            "shot_size": (spec.get("framing") or {}).get("shot_size"),
            "proxy_coverage_pct": coverage_pct,
            # Whether the declared shot size actually bound the framing, or the
            # width of the cast the panel also declared did. See the distance
            # solve above: a tighter size on a multi-body pattern is
            # over-constrained, and this says so instead of leaving the panel
            # to look inexplicably loose.
            "shot_size_bound_by": shot_size_bound_by,
            "camera": [round(v, 4) for v in cam.location],
            "camera_rotation_deg": cam_rotation_deg,
            "lens_mm": spec.get("lens_mm")}


def _kernel_dispatch() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=["greybox"])
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    # Always emit a RESULT_JSON, including on the way out of a raise. Blender
    # exits quietly when a kernel throws: no result reaches the caller, the
    # taskq job still reports done, the route still returns 200, and the
    # previous PNG stays on disk. A silently stale greybox is worse than a
    # failed one, because every downstream check reads it as a fresh render
    # that happened not to change.
    try:
        result = _kernel_greybox(spec)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-2000:]}
    print("RESULT_JSON=" + json.dumps(result))


if __name__ == "__main__":
    _kernel_dispatch()
