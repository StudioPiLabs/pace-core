"""PAI 1.1 schema — SCINE-aligned 4-pillar taxonomy.

This is the SCINE (Stable Cinemetrics) NeurIPS-2025 taxonomy, faithfully
reproduced as Python dataclasses. Every leaf node listed in the paper's
Tables 3-6 (Camera/Lighting/Setup/Events) has a corresponding field here;
every controlled vocabulary is encoded as a Literal so static checkers
catch typos and consumers can enumerate options.

Schema metadata (single source of truth — readers should import these
rather than hard-coding the version string):

  SCHEMA_VERSION       — current version tag emitted by SceneDoc._schema_version
                          (also accepted on read; older versions in
                          SUPPORTED_SCHEMA_VERSIONS are accepted with adapters)
  SUPPORTED_SCHEMA_VERSIONS — versions that the readers (e.g.
                          enrich_scenes_scine.load) will deserialize without
                          warning. Add a new tag here when bumping.

Top-level structure:
    SceneDoc                  — one screenplay scene
      .narrative_meta         — character roster, beats, dialogue (outside pillars)
      .shots: list[Shot]      — ordered list of cinematic shots in this scene
        Shot                    — one cinematic unit (atomic shot)
          .camera: Camera         — Pillar 1 (intrinsics / extrinsics / trajectory / creative)
          .setup: Setup           — Pillar 2 (texture / geometry / space / backdrop / environment / props / subjects)
          .lighting: Lighting     — Pillar 3 (sources / color temp / condition / shadows / position / motion / gels)
          .events: Events         — Pillar 4 (actions / emotions / dialogues / change-in-env / story structure)
          .panels: list[Panel]    — storyboard frames realising this shot
      .compile_hints[]         — per-panel sidecar hints (Flux/GPT-Image-2/Wan-I2V)

Convention for field comments:
  - Most fields are Optional[...] = None → leave None when narrative is silent.
  - Open-set str fields take free-form English; controlled Literals must
    use one of the listed tokens verbatim (the enrichment LLM validates).
  - "PAI extension" tag = added by us on top of stock SCINE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# ════════════════════════════════════════════════════════════════════════
# Pillar 1 — Camera (Table 3)
# ════════════════════════════════════════════════════════════════════════

# Lens focal-length CLASS (not a specific mm). "standard" ≈ 50mm normal
# perspective, "wide" ≈ 18-35mm, "long_lens" ≈ 85-135mm portrait,
# "telephoto" ≈ 200mm+, "fisheye" = ultra-wide distorted.
LensSize       = Literal["standard", "fisheye", "wide", "medium", "long_lens", "telephoto"]
# Depth-of-field bucket. "deep" = everything in focus, "shallow" = subject
# isolated, "rack" = focus pulls during the shot, "split_diopter" =
# foreground+background both sharp via lens trick, "tilt_shift" = miniature look.
DepthOfField   = Literal["deep", "shallow", "soft", "rack", "split_diopter", "tilt_shift"]
# Aperture bucket. "wide" ≈ f/1.4-2.8 (shallow DoF + bright),
# "narrow" ≈ f/8-16 (deep DoF + dim), "medium" between.
Aperture       = Literal["wide", "medium", "narrow"]
# Shutter speed bucket. Bound to motion blur amount + exposure time.
ShutterSpeed   = Literal["slow", "medium", "fast"]
# ISO bucket. "high" = noisy/grainy, "low" = clean but needs more light.
ISOLevel       = Literal["low", "medium", "high"]
# Camera vertical/tilt placement relative to subject.
#   "eye_level"  = neutral, subject's eye line
#   "low"        = camera below subject, looking UP (heroic / imposing)
#   "high"       = camera above subject, looking DOWN (vulnerable / observed)
#   "overhead"   = top-down, perpendicular to ground (god view)
#   "aerial"     = far above (drone / sky)
#   "dutch"      = tilted (canted horizon) — unease / disorientation
#   "shoulder/hip/knee/ground" = body-height datums for low-rise cameras
#   "continuous" = angle changes during the shot (rare)
Angle          = Literal[
    "low", "high", "aerial", "overhead", "dutch",
    "eye_level", "shoulder", "hip", "knee", "ground", "continuous",
]
# Planar (2D) camera moves — no real change in spatial position.
Movement2D     = Literal["pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out"]
# Spatial (3D) camera moves — camera physically translates.
#   push_in/pull_out = dolly toward/away
#   dolly_zoom       = simultaneous dolly + opposing zoom (Vertigo effect)
#   camera_roll      = rotates around its optical axis (canted to upright)
#   tracking         = parallel to subject motion
#   trucking         = lateral travel perpendicular to subject
#   arc              = curved/circular path around subject
#   crane            = vertical translation up/down
Movement3D     = Literal[
    "push_in", "pull_out", "dolly_zoom", "camera_roll",
    "tracking", "trucking", "arc", "crane",
]
# Camera support / rig — affects motion character (steady vs handheld feel).
Gear           = Literal[
    "handheld", "tripod", "pedestal", "cranes", "overhead_rigs", "dolly",
    "stabilizer", "snorricam", "vehicle_mount", "drones",
    "motion_control", "steadicam",
]
# 景别 — how much of the subject + environment is in frame.
#   extreme_close_up = a single feature fills the frame (eye, lips, hand)
#   close_up         = head only
#   medium_close_up  = head + shoulders
#   medium           = waist up
#   medium_full      = mid-thigh up
#   full             = head-to-feet, environment present
#   wide             = subject small in large environment
#   establishing     = environment dominant, subject tiny/absent
#   master           = full-scene blocking shot used in coverage
ShotSize       = Literal[
    "establishing", "master", "wide", "full", "medium_full",
    "medium", "medium_close_up", "close_up", "extreme_close_up",
]
# Composition pattern (who/what is in the frame).
#   single   = one subject
#   two_shot = two subjects, both weighted
#   crowd    = three+ subjects
#   ots      = over-the-shoulder of subject A looking at subject B
#   pov      = first-person — camera IS the character's eyes
#   insert   = detail cut-in on a prop or feature (object-centric)
#   empty    = 空镜 — no human/animal in frame, environment only
#              (use for transitions / mood-set / time-jump beats; distinct
#              from "insert" which still has a clear object focus)
Framing        = Literal["single", "two_shot", "crowd", "ots", "pov", "insert", "empty"]
# Temporal sampling curve between camera-track keyframes.
#   linear        = constant velocity
#   ease_in       = slow start, fast end
#   ease_out      = fast start, slow end (most natural for camera moves)
#   ease_in_out   = slow at both ends — emphasizes the middle
Easing         = Literal["linear", "ease_in", "ease_out", "ease_in_out"]


@dataclass
class CameraIntrinsics:
    """Optical / exposure parameters. SCINE encodes them as 3-bucket
    categoricals (wide/medium/narrow for aperture, etc.); the exact-value
    fields below let real-capture metadata travel alongside without
    conflict, but compilers should read the categorical fields."""
    lens_size:       Optional[LensSize]      = None  # focal-length CLASS (see LensSize)
    depth_of_field:  Optional[DepthOfField]  = None  # focus behavior — shallow/deep/rack/etc
    aperture:        Optional[Aperture]      = None  # wide/medium/narrow f-stop bucket
    shutter_speed:   Optional[ShutterSpeed]  = None  # slow/medium/fast bucket — affects motion blur
    iso:             Optional[ISOLevel]      = None  # low/medium/high sensor sensitivity bucket
    # ── Exact-value layer (real-capture metadata; advisory only) ─────────
    focal_length_mm: Optional[float] = None          # exact lens length, e.g. 35.0
    aperture_f:      Optional[float] = None          # f-stop number, e.g. 2.8
    t_stop:          Optional[float] = None          # cinema lens T-stop (after light loss)
    shutter_angle_deg: Optional[float] = None        # 180° = cinema standard; 90° = sharp; 360° = motion-blurred
    iso_value:       Optional[int]   = None          # exact ISO, e.g. 800
    # pace-0.2 camera.intrinsics.sensor_mm — physical gate [w,h] in mm
    # (default 36×24 full-frame). Resolves the focal-length→FOV ambiguity.
    sensor_mm:       Optional[list[float]] = None     # [w, h] e.g. [36.0, 24.0]; both > 0
    # pace-0.2 camera.intrinsics.focus_distance_m — focus plane distance (m), > 0
    focus_distance_m: Optional[float] = None          # e.g. 2.5

    def __post_init__(self) -> None:
        if self.sensor_mm is not None:
            if len(self.sensor_mm) != 2 or any(v <= 0 for v in self.sensor_mm):
                raise ValueError(
                    f"sensor_mm must be [w_mm, h_mm] with both > 0, got {self.sensor_mm}"
                )
        if self.focus_distance_m is not None and self.focus_distance_m <= 0:
            raise ValueError(
                f"focus_distance_m must be > 0, got {self.focus_distance_m}"
            )


@dataclass
class CameraExtrinsics:
    """Camera placement / orientation."""
    angle: Optional[Angle] = None                    # vertical tilt: eye_level / low / high / overhead / dutch / …
    roll_deg: Optional[float] = None                 # canted horizon about the optical axis, + = clockwise; "dutch" alone implies 15
    # PAI extension — relative position to subject (front / 3-4 / profile /
    # ots / behind). Terms.md §6.2 covers this even though SCINE Table 3
    # only lists `angle`; we keep it because it's a real distinction in
    # the data (e.g. reverse-shot vs front-on).
    position: Optional[Literal["front", "three_quarter", "profile", "ots", "behind"]] = None
    # ↑ horizontal placement: in front of subject / 3⁄4 turn / pure side / over-shoulder / behind back


@dataclass
class CameraTrajectory:
    """Camera motion. SCINE splits movement by gear-capability:
    Static (no motion), 2D (planar — pan/tilt/zoom), 3D (full spatial).
    Composite moves are allowed — list both ["push_in", "tilt_up"] when
    the camera dollies in AND tilts at once."""
    static:        bool                = False             # True ⇒ locked-off, ignore all movement fields
    movement_2d:   list[Movement2D]    = field(default_factory=list)  # planar moves (pan/tilt/zoom)
    movement_3d:   list[Movement3D]    = field(default_factory=list)  # spatial moves (dolly/track/crane/arc/…)
    gear:          Optional[Gear]      = None              # rig used — affects steadiness + motion character
    # PAI extension — temporal sampling curve between start/end keyframes.
    easing:        Easing              = "linear"          # acceleration profile of the move
    # pace-0.2 camera.trajectory.camera_path — artifact ref (assets:// URI) to the
    # computed 6-DoF keyframe JSON (the PRODUCT of camera.program's LAMP recipe).
    camera_path:   Optional[str]       = None             # 6-DoF keyframe JSON artifact reference


@dataclass
class CameraCreativeIntent:
    """Compositional choices that shape narrative/emotional tone."""
    shot_size:    Optional[ShotSize] = None              # jingbie (shot size) — how much of subject/environment is in frame
    framing:      Optional[Framing]  = None              # composition pattern (single/two_shot/empty/…)
    # PAI extension — aspect ratio is a real per-shot choice (2.35:1
    # anamorphic vs 16:9 for inserts) even though SCINE doesn't enumerate it.
    aspect_ratio: str                = "2.35:1"          # e.g. "2.35:1", "16:9", "1.85:1", "4:3"


# DSL provenance for a camera-motion program (pace-0.2 camera.program.source).
ProgramSource = Literal["rule", "llm", "human"]


@dataclass
class FrameRate:
    """pace-0.2 camera.fps — rational frame rate {num/denom}. Rational, NOT
    float: 23.976 fps must be 24000/1001 exactly (floats are banned to keep
    cut-list / timecode math exact). Both terms > 0."""
    num:   int                                       # numerator, e.g. 24000
    denom: int                                       # denominator, e.g. 1001  (24fps = 24/1)

    def __post_init__(self) -> None:
        # schema: type=integer, exclusiveMinimum 0. Enforce int-ness too —
        # the rational form exists precisely so floats never enter timecode math.
        for name, v in (("num", self.num), ("denom", self.denom)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"FrameRate {name} must be an int, got {v!r}")
        if self.num <= 0 or self.denom <= 0:
            raise ValueError(
                f"FrameRate num/denom must both be > 0, got {self.num}/{self.denom}"
            )


@dataclass
class CameraProgram:
    """pace-0.2 camera.program — the *recipe* that produces the camera
    trajectory (truth), as opposed to trajectory.camera_path which is the
    computed 6-DoF keyframe artifact (product). Originated as PAILang's own
    studio field, absorbed into the standard. Schema allows extra keys."""
    lamp_dsl:  Optional[str]           = None        # 24-token LAMP motion DSL → compiled to 6-DoF track
    source:    Optional[ProgramSource] = None        # rule / llm / human — how the DSL was authored
    narrative: Optional[str]           = None        # natural-language intent that produced the DSL (reproducible input)


@dataclass
class Camera:
    """SCINE Pillar 1 — every control related to camera configuration."""
    intrinsics:      CameraIntrinsics      = field(default_factory=CameraIntrinsics)       # lens + exposure
    extrinsics:      CameraExtrinsics      = field(default_factory=CameraExtrinsics)       # angle + position
    trajectory:      CameraTrajectory      = field(default_factory=CameraTrajectory)       # motion + easing
    creative_intent: CameraCreativeIntent  = field(default_factory=CameraCreativeIntent)   # shot_size / framing / aspect
    # pace-0.2 timing/program layer (all Optional — purely additive)
    fps:             Optional[FrameRate]    = None     # rational frame rate {num,denom}
    frame_range:     Optional[list[int]]    = None     # [start, end] inclusive frame indices, both >= 0
    program:         Optional[CameraProgram] = None    # camera-motion recipe (lamp_dsl/source/narrative)

    def __post_init__(self) -> None:
        if self.frame_range is not None:
            # schema: array of exactly 2 integers, each minimum 0.
            ok = (
                len(self.frame_range) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in self.frame_range)
                and all(v >= 0 for v in self.frame_range)
            )
            if not ok:
                raise ValueError(
                    f"frame_range must be [start, end] — two ints, both >= 0, got {self.frame_range}"
                )


# ════════════════════════════════════════════════════════════════════════
# Pillar 2 — Setup (Table 5)
# ════════════════════════════════════════════════════════════════════════

# Image-level contrast (light↔dark spread). "low" = washed out / muted,
# "high" = stark / chiaroscuro.
Contrast       = Literal["low", "high"]
# Blur type if any. gaussian = uniform soft, radial = circular outward,
# motion = directional from camera/subject movement.
BlurKind       = Literal["gaussian", "radial", "motion"]
# Visual noise / grain pattern.
NoiseKind      = Literal["gaussian", "salt_and_pepper", "poisson"]
# Dominant linear direction in the composition.
LineDirection  = Literal["horizontal", "vertical", "diagonal"]
# Regular geometric shapes dominating the frame.
RegularShape   = Literal["square", "circle", "triangle"]
# Natural / organic shape vocabulary (Table 5).
NaturalShape   = Literal["water_like", "cloud_like"]
# How subjects are placed within the frame's mass distribution.
#   rule_of_thirds = subject on a third-grid intersection
#   symmetry       = centered / mirrored composition
#   right/left_heavy = visual mass biased one side
FrameBalance   = Literal["rule_of_thirds", "symmetry", "right_heavy", "left_heavy"]
# Spatial depth feel.
#   deep      = strong foreground/midground/background separation
#   flat      = compressed / 2-plane look (often telephoto)
#   limited   = shallow stage, e.g. close interior
#   ambiguous = depth cues intentionally absent
Depth          = Literal["deep", "flat", "limited", "ambiguous"]
# Interior vs Exterior.
Setting        = Literal["int", "ext"]
# Time-of-day. The 11-value vocab is finer than INT/EXT — picks specific
# light quality bands.
TimeOfDay      = Literal[
    "day", "night", "morning", "evening", "dawn", "dusk",
    "late_night", "midday", "sunrise", "sunset", "afternoon",
]
# Organizational density of the composition.
#   "clean"     = sparse / pristine, few elements in frame
#   "cluttered" = busy / full, many elements competing for attention
# (Renamed from "Positive" — the original SCINE Table 5 term collided
# with the geometric concept of "negative_space" and with "mood" being
# "positive" in sentiment. "DensityLevel" is unambiguous.)
DensityLevel = Literal["clean", "cluttered"]
# Atmospheric / weather elements that change the rendered look.
Element        = Literal["rain", "snow", "fog", "wind", "thunder", "smoke", "dust", "ash", "fire"]
# Prop physical material.
PropMaterial   = Literal["wood", "glass", "gold", "paper", "plastic"]
# Prop surface pattern.
PropPattern    = Literal["grid", "checker", "stripes", "zigzag", "dots", "bricks", "metal", "hexagons"]
# Whether the prop is set dressing or actively used by characters.
PropUtility    = Literal["decorative", "functional"]


@dataclass
class SceneTexture:
    """Setup → Scene → Texture (Table 5 first block).
    All optional — fill only when the look is intentional, not implicit."""
    contrast:      Optional[Contrast]  = None       # low (muted) / high (stark)
    blur:          Optional[BlurKind]  = None       # if a soft/motion/radial blur is part of the look
    noise:         Optional[NoiseKind] = None       # film-grain or noise pattern
    film_grain:    Optional[str]       = None       # open set: stock id, e.g. "kodak_vision3_500t"
    color_palette: Optional[str]       = None       # open set: "warm", "teal_orange", "monochrome_grey", …


@dataclass
class SceneGeometry:
    """Setup → Scene → Geometry. Spatial Location is split into two
    open-set sub-fields per the paper (Positional Accuracy + Relative
    Positioning). Free-text — use `subject.screen_position` for the
    machine-readable per-subject placement instead."""
    lines:                Optional[LineDirection] = None     # dominant line direction in composition
    regular_shapes:       Optional[RegularShape]  = None     # if a regular shape dominates (square/circle/triangle)
    natural_shapes:       Optional[NaturalShape]  = None     # organic shape language
    frame_balance:        Optional[FrameBalance]  = None     # mass distribution: thirds / symmetry / right_heavy / left_heavy
    positional_accuracy:  Optional[str]           = None     # free text: "subject in left third, eyes on upper third line"
    relative_positioning: Optional[str]           = None     # free text: "A in front, B behind" — prefer Subject.screen_position when possible


@dataclass
class SceneSpace:
    """Setup → Scene → Space. SCINE Table 5 lists only Depth as a
    discrete leaf; Figure 2a additionally shows class/material/pattern/
    utility under Space, but the table assigns those to Props. We follow
    the table (Props own material/pattern/utility) and keep Space minimal."""
    depth: Optional[Depth] = None                    # spatial depth feel — deep / flat / limited / ambiguous


@dataclass
class Backdrop:
    """Setup → Set Design → Backdrop. Macro context of the set."""
    setting:     Optional[Setting]   = None          # int (interior) / ext (exterior)
    time_of_day: Optional[TimeOfDay] = None          # 11-value enum — see TimeOfDay
    location:    Optional[str]       = None          # open set: "changan_5th_c_palace_bedroom", "desert_caves" — used to look up location bibles
    # PAI extension — period + region + culture. Lets the renderer pick
    # era-appropriate costume / architecture / props instead of defaulting
    # to "generic Asian historical". Open-set strings: pick the
    # description that most faithfully roots the shot in its world.
    era:         Optional[str] = None                # "350 CE" / "5th c. CE" / "modern_day" / "2099" — open set, human-readable
    region:      Optional[str] = None                # "silk_road_oasis" / "modern_shanghai" — sympathetic to location_ref
    culture:     Optional[str] = None                # "kuchean_buddhist" / "tang_dynasty" / "byzantine" — a coherent cultural-style frame to pull on
    # pace-0.2 setup.backdrop.{weather,season} — open-set strings (no enum upstream)
    weather:     Optional[str] = None                # open: "clear", "rain", "snow", "overcast", "sandstorm", …
    season:      Optional[str] = None                # open: "spring", "summer", "autumn", "winter", …


@dataclass
class Environment:
    """Setup → Set Design → Environment. The "feel" of the place;
    Organization sub-fields are all open sets (paper Table 5)."""
    negative_space: bool                 = False     # True = composition exploits emptiness (open sky / blank wall)
    density:        Optional[DensityLevel] = None    # clean vs cluttered (renamed from "positive" for clarity)
    # Not optional in effect: every image backend compiles the mood into the
    # prompt, so a shot without one does not render neutrally, it renders
    # however the sampler chooses. A screenplay never states it, so it is
    # inferred from what happens in the scene rather than quoted from it, as
    # era, region and culture already are.
    mood:           Optional[str]        = None      # open: "serene", "ominous", "intimate", "frantic", …
    scale:          Optional[str]        = None      # open: "intimate", "monumental", "claustrophobic", …
    style:          Optional[str]        = None      # open: rendering style anchor — "sketch_bw", "inkwash_bw", "chiaroscuro", "photoreal", …
    background:     Optional[str]        = None      # open: backdrop description, e.g. "endless desert dunes at sunset"
    elements:       list[Element]        = field(default_factory=list)  # weather/atmosphere: rain / fog / dust / smoke / …


# Prop physical state — visible condition that changes how it renders.
PropState = Literal[
    "pristine", "weathered", "broken", "burning", "wet", "frozen",
    "bloodied", "dusty", "rusted", "polished",
]
# Relative size — how it scales next to a human subject in the frame.
#   palm_sized   = held in one hand (cup, scroll, coin)
#   wearable     = donned by a character (helmet, robe sash)
#   human_scale  = single-person size (chair, weapon, mirror)
#   two_person   = bench / table that seats two
#   monumental   = larger than human (statue, gate, throne)
#   miniature    = small enough to be lost — figurine, marble
PropSize = Literal[
    "palm_sized", "wearable", "human_scale", "two_person",
    "monumental", "miniature",
]


# Whether an entity is in the picture, the way a storyboard blocking sheet
# lists who and what enters the frame (入画) apart from where things stand.
# "partial" is something the frame cuts -- a shoulder in the foreground, the
# end of a console; "tbd" is the sheet's 待确认, the source does not say,
# which is not the same as "yes".
InFrame = Literal["yes", "partial", "no", "tbd"]


@dataclass
class Prop:
    """Setup → Set Design → Props. One prop per dataclass instance;
    props[] holds all of them on the shot.

    For props that recur across shots, set a stable `prop_id` so downstream
    consumers can keep their appearance + state consistent."""
    description: Optional[str]          = None       # human-readable: "lacquered tea bowl with chip on rim"
    cls:         Optional[str]          = None       # open category: "weapon", "scroll", "tea_bowl", "candle"
    material:    Optional[PropMaterial] = None       # wood/glass/gold/paper/plastic
    pattern:     Optional[PropPattern]  = None       # surface pattern if any
    utility:     Optional[PropUtility]  = None       # decorative (set dressing) vs functional (used by characters)
    # ── PAI extensions (PAI 1.1) ─────────────────────────────────────────
    prop_id:     Optional[str]             = None    # stable id for cross-shot continuity ("andúril_the_sword")
    state:       Optional[PropState]       = None    # current physical state — pristine / weathered / broken / burning / …
    color:       Optional[str]             = None    # open set: "deep_crimson", "soot_black", "celadon_green"
    size:        Optional[PropSize]        = None    # relative scale — palm_sized / human_scale / monumental / …
    held_by:     Optional[str]             = None    # character ref (id@age form, see pai_compat.id_age_to_ref)
    rests_on:    Optional[str]             = None    # character_id the prop lies on ("the car falls on top of him")
    screen_position: Optional["ScreenPosition"] = None  # reuse Subject's frame-placement spec
    count:       int                       = 1       # multiplicity for batches of same prop (a stack of wooden tablets)
    in_frame:    Optional[InFrame]         = None    # in the picture this panel, not just in the scene (see InFrame)
    in_frame_extent: Optional[str]         = None    # for "partial": what the frame shows, "its right end at the bottom edge"


# Named Literals so the enrichment LLM's VOCAB extraction can read them
# via `typing.get_args(GazeTargetType)` etc. without parsing dataclass-
# inline annotations.

# What the subject is looking at:
#   "character" = at another subject (set target_ref = their character_id)
#   "object"    = at a prop (set target_ref = prop name)
#   "feature"   = at a body feature of someone (set target_ref + of_character)
#   "camera"    = direct address (gaze meets the lens — fourth-wall break)
GazeTargetType = Literal["character", "object", "feature", "camera"]
# Off-frame / camera-aware directions (use INSTEAD of target_* when the
# look points outside the frame):
#   off_left / off_right / off_up / off_down = beyond frame edges
#   up / down / left / right                = within frame, no specific target
#   into_camera = at the lens (same as target_type="camera")
#   averted     = deliberately looking away (downcast, evasive)
GazeDirection  = Literal[
    "off_left", "off_right", "off_up", "off_down",
    "up", "down", "left", "right",
    "into_camera", "averted",
]
# Rule-of-thirds zone where the subject sits in the frame. Coarse but
# human-friendly — use (x, y) coords for precise pose-rig output.
ScreenZone = Literal[
    "left", "center", "right",
    "upper", "lower",
    "upper_left", "upper_center", "upper_right",
    "lower_left", "lower_center", "lower_right",
    "center_left", "center_right",
]
# Spatial layer the subject occupies relative to camera.
ScreenDepth = Literal["foreground", "midground", "background"]


@dataclass
class Gaze:
    """PAI extension — where this subject is looking. Captures eyeline so
    the storyboard records "Alice looking at the console" or
    "Bob staring off-frame right" as structured data instead of folding
    it into the free-form `pose` string. Used by the prompt compiler to render
    "looking at X" / "gaze directed off-screen left", and by future
    eyeline-match continuity checks across shots.

    Either `target` (looking at something IN frame) or `direction`
    (looking off-frame in a direction) should be set — not both. If both
    are None the subject's gaze is unspecified."""
    target_type: Optional[GazeTargetType] = None     # character / object / feature / camera — what kind of target
    target_ref:  Optional[str]            = None     # the target's id: character_id, prop name, or feature like "console_edge"
    direction:   Optional[GazeDirection]  = None     # off-frame direction or into-camera — use INSTEAD of target_*
    note:        Optional[str]            = None     # free-form intent: "contemplative", "challenge", "longing", …


@dataclass
class ScreenPosition:
    """PAI extension — where this subject sits in the frame. Spec'd here so
    multi-subject compositions can be machine-read (and later fed to
    ControlNet pose / spatial prompting) instead of relying on
    SceneGeometry.relative_positioning's free-text string.

    Two coordinate systems supported — pick whichever you actually know:
      - `zone`: rule-of-thirds zone keyword (coarse but human-friendly)
      - `(x, y)`: normalized frame coords, 0..1 (precise; for pose rigs)
    `depth` is orthogonal — applies to either."""
    zone:  Optional[ScreenZone]  = None              # rule-of-thirds zone (coarse, human-readable)
    x:     Optional[float]       = None              # 0.0 = left edge of frame, 1.0 = right edge (precise)
    y:     Optional[float]       = None              # 0.0 = top edge of frame, 1.0 = bottom edge (precise)
    depth: Optional[ScreenDepth] = None              # foreground / midground / background


@dataclass
class Subject:
    """Setup → Subjects. The focal characters / creatures in the shot.
    All fields are open sets — captures the visual specifics that a
    Subject LoRA / PuLID reference would later identity-lock."""
    cls:         Optional[str] = None                # subject category, open: "young_woman", "elderly_man", "tabby_cat"
    accessories: Optional[str] = None                # what they're wearing/holding: "prayer beads, walking staff"
    costume:     Optional[str] = None                # garments: "saffron robe", "dust-stained tunic"
    # PAI extension — the wardrobe entry this costume IS, so a garment is a
    # library object with an id and states rather than a sentence retyped per
    # shot. Free text cannot be compared across a cut: one panel said "a
    # fitted grey-blue jacket" and the next delivered a white tunic, and nothing could report it, because there was no identifier for
    # the two to disagree about. Resolves in kb/props.json like any prop_id.
    costume_id:  Optional[str] = None                # wardrobe entry id, e.g. "alice_costume"
    hair:        Optional[str] = None                # "shaved head", "long black braid", "wild grey beard"
    makeup:      Optional[str] = None                # "kohl-rimmed eyes", "war paint", "soot smudges"
    pose:        Optional[str] = None                # body posture: "kneeling, hands clasped", "leaning against doorframe"
    silhouette:  Optional[str] = None                # contour adjective: "imposing", "frail", "rigid"
    proportions: Optional[str] = None                # body type: "wiry", "broad-shouldered", "diminutive"
    # PAI extension — link back to the canonical character registry so
    # Subject appearance can default-inherit from kb/characters.json.
    # Optional; setting just the open-set fields above also works.
    character_id: Optional[str] = None               # id matching the project's character registry, e.g. "alice"
    age_state:    Optional[str] = None               # which life-stage variant of that character, e.g. "adult_50", "adult_18"
    # PAI extension — eyeline + frame placement. Both Optional so existing
    # pai-1.0 files (which lack these) parse unchanged.
    gaze:            Optional[Gaze]           = None    # where this subject is looking
    screen_position: Optional[ScreenPosition] = None    # where this subject sits in the frame
    in_frame:        Optional[InFrame]        = None    # in the picture this panel (see InFrame)
    in_frame_extent: Optional[str]            = None    # for "partial": "right shoulder in the foreground"


# ── Text Generation (PAI 1.1 expansion) ─────────────────────────────────
# Was a single Optional[str] in PAI 1.0; expanded so each on-screen text
# element can specify its physical carrier, language, typography, and
# layout independently. A shot can have multiple TextElements (subtitle
# + sign + handwritten letter in same frame).

TextTarget = Literal[
    "phone_screen", "billboard", "letter", "subtitle", "title_card",
    "sign", "scroll", "book_page", "tattoo", "newspaper",
    "carved_stone", "wall_graffiti", "banner",
]
TextStyle  = Literal[
    "handwritten", "printed", "calligraphy", "neon", "carved",
    "stamped", "embroidered",
]
TextLayout = Literal[
    "single_bubble", "header_only", "input_box", "body_paragraph",
    "list", "headline_only", "two_column",
]


@dataclass
class TextElement:
    """One piece of on-screen text the renderer must draw inside the frame."""
    target:   Optional[TextTarget] = None    # surface that carries the text (phone screen / billboard / letter / …)
    content:  Optional[str]        = None    # the literal text content (the words on the screen)
    language: Optional[str]        = None    # ISO-ish: "zh" / "en" / "sa" (Sanskrit) / "kha" (Kharoṣṭhī) — open set
    style:    Optional[TextStyle]  = None    # typography style — handwritten / printed / calligraphy / …
    layout:   Optional[TextLayout] = None    # text layout pattern — single_bubble / headline_only / two_column / …
    count:    int                  = 1       # number of identical text instances (a row of identical posters)
    note:     Optional[str]        = None    # free-form: "partially faded", "obscured by reflection"


@dataclass
class PrimaryFocus:
    """What dominates the frame. Kept verbatim from v0.3 — sibling of
    Subjects, used by the prompt compiler to lock framing on a single subject."""
    type:           Literal["character", "object", "environment", "feature"] = "character"
    # ↑ what kind of thing is the focus: a character, a prop, the location, or a body feature (like an eye)
    ref:            str = ""                          # id of the focus: character_id, prop name, location id, or feature name
    of_character:   Optional[str] = None              # when type="feature", which character does the feature belong to (character_id)
    coverage_pct:   Optional[int] = None              # how much of the frame area the focus should occupy (0-100)


@dataclass
class Setup:
    """SCINE Pillar 2 — every visible element within the frame."""
    texture:       SceneTexture     = field(default_factory=SceneTexture)     # contrast/blur/noise/film_grain/palette
    geometry:      SceneGeometry    = field(default_factory=SceneGeometry)    # composition geometry (lines, balance)
    space:         SceneSpace       = field(default_factory=SceneSpace)       # spatial depth feel
    backdrop:      Backdrop         = field(default_factory=Backdrop)         # int/ext + time + location id
    environment:   Environment      = field(default_factory=Environment)      # mood/style/atmosphere/elements
    props:         list[Prop]       = field(default_factory=list)             # one entry per visible prop
    subjects:      list[Subject]    = field(default_factory=list)             # focal characters/creatures in this shot
    primary_focus: PrimaryFocus     = field(default_factory=PrimaryFocus)     # what dominates the frame
    secondary_subjects: list[str]   = field(default_factory=list)   # character refs (id@age form), present but not the focus — see pai_compat.ref_to_id_age
    excluded:      list[str]        = field(default_factory=list)             # things NOT to render — fed into the negative prompt
    # On-screen text — expanded from PAI 1.0's single `Optional[str]` into a
    # list of structured elements so each text item (subtitle, sign,
    # handwritten letter, billboard) gets its own target / language /
    # typography. the prompt compiler renders each entry as its own clause.
    text_generation: list["TextElement"] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# Pillar 3 — Lighting (Table 4)
# ════════════════════════════════════════════════════════════════════════

# Sun / moon / fire — the natural sources present.
NaturalLight   = Literal["sunlight", "moonlight", "firelight"]
# Practical light fixtures (lamps visible in shot or shaped to look like
# they're in shot).
PracticalsKind = Literal["led", "hmi", "tungsten", "fluorescent", "hid"]
# Color-temperature bucket (Kelvin band):
#   warm = ~2700-3500K (tungsten, candle, sunset)
#   cool = ~5000-6500K (daylight, fluorescent)
#   cold = ~7000K+ (overcast, deep blue hour)
ColorTempBand  = Literal["warm", "cool", "cold"]
# Lighting genre keyword — 5-value vocab is intentionally narrow.
#   candlelight        = warm intimate INT-night
#   golden_hour        = warm exterior dusk/dawn
#   clear_daylight     = bright exterior day
#   overcast           = diffuse exterior day
#   white_fluorescent  = harsh interior (office / hospital / convenience store)
LightingCondition = Literal[
    "candlelight", "golden_hour", "white_fluorescent",
    "clear_daylight", "overcast",
    # The schema had no way to say the lights are OUT, so a beat asserting
    # "all power in the car goes out" had nowhere to land and the shot kept
    # its `natural: [sunlight]` -- which the compiler then wrote as "lit by
    # sunlight" into a prompt whose own text says the cabin is pitch black.
    "blackout",
]
# How soft shadows are achieved (Table 4).
SoftShadow     = Literal["diffused_light", "high_key_lighting", "reflectors"]
# How hard shadows are achieved.
HardShadow     = Literal["direct_light", "low_key_lighting"]
# Standard 3-point lighting positions (plus key).
#   key_light  = primary light on subject
#   fill_light = softens shadows opposite the key
#   back_light = separates subject from background (rim)
#   side_light = strong directional light from one side
#   top_light  = light from above
LightingPosition = Literal["back_light", "fill_light", "top_light", "side_light", "key_light"]
# Time-varying light behavior.
LightingMotion = Literal["flickering", "pulsing"]


@dataclass
class Lighting:
    """SCINE Pillar 3 — illumination of the shot."""
    natural:           list[NaturalLight]      = field(default_factory=list)    # sun/moon/firelight present in the scene
    practicals:        list[PracticalsKind]    = field(default_factory=list)    # man-made lamps visible (or simulated as visible)
    color_temperature: Optional[ColorTempBand] = None         # overall warm/cool/cold band
    condition:         Optional[LightingCondition] = None     # genre keyword (candlelight / golden_hour / overcast / …)
    soft_shadows:      Optional[SoftShadow]    = None         # how soft shadows arise (diffused / high_key / reflectors)
    hard_shadows:      Optional[HardShadow]    = None         # how hard shadows arise (direct / low_key)
    reflection:        Optional[str]           = None         # free-form: "pool of water reflects sky", "mirror in BG"
    position:          Optional[LightingPosition] = None      # primary 3-point position of the key light
    motion:            Optional[LightingMotion]   = None      # flickering (firelight) / pulsing (sirens, screens)
    color_gels:        Optional[str]           = None         # free-form: "warm amber on key, cyan on fill"
    # PAI extension — exact Kelvin if known from real capture.
    color_temp_k:      Optional[int] = None                    # exact Kelvin, e.g. 3200
    notes:             Optional[str] = None                    # free-form: anything else about the light


# ════════════════════════════════════════════════════════════════════════
# Pillar 4 — Events (Table 6)
# ════════════════════════════════════════════════════════════════════════

# How an action unfolds in time.
#   atomic       = single physical action (head turn, lip quiver) — STRONGEST for models
#   concurrent   = two actions happening simultaneously
#   sequential   = chain of actions in one shot ("open door then walk in")
#   causal       = action B happens because of action A
#   overlapping  = action B starts before A ends
#   cyclic       = action repeats (pacing, breathing)
#   reverse      = unusual: action played backward
ActionTemporal = Literal[
    "atomic", "concurrent", "sequential", "causal",
    "overlapping", "cyclic", "reverse",
]
# How prominent this action is in the frame.
#   focal  = it IS the primary_focus's behavior
#   local  = peripheral, happens to a non-focal subject
#   global = environmental (a crowd moving, wind blowing leaves)
ActionForeground = Literal["local", "global", "focal"]
# Whether the action's outcome is certain.
#   probabilistic = uncertain (a coin toss, a hesitating gesture)
#   deterministic = inevitable (gravity, a falling object)
#   mixed         = both
ActionUncertainty = Literal["probabilistic", "deterministic", "mixed"]

# How an emotion unfolds in time (parallel to ActionTemporal).
EmotionTemporal = Literal["atomic", "concurrent", "sequential", "overlapping", "causal"]
# How prominent this emotion is (parallel to ActionForeground).
EmotionForeground = Literal["local", "global", "focal"]

# How a dialogue line is delivered.
#   dash       = interrupted / cut off
#   ellipsis   = trailing off
#   monologue  = uninterrupted self-narration
DialogueDelivery = Literal["dash", "ellipsis", "monologue"]
DialogueForeground = Literal["local", "global", "focal"]

# Narrative function this shot serves in the scene/film.
StoryStructure = Literal["turning_point", "climax", "foreshadowing", "conflict"]
# Pace of the shot (slow takes / quick cuts).
Pace = Literal["slow", "fast"]
# Whether the rhythm is steady or broken.
Regularity = Literal["regular", "irregular"]


@dataclass
class Action:
    """One action beat. SCINE Table 6 splits actions into Standalone vs
    Interactive (orthogonal open sets); typical usage is to fill one or
    the other for any given Action instance."""
    standalone:   Optional[str]                  = None    # solo physical action (open set): "lips quivering", "raising a cup"
    interactive:  Optional[str]                  = None    # action ON another entity (open set): "handing scroll to disciple"
    temporal:     Optional[ActionTemporal]       = None    # atomic / sequential / causal / cyclic / …
    foreground:   Optional[ActionForeground]     = None    # focal (is the primary subject's act) / local / global
    background:   bool                           = False   # True = this action happens behind the focal subject
    uncertainty:  Optional[ActionUncertainty]    = None    # probabilistic / deterministic / mixed
    # PAI extension — kept from v0.3 for back-compat. Lets the human-
    # authored beat travel alongside the SCINE classification.
    description_zh:  str = ""                              # human-authored Chinese description, e.g. "zuichun xidong" (lips trembling)
    description_en:  str = ""                              # human-authored English description, e.g. "lips quivering"
    beat_features:   list[str] = field(default_factory=list)  # visible texture cues: ["cracked_dry_chapped_lips", "thin_elderly_lips"]
    intensity:       Optional[Literal["subtle", "medium", "dramatic"]] = None  # how strong the beat is on screen
    duration_hint_s: Optional[float] = None                # rough seconds — informs Wan-I2V length, not literal


@dataclass
class Emotion:
    """One emotion beat. Implicit/Explicit are orthogonal — implicit
    captures body-language cues, explicit names the emotion directly."""
    implicit:    Optional[str]               = None    # body language cue (open set): "clenched jaw", "downcast eyes" — preferred
    explicit:    Optional[str]               = None    # direct emotion name (open set): "grief", "joy" — rare; most cinema is implicit
    temporal:    Optional[EmotionTemporal]   = None    # atomic / sequential / overlapping / …
    foreground:  Optional[EmotionForeground] = None    # focal / local / global
    background:  bool                        = False   # True = emotion is on a non-focal subject in the background


@dataclass
class Dialogue:
    """One line of dialogue. Audio is generated separately (ADR);
    this models the FILM-LEVEL classification only."""
    type_of_delivery: Optional[DialogueDelivery]   = None    # dash (interrupted) / ellipsis (trailing) / monologue
    foreground:       Optional[DialogueForeground] = None    # focal / local / global
    # PAI extension — speaker + text so the line can also feed into
    # vo_lines/on_screen_dialogue at the scene level.
    speaker: Optional[str] = None                            # character_id of who's speaking
    text:    Optional[str] = None                            # the line itself (Chinese or English)
    # pace-0.2 events.dialogues[].language — BCP-47-ish open string ("zh","en","ja")
    language: Optional[str] = None                           # spoken language of this line


@dataclass
class EventsAdvanced:
    """SCINE Events → Advanced Controls (Table 6 bottom)."""
    story_structure: Optional[StoryStructure] = None    # narrative function: turning_point / climax / foreshadowing / conflict
    pace:            Optional[Pace]           = None    # slow / fast
    regularity:      Optional[Regularity]     = None    # regular (steady rhythm) / irregular (broken pacing)


@dataclass
class Events:
    """SCINE Pillar 4 — narrative substance of the shot."""
    actions:               list[Action]    = field(default_factory=list)    # one entry per action beat in the shot
    emotions:              list[Emotion]   = field(default_factory=list)    # one entry per emotion beat
    dialogues:             list[Dialogue]  = field(default_factory=list)    # one entry per spoken line
    change_in_environment: Optional[str]   = None    # free-form: "wind rises", "candle gutters out" — a non-character event
    advanced:              EventsAdvanced  = field(default_factory=EventsAdvanced)    # story_structure / pace / regularity


# ════════════════════════════════════════════════════════════════════════
# Shot + Panel + SceneDoc — top-level containers
# ════════════════════════════════════════════════════════════════════════


@dataclass
class PromptOverride:
    """Per-panel hand-written prompt override. Same shape as v0.3.
    When set, the prompt compiler returns this verbatim and skips the structured
    composition — used to inject ad-hoc fixes without re-deriving fields."""
    positive:     str           = ""             # full positive prompt to send to the backend (non-empty wins)
    negative:     str           = ""             # full negative prompt
    authored_at:  Optional[str] = None           # ISO-8601 timestamp when the override was written
    authored_by:  Optional[str] = None           # who wrote it (user handle / "llm" / "auto")
    note:         Optional[str] = None           # why this override exists (the bug it fixes)


@dataclass
class CompileHintsFlux:
    """Flux-specific compile hints (per-panel sidecar)."""
    prompt_override: Optional[PromptOverride] = None    # hand-written prompt that bypasses the prompt compiler's structured assembly


@dataclass
class CompileHints:
    """Per-panel compile_hints sidecar — one entry per backend.
    Kept compatible with v0.3's `compile_hints[]` shape so existing
    sidecar data round-trips."""
    panel_id:     str = ""                                                    # which panel this entry applies to
    flux:         CompileHintsFlux = field(default_factory=CompileHintsFlux)  # Flux-specific overrides
    gpt_image_2:  dict[str, Any]   = field(default_factory=dict)              # GPT-Image-2 overrides (free-form dict)
    wan_i2v:      dict[str, Any]   = field(default_factory=dict)              # Wan-I2V video-stage overrides (free-form dict)


@dataclass
class Panel:
    """One storyboard frame inside a Shot. Inherits the parent shot's
    camera/setup/lighting/events by default; override fields below (all
    optional dicts) hold panel-local diffs only.

    Panels exist when a single shot needs multiple distinct frames in the
    storyboard (e.g. shot_01 starts wide, then a beat later cuts to a
    close-up of the same subject within the same camera setup)."""
    id:              str = ""                                # canonical id, e.g. "scene_01_shot_01_panel_0001"
    panel_number:    int = 1                                 # 1-based ordinal within the shot
    scene_id:        Optional[str] = None                    # cached scene_id for lookup-by-panel-id
    # Optional panel-level overrides — dicts so they can be sparse without
    # carrying every field's default. Each shape mirrors its pillar
    # dataclass (Camera, Setup, Lighting, Events).
    camera_override:   Optional[dict[str, Any]] = None       # sparse Camera-shaped diff vs the parent shot's camera
    setup_override:    Optional[dict[str, Any]] = None       # sparse Setup-shaped diff (e.g. swap `excluded`, change `primary_focus`)
    lighting_override: Optional[dict[str, Any]] = None       # sparse Lighting-shaped diff
    events_override:   Optional[dict[str, Any]] = None       # sparse Events-shaped diff (e.g. different action for this beat)
    # Frequently-overridden flat fields (kept hot for the storyboard sidebar).
    primary_focus: Optional[PrimaryFocus] = None             # quick-access override for what dominates this frame
    notes:         Optional[str] = None                       # free-form: director notes specific to this panel


@dataclass
class Shot:
    """One cinematic shot — the atomic unit per SCINE §3.1. Contains
    all 4 SCINE pillars as first-class sub-trees, plus its panels.

    A shot = one continuous camera run. Multi-camera coverage of the
    same beat = multiple shots, each with its own camera. Panels within
    a shot share the same camera spec (use multiple panels only for
    different time-moments inside that camera setup)."""
    shot_id:  str = ""                                       # canonical id, e.g. "shot_01" within scene_01
    camera:   Camera   = field(default_factory=Camera)       # Pillar 1
    setup:    Setup    = field(default_factory=Setup)        # Pillar 2
    lighting: Lighting = field(default_factory=Lighting)     # Pillar 3
    events:   Events   = field(default_factory=Events)       # Pillar 4
    panels:   list[Panel] = field(default_factory=list)      # storyboard frames realising this shot


@dataclass
class NarrativeMeta:
    """Top-of-scene narrative ledger — character roster, action beats,
    dialogue list. Carried over from v0.3 essentially unchanged; SCINE
    leaves narrative summary outside the 4 pillars."""
    summary:             str            = ""                  # 1-2 sentence prose summary of what happens in this scene
    characters_present:  list[str]      = field(default_factory=list)         # character_ids appearing anywhere in the scene
    character_age_states: dict[str, str] = field(default_factory=dict)        # character_id → age_state used in THIS scene (e.g. "alice": "adult_50")
    key_actions:         list[str]      = field(default_factory=list)         # short prose list of main physical beats
    vo_lines:            list[dict[str, Any]] = field(default_factory=list)   # voice-over lines (off-screen narration) — [{speaker, text, …}]
    on_screen_dialogue:  list[dict[str, Any]] = field(default_factory=list)   # spoken on-screen dialogue — [{speaker, text, …}]
    titles:              list[str]      = field(default_factory=list)         # any on-screen title-cards or chapter headings rendered for this scene
    sfx_notes:           list[str]      = field(default_factory=list)         # sound design notes (used downstream by audio mix)


# ════════════════════════════════════════════════════════════════════════
# Physical layout — scene-wide world coordinates (PAI 1.1 prototype)
# ════════════════════════════════════════════════════════════════════════
#
# Optional layer for scenes where multiple shots share a stage and the
# user wants per-shot screen_position to be derived from camera + world
# coordinates instead of LLM-guessed every time. Also the natural input
# to Blender greybox (`build_scenes.py`).
#
# Use only when the scene has 3+ shots that share blocking. For single-
# shot or one-off scenes, just leave subjects[].screen_position authored
# manually — this layer is extra mileage and not required.


@dataclass
class WorldEntity:
    """A character or prop placed in stage-frame world coordinates.
    `ref` is a character ref ("alice@adult_50") or prop_id ("main_console")."""
    ref:         str                       # character id@age OR prop_id
    world_xy:    list[float] = field(default_factory=list)  # [x, y] in meters; stage top-down
    z:           float = 0.0               # height: 0=on ground, -0.5=sitting, +1=elevated
    facing_deg:  float = 0.0               # 0=east, 90=north, 180=west, 270=south
    scale:       float = 1.0               # 1.0 = real-life size; <1 = miniaturized; >1 = giant


@dataclass
class CameraSetup:
    """One camera position in stage coordinates — what a shot uses."""
    shot_id:        str
    world_xy:       list[float] = field(default_factory=list)  # camera position
    z:              float       = 1.65                          # camera height (eye-level default)
    looking_at_xy:  list[float] = field(default_factory=list)   # what the camera is aimed at (stage point)
    lens_mm:        Optional[float] = None                      # focal length, optional override


@dataclass
class PhysicalLayout:
    """Scene-wide world coordinates. The prompt compiler and a future Blender
    bridge can derive screen positions, depth ordering, and even greybox
    geometry from this single source of truth."""
    frame_of_reference: Literal["stage_top_view", "world_xy"] = "stage_top_view"
    subjects:       list[WorldEntity] = field(default_factory=list)   # entries refer to characters by id@age
    props:          list[WorldEntity] = field(default_factory=list)   # entries refer to props by prop_id
    camera_setups:  list[CameraSetup] = field(default_factory=list)   # one per shot_id that uses world projection


@dataclass
class ShotDefaults:
    """Per-scene defaults that every shot in the scene INHERITS by default.

    Use this for fields that naturally span an entire scene — aspect_ratio,
    backdrop.location, backdrop.{era, region, culture}, environment.style,
    lighting.condition, texture.color_palette. Each Shot can override any
    of these; resolved value is deep_merge(defaults, shot_value) where
    None values in defaults never overwrite filled values in the shot.

    Avoid putting per-shot variables here (shot_size, action, subjects,
    framing). Those are shot-specific by definition.

    Shape mirrors the 3 visual pillars but every nested field is Optional,
    so the dict can be as sparse as you like.

    See `pai_compat.resolve_shot(scene, shot)` for the merge implementation."""
    camera:   Optional[Camera]   = None
    setup:    Optional[Setup]    = None
    lighting: Optional[Lighting] = None


# ─────────────────── PACE pace-0.2 alignment ────────────────────────────
# Ported from pai_platform/docs/pace (registry/fields.json, schemaVersion
# "pace-0.2"), which is the upstream standard this schema now tracks. Of
# PACE's 160 registry fields, 113 already existed here under snake_case
# names; the types below add the active ones that did not, namely the
# `semantics.*` and `manifest.*` pillars and the project/scene artifact
# ledgers. PACE's 31 dormant fields (physicalLayout extras, per-light
# fixtures, lens distortion/encoders, override and render-output blocks)
# are deliberately not modelled until the standard activates them.


@dataclass
class Semantics:
    """PACE `semantics.*` — the OMC-aligned meaning layer.

    Separate from the four visual pillars because it describes what a shot
    is *for* rather than what it looks like: who participates, where the
    beat sits in the story, and which production task produced it.
    """
    task:               Optional[str] = None        # semantics.task — the pipeline task that authored this
    participant:        Optional[list] = None       # semantics.participant — OMC Participant refs
    relationships:      Optional[list] = None       # semantics.relationships — inter-participant relations
    narrative_context:  Optional[dict] = None       # semantics.narrativeContext — beat/act placement
    production_context: Optional[dict] = None       # semantics.productionContext — unit, day, department
    version:            Optional[str] = None        # semantics.version — OMC schema version in force


@dataclass
class FieldMeta:
    """PACE `manifest.fieldMeta[]` — provenance for one field.

    An array rather than a path-keyed map: pace-0.2 removed dynamic key
    sets throughout, so the field path travels inside the record.
    """
    path:      str = ""                             # JSON pointer into the doc, e.g. "/pace/camera/fps"
    source:    Optional[str] = None                 # who set it: "llm" / "human" / "derived"
    confidence: Optional[float] = None
    note:      Optional[str] = None


@dataclass
class PromptRecord:
    """PACE `manifest.prompts[]` — one compiled prompt, addressed by id.

    Also an array in pace-0.2, where earlier drafts keyed by compiler name.
    """
    id:   str = ""                                  # e.g. "v1T2i"
    text: str = ""
    backend: Optional[str] = None


@dataclass
class Artifact:
    """PACE artifact ledger entry — `manifest.artifacts`, and the
    project/scene-level `artifacts` arrays.

    pace-0.2 moved screenplay text and every other binary out of the
    standard project tree and into object storage, referenced by URI, so
    an artifact record is how a document points at its own inputs.
    """
    kind:   str = ""                                # e.g. "source_script_file", "source_script_text"
    uri:    Optional[str] = None                    # "assets://..." object-store reference
    source: Optional[str] = None                    # "uploaded" / "generated"

    # Provenance. `uri` says where an artifact is and `source` says what kind
    # of thing made it; neither says WHICH VERSION of the input produced it,
    # which is the only question a staleness check asks. Without these two a
    # panel can be re-cut from a medium to a close-up and the render made
    # against the old framing stays on disk with nothing marking it expired.
    content_hash: Optional[str] = None              # manifest.artifacts[].contentHash
    derived_from: list[str] = field(default_factory=list)  # upstream content hashes
    note:   Optional[str] = None


@dataclass
class Review:
    """PACE `manifest.review` — an approval, bound to the content it approves.

    `approved_hash` is the load-bearing field. An approval recorded against
    a document *identifier* stays true after the document changes, which
    yields an approved artifact nobody has reviewed in its current form —
    worse than no gate at all, because a gate that cannot go stale stops
    anyone from checking by hand. Binding the approval to the hash makes it
    expire exactly when the content moves.
    """
    status:        str = "unreviewed"        # unreviewed | approved | rejected
    by:            Optional[str] = None
    at:            Optional[str] = None      # ISO 8601
    approved_hash: Optional[str] = None      # content hash this approval covers
    note:          Optional[str] = None


@dataclass
class Manifest:
    """PACE `manifest.*` — the per-document bookkeeping block."""
    artifacts:  list[Artifact] = field(default_factory=list)      # manifest.artifacts
    prompts:    list[PromptRecord] = field(default_factory=list)  # manifest.prompts[]
    field_meta: list[FieldMeta] = field(default_factory=list)     # manifest.fieldMeta[]
    audio:      Optional[dict] = None                             # manifest.audio
    continuity: Optional[dict] = None                             # manifest.continuity
    review:     Optional["Review"] = None                         # manifest.review


# Bumped 1.1 -> 1.2 on adopting PACE pace-0.2 as the upstream standard
# (semantics/manifest pillars, artifact ledgers). 1.0 and 1.1 still read:
# the added blocks are optional, so an older document deserializes with
# them empty rather than failing.
SCHEMA_VERSION: str = "pai-1.2"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("pai-1.0", "pai-1.1", "pai-1.2")
# The upstream standard this schema tracks, recorded so a reader can tell
# which PACE revision the field set corresponds to.
PACE_SCHEMA_VERSION: str = "pace-0.2"


@dataclass
class SceneDoc:
    """One scene — wrapper that holds the SCINE-aligned shot list plus
    narrative meta. Compatible with v0.3's top-level shape so the
    storyboard/api endpoints can deserialize after a small adapter."""
    _schema_version:  str = SCHEMA_VERSION                    # schema version tag — readers branch on this. Bumped 2026-06-05 1.0 → 1.1 to match SCHEMA.md after the SCINE-pillar extensions. 50/51 on-disk scene files already declare "pai-1.1"; enrich_scenes_scine.py:665 accepts both 1.0 and 1.1 for back-compat.
    scene_id:         str = ""                                # canonical id, e.g. "scene_05"
    scene_number:     Optional[int] = None                    # 1-based ordinal in the screenplay
    scene_heading:    Optional[str] = None                    # screenplay slugline: "EXT. DESERT — DAWN"
    act:              Optional[str] = None                    # which act of the film: "I" / "II" / "III" or "prologue" / "epilogue"
    narrative_meta:   NarrativeMeta = field(default_factory=NarrativeMeta)    # roster + beats + dialogue ledger (outside pillars)
    # PAI 1.1 — scene-wide shared fields. Every shot deep-merges these as
    # its baseline before reading its own (sparse) overrides. None means
    # the scene has no defaults; every shot is fully self-describing.
    shot_defaults:    Optional[ShotDefaults] = None
    # PAI 1.1 prototype — optional scene-wide world coordinates. When set,
    # per-subject screen_position can be derived from camera + world_xy
    # instead of authored per shot. See pipeline/derive_screen_position.py.
    physical_layout:  Optional[PhysicalLayout] = None
    shots:            list[Shot]    = field(default_factory=list)             # ordered list of shots in this scene
    compile_hints:    list[CompileHints] = field(default_factory=list)        # per-panel sidecar hints (one entry per panel that needs hints)
    # PACE pace-0.2 blocks. Optional so pai-1.0/1.1 documents load unchanged.
    semantics:        Optional[Semantics] = None
    manifest:         Optional[Manifest] = None
    artifacts:        list[Artifact] = field(default_factory=list)  # scene.artifacts
    _source:          Optional[str] = None                    # path to the source script segment this scene was extracted from
    _generated_at:    Optional[str] = None                    # ISO-8601 timestamp when this scene file was first written


# ════════════════════════════════════════════════════════════════════════
# Field-tier registry — guides authors / LLM-enrichment ordering
# ════════════════════════════════════════════════════════════════════════
#
# Fills out in 3 priority bands:
#
#   required    — without these the shot is unrenderable. Compile_flux
#                 cannot produce a coherent prompt; missing them yields a
#                 generic / wrong-looking frame. Must fill on every shot.
#
#   recommended — significantly shape the look but the shot is still
#                 compilable with sensible defaults. Fill on most shots;
#                 leaving them null degrades quality but doesn't break.
#
#   advanced    — decorative / pro-grade / EXIF-style metadata. Worth
#                 filling when you know the exact values (real capture
#                 EXIF, deliberate lighting design); otherwise null is
#                 fine. compile_flux2 gracefully ignores nulls here.
#
# Dotted paths use `[]` to mark "every element of this list" — e.g.
# `setup.subjects[].character_id` is required for every subject entry.

# Fields a breakdown must supply by inference rather than by quotation. A
# screenplay states none of them in words that can be located in it, and each
# is compiled into every prompt, so returning null does not leave the field
# blank downstream, it leaves the sampler to choose. Keyed "Schema.field", as
# usd_map is. The paper's field specification prints the tier from here.
INFERRED_FIELDS: frozenset[str] = frozenset({
    "Environment.mood",
    "Backdrop.era",
    "Backdrop.region",
    "Backdrop.culture",
})

FIELD_TIER: dict[str, Literal["required", "recommended", "advanced", "requiredIfPresent"]] = {
    # ── REQUIRED ─────────────────────────────────────────────────────────
    "camera.creativeIntent.shotSize":      "required",
    "camera.creativeIntent.framing":        "required",
    "setup.backdrop.location":               "required",
    "setup.backdrop.setting":                "required",
    "setup.backdrop.timeOfDay":            "required",
    "setup.subjects[].characterId":         "required",   # every Subject needs an id
    "setup.subjects[].ageState":            "required",
    "setup.primaryFocus":                   "required",
    "events.actions[].descriptionZh":       "required",
    "events.actions[].descriptionEn":       "required",       # pace-0.2 §events-align (升 required:缺则后端静默回退 zh)
    "setup.props[].propId":                 "required",       # pace-0.2 §full-align
    "setup.environment.style":               "required",       # pace-0.2 §1.E align

    # ── RECOMMENDED ──────────────────────────────────────────────────────
    "camera.extrinsics.angle":               "recommended",
    "camera.extrinsics.position":            "recommended",
    "camera.trajectory.movement2d":         "recommended",
    "camera.trajectory.movement3d":         "recommended",
    "camera.trajectory.gear":                "recommended",
    "camera.trajectory.easing":              "recommended",
    "camera.trajectory.static":              "recommended",   # pace-0.2 §coverage
    "camera.trajectory.cameraPath":         "recommended",   # pace-0.2 §camera-path (program 的产物侧)
    "camera.fps":                            "recommended",   # pace-0.2 §structured
    "camera.frameRange":                    "recommended",   # pace-0.2 §structured
    "camera.creativeIntent.aspectRatio":   "recommended",
    "camera.intrinsics.lensSize":           "recommended",   # pace-0.2 §1.E align
    "camera.intrinsics.depthOfField":      "recommended",   # pace-0.2 §1.E align
    "camera.intrinsics.aperture":            "recommended",   # pace-0.2 §1.E align
    "camera.intrinsics.sensorMm":           "recommended",   # pace-0.2
    "setup.backdrop.era":                    "recommended",
    "setup.backdrop.region":                 "recommended",
    "setup.backdrop.culture":                "recommended",
    "setup.backdrop.weather":                "recommended",   # pace-0.2
    "setup.backdrop.season":                 "recommended",   # pace-0.2
    "setup.environment.mood":                "recommended",
    "setup.environment.elements":            "recommended",
    "setup.environment.background":          "recommended",
    "setup.environment.scale":               "recommended",
    "setup.excluded":                        "recommended",
    "setup.subjects[].cls":                  "recommended",   # pace-0.2 §coverage
    "setup.subjects[].pose":                 "recommended",
    "setup.subjects[].gaze":                 "recommended",
    "setup.subjects[].screenPosition":      "recommended",
    "setup.subjects[].costume":              "recommended",
    "setup.subjects[].costume_id":           "recommended",
    "setup.secondarySubjects":              "recommended",   # pace-0.2 §coverage
    "setup.props[].description":             "recommended",
    "setup.props[].cls":                     "recommended",
    "setup.props[].state":                   "recommended",
    "setup.props[].heldBy":                 "recommended",
    "setup.props[].material":                "recommended",   # pace-0.2 §1.E align
    "setup.props[].utility":                 "recommended",   # pace-0.2 §1.E align
    "setup.props[].size":                    "recommended",   # pace-0.2 §1.E align
    "setup.props[].color":                   "recommended",   # pace-0.2 §1.E align
    "lighting.natural":                      "recommended",
    "lighting.condition":                    "recommended",
    "lighting.position":                     "recommended",
    "lighting.colorTemperature":            "recommended",
    "events.actions[].standalone":           "recommended",   # pace-0.2 §events-align
    "events.actions[].interactive":          "recommended",   # pace-0.2 §events-align
    "events.actions[].temporal":             "recommended",
    "events.actions[].foreground":           "recommended",
    "events.actions[].intensity":            "recommended",
    "events.changeInEnvironment":          "recommended",   # pace-0.2 §coverage
    "events.emotions[].implicit":            "recommended",
    "events.emotions[].foreground":          "recommended",
    "camera.intrinsics.focalLengthMm":     "recommended",   # pace-0.2 §full-align
    "camera.intrinsics.apertureF":          "recommended",   # pace-0.2 §full-align
    "setup.subjects[].hair":                 "recommended",   # pace-0.2 §full-align
    "lighting.practicals":                   "recommended",   # pace-0.2 §full-align
    "events.actions[].durationHintS":      "recommended",   # pace-0.2 §full-align
    "events.actions[].beatFeatures":        "recommended",   # pace-0.2 §full-align

    # ── ADVANCED ─────────────────────────────────────────────────────────
    "camera.program":                        "advanced",   # pace-0.2 §structured
    "camera.intrinsics.shutterSpeed":       "advanced",
    "camera.intrinsics.iso":                 "advanced",
    "camera.intrinsics.tStop":              "advanced",
    "camera.intrinsics.shutterAngleDeg":   "advanced",
    "camera.intrinsics.isoValue":           "advanced",
    "camera.intrinsics.focusDistanceM":    "advanced",     # pace-0.2
    "setup.texture.contrast":                "advanced",
    "setup.texture.blur":                    "advanced",
    "setup.texture.noise":                   "advanced",
    "setup.texture.filmGrain":              "advanced",
    "setup.texture.colorPalette":           "advanced",
    "setup.geometry.lines":                  "advanced",
    "setup.geometry.regularShapes":         "advanced",
    "setup.geometry.naturalShapes":         "advanced",
    "setup.geometry.frameBalance":          "advanced",
    "setup.geometry.positionalAccuracy":    "advanced",
    "setup.geometry.relativePositioning":   "advanced",
    "setup.space.depth":                     "advanced",
    "setup.environment.density":             "advanced",
    "setup.environment.negativeSpace":      "advanced",
    "setup.props[].pattern":                 "advanced",
    "setup.props[].screenPosition":         "advanced",
    "setup.props[].count":                   "advanced",
    "setup.textGeneration":                 "advanced",   # pace-0.2 §coverage (container of text_generation[].*)
    "setup.textGeneration[].target":        "advanced",
    "setup.textGeneration[].content":       "advanced",
    "setup.textGeneration[].style":         "advanced",
    "setup.textGeneration[].layout":        "advanced",
    "setup.subjects[].silhouette":           "advanced",
    "setup.subjects[].proportions":          "advanced",
    "setup.subjects[].accessories":          "advanced",
    "setup.subjects[].makeup":               "advanced",
    "lighting.softShadows":                 "advanced",
    "lighting.hardShadows":                 "advanced",
    "lighting.reflection":                   "advanced",
    "lighting.motion":                       "advanced",
    "lighting.colorGels":                   "advanced",
    "lighting.colorTempK":                 "advanced",
    "lighting.notes":                        "advanced",   # pace-0.2 §coverage
    "events.actions[].uncertainty":          "advanced",
    "events.actions[].background":           "advanced",
    "events.emotions[].explicit":            "advanced",
    "events.emotions[].temporal":            "advanced",
    "events.emotions[].background":          "advanced",   # pace-0.2 §events-align
    "events.dialogues[].typeOfDelivery":   "advanced",
    "events.advanced.storyStructure":       "advanced",
    "events.advanced.pace":                  "advanced",
    "events.advanced.regularity":            "advanced",

    # ── REQUIRED_IF_PRESENT(条件必填:该块存在时其内容才必填)──────────────
    "events.dialogues[]":                    "requiredIfPresent",   # pace-0.2 §events-align
}


# ════════════════════════════════════════════════════════════════════════
# Scene-wide vs shot-local — guidance for scene_defaults usage
# ════════════════════════════════════════════════════════════════════════
#
# Fields that almost always span an entire scene (good candidates for
# scene_defaults) — promote them once at scene level and let shots
# inherit, so you stop authoring them N times:

SCENE_WIDE_FIELDS: list[str] = [
    "camera.creative_intent.aspect_ratio",
    "setup.backdrop.setting",
    "setup.backdrop.time_of_day",
    "setup.backdrop.location",
    "setup.backdrop.era",
    "setup.backdrop.region",
    "setup.backdrop.culture",
    "setup.environment.style",
    "setup.environment.mood",      # mood occasionally shifts within scene; still scene-wide default
    "setup.texture.color_palette",
    "lighting.condition",
    "lighting.color_temperature",
    "lighting.color_temp_k",
]

# Fields that are intrinsically shot-local — never promote these.
SHOT_LOCAL_FIELDS: list[str] = [
    "camera.creative_intent.shot_size",
    "camera.creative_intent.framing",
    "camera.extrinsics.angle",
    "camera.extrinsics.position",
    "camera.trajectory.movement_2d",
    "camera.trajectory.movement_3d",
    "setup.subjects",
    "setup.primary_focus",
    "setup.excluded",
    "events.actions",
    "events.emotions",
    "events.dialogues",
]
