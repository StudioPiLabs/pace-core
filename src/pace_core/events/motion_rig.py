"""Motion rig — sparse, approximately-timed keyframe poses into a dense
per-frame pose track, so that character motion can be *authored* in PAI
terms rather than only replayed from a captured clip.

This is the pose-space counterpart of two mechanisms PAI already has, and
it is deliberately shaped like both. camera_planner compiles a sparse
motion DSL into a dense metric camera track; temporal_keyframes pins known
frames and lets a generative model fill the rest. This module does the same
for a character's articulation: the caller supplies a handful of poses at
approximate times, and gets back one pose per rendered frame, in exactly
the shape vace_pipeline's armature baking already consumes.

Why the timing, and not the posing, is the hard part
────────────────────────────────────────────────────
Goel et al. (Generative Motion Infilling from Imprecisely Timed Keyframes,
EG 2025) make the observation this module is built on: animators can
specify *where* the joints go far more reliably than *when* the pose should
land, and inbetweening systems that treat keyframe times as hard
constraints degrade badly when those times are wrong — the character either
snaps between poses too fast, or fails to reach a pose at all, because no
natural motion satisfies the constraint as stated. Their model therefore
emits a time-warping function that retimes the keyframes, plus spatial
residuals that add detail between them. Disney Research's Generative Motion
Rig wraps that class of model in an artist-facing control surface: sparse
poses and handles, a window length, and a noise seed for variation.

We adopt the *decomposition* — retime first, then fill, then add residual
detail — because it is what makes sparse authoring usable, and because each
stage is separately inspectable. What we do not adopt is a learned motion
prior: `solve()` below computes the warp from a reachability budget rather
than from data, and its residuals are procedural anticipation/follow-through
rather than sampled detail. That is an honest downgrade, not an equivalence.
It ships because it is deterministic, testable without a GPU, and correct
about the thing that matters most in previs — that a pose the character
cannot physically reach in the time allotted gets more time, rather than
being hit by teleporting. `MotionBackend` is the seam where a real
generative prior replaces the procedural core without any caller changing.

Pure functions, no bpy, no I/O — the same rule camera_planner and
composition_solver follow, so the whole module is testable off-GPU.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Protocol, Sequence

# A pose is {bone_name: (w, x, y, z)} unit quaternions, optionally carrying
# the reserved key ROOT_KEY for world-space root translation in metres.
# Quaternions (not Euler) because interpolation between two Euler triples is
# path-dependent and gimbal-prone; vace_pipeline keys pose bones as
# rotation_quaternion already, so this matches what the renderer consumes.
Pose = dict
ROOT_KEY = "__root_loc__"

# Default reachability budget, interpreted as a bound on the PEAK per-frame
# angular speed a joint reaches, not on its segment average. A human joint
# rotating faster than this in a previs proxy reads as a snap rather than a
# move; it is the knob that decides whether a requested keyframe time is
# honoured or stretched.
DEFAULT_MAX_ANGULAR_SPEED_DPS = 540.0

# Ratio of peak to mean angular speed across a segment, given this module's
# ease curve and residual. Sizing a segment by its average speed alone lets
# the delivered peak run far over budget, which is a budget that does not
# bind: smoothstep alone peaks at 1.5x its mean (its derivative 6t-6t^2
# maxes at 1.5), and the seeded asymmetry plus anticipation/overshoot push
# that to ~3.4x, measured across seeds and travel distances. We size
# segments against this so the constraint holds on the delivered track
# rather than on the solver's own idea of it — the same measure-the-output
# discipline the composition check uses. Recompute this if _ease or
# ProceduralBackend's residual changes.
EASE_PEAK_FACTOR = 3.5


class MotionBackend(Protocol):
    """The seam a learned motion prior plugs into.

    An implementation receives the retimed keyframes and the frame count and
    returns one pose per frame. `ProceduralBackend` below is the shipped
    default; a diffusion-style inbetweener would implement the same call and
    ignore nothing, since retiming has already made the constraints
    physically satisfiable — which is precisely the condition Goel et al.
    show such models need in order not to degrade.
    """

    def infill(self, keys: list[tuple[int, Pose]], n_frames: int,
               *, seed: int, seed_pos: int = 0) -> list[Pose]: ...


# ─────────────────────────── quaternion helpers ───────────────────────────

def _q_norm(q: Sequence[float]) -> tuple[float, float, float, float]:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(c / n for c in q)  # type: ignore[return-value]


def _q_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[i] * b[i] for i in range(4))


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Geodesic angle in degrees between two orientations. Uses the short
    arc, so q and -q (the same orientation, opposite sign) read as zero.

    Computed as 2*atan2(|vec(r)|, |w(r)|) on the relative rotation r rather
    than the textbook 2*acos(|dot|). The acos form is ill-conditioned
    exactly where this function is used hardest: near zero its derivative
    diverges, so two *bit-identical* quaternions can report ~2e-6 degrees
    apart purely from rounding in the dot product. Every "the authored pose
    is hit exactly" claim in this codebase rests on this function, and an
    error bar seven orders of magnitude above the true value would make
    those claims untestable. The atan2 form stays accurate at both ends.
    """
    qa, qb = _q_norm(a), _q_norm(b)
    if _q_dot(qa, qb) < 0.0:                    # take the short arc
        qb = tuple(-c for c in qb)
    # r = conj(qa) * qb; its vector part is small near identity and stays
    # accurate there, which is the whole reason for this formulation.
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    rw = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    rx = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    ry = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    rz = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2
    return math.degrees(2.0 * math.atan2(math.sqrt(rx * rx + ry * ry + rz * rz),
                                         abs(rw)))


def slerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple:
    """Shortest-arc spherical interpolation. Falls back to normalized lerp
    when the inputs are nearly parallel, where slerp is numerically unstable
    and the two agree to well beyond rendering precision anyway."""
    qa, qb = _q_norm(a), _q_norm(b)
    d = _q_dot(qa, qb)
    if d < 0.0:                      # take the short way round
        qb, d = tuple(-c for c in qb), -d
    if d > 0.9995:
        return _q_norm(tuple(qa[i] + t * (qb[i] - qa[i]) for i in range(4)))
    th0 = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th0)
    f0, f1 = math.sin((1.0 - t) * th0) / s, math.sin(t * th0) / s
    return _q_norm(tuple(f0 * qa[i] + f1 * qb[i] for i in range(4)))


def _lerp3(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def _ease(t: float, k: float = 0.0) -> float:
    """Smoothstep with a seeded asymmetry k ∈ [-0.35, 0.35]. k>0 leaves the
    pose late and arrives fast (a snap), k<0 leaves early and settles (a
    drift). Real motion is rarely symmetric in and out of a pose, and this
    is the cheapest handle that varies a take without moving any keyframe."""
    t = min(1.0, max(0.0, t))
    s = t * t * (3.0 - 2.0 * t)
    return min(1.0, max(0.0, s + k * s * (1.0 - s) * 2.0))


# ───────────────────────────── time warping ──────────────────────────────

def solve_time_warp(keys: list[tuple[int, Pose]], n_frames: int, *,
                    fps: int = 16,
                    max_speed_dps: float = DEFAULT_MAX_ANGULAR_SPEED_DPS,
                    ) -> dict:
    """Retime approximately-placed keyframes so every segment is reachable.

    For each consecutive pair, the largest per-bone rotation Δθ implies a
    minimum duration Δθ · EASE_PEAK_FACTOR / max_speed_dps seconds — the
    peak factor is there because the budget bounds the fastest frame, not
    the segment average, and an eased move is markedly faster in its middle
    than end to end. Where the requested spacing is shorter than that, the
    segment cannot be performed as asked and is widened; slack is then taken
    from segments that have room, preserving order. The result is a
    monotonically increasing map from each authored keyframe index to the
    frame it should actually land on — the same artifact Goel et al.'s model
    learns, computed here from a budget.

    Returns {"frames", "shifted", "required_frames", "feasible", "scale"}.
    `feasible` is False when the whole motion cannot fit in n_frames even
    after redistribution; the warp is still returned (uniformly compressed,
    the least-bad option) and the caller is told rather than silently given
    a motion that snaps.
    """
    if len(keys) < 2:
        return {"frames": [k for k, _ in keys], "shifted": [0] * len(keys),
                "required_frames": 0, "feasible": True, "scale": 1.0}

    src = [int(k) for k, _ in keys]
    poses = [p for _, p in keys]

    # Minimum frames each segment needs to be performable.
    need: list[float] = []
    for i in range(len(poses) - 1):
        dtheta = max(
            [angle_between(poses[i][b], poses[i + 1][b])
             for b in poses[i] if b != ROOT_KEY and b in poses[i + 1]] or [0.0])
        need.append(dtheta * EASE_PEAK_FACTOR / max(max_speed_dps, 1e-6) * fps)

    want = [max(1.0, float(src[i + 1] - src[i])) for i in range(len(src) - 1)]
    # Honour the request where it is already performable; widen where not.
    seg = [max(w, nd) for w, nd in zip(want, need)]

    total = sum(seg)
    span = max(1.0, float(n_frames - 1))
    scale = 1.0
    feasible = True
    if total > span:
        # Widening the tight beats has overrun the window. Reclaim the
        # overrun from segments that asked for more than they need, in
        # proportion to how much slack each holds — a long hold gives up
        # frames so a fast beat can become performable, which is what an
        # animator would do by hand. Compressing everything uniformly here
        # would instead steal from the beats that were already too tight,
        # making the exact problem retiming exists to fix worse.
        slack = [s - nd for s, nd in zip(seg, need)]
        avail = sum(slack)
        deficit = total - span
        if deficit <= avail and avail > 0:
            seg = [s - deficit * (sl / avail) for s, sl in zip(seg, slack)]
        else:
            # No slack anywhere: the motion genuinely does not fit. Fall back
            # to the least-bad layout (each segment scaled off its own need,
            # so the ordering of demands is at least preserved) and say so.
            need_total = sum(need) or 1.0
            scale = span / need_total
            seg = [nd * scale for nd in need]
            feasible = False
    elif total < span:
        # Room to spare: distribute it in proportion to what each segment
        # originally asked for, so authored rhythm survives the retime.
        extra = span - total
        wsum = sum(want)
        seg = [s + extra * (w / wsum) for s, w in zip(seg, want)]

    out = [float(src[0] if src[0] < n_frames else 0)]
    for s in seg:
        out.append(out[-1] + s)

    # Snap to integer frames, keeping the order the artist authored. Each
    # key reserves one frame for every key still to be placed after it,
    # otherwise heavy compression piles the tail onto the last frame and
    # two distinct poses collapse onto one — which reads as a dropped
    # keyframe rather than as the tight timing it actually is. When the
    # window has fewer frames than keys no distinct placement exists at
    # all; we allow the collision there and let `feasible` carry the news.
    n_keys = len(out)
    room = n_frames >= n_keys
    frames: list[int] = []
    prev = -1
    for i, v in enumerate(out):
        f = int(round(min(max(v, 0.0), float(n_frames - 1))))
        if room:
            lo = prev + 1
            hi = n_frames - 1 - (n_keys - 1 - i)   # leave room for the rest
            f = min(max(f, lo), max(lo, hi))
        elif f <= prev:
            f = min(prev + 1, n_frames - 1)
        frames.append(f)
        prev = f
    return {"frames": frames,
            "shifted": [frames[i] - src[i] for i in range(len(src))],
            "required_frames": int(math.ceil(sum(need))) + 1,
            "feasible": feasible, "scale": scale}


# ─────────────────────────────── backends ────────────────────────────────

class ProceduralBackend:
    """Shipped default: eased slerp between retimed keys, plus seeded
    anticipation/follow-through as the residual detail.

    The residual occupies the slot a learned model fills with sampled
    submovement. Here it is one animation principle applied honestly: a
    joint about to make a large move first drifts slightly *against* it
    (anticipation) and slightly past its target on arrival (overshoot),
    scaled by how far that joint travels, so small adjustments stay clean
    and large gestures get weight. It is not sampled from motion data and
    will not invent a submovement the keyframes do not imply.
    """

    def __init__(self, *, overshoot: float = 0.12, anticipation: float = 0.06,
                 bow: float = 0.18):
        self.overshoot = overshoot
        self.anticipation = anticipation
        self.bow = bow

    def infill(self, keys: list[tuple[int, Pose]], n_frames: int,
               *, seed: int, seed_pos: int = 0) -> list[Pose]:
        """`seed` resamples the orientational take, `seed_pos` the positional
        one. Buhmann et al. report these behave differently and are worth
        separating: resampling orientation keeps the motion's overall shape
        and changes the pose at a given instant, while resampling position
        changes the outline of the whole move. We reproduce that split
        rather than the mechanism, since the two act on genuinely different
        parts of a pose here: orientation varies the ease profile between
        keys, position bows the root's path off the straight line between
        them. Both leave the authored keys exactly where they were."""
        if not keys:
            return []
        if len(keys) == 1:
            return [dict(keys[0][1]) for _ in range(n_frames)]

        # Deterministic per-seed ease asymmetry, one value per segment, so
        # the same seed always reproduces the same take.
        rnd = _seeded_stream(seed)
        ks = [rnd() * 0.7 - 0.35 for _ in range(len(keys) - 1)]
        # Positional stream: a per-segment lateral bow, zero at both ends so
        # every authored root position is still hit exactly.
        rndp = _seeded_stream(seed_pos)
        bows = [[(rndp() * 2.0 - 1.0) * self.bow for _ in range(3)]
                for _ in range(len(keys) - 1)]

        out: list[Pose] = []
        for f in range(n_frames):
            i = _segment_index([k for k, _ in keys], f)
            f0, p0 = keys[i]
            f1, p1 = keys[min(i + 1, len(keys) - 1)]
            span = max(1, f1 - f0)
            t = _ease(min(1.0, max(0.0, (f - f0) / span)), ks[min(i, len(ks) - 1)])
            pose: Pose = {}
            for b in p0:
                if b == ROOT_KEY:
                    continue
                if b not in p1:
                    pose[b] = p0[b]
                    continue
                trav = angle_between(p0[b], p1[b])
                pose[b] = self._with_residual(p0[b], p1[b], t, trav)
            if ROOT_KEY in p0 and ROOT_KEY in p1:
                base = _lerp3(p0[ROOT_KEY], p1[ROOT_KEY], t)
                # sin(pi*t) is 0 at both keys, so the bow reshapes the path
                # between them without ever moving an authored position.
                span_len = math.dist(p0[ROOT_KEY], p1[ROOT_KEY])
                if span_len > 1e-9:
                    b = bows[min(i, len(bows) - 1)]
                    amp = math.sin(math.pi * min(1.0, max(0.0, t))) * span_len
                    base = [base[k] + b[k] * amp for k in range(3)]
                pose[ROOT_KEY] = base
            elif ROOT_KEY in p0:
                pose[ROOT_KEY] = list(p0[ROOT_KEY])
            out.append(pose)
        return out

    def _with_residual(self, a, b, t: float, travel_deg: float):
        """Push t slightly out of [0,1] at the ends of a large move. The
        magnitude scales with travel so a 5-degree correction does not
        acquire a wind-up it never had in the authored poses."""
        if travel_deg < 1e-3:
            return slerp(a, b, t)
        w = min(1.0, travel_deg / 90.0)          # full effect at a quarter turn
        if t < 0.25:
            t = t - self.anticipation * w * math.sin(math.pi * (t / 0.25))
        elif t > 0.75:
            t = t + self.overshoot * w * math.sin(math.pi * ((t - 0.75) / 0.25))
        return slerp(a, b, t)


def _seeded_stream(seed: int) -> Callable[[], float]:
    """Tiny deterministic LCG. `random` is avoided so a caller cannot
    perturb a take by seeding the global RNG somewhere unrelated — a real
    hazard when this runs inside a pipeline that also samples diffusion."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt() -> float:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 33) & 0x7FFFFFFF) / float(0x7FFFFFFF)
    return nxt


def _segment_index(frames: Sequence[int], f: int) -> int:
    for i in range(len(frames) - 1):
        if f < frames[i + 1]:
            return i
    return max(0, len(frames) - 2)


# ──────────────────────────────── entry ──────────────────────────────────

def pose_at(track: list[dict], frame: int) -> Pose:
    """One pose lifted out of a solved track.

    This is the whole of the generative-to-traditional transfer, and the
    narrowness is the point. Buhmann et al. observe that handing a
    traditional rig the entire generated motion would key every frame and
    leave the rig uneditable — the artist gains a take and loses the
    ability to work on it. Transferring a single pose instead lets them
    pick the key poses worth polishing by hand. There is deliberately no
    function here that exports a whole track as constraints.
    """
    if not track:
        raise ValueError("empty track")
    f = max(0, min(int(frame), len(track) - 1))
    entry = track[f]
    pose: Pose = {b: tuple(q) for b, q in (entry.get("bones") or {}).items()}
    if "root" in entry:
        pose[ROOT_KEY] = list(entry["root"])
    return pose


def as_keyframe(pose: Pose, frame: int) -> dict:
    """Wrap a pose as an authored keyframe — the traditional-to-generative
    direction, where the artist's current hand-made pose becomes a
    full-body constraint the next solve must satisfy."""
    return {"frame": int(frame), "pose": {k: list(v) for k, v in pose.items()}}


def blend_layers(generative: list[Pose], traditional: list[Pose],
                 weights) -> list[Pose]:
    """Composite a generative layer over a traditional one, per frame.

    `weights` is a scalar or one value per frame; 0 takes the traditional
    layer, 1 the generative. Keeping both layers and a weight curve, rather
    than baking one into the other, is what lets a shot use the rig for
    timing and layout and hand-animation for the parts that need to leave
    physical plausibility behind — the case the paper names as motivating
    layers in the first place.
    """
    n = min(len(generative), len(traditional))
    if n == 0:
        return []
    if isinstance(weights, (int, float)):
        w = [float(weights)] * n
    else:
        w = [float(x) for x in weights]
        if len(w) < n:
            w += [w[-1] if w else 1.0] * (n - len(w))
    return [_blend_pose(traditional[i], generative[i],
                        min(1.0, max(0.0, w[i]))) for i in range(n)]


def _blend_pose(a: Pose, b: Pose, t: float) -> Pose:
    """Blend two poses, t=0 giving `a`. Used only at inpainting seams."""
    out: Pose = {}
    for k in a:
        if k == ROOT_KEY:
            out[k] = _lerp3(a[k], b.get(k, a[k]), t)
        else:
            out[k] = slerp(a[k], b.get(k, a[k]), t)
    for k in b:
        out.setdefault(k, b[k])
    return out


def solve(keyframes: list[dict], *, n_frames: int, fps: int = 16,
          seed: int = 0, seed_pos: Optional[int] = None,
          max_speed_dps: float = DEFAULT_MAX_ANGULAR_SPEED_DPS,
          backend: Optional[MotionBackend] = None,
          base_motion: Optional[list] = None,
          preserve: Optional[list] = None,
          bounds: Optional[tuple] = None,
          blend_frames: int = 3) -> dict:
    """Sparse authored poses → one pose per frame.

    `keyframes` = [{"frame": int, "pose": {bone: quat, ...}}], where `frame`
    is the *approximate* time the artist wants the pose to land. Poses need
    not agree on bone sets; a bone missing from the next key holds.

    `n_frames` is the window length — the Disney rig's authoring handle for
    how much room the motion has, and the quantity retiming is solved
    against. `seed` selects a take: it varies ease asymmetry and residual
    phase, never the poses themselves, so every take still hits what was
    authored.

    `base_motion` turns generation into editing: pass an existing dense
    track (a mocap take, or a previous result) and the solve preserves the
    frames named authoritative — everything outside `bounds`, plus any
    (lo, hi) range in `preserve` — while generating the rest to satisfy the
    keyframes. `bounds` is the evaluation window the artist drags on the
    timeline; frames beyond it pass through untouched, which is also how
    motion extension works, by placing bounds past the end of the
    constraints. Seams get a `blend_frames`-long ease so a regenerated
    span does not pop against the take it is spliced into.

    Returns {"track", "warp", "n_frames", "seed", "feasible", "edited",
    "preserved_frames"}. `track` is
    [{frame_index, bones, root?}], one entry per frame, directly consumable
    by vace_pipeline's armature baking. `warp` reports how far each authored
    keyframe was moved, so a UI can draw the retiming rather than silently
    applying it — the paper's own presentation choice, and the reason a
    stretched segment reads as a decision instead of a bug.
    """
    keys_in = sorted(
        [(int(k["frame"]), dict(k["pose"])) for k in keyframes],
        key=lambda kv: kv[0])
    if not keys_in:
        raise ValueError("at least one keyframe is required")
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    warp = solve_time_warp(keys_in, n_frames, fps=fps, max_speed_dps=max_speed_dps)
    retimed = [(warp["frames"][i], keys_in[i][1]) for i in range(len(keys_in))]

    be = backend or ProceduralBackend()
    poses = be.infill(retimed, n_frames, seed=seed,
                      seed_pos=(seed if seed_pos is None else seed_pos))

    # ── motion editing ────────────────────────────────────────────────────
    # Buhmann et al. guide generation toward an existing clip through an
    # inpainting strategy, so the same controls that author new motion also
    # stitch, extend, and edit captured motion — the difference between a
    # tool that starts from nothing and one that can be pointed at a take
    # you already have. The pattern is the one this pipeline already uses
    # for pixels: name the frames that are authoritative, generate the
    # rest. Here the authoritative frames are those outside the evaluation
    # bounds plus any explicitly preserved range.
    edited = False
    authoritative: set = set()
    if base_motion is not None:
        if len(base_motion) != n_frames:
            raise ValueError(
                f"base_motion has {len(base_motion)} frames, expected {n_frames}")
        edited = True
        lo, hi = bounds if bounds else (0, n_frames - 1)
        lo, hi = max(0, int(lo)), min(n_frames - 1, int(hi))
        authoritative |= {f for f in range(n_frames) if f < lo or f > hi}
        for rng in (preserve or []):
            a, b = int(rng[0]), int(rng[1])
            authoritative |= {f for f in range(max(0, a), min(n_frames - 1, b) + 1)}

        merged: list[Pose] = []
        for f in range(n_frames):
            if f in authoritative:
                merged.append(dict(base_motion[f]))
                continue
            # Ease across a seam rather than cutting at it. The paper flags
            # exactly this failure — re-conditioning on a previously
            # predicted pose shifts the generated motion and shows up as a
            # pop — so the boundary gets a short blend instead of a splice.
            d = min((abs(f - g) for g in authoritative), default=None)
            if d is not None and blend_frames > 0 and d <= blend_frames:
                w = d / float(blend_frames + 1)      # 0 at the seam, →1 away
                merged.append(_blend_pose(base_motion[f], poses[f], w))
            else:
                merged.append(poses[f])
        poses = merged

    track = []
    for f, p in enumerate(poses):
        entry: dict = {"frame_index": f,
                       "bones": {b: list(q) for b, q in p.items() if b != ROOT_KEY}}
        if ROOT_KEY in p:
            entry["root"] = list(p[ROOT_KEY])
        track.append(entry)
    return {"track": track, "warp": warp, "n_frames": n_frames,
            "seed": seed, "seed_pos": (seed if seed_pos is None else seed_pos),
            "feasible": warp["feasible"],
            "edited": edited, "preserved_frames": sorted(authoritative)}
