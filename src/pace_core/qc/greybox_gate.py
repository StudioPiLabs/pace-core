#!/usr/bin/env python3
"""GreyboxGate: is this control frame fit to generate from?

The design (Script -> Beat -> Beat Image v1.1, section 7.5) puts a hard gate
between the 3D anchor and the generator: only a greybox that passes enters
image generation. Nothing enforced that here. Every panel that has ever been
generated in this project was generated from whatever the geometry produced,
including the ones where it produced the wrong thing -- and the verifier
downstream then measured a delivered frame against an anchor nobody had
checked was worth delivering.

The design states five clauses. Three are measurable from what the kernel
already renders, one is measurable only in part, and one is not measurable at
all. Which is which is stated here rather than papered over, because a gate
that reports a number it cannot compute is worse than a gate that says it
cannot compute it.

  required_entity_visibility   MEASURED. Every required subject has a matte
                               and that matte has pixels. The kernel renders
                               one alpha matte per declared body, so a body
                               that ended up outside the frustum, behind the
                               shell, or never staged at all comes back empty
                               and is caught here rather than in the delivery.

  projected_subject_area       MEASURED. Each required subject's share of the
                               frame, from its own matte.

  critical_relation_satisfaction
                               PARTLY MEASURED. The design means the full
                               relation set -- LEFT_OF, NEAR, ON, FACING,
                               CONTACT. We have no Relation IR, so most of
                               those have no ground truth to check against.
                               Exactly one relation IS declared per panel and
                               is checkable: the left-to-right screen order of
                               the cast, from `screen_position.x`. That is
                               what this measures, and it is reported under
                               its own name, `screen_order`, rather than under
                               the design's broader one.

  control_versions_are_consistent
                               MEASURED. Depth and the mattes must come from
                               the build that produced the beauty frame; one
                               written BEFORE its own frame came from an
                               earlier build of a different staging. It finds
                               nothing on this corpus today, which is the
                               result and not a reason to drop it: it is the
                               guard against a caller re-rendering one pass on
                               its own, and passing is what it should report
                               until someone does.

  focal_action_readability     NOT MEASURED. It asks whether the focal
                               action reads -- whether the contact points of
                               a HELP_STAND are legible. Our subjects are
                               static SMPL-X proxies with no rig and no
                               contact points, so there is nothing to measure
                               against. It is reported as None and does not
                               vote. Inventing a proxy for it -- silhouette
                               overlap, say -- would measure whether two
                               bodies are near each other, and then report
                               that as whether an action reads.

Thresholds come from the corpus, not from preference; see the constants.

    uv run python -m pace_core.qc.greybox_gate --project AutomaticDrive
    uv run python -m pace_core.qc.greybox_gate --project AutomaticDrive --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Measured over AutomaticDrive's 81 staged bodies across 37 panels: the
# smallest any subject has ever projected is 0.0231 of the frame and the 5th
# percentile is 0.0359. So 0.01 passes every body in the corpus today. That is
# deliberate -- this is a floor against a subject VANISHING (staged outside the
# frustum, occluded to nothing by the shell), not a tuning knob for framing.
# A discriminating threshold would have to be argued from what reads at
# 1280x544, and nothing here has measured that.
MIN_SUBJECT_AREA = 0.01

# Alpha above which a matte pixel counts as the body.
ALPHA_ON = 0.5

# How much of one subject's silhouette may lie under another's before the
# panel stops showing two people and starts showing one.
#
# The mattes are rendered one body at a time, so each is the silhouette that
# body WOULD have alone; where two overlap, the further one is that much
# hidden. Measured on this corpus the spread is not subtle: a front-facing
# three-hander overlaps 17%, while scene 2's three-quarter azimuth stacks two
# subjects on one sight line at 57% and 67%. Both passed every other clause,
# which is how a panel that renders a declared subject as a sliver of forehead
# was called an anchor.
#
# 0.35 sits in the gap between those two populations. It is a floor against a
# subject being effectively absent, not a composition preference: overlap is
# how depth reads, and a gate that demanded zero would forbid a two-shot.
MAX_SUBJECT_OVERLAP = 0.35

# How much of a subject's READ POINT -- the head, the thing a frame is
# composed around -- has to survive the set standing in front of it.
#
# Body-vs-body overlap is measured on mattes rendered with everything else
# hidden, so it is blind to furniture: a wall panel between camera and cast
# costs a subject nothing there. The kernel now also renders each body against
# the standing set, and the visible head is that pass intersected with the
# isolated head matte -- no extra render, and it is the fraction that decides
# whether a panel has a read point at all.
#
# Measured over the corpus's 76 staged subjects the answer is bimodal and not
# close: 72 heads are 100% visible and 4 are 0%, with nothing between. The
# threshold is therefore not doing subtle work; it separates "there is a head
# to compose around" from "there is not".
MIN_READ_POINT_VISIBLE = 0.5

# A control is stale when it predates its own beauty frame. Within one build
# the kernel writes the frame first and the derived passes after, so positive
# skew is normal and large (p95 = 30 s across the corpus, the mattes being
# one render each). Negative skew is not: it means the file on disk was
# written by an earlier build. Five seconds absorbs filesystem timestamp
# noise without absorbing a rebuild.
STALE_TOLERANCE_S = 5.0

# The passes the kernel derives from a build, by suffix on the frame's path.
# Only passes the kernel writes ONCE, at build time, belong here.
#
# `.structure.png` deliberately does not, and it is the interesting exclusion.
# It is what the sampler actually starts from, so an init from an older build
# would mean the generator began from a picture of a different staging than
# the depth and the mattes describe -- exactly the failure this clause exists
# for. 23 of AutomaticDrive's 33 built panels have one on disk, the oldest 41
# hours older than its own frame. None of them can reach a render: the init is
# derived lazily by `structure_init.structure_init_for`, which rebuilds it
# whenever the greybox is newer, and that is the only path any consumer takes
# to reach one. Checking it here would report 23 failures for a hazard another
# module already closes, which is worse than not checking it -- a gate whose
# failures do not mean anything stops being read.
DERIVED_SUFFIXES = (".depth.png", ".set.png", ".joints.json", ".props.json")

# A prop the build records in frame with less than this share of its
# projected box inside the frame is cut by an edge: staged "partial".
PROP_WHOLE_SHOWN = 0.98

# ── composition clauses ──
#
# The clauses above ask whether the frame stages what it declares. The five
# below ask whether it is composed the way a camera operator would frame it;
# each names one fault a storyboard artist corrects by eye. Their thresholds
# are starting values in shares of the frame, not corpus measurements, and
# are reported against the rebuilt corpus rather than tuned to it.
#
# A background edge counts as vertical where the grey level steps by more
# than LINE_EDGE_STEP across it, and as a line into a head where it is
# present over LINE_MIN_COVER of the band from LINE_ABOVE_CROWN of the frame
# above the crown to a quarter of the head below it.
LINE_EDGE_STEP = 12
LINE_MIN_COVER = 0.85
LINE_ABOVE_CROWN = 0.10
# A frame edge cuts at a joint when the joint projects within this share of
# the frame of an edge the body runs off. The joints are the ones a cut
# through reads as an amputation.
JOINT_CUT_TOL = 0.025
CUT_JOINTS = ("neck", "elbow", "wrist", "hip", "knee", "ankle")
# Two silhouettes are tangent when they come within TANGENT_GAP of the frame
# width of each other while overlapping by less than TANGENT_MAX_OVERLAP of
# the smaller one; a body grazes the top or a side edge when it stops within
# that distance of it, or crosses it along less than twice that.
TANGENT_GAP = 0.006
TANGENT_MAX_OVERLAP = 0.01
# Three or more heads stand in a row when their centres sit within this share
# of the frame height of one another at sizes -- depths -- within this ratio.
LINEUP_MAX_Y_SPREAD = 0.03
LINEUP_MAX_SIZE_RATIO = 1.25
# The singles of one shot / reverse-shot exchange hold the eye line: the eyes
# may differ by at most this share of the frame height between them. The eyes
# are taken EYE_FROM_TOP of the way down the head matte.
EYE_LINE_MAX_DIFF = 0.05
EYE_FROM_TOP = 0.42


@dataclass
class Clause:
    """One gate condition. `ok is None` means it could not be measured."""
    name: str
    ok: bool | None
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "value": self.value,
                "threshold": self.threshold, "detail": self.detail}


@dataclass
class GateReport:
    panel_id: str
    anchor_version: str | None
    clauses: list[Clause] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Passes when no measurable clause failed.

        An unmeasurable clause does not block. It also does not pass: it is
        absent, and `unmeasured` names it so a caller cannot mistake this
        report for the full gate the design specifies.
        """
        return all(c.ok is not False for c in self.clauses)

    @property
    def failures(self) -> list[Clause]:
        return [c for c in self.clauses if c.ok is False]

    @property
    def unmeasured(self) -> list[str]:
        return [c.name for c in self.clauses if c.ok is None]

    def as_dict(self) -> dict:
        return {"panel_id": self.panel_id, "ok": self.ok,
                "anchor_version": self.anchor_version,
                "unmeasured": self.unmeasured,
                "clauses": [c.as_dict() for c in self.clauses]}


def matte_stats(path: str | Path) -> dict | None:
    """Area and horizontal centroid of one body matte, or None if unreadable.

    A matte IS its alpha channel. If the file has no alpha the render lost
    `film_transparent` and every body in it reads as fully opaque, which would
    silently pass both the visibility and the area clause for a body that may
    not be in frame at all -- so that case returns a explicit failure marker
    rather than a measurement.
    """
    from PIL import Image
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return None
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or a.shape[2] != 4:
        return {"no_alpha": True, "area": None, "centroid_x": None}
    m = (a[..., 3] / 255.0) > ALPHA_ON
    if not m.any():
        return {"no_alpha": False, "area": 0.0, "centroid_x": None}
    xs = np.nonzero(m)[1]
    return {"no_alpha": False, "area": float(m.mean()),
            "centroid_x": float(xs.mean() / m.shape[1])}


def _mask(path):
    """A body matte as a boolean array, or None if absent or alpha-less."""
    from PIL import Image
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return None
    a = np.asarray(Image.open(p))
    if a.ndim != 3 or a.shape[2] != 4:
        return None
    return (a[..., 3] / 255.0) > ALPHA_ON


def _mask_l(path):
    """A single-channel visible-matte as a boolean array, or None if absent.

    The visible pass is rendered flat two-tone against the standing set rather
    than with alpha, because Workbench has neither holdout nor cryptomatte, so
    it is read off luminance.
    """
    from PIL import Image
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return None
    return np.asarray(Image.open(p).convert("L")) > 127


def screen_order_agreement(declared: list[tuple[str, float]],
                           measured: dict[str, float]) -> tuple[float, list[str]]:
    """Fraction of declared left-to-right pairs the render actually delivered.

    Pairwise rather than a rank correlation so the failures name themselves:
    a report saying "ethan renders left of ryan, declared right" is actionable
    where a coefficient is not. Pairs that declare the same x are skipped --
    they assert no order, so the geometry cannot violate one.
    """
    known = [(cid, x) for cid, x in declared
             if measured.get(cid) is not None and x is not None]
    ok = tot = 0
    bad: list[str] = []
    for i in range(len(known)):
        for j in range(i + 1, len(known)):
            (a, xa), (b, xb) = known[i], known[j]
            if xa == xb:
                continue
            tot += 1
            want_a_left = xa < xb
            got_a_left = measured[a] < measured[b]
            if want_a_left == got_a_left:
                ok += 1
            else:
                left, right = (a, b) if want_a_left else (b, a)
                bad.append(f"{left} declared left of {right}, renders right of it")
    return (ok / tot if tot else 1.0), bad


def _derived_controls(out: Path) -> list[Path]:
    """Every pass on disk derived from this frame, by naming convention."""
    found = [out.parent / (out.name + s) for s in DERIVED_SUFFIXES]
    found += sorted(out.parent.glob(out.name + ".body_*.png"))
    found += sorted(out.parent.glob(out.name + ".head_*.png"))
    return [p for p in found if p.is_file()]


def control_provenance(out: str | Path) -> Clause:
    """Do this frame's derived controls come from the build that made it?

    Timestamps are the only provenance the passes carry -- the kernel writes
    them as plain PNGs with no anchor stamped inside. That is a weaker check
    than comparing anchor versions and it is named as such: it catches a pass
    left over from an earlier build, and it would not catch two builds of
    different staging that happened to run back to back. Stamping the anchor
    into the passes at build time would make this exact; nothing does yet.
    """
    p = Path(out)
    if not p.is_file():
        return Clause("control_versions_are_consistent", False,
                      detail=f"beauty frame missing: {p.name}")
    t0 = p.stat().st_mtime
    stale = [(c.name.split(".png.", 1)[-1], (c.stat().st_mtime - t0) / 3600.0)
             for c in _derived_controls(p)
             if c.stat().st_mtime - t0 < -STALE_TOLERANCE_S]
    if stale:
        worst = min(stale, key=lambda s: s[1])
        return Clause(
            "control_versions_are_consistent", False, value=float(len(stale)),
            detail=(f"{len(stale)} control(s) predate this frame; "
                    f"oldest {worst[0]} by {abs(worst[1]):.1f} h"))
    return Clause("control_versions_are_consistent", True, value=0.0)


def _dilate(m, r: int):
    """Square binary dilation by `r` pixels, done separably, without scipy."""
    out = m.copy()
    for d in range(1, r + 1):
        out[:, d:] |= m[:, :-d]
        out[:, :-d] |= m[:, d:]
    wide = out.copy()
    for d in range(1, r + 1):
        out[d:, :] |= wide[:-d, :]
        out[:-d, :] |= wide[d:, :]
    return out


def eye_line(out: str | Path, cid: str) -> float | None:
    """Where one subject's eyes sit, as a share of frame height from the top."""
    import numpy as np

    out = Path(out)
    m = _mask(out.parent / (out.name + f".head_{cid}.png"))
    if m is None or not m.any():
        return None
    rows = np.nonzero(m.any(axis=1))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    return (top + EYE_FROM_TOP * (bottom - top + 1)) / m.shape[0]


def background_line_clause(out: Path, need: set[str]) -> Clause:
    """Does a vertical edge of the set run into a head from above?"""
    from PIL import Image
    import numpy as np

    name = "head_clear_of_background_lines"
    set_img = out.parent / (out.name + ".set.png")
    if not set_img.is_file():
        return Clause(name, None, detail="no set pass on this build; rebuild "
                                          "the greybox to measure")
    g = np.asarray(Image.open(set_img).convert("L")).astype(np.int16)
    H = g.shape[0]
    edge = np.zeros(g.shape, dtype=bool)
    edge[:, 1:] = np.abs(g[:, 1:] - g[:, :-1]) > LINE_EDGE_STEP
    hits, measured = [], 0
    for cid in sorted(need):
        head = _mask(out.parent / (out.name + f".head_{cid}.png"))
        if head is None or not head.any():
            continue
        vis = _mask_l(out.parent / (out.name + f".visible_{cid}.png"))
        if vis is not None and (vis & head).sum() < MIN_READ_POINT_VISIBLE * head.sum():
            continue                  # a hidden head is read_point_visible's failure
        ys, xs = np.nonzero(head)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        ya = max(0, y0 - int(LINE_ABOVE_CROWN * H))
        yb = min(H, y0 + int(0.25 * (y1 - y0 + 1)))
        if yb - ya < 4:
            continue
        measured += 1
        inset = int(0.15 * (x1 - x0 + 1))
        band = edge[ya:yb]
        # A line may wander by a pixel: any of three neighbouring columns.
        cover = max((float(band[:, max(0, x - 1):x + 2].any(axis=1).mean())
                     for x in range(x0 + inset, x1 - inset + 1)), default=0.0)
        if cover >= LINE_MIN_COVER:
            hits.append(f"{cid} ({cover:.0%})")
    if not measured:
        return Clause(name, None, detail="no visible head to measure against the set")
    return Clause(name, not hits, value=round(1 - len(hits) / measured, 4),
                  threshold=LINE_MIN_COVER,
                  detail=("a vertical line of the set runs into the head: "
                          + ", ".join(hits)) if hits else "")


def joint_cut_clause(out: Path, need: set[str]) -> Clause:
    """Does a frame edge a body runs off cross it at a joint?"""
    name = "frame_not_cut_at_joint"
    jf = out.parent / (out.name + ".joints.json")
    if not jf.is_file():
        return Clause(name, None, detail="no joint projection on this build; "
                                          "rebuild the greybox to measure")
    joints = json.loads(jf.read_text())
    cuts, measured = [], 0
    for cid in sorted(need):
        js = joints.get(cid)
        m = _mask(out.parent / (out.name + f".body_{cid}.png"))
        if not js or m is None or not m.any():
            continue
        measured += 1
        runs_off = {"bottom": bool(m[-1].any()), "top": bool(m[0].any()),
                    "left": bool(m[:, 0].any()), "right": bool(m[:, -1].any())}
        for joint, (x, y, z) in js.items():
            kind = joint.split("_")[0]
            if kind not in CUT_JOINTS or z <= 0:
                continue
            for edge, off in runs_off.items():
                if not off:
                    continue
                if edge in ("bottom", "top"):
                    dist, along = (abs(y - 1.0) if edge == "bottom" else abs(y)), x
                else:
                    dist, along = (abs(x) if edge == "left" else abs(x - 1.0)), y
                if dist <= JOINT_CUT_TOL and 0.0 <= along <= 1.0:
                    cut = f"the {edge} edge cuts {cid} at the {kind}"
                    if cut not in cuts:
                        cuts.append(cut)
    if not measured:
        return Clause(name, None, detail="no body with both a matte and joints")
    return Clause(name, not cuts, value=float(len(cuts)), threshold=0.0,
                  detail="; ".join(cuts))


def tangency_clause(out: Path, need: set[str]) -> Clause:
    """Do two outlines, or an outline and the frame, just touch?"""
    import numpy as np

    name = "silhouettes_not_tangent"
    masks = {}
    for cid in sorted(need):
        m = _mask(out.parent / (out.name + f".body_{cid}.png"))
        if m is not None and m.any():
            masks[cid] = m
    if not masks:
        return Clause(name, None, detail="no readable body matte")
    W = next(iter(masks.values())).shape[1]
    r = max(2, int(round(TANGENT_GAP * W)))
    found = []
    ids = sorted(masks)
    grown = {c: _dilate(masks[c], r) for c in ids} if len(ids) > 1 else {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = masks[ids[i]], masks[ids[j]]
            inter = int((a & b).sum())
            if (grown[ids[i]] & b).any() and \
                    inter <= TANGENT_MAX_OVERLAP * min(int(a.sum()), int(b.sum())):
                found.append(f"{ids[i]} and {ids[j]} meet at their outlines")
    # The bottom edge is left out: a body running off the bottom is the shot
    # size working, and Section 6.4.3's containment measure counts it so.
    for cid, m in masks.items():
        rows = np.nonzero(m.any(axis=1))[0]
        cols = np.nonzero(m.any(axis=0))[0]
        for edge, gap, contact in (("top", int(rows[0]), int(m[0].sum())),
                                   ("left", int(cols[0]), int(m[:, 0].sum())),
                                   ("right", W - 1 - int(cols[-1]), int(m[:, -1].sum()))):
            if 0 < gap <= r or (gap == 0 and 0 < contact < 2 * r):
                found.append(f"{cid} grazes the {edge} edge")
    return Clause(name, not found, value=float(len(found)), threshold=0.0,
                  detail="; ".join(found))


def lineup_clause(out: Path, need: set[str]) -> Clause:
    """Do three or more heads stand at one height and one depth?"""
    import numpy as np

    name = "cast_not_in_a_row"
    heads = {}
    for cid in sorted(need):
        m = _mask(out.parent / (out.name + f".head_{cid}.png"))
        if m is not None and m.any():
            heads[cid] = m
    if len(heads) < 3:
        return Clause(name, None, detail="fewer than three heads; a row needs three")
    H = next(iter(heads.values())).shape[0]
    cy = [float(np.nonzero(m)[0].mean()) / H for m in heads.values()]
    size = [int(m.sum()) for m in heads.values()]
    spread = max(cy) - min(cy)
    ratio = max(size) / max(min(size), 1)
    lined = spread <= LINEUP_MAX_Y_SPREAD and ratio <= LINEUP_MAX_SIZE_RATIO
    return Clause(name, not lined, value=round(spread, 4), threshold=LINEUP_MAX_Y_SPREAD,
                  detail=(f"{len(heads)} heads within {spread:.1%} of the frame "
                          f"height at sizes within {ratio:.2f}x") if lined else "")


def eye_line_clause(out: Path, need: set[str],
                    reverse_shots: list[dict] | None) -> Clause:
    """Do the singles of one shot / reverse-shot exchange hold the eye line?"""
    name = "reverse_shot_eye_lines_match"
    if not reverse_shots:
        return Clause(name, None, detail="not a single with a reverse shot built")
    own = [e for e in (eye_line(out, c) for c in sorted(need)) if e is not None]
    diffs = [(r["panel_id"], abs(own[0] - r["eye_y"])) for r in reverse_shots
             if r.get("eye_y") is not None] if len(own) == 1 else []
    if not diffs:
        return Clause(name, None, detail="no eye line to compare")
    pid, worst = max(diffs, key=lambda d: d[1])
    ok = worst <= EYE_LINE_MAX_DIFF
    return Clause(name, ok, value=round(worst, 4), threshold=EYE_LINE_MAX_DIFF,
                  detail="" if ok else f"eye line {worst:.1%} of the frame height "
                                        f"away from {pid}")


def declared_in_frame_clause(out: Path, declared: dict | None) -> Clause:
    """Does what the panel declares in frame (入画) match what was staged?

    `declared` maps ("prop" | "subject", id) to yes / partial / no / tbd. A
    prop is read off the build's own record (.props.json), a subject off its
    matte. "tbd" -- the source does not say -- is named in the detail and is
    never counted as a pass.
    """
    name = "declared_in_frame_matches_staging"
    if not declared:
        return Clause(name, None, detail="the panel declares nothing about what is in "
                                         "frame; the compiler reads the build's record instead")
    try:
        rec = json.loads((out.parent / (out.name + ".props.json")).read_text())
    except (OSError, ValueError):
        rec = {}
    bad, tbd, measured = [], [], 0
    for (kind, ident), want in sorted(declared.items()):
        if want == "tbd":
            tbd.append(ident)
            continue
        if kind == "prop":
            r = rec.get(ident)
            if not isinstance(r, dict):
                continue
            staged = ("no" if not r.get("in_frame")
                      else "partial" if (r.get("shown") or 0.0) < PROP_WHOLE_SHOWN else "yes")
            match = staged == want or (want == "yes" and staged == "partial")
        else:
            st = matte_stats(out.parent / (out.name + f".body_{ident}.png"))
            if st is None or st.get("no_alpha"):
                continue
            staged = "yes" if (st.get("area") or 0.0) > 0 else "no"
            match = (staged == "no") == (want == "no")
        measured += 1
        if not match:
            bad.append(f"{ident} declared {want}, staged {staged}")
    note = [f"to be confirmed: {', '.join(tbd)}"] if tbd else []
    if not measured:
        return Clause(name, None, detail="; ".join(note) or
                      "no declared entity has a staging record to compare")
    return Clause(name, not bad, value=round(1 - len(bad) / measured, 4), threshold=1.0,
                  detail="; ".join(bad + note))


def evaluate(spec: dict, out: str | Path, *,
             required: set[str] | None = None,
             min_area: float = MIN_SUBJECT_AREA,
             reverse_shots: list[dict] | None = None,
             declared_in_frame: dict | None = None) -> GateReport:
    """Run the gate on one built greybox.

    `spec` is what `panel_greybox.build_spec` produced; `out` is the beauty
    frame it was rendered to. `required` names the entities that must be
    visible -- from a Beat Visual IR's required facts when there is one.
    Defaults to every subject the spec declares, which is the honest default:
    the geometry staged them, so it is asserting they are there.
    """
    from pace_core.node.panel_greybox import anchor_version

    out = Path(out)
    subs = spec.get("subjects") or []
    declared = [(s.get("character_id"), s.get("declared_x")) for s in subs]
    need = required if required is not None else {c for c, _ in declared if c}

    stats: dict[str, dict | None] = {}
    for cid, _ in declared:
        if cid:
            stats[cid] = matte_stats(out.parent / (out.name + f".body_{cid}.png"))

    clauses: list[Clause] = []

    # 1. required_entity_visibility
    absent = [c for c in sorted(need) if stats.get(c) is None]
    broken = [c for c in sorted(need) if (stats.get(c) or {}).get("no_alpha")]
    empty = [c for c in sorted(need)
             if (stats.get(c) or {}).get("area") == 0.0]
    seen = len(need) - len(set(absent) | set(broken) | set(empty))
    vis = seen / len(need) if need else 1.0
    why = []
    if absent:
        why.append(f"no matte: {', '.join(absent)}")
    if broken:
        why.append(f"matte has no alpha (film_transparent lost): {', '.join(broken)}")
    if empty:
        why.append(f"matte empty, body not in frame: {', '.join(empty)}")
    clauses.append(Clause("required_entity_visibility", vis == 1.0,
                          value=round(vis, 4), threshold=1.0,
                          detail="; ".join(why)))

    # 2. projected_subject_area
    areas = {c: (stats.get(c) or {}).get("area") for c in sorted(need)}
    measured = {c: a for c, a in areas.items() if a is not None}
    if measured:
        worst_id = min(measured, key=lambda c: measured[c])
        small = [f"{c} {measured[c]:.4f}" for c in sorted(measured)
                 if measured[c] < min_area]
        clauses.append(Clause(
            "projected_subject_area", not small,
            value=round(measured[worst_id], 5), threshold=min_area,
            detail=(f"below floor: {', '.join(small)}" if small
                    else f"smallest is {worst_id}")))
    else:
        clauses.append(Clause("projected_subject_area", None,
                              detail="no readable matte to measure"))

    # 3. critical_relation_satisfaction -> the one relation we declare
    centroids = {c: (stats.get(c) or {}).get("centroid_x")
                 for c in stats if stats.get(c)}
    centroids = {c: v for c, v in centroids.items() if v is not None}
    if len(centroids) >= 2:
        frac, bad = screen_order_agreement(declared, centroids)
        clauses.append(Clause("screen_order", frac == 1.0,
                              value=round(frac, 4), threshold=1.0,
                              detail="; ".join(bad)))
    else:
        clauses.append(Clause(
            "screen_order", None,
            detail="fewer than two bodies with a measured position; "
                   "a single subject declares no left-to-right order"))

    # 4. subject_occlusion -- the design's "key actor not severely occluded"
    masks = {}
    for cid in sorted(need):
        f = out.parent / (out.name + f".body_{cid}.png")
        if f.is_file():
            m = _mask(f)
            if m is not None:
                masks[cid] = m
    worst, pairs = 0.0, []
    ids = sorted(masks)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = masks[ids[i]], masks[ids[j]]
            inter = int((a & b).sum())
            if not inter:
                continue
            for who, other, frac in ((ids[i], ids[j], inter / max(int(a.sum()), 1)),
                                     (ids[j], ids[i], inter / max(int(b.sum()), 1))):
                if frac > worst:
                    worst = frac
                if frac > MAX_SUBJECT_OVERLAP:
                    pairs.append(f"{frac:.0%} of {who} is under {other}")
    if len(masks) >= 2:
        clauses.append(Clause("subject_occlusion", not pairs,
                              value=round(worst, 4), threshold=MAX_SUBJECT_OVERLAP,
                              detail="; ".join(pairs)))
    else:
        clauses.append(Clause(
            "subject_occlusion", None,
            detail="fewer than two bodies with a matte; one subject cannot "
                   "occlude itself"))

    # 5. read_point_visible -- can the set be seen past?
    hidden = []
    measured = 0
    for cid in sorted(need):
        vis = _mask_l(out.parent / (out.name + f".visible_{cid}.png"))
        head = _mask(out.parent / (out.name + f".head_{cid}.png"))
        if vis is None or head is None or not head.any():
            continue
        measured += 1
        frac = float((vis & head).sum()) / float(head.sum())
        if frac < MIN_READ_POINT_VISIBLE:
            hidden.append(f"{cid} {frac:.0%}")
    if measured:
        clauses.append(Clause("read_point_visible", not hidden,
                              value=round(1 - len(hidden) / measured, 4),
                              threshold=MIN_READ_POINT_VISIBLE,
                              detail=("head behind the set: " + ", ".join(hidden))
                                     if hidden else ""))
    else:
        clauses.append(Clause(
            "read_point_visible", None,
            detail="no visible-matte pass on this build; rebuild the greybox "
                   "to measure whether the set stands in front of a head"))

    # 6. control_versions_are_consistent
    clauses.append(control_provenance(out))

    # 7. focal_action_readability -- see the module docstring.
    clauses.append(Clause(
        "focal_action_readability", None,
        detail="no rig and no contact points on a static proxy; nothing to "
               "measure the legibility of an ACTION against. Whether the "
               "subject can be seen at all is measured, separately, by "
               "read_point_visible"))

    # 8-12. composition -- see the constants above.
    clauses.append(background_line_clause(out, need))
    clauses.append(joint_cut_clause(out, need))
    clauses.append(tangency_clause(out, need))
    clauses.append(lineup_clause(out, need))
    clauses.append(eye_line_clause(out, need, reverse_shots))

    # 13. what the panel declares in frame, against what was staged
    clauses.append(declared_in_frame_clause(out, declared_in_frame))

    return GateReport(panel_id=spec.get("panel_id") or out.stem,
                      anchor_version=anchor_version(spec), clauses=clauses)


def find_built(project: str, panel_id: str) -> Path | None:
    """The beauty frame a panel's greybox was built to, if there is one.

    Matched by glob on the panel id rather than by reconstructing the studio
    router's filename convention: the naming belongs to the caller that built
    it, and a checker that hardcodes a second copy of it goes stale silently
    the first time that convention moves.

    The beauty frame is the one with no derived suffix. Everything the build
    and the style experiments write beside it -- `.depth.png`, `.body_x.png`,
    `.flat12.png`, `.toon16.png` -- inserts another dot, so counting dots
    separates them. Counting `.png` does not: `.flat12.png` contains exactly
    one, and picking it made the gate look for mattes named after a frame
    nothing had rendered mattes for, then report three present bodies as
    missing.
    """
    from pace_core.paths import paths_for

    d = Path(paths_for(project).greyboxes_dir)
    hits = [p for p in sorted(d.glob(f"{panel_id}_*.png"))
            if p.name.count(".") == 1]
    return hits[0] if hits else None


def gate_panel(project: str, scene_id: str, panel_id: str,
               out: str | Path) -> GateReport:
    """Gate one panel's build, resolving its spec from the KB.

    The spec is rebuilt rather than threaded through from the build that made
    the frame: `build_spec` is a pure read of the same KB documents, so it
    returns what was staged, and taking it as an argument would make every
    caller carry a 40-field dict to ask a yes/no question.
    """
    from pace_core.node.panel_greybox import build_spec

    spec = build_spec(project, scene_id, panel_id, out)
    spec.setdefault("panel_id", panel_id)
    return evaluate(spec, out, reverse_shots=reverse_shot_partners(
        project, scene_id, panel_id, spec),
        declared_in_frame=declared_in_frame_of(project, scene_id, panel_id))


def reverse_shot_partners(project: str, scene_id: str, panel_id: str,
                          spec: dict) -> list[dict]:
    """The other singles of this panel's shot, with their built eye lines.

    A shot split into one close-up per subject is a shot / reverse-shot
    exchange, and its panels are cut together, so the eye line has to hold
    across them. Only singles of a different subject count: a later frame of
    the same subject is not its reverse.
    """
    from pace_core.pai_compat import resolve_shot
    from pace_core.paths import paths_for

    subs = spec.get("subjects") or []
    if len(subs) != 1:
        return []
    me = subs[0].get("character_id")
    try:
        scene = json.loads((Path(paths_for(project).scenes_dir)
                            / f"{scene_id}.json").read_text())
    except (OSError, ValueError):
        return []
    shot = next((sh for sh in scene.get("shots") or []
                 if any(p.get("id") == panel_id for p in sh.get("panels") or [])), None)
    if shot is None:
        return []
    out = []
    for pl in shot.get("panels") or []:
        if pl.get("id") == panel_id:
            continue
        ss = (resolve_shot(scene, shot, pl).get("setup") or {}).get("subjects") or []
        if len(ss) != 1 or ss[0].get("character_id") == me:
            continue
        built = find_built(project, pl["id"])
        if built is None:
            continue
        cid = ss[0].get("character_id")
        out.append({"panel_id": pl["id"], "character_id": cid,
                    "eye_y": eye_line(built, cid)})
    return out


def declared_in_frame_of(project: str, scene_id: str, panel_id: str) -> dict:
    """The panel's in-frame declarations, keyed ("prop" | "subject", id)."""
    from pace_core.pai_compat import declared_in_frame, resolve_shot
    from pace_core.paths import paths_for

    try:
        scene = json.loads((Path(paths_for(project).scenes_dir)
                            / f"{scene_id}.json").read_text())
    except (OSError, ValueError):
        return {}
    for sh in scene.get("shots") or []:
        for pl in sh.get("panels") or []:
            if pl.get("id") != panel_id:
                continue
            setup = resolve_shot(scene, sh, pl).get("setup") or {}
            out = {}
            for kind, key, idk in (("subject", "subjects", "character_id"),
                                   ("prop", "props", "prop_id")):
                for e in setup.get(key) or []:
                    v = declared_in_frame(e) if isinstance(e, dict) else None
                    if v and e.get(idk):
                        out[(kind, e[idk])] = v
            return out
    return {}


def gate_project(project: str, scene_id: str | None = None) -> list[GateReport]:
    """Run the gate over every built greybox in scope.

    Panels the geometry cannot stage at all are skipped rather than failed:
    `eligible_panels` already says why those have no greybox, and reporting
    them again as gate failures would drown the panels that DO have one and
    are wrong.
    """
    from pace_core.node.panel_greybox import eligible_panels

    out: list[GateReport] = []
    for row in eligible_panels(project, scene_id):
        if not row.get("eligible"):
            continue
        built = find_built(project, row["panel_id"])
        if built is None:
            continue
        out.append(gate_panel(project, row["scene_id"], row["panel_id"], built))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--scene", default=None, help="limit to one scene id")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when any panel fails the gate")
    a = ap.parse_args(argv)

    reports = gate_project(a.project, a.scene)
    if a.json:
        print(json.dumps([r.as_dict() for r in reports], indent=1))
    else:
        failed = [r for r in reports if not r.ok]
        for r in failed:
            print(f"{r.panel_id}  anchor={r.anchor_version}")
            for c in r.failures:
                v = ("" if c.value is None or c.threshold is None
                     else f" ({c.value} vs {c.threshold})")
                print(f"    {c.name}{v}: {c.detail}")
        print(f"\n{len(reports) - len(failed)}/{len(reports)} panels pass the gate")
        by_clause: dict[str, int] = {}
        for r in reports:
            for c in r.failures:
                by_clause[c.name] = by_clause.get(c.name, 0) + 1
        for name, n in sorted(by_clause.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3} failed {name}")
        unmeasured: dict[str, int] = {}
        for r in reports:
            for name in r.unmeasured:
                unmeasured[name] = unmeasured.get(name, 0) + 1
        for name, n in sorted(unmeasured.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3} could not measure {name}")
    return 1 if (a.strict and any(not r.ok for r in reports)) else 0


if __name__ == "__main__":
    sys.exit(main())
