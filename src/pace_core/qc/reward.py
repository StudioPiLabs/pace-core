"""The reward a policy is trained against, and why it needs two terms.

The gate measures whether a frame stages what its declaration asked for. That
makes it a verifier, and a verifier is not yet a reward: measured against
graded perturbations of real declarations, every weighting of the gate's own
clauses turns out to be non-monotone in how bad the staging choice was, and
several of them *rise* as the choice gets worse. The reason is structural
rather than a matter of weights. A worse declaration is re-solved into a
different, self-consistent frame, and a self-consistency check cannot see that
the frame it is looking at came from a worse decision than the one before.

So conformance is the floor and not the signal. This module keeps the two
apart:

  `conformance`  what the gate measures, over the clauses that both vary and
                 can be measured on this panel. Bounded, comparable across
                 panels, and undefined -- not zero -- where nothing is
                 measurable.
  `agreement`    how far the declaration is from the one a person made for
                 this beat: shot size, framing, and who is in frame. This is
                 the term the gate cannot supply, and it is what makes the
                 reward monotone in the quality of the decision.

Neither alone is the reward. `reward()` combines them and reports the parts,
because a scalar that hides which half moved is not debuggable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

# Clauses that were observed to vary across graded perturbations of real
# declarations. Only these carry gradient; the rest are constants or are never
# measurable, and including them dilutes the signal without adding to it.
DISCRIMINATIVE: frozenset[str] = frozenset({
    "frame_not_cut_at_joint",
    "subject_occlusion",
    "screen_order",
    "cast_not_in_a_row",
    "reverse_shot_eye_lines_match",
    # Included by construction rather than by that measurement: it reports a
    # residual in metres against its own tolerance, so it grades rather than
    # passing or failing, and it is undefined on a panel where nobody declared
    # a mark -- which is most of them -- so it cannot dilute a group it has
    # nothing to say about. It has not yet been through the perturbation test
    # the other five passed, because the staging it measures only became
    # observable with it.
    "subject_on_declared_mark",
})

# Clauses that passed on every panel of the corpus. They stay in the gate as
# constraint checks -- a failure is worth investigating -- but a constant
# contributes no gradient, and weighting one above the rest measurably reduced
# the within-group spread the objective depends on.
CONSTANT: frozenset[str] = frozenset({
    "required_entity_visibility",
    "projected_subject_area",
    "read_point_visible",
    "head_clear_of_background_lines",
    "silhouettes_not_tangent",
})

# Shot sizes, ordered, so disagreement can be graded by how far apart two
# choices are rather than scored as a bare mismatch. Kept for the fallback
# path; the distance actually used is geometric -- see `_size_distance`.
SIZE_ORDER: tuple[str, ...] = (
    "extreme_close_up", "close_up", "medium_close_up", "medium",
    "medium_full", "full", "wide", "establishing", "master",
)

# Two shot sizes that resolve to the same camera distance are the same framing
# under two names. Scoring that as a perfect match would make a wrong answer
# indistinguishable from the right one, so it costs this much instead: enough
# to break the tie, little enough to say the error is small.
SAME_DISTANCE_PENALTY: float = 0.02


@dataclass
class Parts:
    """The reward and the pieces it is made of."""
    reward: Optional[float]
    conformance: Optional[float]
    agreement: Optional[float]
    measured_clauses: tuple[str, ...] = ()
    rejected: bool = False
    note: str = ""

    @property
    def scorable(self) -> bool:
        return self.reward is not None


# Clauses whose residual has no natural ceiling. A screen-position error is
# bounded by the frame, so linear credit that reaches zero at the tolerance
# loses nothing past it. A floor mark is in metres and can be wrong by any
# amount, and there the linear form has a dead zone: everything past the
# tolerance scores exactly zero, so half a metre off and thirty metres off are
# the same reward and nothing drives the policy back. These decay instead.
UNBOUNDED_RESIDUAL: frozenset[str] = frozenset({"subject_on_declared_mark"})


def _credit(ok: Optional[bool], value: Optional[float],
            threshold: Optional[float], name: str = "") -> float:
    """One clause's credit. A clause that reports a measured quantity and its
    own tolerance is graded against that tolerance; otherwise it is its verdict.

    Two gradings, because two kinds of residual. The linear one reaches zero at
    the tolerance and stays there, which is right for a quantity the frame
    already bounds. The decaying one, `1 / (1 + (v/t)^2)`, is 1 on the mark,
    0.5 at the tolerance and never quite zero, which is what a distance in
    metres needs: it keeps a gradient at any error instead of giving up past
    one pace. Only clauses in `UNBOUNDED_RESIDUAL` take it, so every figure
    measured under the linear form stands.
    """
    if value is not None and threshold not in (None, 0):
        ratio = abs(value) / abs(threshold)
        if name in UNBOUNDED_RESIDUAL:
            return 1.0 / (1.0 + ratio * ratio)
        return max(0.0, 1.0 - ratio)
    return 1.0 if ok else 0.0


def conformance(clauses: Iterable) -> tuple[Optional[float], tuple[str, ...]]:
    """What the gate measures, over the clauses that can carry a gradient.

    Returns (score, names). The score is None when no discriminative clause was
    measurable on this panel -- a single, for instance, cannot be measured for
    occlusion or screen order. That is a property of the shot, not of the
    policy's output, so scoring it zero would be a penalty the policy cannot
    avoid and the group-relative baseline would turn into a spurious negative
    advantage. The caller drops such a rollout from the group instead.
    """
    kept = [c for c in clauses
            if getattr(c, "name", None) in DISCRIMINATIVE
            and getattr(c, "ok", None) is not None]
    if not kept:
        return None, ()
    total = sum(_credit(c.ok, getattr(c, "value", None),
                        getattr(c, "threshold", None),
                        getattr(c, "name", "")) for c in kept)
    return total / len(kept), tuple(c.name for c in kept)


def _size_distance(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """How far apart two shot sizes are, in [0, 1].

    Measured as a ratio of subject distances rather than as a count of steps
    along the name list, for two reasons that an ordinal scale gets wrong.
    The steps are not equal: 0.4 m to 0.7 m is a far larger change in what the
    frame holds than 3 m to 5 m is, and the names do not even agree on their
    own order -- `SIZE_ORDER` puts `master` at the wide end while the camera
    planner places it at the same distance as `full`, which is where a master
    actually sits. Both follow from the fact that a shot size is a word for a
    camera distance, so the distance is what to compare.

    A ratio, in logarithm, because framing changes multiplicatively: halving
    the distance fills twice the frame whether it starts at 1 m or at 10 m.

    The table is the camera planner's own, imported rather than restated, so
    that what the reward calls a size difference is what the renderer would
    actually do differently.
    """
    if a is None or b is None:
        return None
    if a == b:
        return 0.0

    from pace_core.camera.camera_planner import SHOT_SIZE_DISTANCE_M as TABLE

    da, db = TABLE.get(a), TABLE.get(b)
    if da is None or db is None:
        # A size the planner does not know: fall back to the name order, and
        # to a full mismatch if it is not even a name we recognise.
        try:
            return abs(SIZE_ORDER.index(a) - SIZE_ORDER.index(b)) / (len(SIZE_ORDER) - 1)
        except ValueError:
            return 1.0

    known = [TABLE[k] for k in SIZE_ORDER if k in TABLE]
    span = math.log(max(known) / min(known))
    d = abs(math.log(da) - math.log(db)) / span
    # Clamped: the widest pair divides out to 1 in exact arithmetic and to a
    # shade over it in floating point, which would make a credit of 0 come
    # back very slightly negative.
    return min(1.0, max(d, SAME_DISTANCE_PENALTY))


def agreement(proposed: dict, human: dict) -> Optional[float]:
    """How close a proposed declaration is to the one a person made.

    Three comparable decisions, because they are the three a breakdown records
    and a renderer cannot infer: how much of the subject the frame holds, how
    the frame is organised around them, and who is in it. Each is scored only
    when the human record states it, so a breakdown that left a cell blank
    neither rewards nor penalises.

    Returns None when the human record states none of the three.
    """
    scores: list[float] = []

    d = _size_distance(proposed.get("shot_size"), human.get("shot_size"))
    if d is not None:
        scores.append(1.0 - d)

    if human.get("framing") is not None:
        scores.append(1.0 if proposed.get("framing") == human["framing"] else 0.0)

    want = human.get("cast")
    if want:
        got = set(proposed.get("cast") or ())
        want = set(want)
        union = want | got
        scores.append(len(want & got) / len(union) if union else 1.0)

    return sum(scores) / len(scores) if scores else None


def reward(clauses: Iterable, proposed: Optional[dict] = None,
           human: Optional[dict] = None, *,
           beta: float = 0.5, penalty: float = 1.0,
           rejected: bool = False) -> Parts:
    """Conformance, plus agreement with the human decision where one exists.

    `beta` is the weight on agreement. It is not a tuning knob to be swept
    blindly: conformance alone was measured non-monotone in decision quality,
    so beta = 0 reproduces a reward that rises as the staging choice gets
    worse. A positive beta is what makes the objective order the decisions.

    `rejected` is for a declaration the schema refuses -- declaring a subject
    both static and moving, say. That is a flat penalty rather than a low
    score, because it is not a worse decision but an unreadable one.
    """
    conf, names = conformance(clauses)
    agr = agreement(proposed, human) if (proposed and human) else None

    if rejected:
        return Parts(-penalty, conf, agr, names, True,
                     "declaration rejected by the schema")
    if conf is None and agr is None:
        return Parts(None, None, None, names, False,
                     "no discriminative clause was measurable and no human "
                     "record to compare against; drop from the group")
    if agr is None:
        return Parts(conf, conf, None, names, False,
                     "no human record for this beat; conformance only, which "
                     "is known not to order decisions on its own")
    if conf is None:
        return Parts(beta * agr, None, agr, names, False,
                     "nothing measurable in the frame; agreement only")
    return Parts((conf + beta * agr) / (1.0 + beta), conf, agr, names)
