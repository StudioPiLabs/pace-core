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
# choices are rather than scored as a bare mismatch.
SIZE_ORDER: tuple[str, ...] = (
    "extreme_close_up", "close_up", "medium_close_up", "medium",
    "medium_full", "full", "wide", "establishing", "master",
)


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


def _credit(ok: Optional[bool], value: Optional[float],
            threshold: Optional[float]) -> float:
    """One clause's credit. A clause that reports a measured quantity and its
    own tolerance is graded against that tolerance; otherwise it is its verdict."""
    if value is not None and threshold not in (None, 0):
        return max(0.0, 1.0 - abs(value) / abs(threshold))
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
                        getattr(c, "threshold", None)) for c in kept)
    return total / len(kept), tuple(c.name for c in kept)


def _size_distance(a: Optional[str], b: Optional[str]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    try:
        return abs(SIZE_ORDER.index(a) - SIZE_ORDER.index(b)) / (len(SIZE_ORDER) - 1)
    except ValueError:
        return 1.0


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
