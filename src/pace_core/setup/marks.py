"""Where each subject actually stands, and whether anyone said so.

`Subject.blocking.world_xy` is the floor mark a breakdown states, and until
now nothing read it: the scene builder spaced bodies evenly on a circle and
the composition solver aimed the camera at a separate table of template slots.
Both were plausible in isolation and they did not agree with each other, so a
frame could be composed against a position the render never used.

This module is the one place that answers the question. It prefers the
declared mark, falls back to whatever layout the caller would otherwise have
used, and reports which happened per subject, so a consumer -- a gate clause,
a report -- can tell a staged mark from an invented one rather than having to
assume.

Pure functions, no I/O and no Blender, so the scene builder and the camera
planner can both call it and provably agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Mark:
    """One subject's floor position, and where it came from."""
    xy: tuple[float, float]
    z: float = 0.0
    facing_deg: Optional[float] = None
    declared: bool = False          # True when a document stated this mark

    def as_xyz(self) -> tuple[float, float, float]:
        return (self.xy[0], self.xy[1], self.z)


def _xy_of(value: object) -> Optional[tuple[float, float]]:
    """A two-number coordinate, or None for anything else.

    Written to reject rather than coerce: a mark with one element, three, or a
    string in it is a document error, and silently taking the first two
    numbers would place a body somewhere nobody asked for.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def declared_marks(scene_doc: dict) -> dict[str, Mark]:
    """Every character's declared mark, from the first shot that states one.

    A scene is built once but blocking is declared per shot, so the first
    declaration wins -- the same rule the pose lookup already uses. A
    character no shot gives a mark for is absent from the result, not present
    with a default, so the caller can tell silence from a stated origin.
    """
    out: dict[str, Mark] = {}
    for shot in scene_doc.get("shots") or []:
        for sub in ((shot.get("setup") or {}).get("subjects") or []):
            cid = sub.get("character_id")
            blocking = sub.get("blocking") or {}
            if not cid or cid in out:
                continue
            xy = _xy_of(blocking.get("world_xy") or blocking.get("worldXy"))
            if xy is None:
                continue
            z = blocking.get("z")
            facing = blocking.get("facing_deg")
            if facing is None:
                facing = blocking.get("facingDeg")
            out[cid] = Mark(xy=xy, z=float(z) if z is not None else 0.0,
                            facing_deg=float(facing) if facing is not None
                            else None, declared=True)
    return out


def resolve_positions(character_ids: Sequence[str],
                      declared: dict[str, Mark],
                      fallback: Iterable[tuple[float, float, float]],
                      fallback_facing: Optional[Sequence[float]] = None,
                      ) -> list[Mark]:
    """One Mark per character, declared where stated and laid out otherwise.

    `fallback` supplies a position per character in the same order, for the
    subjects nobody placed; it is the caller's existing layout, passed in
    rather than reimplemented here, because what a sensible default looks
    like depends on the set and this module should not have opinions about
    sets.

    A partially declared scene resolves per subject: one character on a stated
    mark and another on the fallback is the normal case while a corpus is
    being filled in, and mixing them is correct.
    """
    laid_out = list(fallback)
    facings = list(fallback_facing or ())
    out: list[Mark] = []
    for i, cid in enumerate(character_ids):
        got = declared.get(cid)
        if got is not None:
            out.append(got)
            continue
        if i < len(laid_out):
            x, y, z = laid_out[i]
        else:
            x, y, z = 0.0, 0.0, 0.0
        out.append(Mark(xy=(x, y), z=z,
                        facing_deg=facings[i] if i < len(facings) else None,
                        declared=False))
    return out


def mark_error_m(staged_xy: Sequence[float],
                 declared_xy: Sequence[float]) -> float:
    """Distance between where a subject was staged and where it was declared.

    In metres on the floor plane, which is the unit the declaration is in and
    the unit a director would argue about. Section "Annotating a floor
    position" validates the same comparison against films that record the
    answer; this is that comparison run on our own renders.
    """
    dx = float(staged_xy[0]) - float(declared_xy[0])
    dy = float(staged_xy[1]) - float(declared_xy[1])
    return (dx * dx + dy * dy) ** 0.5
