"""Pass 0: a screenplay into typed elements that still know where they came from.

The existing splitter (`split_script.py`) hands the whole script to an LLM and
gets scene boundaries back. That works, and it discards the one thing a
verifier needs: the offset. Nothing downstream can answer "which words in the
screenplay support this field", so a value invented by a lookup table is
indistinguishable from a value read off the page -- which is how every
`screen_position.x` in the corpus came to be a constant selected by list index
without anyone noticing.

This pass is deterministic and free. It does not interpret; it segments, and
every element it emits carries `(start, end)` into the exact text it parsed,
so an evidence span can be checked by slicing the source rather than by
trusting a model that it exists.

Grammar is the usual American screenplay one, with the caveat that text
extracted from a PDF has lost its indentation, so cues are recognised by case
and context rather than by column:

    scene_heading   INT./EXT./EST. ... - DAY
    transition      FADE IN:, CUT TO:, DISSOLVE TO:, FADE OUT.
    super           SUPER: 2035
    character       an all-caps short line that is followed by speech
    parenthetical   (singing), (to the image of Theo)
    dialogue        the lines under a character cue
    action          everything else
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Iterator

SCENE_RE = re.compile(r"^(INT\.?/EXT\.?|EXT\.?/INT\.?|INT\.?|EXT\.?|EST\.?)[\s.]", re.I)
TRANSITION_RE = re.compile(r"^(FADE (IN|OUT)|CUT TO|DISSOLVE TO|SMASH CUT|MATCH CUT|BACK TO)\b.*[:.]?$", re.I)
SUPER_RE = re.compile(r"^(SUPER|TITLE|CAPTION)\s*:", re.I)
PAREN_RE = re.compile(r"^\(.*\)?$")
# A page number the PDF left behind: "2." or "12" alone on a line.
PAGE_RE = re.compile(r"^\d{1,3}\.?$")
# (CONT'D), (V.O.), (O.S.), (to the image of Theo) trailing a cue.
CUE_SUFFIX_RE = re.compile(r"\s*\((CONT'?D|V\.?O\.?|O\.?S\.?|O\.?C\.?)\)\s*$", re.I)


@dataclass
class Element:
    kind: str                 # scene_heading|action|character|parenthetical|dialogue|transition|super
    text: str
    start: int                # character offset into the parsed source
    end: int
    scene_index: int | None = None   # which scene heading governs this element
    speaker: str | None = None       # for parenthetical/dialogue: whose

    def as_dict(self) -> dict:
        return asdict(self)


def _is_cue(line: str) -> bool:
    """An all-caps line short enough to be a name.

    Length matters: an action line can be shouted in caps, but a cue is a
    name, so anything long is prose. The check runs on the cue with its
    (CONT'D)/(V.O.) suffix removed, since those are lower-case-bearing.
    """
    s = CUE_SUFFIX_RE.sub("", line).strip()
    if not s or len(s) > 38:
        return False
    if not re.search(r"[A-Z一-鿿]", s):
        return False
    # Allow a name plus a parenthetical age note: "NINA and OMAR(50's)"
    core = re.sub(r"\([^)]*\)", "", s)
    letters = [c for c in core if c.isalpha()]
    if not letters:
        return False
    if not all(c.isupper() for c in letters if c.isascii()):
        return False
    return not s.endswith((".", ",", ";"))


def _lines_with_offsets(text: str) -> Iterator[tuple[str, int, int]]:
    at = 0
    for raw in text.split("\n"):
        yield raw, at, at + len(raw)
        at += len(raw) + 1


def parse(text: str) -> list[Element]:
    """Segment a screenplay. Offsets index into `text` exactly as given."""
    out: list[Element] = []
    scene_i = -1
    speaker: str | None = None
    # A cue only becomes a cue if speech follows it; until then it is pending.
    pending: tuple[str, int, int] | None = None

    def flush_pending_as_action():
        nonlocal pending
        if pending is not None:
            t, s, e = pending
            out.append(Element("action", t, s, e, scene_i or None if scene_i >= 0 else None))
            pending = None

    for raw, s, e in _lines_with_offsets(text):
        line = raw.strip()
        if not line:
            # A blank line ends a speech block but not a pending cue's chance.
            speaker = None
            continue
        if PAGE_RE.match(line):
            continue

        if SCENE_RE.match(line):
            flush_pending_as_action()
            scene_i += 1
            out.append(Element("scene_heading", line, s, e, scene_i))
            speaker = None
            continue
        if TRANSITION_RE.match(line):
            flush_pending_as_action()
            out.append(Element("transition", line, s, e, scene_i if scene_i >= 0 else None))
            continue
        if SUPER_RE.match(line):
            flush_pending_as_action()
            out.append(Element("super", line, s, e, scene_i if scene_i >= 0 else None))
            continue

        if pending is not None:
            # The line after a candidate cue decides what the cue was.
            t, ps, pe = pending
            pending = None
            if PAREN_RE.match(line) or not _is_cue(line):
                speaker = CUE_SUFFIX_RE.sub("", t).strip()
                out.append(Element("character", t, ps, pe, scene_i if scene_i >= 0 else None))
                kind = "parenthetical" if PAREN_RE.match(line) else "dialogue"
                out.append(Element(kind, line, s, e, scene_i if scene_i >= 0 else None, speaker))
                continue
            # Two cue-shaped lines in a row: the first was shouted action.
            out.append(Element("action", t, ps, pe, scene_i if scene_i >= 0 else None))

        if _is_cue(line):
            pending = (line, s, e)
            continue

        if speaker is not None:
            kind = "parenthetical" if PAREN_RE.match(line) else "dialogue"
            out.append(Element(kind, line, s, e, scene_i if scene_i >= 0 else None, speaker))
        else:
            out.append(Element("action", line, s, e, scene_i if scene_i >= 0 else None))

    flush_pending_as_action()
    return out


def scenes(elements: list[Element]) -> list[dict]:
    """Group elements under their scene heading, with the scene's own span."""
    out: list[dict] = []
    for el in elements:
        if el.kind == "scene_heading":
            out.append({"index": el.scene_index, "heading": el.text,
                        "start": el.start, "end": el.end, "elements": []})
        elif out:
            out[-1]["elements"].append(el)
            out[-1]["end"] = max(out[-1]["end"], el.end)
    return out


def verify_spans(text: str, elements: list[Element]) -> list[str]:
    """Every element must slice back to itself. A parser whose offsets drift
    produces evidence that points at the wrong words, which is worse than no
    evidence at all."""
    bad = []
    for el in elements:
        if text[el.start:el.end].strip() != el.text:
            bad.append(f"{el.kind}@{el.start}:{el.end} {el.text[:40]!r}")
    return bad
