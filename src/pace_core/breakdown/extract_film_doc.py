#!/usr/bin/env python3
"""The treatment's top matter, read off the document that is already on disk.

A production document states what the film IS before it states what happens:
題材 (genre), 核心溯源 (source), 世界观设定 (worldview), 核心主题 (theme),
人物设定 (characters). In every project here that top matter sits on page one
of the same file the breakdown reads, above the first scene heading -- so the
pipeline has always had these words and has always thrown them away, because
there was no scene above the first scene to put them in and no field to hold
them.

This is not a model call. The labels are literal (`题材：`, `核心溯源：`, ...),
so the extraction is a parse, and a parse is worth preferring here for a
reason beyond cost: a model asked to summarise a worldview writes a NEW
worldview, and what belongs in `film.json` is the author's own sentence. What
this returns is verbatim, minus the label.

The head is everything before the first scene heading, and finding that is the
only judgment call: a Chinese shooting script numbers scenes `1、夜`, an
English one opens `INT. ...`. Both are matched. Text with no recognisable
first scene is treated as ALL head only when it is short enough to be a
treatment rather than a script, so a screenplay this parser does not
understand degrades to "no film document" rather than to "the whole film is
the worldview".

    uv run python -m pace_core.breakdown.extract_film_doc --project <slug>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.paths import paths_for                                  # noqa: E402

#: A scene heading opens the screenplay and closes the treatment. `1、夜` is
#: the Chinese shooting-script form; `INT./EXT.` the English one.
_SCENE_START = re.compile(
    r"^\s*(?:\d{1,3}\s*[、.．]\s*\S|(?:INT|EXT|I/E)[\.\s])", re.M | re.I)

#: Labels, in the order a treatment writes them. The bullet glyph varies with
#: whatever the .docx used ("v" is a Wingdings check that pdftotext renders
#: literally), so the label itself is the anchor, not the bullet.
_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("genre",       ("题材", "類型", "类型")),
    ("source",      ("核心溯源", "溯源", "出处", "出處")),
    ("worldview",   ("世界观设定", "世界觀設定", "世界观", "世界觀")),
    ("theme",       ("核心主题", "核心主題", "主题", "主題")),
    ("characters",  ("人物设定", "人物設定", "角色设定", "角色設定")),
)

#: 科幻、惊悚 -> ["科幻", "惊悚"]. Genre is a list; everything else is prose.
_GENRE_SPLIT = re.compile(r"[、，,/·\s]+")

#: A treatment that never reaches a scene heading. Beyond this it is far more
#: likely to be a script we failed to parse than a very long treatment.
_HEAD_ONLY_MAX_CHARS = 6000


def film_head(text: str) -> str:
    """Everything above the first scene heading."""
    text = (text or "").replace("\r\n", "\n")
    m = _SCENE_START.search(text)
    if m:
        return text[:m.start()].strip()
    return text.strip() if len(text.strip()) <= _HEAD_ONLY_MAX_CHARS else ""


#: A .docx bullet reaches the text layer as whatever glyph the list style
#: used -- a Wingdings check comes through as a literal "v". Each field is cut
#: at the NEXT label, so the next bullet is the last token inside it.
#:
#: Two extractions have to be handled, because the one this parser was first
#: written against is not the one the pipeline uses. `pdftotext` keeps the
#: document's line breaks, so a bullet lands alone on a line. `pypdf` -- what
#: `script_bytes_to_text` actually calls -- returns the whole treatment on ONE
#: line with the bullets inline as " v ", which a bullet-only-LINE filter
#: never sees. Hence both: bullet lines are dropped, and a trailing bullet
#: token is stripped from every field.
_BULLETS = "v❖•▪◆*\\-–‣»·"
_BULLET_ONLY = frozenset("v❖•▪◆*-–‣»·")
_TRAILING_BULLET = re.compile(rf"(?:\s+[{_BULLETS}])+\s*$")

#: A page number pypdf emits as its own line ("1 \nAI 短片…"), which would
#: otherwise be joined onto the title.
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")


def _clean(v: str) -> str:
    """Join the wrapped lines of one field back into its sentence.

    A PDF breaks a paragraph wherever the column ended, and those breaks are
    not the author's. Chinese does not put spaces between characters, so the
    lines are joined bare; a line ending mid-word in a Latin run keeps its
    space.
    """
    out = ""
    for line in (l.strip() for l in v.split("\n")):
        if (not line or (len(line) == 1 and line in _BULLET_ONLY)
                or _PAGE_NUMBER.match(line)):
            continue
        if out and (out[-1].isascii() and out[-1].isalnum()) and line[0].isascii():
            out += " "
        out += line
    return _TRAILING_BULLET.sub("", out).strip(" :：　")


def parse_head(head: str) -> dict:
    """The labelled fields of a treatment's top matter, verbatim."""
    if not head.strip():
        return {}
    # Where each label starts, so a field runs to the next one rather than to
    # a guessed line count -- 世界观设定 is routinely five wrapped lines.
    hits: list[tuple[int, int, str]] = []
    for key, names in _LABELS:
        for name in names:
            m = re.search(rf"{re.escape(name)}\s*[:：]", head)
            if m:
                hits.append((m.start(), m.end(), key))
                break
    if not hits:
        return {}
    hits.sort()
    out: dict = {}
    for i, (_start, end, key) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(head)
        out[key] = _clean(head[end:stop])
    # The title is whatever stands above the first label.
    title = _clean(head[:hits[0][0]])
    if title:
        out["title"] = title
    if out.get("genre"):
        out["genre"] = [g for g in _GENRE_SPLIT.split(out["genre"]) if g]
    return out


def film_doc(parsed: dict) -> dict:
    """`kb/film.json`, from the parsed top matter.

    `worldview` goes to `design_language`, which IS compiled into every
    prompt, rather than to `premise`, which is not. A treatment written like
    this one states its world as rules with the look implied, and the point of
    reading it at all is that the world reaches the frame. `premise` stays for
    a project that separates the two itself.
    """
    wv = parsed.get("worldview") or ""
    return {
        "_note": ("Read from the treatment's top matter by "
                  "pace_core.breakdown.extract_film_doc. Verbatim, minus "
                  "the label. Edit freely -- nothing regenerates this."),
        "title": parsed.get("title", ""),
        "genre": parsed.get("genre") or [],
        "source": parsed.get("source", ""),
        "worldview": {
            "premise": "",
            "design_language": [wv] if wv else [],
            "invariants": [],
        },
        "theme": parsed.get("theme", ""),
        # Not a film field -- the character sheet belongs in characters.json.
        # Carried here so the words are not lost while that path is built.
        "_characters_raw": parsed.get("characters", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--script", type=Path, default=None,
                    help="script file (default: the newest under script/)")
    ap.add_argument("--write", action="store_true", help="write kb/film.json")
    a = ap.parse_args()

    p = paths_for(a.project)
    src = a.script
    if src is None:
        cands = sorted(Path(p.script_dir).glob("*"), key=lambda f: f.stat().st_mtime)
        if not cands:
            print(f"no script uploaded for {a.project!r}", file=sys.stderr)
            return 1
        src = cands[-1]

    raw = src.read_bytes()
    if raw[:4] == b"%PDF":
        import subprocess
        text = subprocess.run(["pdftotext", str(src), "-"], check=True,
                              capture_output=True, text=True).stdout
    else:
        text = raw.decode("utf-8", errors="replace")

    parsed = parse_head(film_head(text))
    if not parsed:
        print(f"no labelled top matter in {src.name}", file=sys.stderr)
        return 1
    doc = film_doc(parsed)
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    if a.write:
        p.film_file.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
        print(f"\nwrote {p.film_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
