"""Emit the paper's field specification from the schema, so it cannot drift.

The appendix prints every sub-schema of `types_v1`, in declaration order, with
the comment written beside each field as its meaning. It says of itself that
it is emitted rather than transcribed. It stopped being either: the generator
was lost, the file was hand-patched, and by the time anyone checked it was
seven fields behind the schema and had rendered `CompileHints` as
`CompileHintegers`, from a blind `int` substitution nobody could see because
nothing re-ran.

This module is that generator, back, with a test that counts its rows against
the dataclasses. It reads the source text rather than importing the module,
because the meanings live in comments and comments do not survive `import`.

    python -m pace_core.spec_appendix --spec  > spec_appendix.tex
    python -m pace_core.spec_appendix --usd   > usd_correspondence_table.tex
"""
from __future__ import annotations

import argparse
import ast
import io
import pathlib
import re
import sys
import tokenize
from typing import Iterator

SOURCE = pathlib.Path(__file__).with_name("types_v1.py")

# LaTeX means something by these; a schema comment does not.
_ESCAPE = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
           "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
# The documents this feeds are set in Times under T1, where anything outside
# Latin-1 prints as a wrong glyph or as nothing: the em dashes and arrows the
# schema's comments are written with came out as gaps, and every `§` as `ğ`.
# Spelled as commands, they typeset in any encoding.
_SYMBOL = {
    # TeX forbids a break after an em dash, and these columns are narrow.
    "—": r"---\allowbreak{}", "–": r"--\allowbreak{}", "…": r"\ldots{}",
    "→": r"$\to$", "←": r"$\gets$", "↑": r"$\uparrow$", "↓": r"$\downarrow$",
    "⇒": r"$\Rightarrow$", "⇐": r"$\Leftarrow$",
    "×": r"$\times$", "°": r"$^\circ$", "±": r"$\pm$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≠": r"$\neq$",
    "§": r"\S{}", "·": r"$\cdot$",
    # Not a substitution: a break point. These columns are a quarter of the
    # text width and the schema writes its enumerations as slash runs
    # (pan/tilt/zoom, wide/medium/narrow), which TeX treats as one long word.
    "/": r"/\allowbreak{}",
    # Romanisations the schema quotes, which T1 has no glyph for.
    "ī": r"\={\i}", "ū": r"\=u", "ā": r"\=a",
    "ṣ": r"\d{s}", "ṭ": r"\d{t}", "ṇ": r"\d{n}", "ṃ": r"\d{m}",
    "ú": r"\'u", "é": r"\'e", "ö": r'\"o',
}

# Scalar annotations, as the appendix has always named them.
_SCALAR = {"str": "text", "int": "integer", "float": "number", "bool": "true/false",
           "Any": "Any", "dict": "dict", "list": "list"}


def tex(s: str) -> str:
    out = "".join(_ESCAPE.get(c, _SYMBOL.get(c, c)) for c in s)
    return re.sub(r"\s+", " ", out).strip()


def breakable(s: str, lists: bool = True) -> str:
    """The same, with break points inside an identifier.

    The field and type columns are a quarter of the text width each, and a
    name like `change_in_environment` or a Literal listing five values has no
    space in it to break at, so it runs into the next column. TeX will not
    hyphenate inside \\texttt either. An underscore and a list comma are the
    two places a reader already sees a seam.
    """
    out = tex(s).replace(r"\_", r"\_\allowbreak ")
    return out.replace('", "', '", \\allowbreak "') if lists else out


# ── reading the source ───────────────────────────────────────────────────
def _comments(src: str) -> dict[int, str]:
    """Line number -> comment text, for every comment in the file.

    Tokenising rather than regexing: a `#` inside a string literal is not a
    comment, and the schema has several ("modern_day" / "2099" — open set).
    """
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.start[0]] = tok.string.lstrip("#").strip()
    return out


def _is_rule(text: str) -> bool:
    """A box-drawing separator comment is punctuation, not documentation."""
    return bool(re.fullmatch(r"[─—\-=·]*\s*.*?\s*[─—\-=·]{3,}", text)) or \
        set(text) <= set("─—-=· ")


def _meaning(node: ast.AnnAssign, comments: dict[int, str], lines: list[str]) -> str:
    """The comment beside a field, preceded by any comment block above it.

    Both are how this schema documents a field: a short note goes on the line,
    and anything longer goes on the lines above. The appendix has to print
    both or it prints half of what the schema says.
    """
    trailing = comments.get(node.end_lineno, "")
    # Only a line that is nothing but a comment: a comment sitting at the end
    # of the previous field's line documents that field, not this one.
    above, n = [], node.lineno - 1
    while n >= 1 and n in comments and lines[n - 1].lstrip().startswith("#"):
        if _is_rule(comments[n]):
            break
        above.append(comments[n])
        n -= 1
    above.reverse()
    return " ".join([*above, trailing]).strip()


def _render_type(node: ast.expr, src: str) -> tuple[str, bool]:
    """(rendered type, is-optional)."""
    if isinstance(node, ast.Subscript):
        base = ast.unparse(node.value)
        if base in ("Optional",):
            inner, _ = _render_type(node.slice, src)
            return inner, True
        if base in ("list", "List"):
            inner, _ = _render_type(node.slice, src)
            return f"list of {inner}", False
        if base in ("dict", "Dict"):
            return "map", False
        if base == "Literal":
            return ast.get_source_segment(src, node) or ast.unparse(node), False
        return ast.unparse(node), False
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False                     # a forward reference
    name = ast.unparse(node)
    return _SCALAR.get(name, name), False


def schemas(src: str | None = None) -> Iterator[tuple[str, str, list[tuple[str, str, bool, str]]]]:
    """(class name, its docstring, [(field, type, optional, meaning)])."""
    src = src if src is not None else SOURCE.read_text()
    tree, comments, lines = ast.parse(src), _comments(src), src.splitlines()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = [s for s in node.body if isinstance(s, ast.AnnAssign)]
        if not fields:
            continue
        rows = []
        for f in fields:
            kind, opt = _render_type(f.annotation, src)
            rows.append((ast.unparse(f.target), kind, opt,
                         _meaning(f, comments, lines)))
        yield node.name, (ast.get_docstring(node) or "").strip(), rows


# ── the field specification ──────────────────────────────────────────────
_HEAD = r"""\section{The full field specification}
\label{app:spec}

The table below is generated from the schema source: %d sub-schemas, %d fields. It is emitted from the source rather than transcribed, so it cannot drift from the specification it documents. Each field's explanation is the comment written beside its declaration, and fields appear in declaration order, which is the order the schema reads in.

"""
_TABULAR = (r"\begin{longtable}{@{}R{0.24\textwidth}R{0.21\textwidth}R{0.49\textwidth}@{}}"
            "\n" r"\toprule \textbf{field} & \textbf{type} & \textbf{meaning} \\ \midrule"
            "\n" r"\endhead")


def spec_appendix(src: str | None = None) -> str:
    from pace_core.types_v1 import INFERRED_FIELDS

    blocks, n_fields, n_schemas = [], 0, 0
    for name, doc, rows in schemas(src):
        n_schemas += 1
        n_fields += len(rows)
        body = [f"\\subsection*{{\\texttt{{{tex(name)}}}}}"]
        if doc:
            body.append(f"\\noindent {breakable(doc, lists=False)}\n")
        body.append(_TABULAR)
        for field, kind, opt, meaning in rows:
            if f"{name}.{field}" in INFERRED_FIELDS:
                suffix = r", \emph{inferred}"
            else:
                suffix = ", optional" if opt else ""
            body.append(f"\\texttt{{{breakable(field)}}} & {breakable(kind)}{suffix} "
                        f"& {breakable(meaning, lists=False)} \\\\")
        body.append("\\bottomrule\n\\end{longtable}")
        blocks.append("\n".join(body))
    return (_HEAD % (n_schemas, n_fields)) + "\n\n".join(blocks) + "\n"


# ── the OpenUSD correspondence ───────────────────────────────────────────
# Which pillar a sub-schema belongs to, for the correspondence table. The
# grouping is the paper's, and a schema named here that the file no longer
# declares fails the test rather than quietly dropping its fields.
PILLARS: dict[str, tuple[str, ...]] = {
    "Camera \\& Lighting": (
        "CameraIntrinsics", "CameraExtrinsics", "CameraTrajectory",
        "CameraCreativeIntent", "FrameRate", "CameraProgram", "Camera",
        "Lighting", "CameraSetup"),
    "Setup": (
        "Setup", "Backdrop", "Environment", "SceneTexture", "SceneGeometry",
        "SceneSpace", "Prop", "WorldEntity", "PhysicalLayout"),
    # Entities that appear in the picture, text included: an on-screen title
    # is placed and framed the way a subject is.
    "Characters": ("Subject", "Gaze", "ScreenPosition", "PrimaryFocus",
                   "TextElement"),
    "Events": ("Events", "Action", "Emotion", "Dialogue", "EventsAdvanced"),
}
_USD_HEAD = r"""% Generated by pace_core.spec_appendix --usd. Do not hand-edit: the
% verdicts come from pace_core.usd_map, which pace_core.usd_export also reads,
% so a field that gains a USD target starts being exported and starts counting
% here in the same commit.
\begin{tabularx}{\textwidth}{@{}lrrrrX@{}}
\toprule
pillar & native & derived & custom & total & what USD carries natively \\
\midrule"""
_NATIVELY = {
    "Camera \\& Lighting": r"\texttt{focalLength}, \texttt{fStop}, \texttt{horizontalAperture}, \texttt{xformOp}, \texttt{inputs:colorTemperature}",
    "Setup": r"\texttt{xformOp} translate/rotate/scale, material binding, prim paths",
    "Characters": "prim path as character identifier",
    "Document / provenance": "none",
    "Events": "none",
}


def usd_table(src: str | None = None) -> str:
    from pace_core.usd_map import verdict_for

    of_schema = {s: p for p, ss in PILLARS.items() for s in ss}
    tally = {p: [0, 0, 0] for p in (*PILLARS, "Document / provenance")}
    col = {"native": 0, "derived": 1, "custom": 2}
    for name, _doc, rows in schemas(src):
        pillar = of_schema.get(name, "Document / provenance")
        for field, *_ in rows:
            tally[pillar][col[verdict_for(name, field)[0]]] += 1
    # The paper's order: the two pillars USD models, then the ones it does not.
    order = ["Camera \\& Lighting", "Setup", "Characters",
             "Document / provenance", "Events"]
    out, total = [_USD_HEAD], [0, 0, 0]
    for p in order:
        n, d, c = tally[p]
        for i, v in enumerate((n, d, c)):
            total[i] += v
        out.append(f"{p} & {n} & {d} & {c} & {n + d + c} & {_NATIVELY[p]} \\\\")
    out.append(r"\midrule")
    out.append(f"\\textbf{{all}} & \\textbf{{{total[0]}}} & \\textbf{{{total[1]}}} "
               f"& \\textbf{{{total[2]}}} & \\textbf{{{sum(total)}}} & \\\\")
    out.append("\\bottomrule\n\\end{tabularx}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usd", action="store_true",
                    help="emit the OpenUSD correspondence table instead")
    ap.add_argument("-o", "--out", type=pathlib.Path)
    a = ap.parse_args(argv)
    text = usd_table() if a.usd else spec_appendix()
    if a.out:
        a.out.write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
