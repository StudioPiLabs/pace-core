"""The paper's field specification is emitted from the schema, as it claims.

It used to be. The generator was lost, the .tex was hand-patched, and by the
time anyone counted it was seven fields behind and had rendered `CompileHints`
as `CompileHintegers` and `ActionUncertainty` as `ActionUncertaintegery`, from
a blind int->integer substitution that nothing re-ran to catch. A document
that says of itself that it cannot drift has to be checked that it has not.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core import spec_appendix as S                          # noqa: E402
from pace_core.types_v1 import INFERRED_FIELDS                    # noqa: E402

SRC = S.SOURCE.read_text()
TEX = S.spec_appendix()
CLASSES = [n for n in ast.parse(SRC).body
           if isinstance(n, ast.ClassDef)
           and any(isinstance(s, ast.AnnAssign) for s in n.body)]
N_FIELDS = sum(sum(isinstance(s, ast.AnnAssign) for s in c.body) for c in CLASSES)


def test_every_declared_field_has_a_row():
    """The count in the opening sentence is the count of rows below it, and
    both are the schema's: a field added to types_v1 and missing here is the
    exact defect this file exists for."""
    rows = len(re.findall(r"^\\texttt\{", TEX, re.M))
    assert rows == N_FIELDS, f"{rows} rows for {N_FIELDS} fields"
    assert f"{len(CLASSES)} sub-schemas, {N_FIELDS} fields" in TEX


def test_every_sub_schema_has_a_section():
    for c in CLASSES:
        assert f"\\subsection*{{\\texttt{{{c.name}}}}}" in TEX, c.name


@pytest.mark.parametrize("mangled", ["Hintegers", "Uncertaintegery", "printeger"])
def test_a_type_name_is_not_mangled_by_substituting_inside_it(mangled):
    """int -> integer applied to the whole string, rather than to a type, is
    how the old file got CompileHintegersFlux."""
    assert mangled not in TEX


def test_a_literal_keeps_its_closing_bracket():
    """Two Literal rows printed as `Literal["a", "b"` with the bracket eaten
    by a naive truncation, which reads as a syntax error in a specification."""
    for lit in re.findall(r"& (Literal\[[^&]*) &", TEX):
        assert lit.rstrip().rstrip(", optional").rstrip().endswith("]"), lit


def test_an_inferred_field_is_marked_inferred_not_optional():
    """`mood`, `era`, `region` and `culture` are Optional in Python and are
    not optional in effect: every backend compiles them into the prompt, so a
    null is not an absent value, it is the sampler choosing."""
    assert INFERRED_FIELDS, "nothing is declared inferred"
    for key in INFERRED_FIELDS:
        schema, field = key.split(".")
        row = next(r for r in re.findall(r"^\\texttt\{[^\n]*$", TEX, re.M)
                   if r.startswith("\\texttt{" + field.replace("_", "\\_") + "}"))
        assert r"\emph{inferred}" in row, key


def test_a_meaning_comes_from_the_comment_beside_the_declaration():
    got = dict((f, m) for _n, _d, rows in S.schemas(SRC) for f, _t, _o, m in rows)
    assert got["focal_length_mm"] == "exact lens length, e.g. 35.0"
    # A comment block above the field is documentation too, and the old file
    # dropped all of it.
    assert got["rests_on"].startswith("character_id the prop lies on")
    assert "wardrobe entry" in got["costume_id"]


def test_a_comment_on_the_previous_line_belongs_to_the_previous_field():
    """The first attempt walked upward through trailing comments and gave
    every field in a block the concatenation of all the ones above it."""
    got = dict((f, m) for _n, _d, rows in S.schemas(SRC) for f, _t, _o, m in rows)
    assert got["aperture_f"] == "f-stop number, e.g. 2.8"


def test_latex_special_characters_are_escaped():
    assert S.tex("a_b & c% d#e") == r"a\_b \& c\% d\#e"


# ── the OpenUSD correspondence, from the same source ─────────────────────
def test_the_correspondence_covers_every_field_exactly_once():
    usd = S.usd_table()
    total = int(re.search(r"\\textbf\{all\}.*?\\textbf\{(\d+)\}\s*&\s*\\\\", usd,
                          re.S).group(0).split("textbf{")[-1].split("}")[0])
    assert total == N_FIELDS


def test_the_verdicts_are_the_exporter_s_verdicts():
    """The table and pace_core.usd_export read one mapping, so a field that
    gains a USD target starts being exported and starts counting here in the
    same commit."""
    from pace_core.usd_map import MAPPING

    usd = S.usd_table()
    native = sum(1 for v in MAPPING.values() if v[0] == "native")
    derived = sum(1 for v in MAPPING.values() if v[0] == "derived")
    row = re.search(r"\\textbf\{all\} & \\textbf\{(\d+)\} & \\textbf\{(\d+)\}", usd)
    assert (int(row.group(1)), int(row.group(2))) == (native, derived)


def test_every_pillar_names_schemas_that_exist():
    """A pillar naming a class the schema dropped silently loses that class's
    fields into 'document and provenance'."""
    have = {c.name for c in CLASSES}
    for pillar, names in S.PILLARS.items():
        missing = [n for n in names if n not in have]
        assert not missing, f"{pillar}: {missing}"
