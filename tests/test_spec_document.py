"""The field specification in docs/spec is the generator's output, verbatim.

It says of itself that it is emitted from the schema and cannot drift. That
held only while something re-ran the generator; a checked-in copy nobody
regenerates drifts silently, so this compares the two.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.spec_appendix import spec_appendix  # noqa: E402


def test_the_checked_in_body_is_what_the_schema_generates():
    body = ROOT / "docs" / "spec" / "spec_appendix.tex"
    assert body.read_text() == spec_appendix(), (
        "docs/spec/spec_appendix.tex is stale; run docs/spec/build.sh")


def test_the_document_inputs_that_body():
    assert r"\input{spec_appendix}" in (ROOT / "docs" / "spec" / "pace_spec.tex").read_text()
