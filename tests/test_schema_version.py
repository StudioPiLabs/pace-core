"""A document keeps reading after the schema is renamed.

The version tag is only worth writing if a reader can act on it, and readers
asked the question by writing the supported list out themselves. The copies
went stale the moment a version was added: one still testing for 1.0 and 1.1
silently skipped every 1.2 document it was handed, and half a corpus stopped
being enriched without anything reporting it.

These pin the two things that stops: the list has one home, and the versions
written under the project's earlier name are still accepted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.types_v1 import (  # noqa: E402
    SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, UPSTREAM_REGISTRY_VERSION,
    SceneDoc, is_supported_schema_version,
)


def test_new_documents_carry_the_current_version():
    assert SceneDoc()._schema_version == SCHEMA_VERSION
    assert SCHEMA_VERSION == "pace-1.3"


def test_the_current_version_is_supported_by_its_own_reader():
    assert is_supported_schema_version(SCHEMA_VERSION)


def test_documents_written_under_the_earlier_name_still_read():
    """Renaming the schema does not orphan what is already on disk."""
    for old in ("pai-1.0", "pai-1.1", "pai-1.2"):
        assert is_supported_schema_version(old), old


def test_an_unknown_version_is_not_supported():
    assert not is_supported_schema_version("pai-0.3")
    assert not is_supported_schema_version("pace-9.9")
    assert not is_supported_schema_version(None)
    assert not is_supported_schema_version("")


def test_the_registry_version_is_a_separate_number():
    """Two constants reading `pace-x.y` for different things is the mistake
    this name avoids: one numbers the document, the other the field registry."""
    assert UPSTREAM_REGISTRY_VERSION != SCHEMA_VERSION
    assert UPSTREAM_REGISTRY_VERSION not in SUPPORTED_SCHEMA_VERSIONS


def test_every_writer_emits_the_constant():
    """A writer that spells the version out drifts behind it; three had."""
    import re
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        for m in re.finditer(r'"_?schema_version":\s*"(pai|pace)-[0-9.]+"', py.read_text()):
            offenders.append(f"{py.relative_to(ROOT)}: {m.group(0)}")
    assert not offenders, offenders
