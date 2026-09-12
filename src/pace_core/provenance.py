"""Content hashing and staleness, so a gate is a computation and not a habit.

A pipeline that runs its stages unconditionally cannot tell an artifact
that is current from one whose input has since been re-cut. This module
supplies the two operations that make that difference decidable:

  content_hash(obj)     a canonical hash of a document, stable under key
                        order and formatting, so an artifact can name the
                        exact version of the input it was produced from.

  stale_artifacts(...)  compares recorded upstream hashes against the
                        current ones and returns what no longer matches.

The pairing with approval matters as much as the hash. `Review.approved_hash`
records the version an approval covers, so the approval expires when the
content moves. An approval recorded against a document *identifier* stays
true across an edit, which produces an approved artifact nobody reviewed in
its present form.

Hashing is deliberately narrow. Only authored content is hashed; derived
bookkeeping (the manifest, artifact ledgers, timestamps) is excluded,
because including a document's own provenance in its hash makes writing the
provenance change the hash it just recorded.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

HASH_LEN = 12
# Excluded from a document's own content hash: writing provenance must not
# perturb the hash that provenance records.
DERIVED_KEYS = frozenset({
    "manifest", "artifacts", "_schema_version", "semantics",
    "created_at", "updated_at", "last_modified", "_extracted",
})


def _canonical(obj: Any, *, drop: frozenset = DERIVED_KEYS) -> Any:
    """Strip derived bookkeeping so the hash covers authored content only."""
    if isinstance(obj, dict):
        return {k: _canonical(v, drop=drop) for k, v in sorted(obj.items())
                if k not in drop}
    if isinstance(obj, list):
        return [_canonical(v, drop=drop) for v in obj]
    return obj


def content_hash(obj: Any, *, drop: frozenset = DERIVED_KEYS) -> str:
    """Stable short hash of authored content.

    Stable across key order and whitespace, so reserialising a document
    does not invalidate everything downstream of it.
    """
    payload = json.dumps(_canonical(obj, drop=drop), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:HASH_LEN]


def upstream_ref(kind: str, digest: str) -> str:
    """Format an upstream pointer, e.g. `shot:f29cfa57fe3b`."""
    return f"{kind}:{digest}"


def parse_ref(ref: str) -> tuple[str, str]:
    kind, _, digest = ref.partition(":")
    return kind, digest


def is_stale(artifact: dict, current: dict[str, str]) -> tuple[bool, list[str]]:
    """Does `artifact` record upstreams that no longer match `current`?

    `current` maps upstream kind -> present hash. Returns (stale, reasons).

    An artifact recording no upstream at all is reported as stale with an
    explicit reason rather than as current: an artifact that cannot say what
    produced it has not been shown to be up to date, and silently treating
    unknown as current is the failure this module exists to remove.
    """
    refs = artifact.get("derived_from") or []
    if not refs:
        return True, ["records no upstream, so currency cannot be established"]
    reasons = []
    for ref in refs:
        kind, digest = parse_ref(ref)
        now = current.get(kind)
        if now is None:
            reasons.append(f"upstream {kind!r} is no longer present")
        elif now != digest:
            reasons.append(f"{kind} changed: {digest} -> {now}")
    return bool(reasons), reasons


def approval_valid(review: dict | None, current_hash: str) -> tuple[bool, str]:
    """Is an approval still valid for the content as it now stands?"""
    if not review:
        return False, "no review recorded"
    if review.get("status") != "approved":
        return False, f"status is {review.get('status', 'unreviewed')!r}"
    approved = review.get("approved_hash")
    if not approved:
        return False, "approval names no content hash, so it cannot expire"
    if approved != current_hash:
        return False, f"content moved since approval: {approved} -> {current_hash}"
    return True, "approved for the current content"


def stale_artifacts(artifacts: Iterable[dict],
                    current: dict[str, str]) -> list[tuple[dict, list[str]]]:
    out = []
    for a in artifacts:
        bad, why = is_stale(a, current)
        if bad:
            out.append((a, why))
    return out
