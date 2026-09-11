"""What shape of space a location is, read from what the KB declares.

Two consumers need this answer and neither should own it: the scene assembler,
which builds a room proxy inside Blender, and the panel greybox, which decides
whether it can stage a panel at all. It lives here because it is a reading of
authored data, not geometry -- scene_proxies imports bpy at module scope, so
anything defined there is unavailable to the host side and untestable without
Blender.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

# The shape vocabulary. SHAPES is the runtime source; LocationShape is its
# static mirror, because a Literal cannot be built from a tuple. A test asserts
# the two agree, so a shape added to one and not the other fails loudly rather
# than type-checking clean and failing at a call site.
LocationShape = Literal["chamber", "subway", "office", "outdoor"]
SHAPES: tuple[str, ...] = ("chamber", "subway", "office", "outdoor")

# environment_type -> shape. The KB already declares an environment for every
# location; this is the reading of it, not a second list of location ids.
#
# What this replaces: LOCATION_SHAPE_MAP, a dict of THIS production's location
# ids, consulted with `.get(ref, "chamber")`. Any location it had not been
# taught -- every location of every other screenplay -- silently became a small
# wooden room. A motorway built as a chamber raises no error anywhere; it just
# emits a depth signal that cannot be right.
# These are DEFAULTS, not the definition. A production whose screenplay uses
# environment vocabulary this table has never heard of should not need a Python
# edit: the KB's stub document may carry its own `shape_map` and
# `shape_prefix_map`, merged over these by shape_maps_from_doc(). Nothing here
# names a location id -- that was the bug this module replaced — and nothing
# here is the last word either.
ENVIRONMENT_SHAPE: dict[str, LocationShape] = {
    "vehicle_interior": "subway",   # the library's shape for seated rows in a
                                    # moving vehicle: a substitution, not a
                                    # match, and named as one at the mapping
                                    # rather than invisibly at the call site
    "office":           "office",
    "chamber":          "chamber",
}
# Whole families, for environments no table will ever finish enumerating.
ENVIRONMENT_PREFIX_SHAPE: list[tuple[str, LocationShape]] = [
    ("exterior", "outdoor"),
    ("vehicle",  "subway"),
    ("interior", "chamber"),
]


def shape_maps_from_doc(doc: dict | None) -> tuple[dict, list]:
    """A project's own environment→shape mapping, over the built-in defaults.

    The stub document may declare `shape_map` (exact environment_type) and
    `shape_prefix_map` (families, as a list of [prefix, shape] pairs or an
    ordered object). Both are optional, both merge over the defaults, and both
    are filtered to the known shape vocabulary — a typo in the KB should not
    silently introduce a shape no builder can build.
    """
    env = dict(ENVIRONMENT_SHAPE)
    prefixes = list(ENVIRONMENT_PREFIX_SHAPE)
    if not isinstance(doc, dict):
        return env, prefixes

    for k, v in (doc.get("shape_map") or {}).items():
        if isinstance(v, str) and v in SHAPES:
            env[str(k).strip().lower()] = v                 # type: ignore[assignment]

    raw = doc.get("shape_prefix_map")
    pairs = raw.items() if isinstance(raw, dict) else (raw or [])
    extra = [(str(a).strip().lower(), b) for a, b in pairs
             if isinstance(b, str) and b in SHAPES]
    # Project prefixes are consulted before the built-ins, so a production can
    # narrow a family the defaults would have swallowed ("exterior_soundstage"
    # is not outdoors) without having to restate the ones it agrees with.
    return env, extra + prefixes


def shape_from_stub(stub: dict, *, env_shape: dict | None = None,
                    prefix_shape: list | None = None) -> LocationShape | None:
    """The shape a location declares, or None if it declares none.

    Order: an explicit `shape` on the stub wins, since it is the author saying
    so outright; otherwise the environment_type is read, through the project's
    maps when given and the built-in defaults otherwise. None is a real answer
    and the caller has to handle it -- the previous default of "chamber"
    turned "this location is unclassified" into "this location is a small
    room", which is the same sentence to Blender and a very different one to
    everybody else.
    """
    if not isinstance(stub, dict):
        return None
    explicit = (stub.get("shape") or "").strip()
    if explicit in SHAPES:
        return explicit                                    # type: ignore[return-value]
    env = (stub.get("environment_type") or "").strip().lower()
    if not env:
        return None
    table = ENVIRONMENT_SHAPE if env_shape is None else env_shape
    prefixes = ENVIRONMENT_PREFIX_SHAPE if prefix_shape is None else prefix_shape
    if env in table:
        return table[env]
    for prefix, shape in prefixes:
        if env.startswith(prefix):
            return shape
    return None


def stub_doc(project: str | None = None) -> dict:
    """The project's whole stub document, or {} — it carries the shape maps as
    well as the stubs, so both readings come from one load."""
    import json as _json
    if not project:
        return {}
    try:
        from pace_core.paths import paths_for
        f = paths_for(project).loc_stubs_file
        if not Path(f).is_file():
            return {}
        return _json.loads(Path(f).read_text()) or {}
    except Exception:                                          # noqa: BLE001
        return {}


def location_stub(location_ref: str, project: str | None = None) -> dict:
    """One location's stub from the project KB, or {} if there is none."""
    if not location_ref:
        return {}
    stubs = stub_doc(project).get("stubs") or {}
    stub = stubs.get(location_ref)
    return stub if isinstance(stub, dict) else {}


def location_shape_for(location_ref: str,
                       project: str | None = None) -> LocationShape | None:
    """Shape for a location, read from its own stub and the project's maps."""
    doc = stub_doc(project)
    stubs = doc.get("stubs") or {}
    stub = stubs.get(location_ref)
    env_shape, prefix_shape = shape_maps_from_doc(doc)
    return shape_from_stub(stub if isinstance(stub, dict) else {},
                           env_shape=env_shape, prefix_shape=prefix_shape)
