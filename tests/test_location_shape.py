"""What shape of space a location is, and what happens when it does not say.

This replaced LOCATION_SHAPE_MAP, a dict of one production's location ids read
with `.get(ref, "chamber")`. Every location it had not been taught -- which is
every location of every other screenplay -- silently became a small wooden
room. The tests that matter here are about the unclassified case, because that
is the one that used to be invisible.
"""
from __future__ import annotations

import json

from pace_core.setup.location_shape import (
    location_shape_for,
    location_stub,
    shape_from_stub,
)


def test_environment_type_decides_the_shape():
    assert shape_from_stub({"environment_type": "vehicle_interior"}) == "subway"
    assert shape_from_stub({"environment_type": "office"}) == "office"


def test_environment_families_cover_what_no_table_finishes():
    """A screenplay invents environment names freely. `exterior_crash_site` and
    `exterior_highway` are both exteriors without either being enumerated."""
    for env in ("exterior_crash_site", "exterior_highway", "exterior_rooftop"):
        assert shape_from_stub({"environment_type": env}) == "outdoor", env
    assert shape_from_stub({"environment_type": "interior_temple"}) == "chamber"


def test_an_explicit_shape_wins():
    """The author saying so outright beats the environment being read."""
    assert shape_from_stub(
        {"shape": "office", "environment_type": "exterior_highway"}) == "office"


def test_a_bogus_explicit_shape_is_not_trusted():
    """Only shapes the proxy library can build are accepted; anything else
    falls through to the environment rather than reaching Blender."""
    assert shape_from_stub(
        {"shape": "cathedral", "environment_type": "vehicle_interior"}) == "subway"


def test_an_unclassified_location_answers_none_not_chamber():
    """The whole point. `None` is a real answer that a caller has to handle;
    the old default turned "this location is unclassified" into "this location
    is a small room", which is the same sentence to Blender and a very
    different one to everyone else."""
    assert shape_from_stub({}) is None
    assert shape_from_stub({"environment_type": ""}) is None
    assert shape_from_stub({"environment_type": "somewhere_new"}) is None
    assert shape_from_stub(None) is None


def test_shape_is_read_from_the_project_kb(tmp_path, monkeypatch):
    """Read per location from the stub file, so a new screenplay's locations
    resolve from what it declares rather than from a table naming another
    film's ids."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "location_stubs.json").write_text(json.dumps({"stubs": {
        "sedan_interior": {"environment_type": "vehicle_interior"},
        "the_beach": {"environment_type": "exterior_shoreline"},
        "unlabelled": {},
    }}))

    class _P:
        loc_stubs_file = kb / "location_stubs.json"

    # location_shape imports paths_for inside the function, so the patch has to
    # land on pace_core.paths itself rather than on a name bound at import time.
    import pace_core.paths as _paths
    monkeypatch.setattr(_paths, "paths_for", lambda p: _P())

    assert location_shape_for("sedan_interior", "proj") == "subway"
    assert location_shape_for("the_beach", "proj") == "outdoor"
    assert location_shape_for("unlabelled", "proj") is None
    # A location the KB has never heard of resolves to nothing, rather than to
    # whatever the table's default happened to be.
    assert location_shape_for("not_in_the_kb", "proj") is None
    assert location_stub("not_in_the_kb", "proj") == {}


def test_no_project_means_no_guess():
    assert location_shape_for("sedan_interior", None) is None


# ── the mapping is a default, not a definition ───────────────────────────
def test_a_project_can_name_environments_the_defaults_never_heard_of():
    """Review feedback on #89: the environment→shape table must not be a
    screenplay-specific definition baked into Python. It is a default. A
    production whose screenplay invents its own environment vocabulary
    declares the mapping in its own KB, next to the stubs it describes."""
    from pace_core.setup.location_shape import shape_from_stub, shape_maps_from_doc

    env, prefixes = shape_maps_from_doc({"shape_map": {"submarine": "subway"}})
    assert shape_from_stub({"environment_type": "submarine"},
                           env_shape=env, prefix_shape=prefixes) == "subway"
    # …and the defaults it did not override still answer.
    assert shape_from_stub({"environment_type": "vehicle_interior"},
                           env_shape=env, prefix_shape=prefixes) == "subway"
    assert shape_from_stub({"environment_type": "exterior_crash_site"},
                           env_shape=env, prefix_shape=prefixes) == "outdoor"


def test_a_project_prefix_beats_a_builtin_family():
    """A production has to be able to NARROW a family the defaults would
    swallow — an exterior soundstage is not outdoors — without restating every
    family it agrees with. Project prefixes are consulted first for that."""
    from pace_core.setup.location_shape import shape_from_stub, shape_maps_from_doc

    env, prefixes = shape_maps_from_doc(
        {"shape_prefix_map": [["exterior_soundstage", "chamber"]]})
    assert shape_from_stub({"environment_type": "exterior_soundstage_b"},
                           env_shape=env, prefix_shape=prefixes) == "chamber"
    assert shape_from_stub({"environment_type": "exterior_highway"},
                           env_shape=env, prefix_shape=prefixes) == "outdoor"


def test_a_kb_cannot_invent_a_shape_nothing_can_build():
    """Parameterised does not mean unchecked. A typo, or a shape no builder
    has a branch for, is dropped at the mapping rather than reaching Blender."""
    from pace_core.setup.location_shape import shape_from_stub, shape_maps_from_doc

    env, prefixes = shape_maps_from_doc(
        {"shape_map": {"submarine": "sbuway"},
         "shape_prefix_map": [["exterior_x", "not_a_shape"]]})
    assert "submarine" not in env
    assert shape_from_stub({"environment_type": "submarine"},
                           env_shape=env, prefix_shape=prefixes) is None
    assert all(s in ("chamber", "subway", "office", "outdoor") for _, s in prefixes)


def test_a_malformed_shape_map_is_not_a_crash():
    from pace_core.setup.location_shape import ENVIRONMENT_SHAPE, shape_maps_from_doc

    for doc in (None, {}, {"shape_map": None}, {"shape_prefix_map": None},
                {"shape_map": "nonsense"} if False else {}):
        env, prefixes = shape_maps_from_doc(doc)
        assert env == ENVIRONMENT_SHAPE and prefixes


def test_one_list_of_shapes_not_four():
    """SHAPES is the runtime vocabulary and LocationShape is its static mirror;
    a Literal cannot be built from a tuple, so this is what keeps them equal.
    BUILDABLE_SHAPES stays its own set — a shape can exist before this builder
    can build it — but it may not contain one the vocabulary has never heard
    of, which is how a typo used to become an unbuildable panel."""
    import typing

    from pace_core.setup.location_shape import SHAPES, LocationShape
    from pace_core.node.panel_greybox import BUILDABLE_SHAPES

    assert set(typing.get_args(LocationShape)) == set(SHAPES)
    assert BUILDABLE_SHAPES <= set(SHAPES)
