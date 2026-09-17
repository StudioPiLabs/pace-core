"""A subject the lens sees from behind is described without a face.

On the over-the-shoulder of one scene's "Dad swivels his chair around to face
Hal", with geometry, proxy hair and seed held fixed, the only variable that
decided whether Gus stayed turned toward his son was his description: given
his face ("weathered features and a determined expression") the sampler turned
him to the lens or painted a face onto the back of his head; described from
behind, it kept him there.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.compilers.compile_common import (  # noqa: E402
    _character_back_hint, _hair_phrase, seen_from_behind,
)

KB = {
    "gus": {"sex": "male", "trigger": "gus_male",
             "anchor": "athletic build, Caucasian, with short greying dark hair, "
                       "weathered features and a determined expression.",
             "costumes": {"default": "a plain charcoal cotton t-shirt and dark denim jeans"}},
    "fay": {"sex": "female",
              "generic_anchor": "slender build, Caucasian, with long brown hair "
                                "greying at the temples and bright, lively eyes"},
    "bald": {"sex": "male", "generic_anchor": "stocky build, with a broad smile"},
}


def _shot(position: str | None) -> dict:
    return {"camera": {"extrinsics": {"position": position}}}


def test_the_back_view_keeps_hair_build_and_clothing():
    out = _character_back_hint("gus@adult_50", KB)
    assert out.startswith("50-year-old man, seen from behind")
    assert "his face not visible" in out
    assert "athletic build" in out
    assert "the back of his head with short greying dark hair" in out
    assert "wearing a plain charcoal cotton t-shirt and dark denim jeans" in out


def test_the_back_view_drops_the_face():
    out = _character_back_hint("gus@adult_50", KB)
    for face in ("weathered features", "determined expression", "gus_male"):
        assert face not in out


def test_hair_is_cut_where_the_face_clause_starts():
    assert _hair_phrase(KB["gus"]["anchor"]) == "short greying dark hair"
    assert _hair_phrase(KB["fay"]["generic_anchor"]) == \
        "long brown hair greying at the temples"
    assert _hair_phrase("average build, Hispanic, with short black hair "
                        "flecked with grey and a relaxed demeanor") == \
        "short black hair flecked with grey"


def test_a_descriptor_without_hair_says_nothing_about_hair():
    out = _character_back_hint("bald@adult_40", KB)
    assert "head with" not in out
    assert "broad smile" not in out


def test_behind_sees_every_back():
    assert seen_from_behind(_shot("behind"), ["gus", "fay"], {}) == {"gus", "fay"}


def test_an_ots_sees_the_backs_of_everyone_but_the_focus():
    pf = {"type": "character", "ref": "hal"}
    assert seen_from_behind(_shot("ots"), ["gus", "hal"], pf) == {"gus"}


def test_an_ots_without_a_focus_names_nobody():
    """The greybox picks the near subject from seat depth; the text compiler
    cannot see seats, and guessing would describe the wrong person's back."""
    assert seen_from_behind(_shot("ots"), ["gus", "hal"], {}) == set()


def test_positions_that_face_the_cast_see_no_backs():
    for pos in ("front", "three_quarter", "profile", None):
        assert seen_from_behind(_shot(pos), ["gus", "hal"],
                                {"type": "character", "ref": "hal"}) == set()
