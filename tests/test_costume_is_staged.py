"""A registered costume reaches the greybox as garment tones.

On the paper's beat pilot the same declared shirt came back a different shirt
in every panel, with the character's plate as a reference and without: the
greybox is one flat grey, and denoised from it the sampler chose the garment
from the body's shape. The costume's own words now set the tone of the proxy's
shirt and trousers in the control image.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.pai_compat import costume_text  # noqa: E402
from pace_core.node.panel_greybox import garment_tones  # noqa: E402


def test_a_dark_shirt_and_dark_jeans():
    t = garment_tones("a plain charcoal cotton t-shirt and dark denim jeans")
    assert t["top"] == {"tone": 0.22, "sleeve": "short"}
    assert t["bottom"] == {"tone": 0.28}


def test_a_faded_shirt_reads_lighter_than_the_greybox():
    t = garment_tones("a faded graphic t-shirt and dark jeans")
    assert t["top"]["tone"] > 0.55 > t["bottom"]["tone"]


def test_a_long_sleeved_garment_is_long_sleeved():
    assert garment_tones("a black wool sweater and grey trousers")["top"]["sleeve"] == "long"


def test_a_garment_named_without_a_colour_stays_grey():
    assert garment_tones("a cotton t-shirt and jeans") == {}
    assert garment_tones("a white shirt and jeans") == {"top": {"tone": 0.88, "sleeve": "long"}}


def test_no_costume_no_tones():
    assert garment_tones(None) == {}
    assert garment_tones("") == {}


def test_the_costume_follows_the_age_state_then_the_default():
    entry = {"costumes": {"default": "a grey coat", "child_7": "a red jumper"}}
    assert costume_text(entry, "child_7") == "a red jumper"
    assert costume_text(entry, "adult_50") == "a grey coat"
    assert costume_text({"costumes": " a white shirt. "}, None) == "a white shirt."
    assert costume_text({}, "adult_50") is None
    assert costume_text(None) is None


# ── the garment is a library entry, and the shot names it ────────────────
def test_a_named_wardrobe_entry_beats_the_character_default():
    """`costume_id` is what a shot says about this garment; the registry is
    only what the character wears when no shot says anything. A field that
    nothing preferred would be decoration."""
    from pace_core.pai_compat import wardrobe_of

    props = {"props": {"abigail_costume": {
        "id": "abigail_costume", "category": "wearable",
        "anchor": "a fitted slate-blue technical jacket"}}}
    entry = {"costumes": {"default": "a grey coat"}}
    wd = wardrobe_of(props, {"character_id": "abigail",
                             "costume_id": "abigail_costume"})
    assert costume_text(entry, None, wardrobe=wd) == \
        "a fitted slate-blue technical jacket"
    assert costume_text(entry, None, wardrobe=None) == "a grey coat"


def test_an_unresolvable_costume_id_falls_back_rather_than_blanks():
    """A typo in the id must not undress the character in the control image."""
    from pace_core.pai_compat import wardrobe_of

    entry = {"costumes": {"default": "a grey coat"}}
    for subject in ({"costume_id": "no_such_costume"}, {"costume_id": "  "}, {}):
        wd = wardrobe_of({"props": {}}, subject)
        assert costume_text(entry, None, wardrobe=wd) == "a grey coat"


def test_a_props_library_given_as_a_list_resolves_the_same():
    from pace_core.pai_compat import wardrobe_of

    entry = {"id": "lucas_costume", "anchor": "a soft olive sweatshirt"}
    assert wardrobe_of([entry], {"costume_id": "lucas_costume"}) is entry


# ── hair, from the same description the prompt reads ─────────────────────
def test_any_named_hair_is_capped():
    """A lens behind a bald proxy delivers a bald back of the head under a
    prompt that names hair: short, curly or shoulder-length. A reverse shot
    over Abigail's shoulder came back as a helmet before long hair had a
    shell."""
    from pace_core.node.panel_greybox import HAIR_STYLES, hair_style

    assert hair_style("average build, Hispanic, with short black hair flecked "
                      "with grey and a relaxed demeanor.") == "short"
    assert hair_style("slim, with tight curly hair and a wide smile") == "curly"
    assert hair_style("average build, Caucasian, with shoulder-length blonde "
                      "hair and a thoughtful expression") == "long"
    assert hair_style("a bald man") is None and hair_style(None) is None
    assert set(filter(None, (hair_style("short hair"), hair_style("curly hair")))) <= set(HAIR_STYLES)


def test_a_garment_named_by_hue_gets_a_tone():
    """An olive sweatshirt and a slate-blue jacket were read as no garment
    and a white tee respectively: the hue was unknown, and the tee under
    the jacket was the first garment with a colour it recognised."""
    lucas = garment_tones("a soft olive crew-neck sweatshirt and charcoal tapered trousers")
    assert lucas["top"]["sleeve"] == "long" and 0.3 < lucas["top"]["tone"] < 0.5
    abigail = garment_tones("a fitted slate-blue technical jacket with a standing "
                            "collar over a white tee, and dark tapered trousers")
    assert abigail["top"] == {"tone": 0.38, "sleeve": "long"}
    assert garment_tones("a faded graphic t-shirt and dark jeans")["top"]["sleeve"] == "short"
