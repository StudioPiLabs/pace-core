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
