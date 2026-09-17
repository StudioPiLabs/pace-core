"""A pose keyword must match a word, not a run of letters inside one.

`pose_key_for` tested its keywords with `in`, so "lay" matched "player",
"layer" and "overlay", and "lies"/"lying" matched "flies"/"flying". Because
`lying` is the first bucket tried, every one of those staged the subject flat
on the floor -- including a shot whose only body verb is "approaches".
"""
from __future__ import annotations

from pace_core.node.panel_greybox import pose_key_for


def test_lay_inside_another_word_does_not_lay_the_subject_down():
    for text in ("Lena breaks into a playful smile, tilting her head up",
                 'A screen reads "opening access layer"; Lena speaks quietly.',
                 "Charts overlay a screen, sharp lines and chaotic rising curves"):
        assert pose_key_for(text) == "standing", text


def test_flying_and_flies_are_not_lying_and_lies():
    assert pose_key_for("Lena drives a flying knee hard into Elias's stomach") == "standing"
    assert pose_key_for("sparks flying from the cable joints") == "standing"
    assert pose_key_for("As less data flies past, the pressure eases") == "standing"


def test_a_real_verb_still_wins_over_a_substring_in_the_same_line():
    """`lying` is tried before `walking`; a spurious hit used to take the shot."""
    assert pose_key_for("Dave approaches the wall; a floating warning: RESTRICTED LAYER") == "walking"
    assert pose_key_for("Dave reluctantly reaches out; it flies along his gesture") == "reaching"


def test_the_poses_that_were_meant_still_match():
    assert pose_key_for("Dave kneels on one knee, gasping") == "kneeling"
    assert pose_key_for("she lies motionless on the floor") == "lying"
    assert pose_key_for("the body lay on the ground") == "lying"
    assert pose_key_for("he sits at the table") == "sitting"
    assert pose_key_for("she walks toward the door") == "walking"
    assert pose_key_for("he pushes the door open") == "reaching"
    assert pose_key_for("she stands by the window") == "standing"


def test_multi_word_keys_still_match():
    assert pose_key_for("he holds out a hand") == "reaching"
    assert pose_key_for("the figure is sprawled on the ground") == "lying"


def test_empty_and_missing_fall_back_to_the_default():
    assert pose_key_for(None) == "standing"
    assert pose_key_for("   ") == "standing"
    assert pose_key_for("no posture words here at all", default="sitting") == "sitting"
