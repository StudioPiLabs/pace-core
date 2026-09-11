"""Character-name resolution between shot documents and the registry.

Shot documents name people the way the screenplay does ("李明"); the
registry is keyed on an ASCII slug ("li_ming"). Before `resolve_character`
these were compared with a plain dict lookup, which missed silently: every
caller read the miss as "this character has no identity data", so panels
rendered with nothing locking appearance and the character came out as a
different person in each scene. These tests pin the resolution order and,
just as importantly, that a genuinely unknown name still fails.
"""
from pace_core.pai_compat import (resolve_character, unresolved_characters,
                                     lora_trigger_for)

KB = {
    "_schema_version": "pai-0.3",           # metadata keys must be skipped
    "li_ming": {
        "trigger": "li_ming_man",
        "aliases": ["李明"],
        "lora": {"trigger_word": "li_ming_man", "path": "/x/li_ming.safetensors"},
    },
    "shen_mi_ren": {"trigger": "shen_mi_ren_man", "name_zh": "神秘人"},
    "plain": {"trigger": "plain_man"},
}


def test_exact_key_wins():
    assert resolve_character("li_ming", KB)["trigger"] == "li_ming_man"


def test_alias_resolves():
    assert resolve_character("李明", KB)["trigger"] == "li_ming_man"


def test_name_zh_resolves():
    assert resolve_character("神秘人", KB)["trigger"] == "shen_mi_ren_man"


def test_trigger_resolves():
    assert resolve_character("plain_man", KB)["trigger"] == "plain_man"


def test_unknown_name_still_misses():
    # The point of the fix is to resolve real aliases, not to match loosely.
    assert resolve_character("幽灵", KB) == {}
    assert resolve_character("", KB) == {}


def test_metadata_keys_are_not_characters():
    assert resolve_character("pai-0.3", KB) == {}


def test_unresolved_characters_reports_only_the_bad_ones():
    assert unresolved_characters(["李明", "神秘人", "幽灵"], KB) == ["幽灵"]


def test_unresolved_strips_age_state_suffix():
    assert unresolved_characters(["李明@adult"], KB) == []


def test_lora_trigger_follows_an_alias():
    # The regression that mattered: a display-name reference used to yield
    # no trigger and therefore no identity conditioning.
    assert lora_trigger_for(["李明"], {}, KB) == "li_ming_man"


def test_lora_trigger_none_for_unknown():
    assert lora_trigger_for(["幽灵"], {}, KB) is None


# ── appearance references ────────────────────────────────────────────

REF_KB = {
    "li_ming": {"aliases": ["李明"],
                "reference_images": ["shared/a_front.png", "shared/a_side.png"]},
    "shen_mi_ren": {"name_zh": "神秘人",
                    "reference_images": "shared/b_front.png"},   # bare string
    "aged": {"reference_images": ["default.png"],
             "reference_images_by_age_state": {"old": ["old.png"]}},
    "no_refs": {"trigger": "x"},
}


def test_reference_resolves_through_alias():
    from pace_core.pai_compat import reference_images_for
    assert reference_images_for(["李明"], {}, REF_KB) == ["shared/a_front.png"]


def test_reference_accepts_a_bare_string():
    from pace_core.pai_compat import reference_images_for
    assert reference_images_for(["神秘人"], {}, REF_KB) == ["shared/b_front.png"]


def test_every_character_in_a_two_hander_contributes():
    # A scene with two people needs both faces held, not just the lead's.
    from pace_core.pai_compat import reference_images_for
    assert reference_images_for(["李明", "神秘人"], {}, REF_KB) == [
        "shared/a_front.png", "shared/b_front.png"]


def test_per_character_count_is_respected():
    from pace_core.pai_compat import reference_images_for
    assert reference_images_for(["李明"], {}, REF_KB, per_character=2) == [
        "shared/a_front.png", "shared/a_side.png"]


def test_age_state_reference_wins_over_default():
    from pace_core.pai_compat import reference_images_for
    assert reference_images_for(["aged"], {"aged": "old"}, REF_KB) == ["old.png"]
    assert reference_images_for(["aged"], {"aged": "young"}, REF_KB) == ["default.png"]


def test_characters_without_references_contribute_nothing():
    from pace_core.pai_compat import reference_images_for
    assert reference_images_for(["no_refs", "幽灵"], {}, REF_KB) == []


def test_limit_caps_the_multi_reference_channel():
    from pace_core.pai_compat import reference_images_for
    kb = {f"c{i}": {"reference_images": [f"{i}.png"]} for i in range(15)}
    assert len(reference_images_for(list(kb), {}, kb, limit=10)) == 10


def test_duplicate_references_are_not_stacked():
    from pace_core.pai_compat import reference_images_for
    kb = {"a": {"reference_images": ["same.png"]},
          "b": {"reference_images": ["same.png"]}}
    assert reference_images_for(["a", "b"], {}, kb) == ["same.png"]
