"""Scene boundaries and quotes stay tied to the screenplay, not model defaults."""
import json

import pytest

from pace_core.breakdown import split_script as split
from pace_core.breakdown.extract_script_ir import ground


def run_split(tmp_path, monkeypatch, text, replies):
    from pace_core import llm_client

    registry = tmp_path / 'models.json'
    registry.write_text(json.dumps({'test': {}}))
    responses = iter(replies)
    monkeypatch.setattr(llm_client, 'call_model', lambda *args: (json.dumps({'scenes': next(responses)}), 0))
    return split.split_text_to_pai(text, source_label='script.txt', model='test',
                                   models_file=registry, locations_dir=tmp_path / 'locations',
                                   output_dir=tmp_path / 'scenes')


def test_identical_numbered_headings_remain_separate_with_notes(tmp_path, monkeypatch):
    text = ('1 EXT. RIDGE - DAY\n\nAda falls.\n\n[1] Keep the weapon in her left hand.\n\n'
            '2 EXT. RIDGE - DAY\n\nBen catches Ada.\n')
    first = {'heading': 'invented', 'location_raw': 'ridge', 'time_of_day': 'day',
             'key_actions': ['Ada falls'], 'action_evidence': [{'quote': 'Ada falls.'}],
             'source_notes': [{'kind': 'footnote', 'quote': '[1] Keep the weapon in her left hand.'}]}
    second = {'heading': 'invented', 'location_raw': 'ridge', 'time_of_day': 'day',
              'key_actions': ['Ben catches Ada'], 'action_evidence': [{'quote': 'Ben catches Ada.'}]}
    result = run_split(tmp_path, monkeypatch, text, [[first], [second]])
    assert len(result['scenes']) == 2
    assert [s['scene_heading'] for s in result['scenes']] == ['1 EXT. RIDGE - DAY', '2 EXT. RIDGE - DAY']
    evidence = json.loads((tmp_path / 'scenes/breakdown_evidence.json').read_text())
    note = evidence['scenes'][0]['notes'][0]
    assert note['grounded'] and text[note['evidence']['start']:note['evidence']['end']] == note['evidence']['source_text']
    assert not evidence['human_verified'] and not evidence['training_eligible']
    assert 'source_notes' not in result['scenes'][0]  # Provenance is separate from canonical fields.


def test_long_scene_fragments_preserve_repeated_actions(tmp_path, monkeypatch):
    original_parts = split._script_parts
    monkeypatch.setattr(split, '_script_parts', lambda text: original_parts(text, 55))
    text = '1 EXT. RIDGE - DAY\n\nAda waits.\n\n' + 'Cold wind blows. ' * 4 + '\n\nAda waits.'
    parts = original_parts(text, 55)
    replies = [[{'key_actions': ['Ada waits'], 'action_evidence': [{'quote': 'Ada waits.'}]}]
               if 'Ada waits.' in p['text'] else [{'key_actions': []}] for p in parts]
    result = run_split(tmp_path, monkeypatch, text, replies)
    assert len(result['scenes']) == 1
    assert result['scenes'][0]['narrative_meta']['key_actions'] == ['Ada waits', 'Ada waits']
    actions = result['source_evidence']['scenes'][0]['actions']
    assert all(a['grounded'] for a in actions)
    assert actions[0]['evidence']['start'] < actions[1]['evidence']['start']


def test_a_model_cannot_split_one_explicit_heading(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='split into 2 scenes'):
        run_split(tmp_path, monkeypatch, '1 EXT. RIDGE - DAY\n\nAda falls.', [[{}, {}]])


def test_missing_and_wrong_scene_quotes_stay_ungrounded(tmp_path, monkeypatch):
    text = '1 EXT. RIDGE - DAY\n\nAda falls.\n\n2 INT. ROOM - NIGHT\n\nBen sleeps.'
    result = run_split(tmp_path, monkeypatch, text, [[{
        'key_actions': ['Ben sleeps', 'Ada flies'], 'action_evidence': [{'quote': 'Ben sleeps.'}],
        'source_notes': [{'kind': 'footnote', 'quote': 'A camera flies overhead.'}]}], [{'key_actions': []}]])
    record = result['source_evidence']['scenes'][0]
    assert all(not n['grounded'] for n in record['actions'] + record['notes'])


def test_whitespace_normalized_quotes_still_have_real_offsets():
    text = 'Earlier scene.\n\n她扶起\n  受伤的人。'
    ir = ground({'events': [{'local_id': 'e1', 'quote': '她扶起 受伤的人。'}]}, text,
                {'start': 16, 'end': len(text)})
    event = ir['events'][0]
    assert event['grounded']
    ev = event['evidence']
    assert text[ev['start']:ev['end']] == ev['source_text'] == '她扶起\n  受伤的人。'


def test_prose_without_headings_remains_supported(tmp_path, monkeypatch):
    text = 'Ada walks through snow. Later, Ben lights a fire.'
    result = run_split(tmp_path, monkeypatch, text, [[
        {'key_actions': ['Ada walks'], 'action_evidence': [{'quote': 'Ada walks through snow.'}]},
        {'key_actions': ['Ben lights a fire'], 'action_evidence': [{'quote': 'Ben lights a fire.'}]}]])
    assert len(result['scenes']) == 2
    assert all(s['actions'][0]['grounded'] for s in result['source_evidence']['scenes'])


def test_page_numbers_are_not_usable_footnotes(tmp_path, monkeypatch):
    result = run_split(tmp_path, monkeypatch, 'INT. HUT - DAY\n\n2\n\nMira kneels.', [[{
        'key_actions': ['Mira kneels'], 'action_evidence': [{'quote': 'Mira kneels.'}],
        'source_notes': [{'kind': 'footnote', 'quote': '2'}]}]])
    note = result['source_evidence']['scenes'][0]['notes'][0]
    assert note['grounded'] and not note['usable_as_context']
    assert note['issue'] == 'page_number_or_note_marker_without_content'


def test_wrong_action_shape_is_rejected_before_any_scene_is_published(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='list of strings'):
        run_split(tmp_path, monkeypatch, 'INT. HUT - DAY\n\nMira kneels.', [[{
            'key_actions': [{'action': 'Mira kneels', 'quote': 'Mira kneels.'}]}]])
    assert not (tmp_path / 'scenes').exists()
