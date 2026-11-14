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


def test_every_line_lands_on_its_shot_and_undesigned_camera_stays_unknown(tmp_path, monkeypatch):
    text = '1 INT. SHOP - DAY\n\nada opens the door.\n\nADA: Morning.\n\nben wipes the counter.\n\nBEN (O.S.): Shut it.\n'
    scene = {'heading': '1 INT. SHOP - DAY', 'location_raw': 'shop', 'time_of_day': 'day',
             'characters_present': ['ada', 'ben'],
             'key_actions': ['ada opens the door', 'ben wipes the counter'],
             'action_evidence': [{'quote': 'ada opens the door.'}, {'quote': 'ben wipes the counter.'}],
             'dialogue': [{'speaker': 'ada', 'text': 'Morning.', 'on_screen': True, 'after_action': 0},
                          {'speaker': 'ben', 'text': 'Shut it.', 'on_screen': False, 'after_action': 1}]}
    run_split(tmp_path, monkeypatch, text, [[scene]])
    doc = json.loads(next((tmp_path / 'scenes').glob('*.json')).read_text())
    first, second = doc['shots']
    assert [d['text'] for d in first['events']['dialogues']] == ['Morning.']
    assert (second['events']['dialogues'][0]['speaker'], second['events']['dialogues'][0]['on_screen']) == ('ben', False)
    assert [line['kind'] for line in doc['narrative_meta']['vo_lines']] == ['vo']
    assert first['camera']['creative_intent']['shot_size'] is None
    assert first['setup']['primary_focus']['ref'] == 'ada'
    assert doc['shot_defaults']['setup']['environment']['style'] is None


def test_a_camera_frame_range_sets_the_shot_length_and_playback_frames():
    from pace_core.events.shot_duration import playback_frames, shot_duration_s

    shot = {'camera': {'fps': {'num': 24, 'denom': 1}, 'frame_range': [0, 71]},
            'events': {'actions': [{'duration_hint_s': 5}]}}
    assert shot_duration_s(shot) == (3.0, 'camera')
    assert playback_frames(shot) == (72, 24.0)


def test_an_exchange_is_covered_shot_reverse_shot_and_bad_coverage_falls_back(tmp_path, monkeypatch):
    text = ('1 INT. SHOP - DAY\n\nada leans on the counter.\n\nADA: Pay up.\n\nBEN: Tomorrow.\n\nADA: Today.\n')
    exchange = [{'speaker': 'ada', 'text': 'Pay up.', 'after_action': 0},
                {'speaker': 'ben', 'text': 'Tomorrow.', 'after_action': 0},
                {'speaker': 'ada', 'text': 'Today.', 'after_action': 0}]

    def shot(lines, who, to, side, actions=()):
        return {'actions': list(actions), 'lines': lines, 'shot_size': 'medium_close_up', 'angle': 'eye_level',
                'position': 'ots', 'camera_side': side, 'focus': who, 'speaker': who, 'eyeline_to': to,
                'basis': 'Pay up.'}
    scene = {'heading': '1 INT. SHOP - DAY', 'location_raw': 'shop', 'time_of_day': 'day',
             'characters_present': ['ada', 'ben'], 'key_actions': ['ada leans on the counter'],
             'action_evidence': [{'quote': 'ada leans on the counter.'}], 'dialogue': exchange,
             'axis': {'between': ['ada', 'ben']},
             'coverage': [shot([0], 'ada', 'ben', 'A', actions=[0]), shot([1], 'ben', 'ada', 'A'), shot([2], 'ada', 'ben', 'A')]}
    run_split(tmp_path, monkeypatch, text, [[scene]])
    doc = json.loads(next((tmp_path / 'scenes').glob('*.json')).read_text())
    assert [[d['speaker'] for d in s['events']['dialogues']] for s in doc['shots']] == [['ada'], ['ben'], ['ada']]
    reverse = doc['shots'][1]
    assert reverse['camera']['creative_intent']['shot_size'] == 'medium_close_up'
    assert reverse['camera']['extrinsics']['axis_side'] == 'A'
    assert reverse['setup']['subjects'][0]['gaze']['target_ref'] == 'ada'
    assert reverse['camera']['provenance']['shot_size'] == 'estimated'
    assert doc['narrative_meta']['coverage'] == {'source': 'skill', 'warnings': []}

    scene['coverage'] = [shot([0, 2], 'ada', 'ben', 'A', actions=[0]), shot([1], 'ben', 'ada', 'A')]
    for old in (tmp_path / 'scenes').glob('*.json'):
        old.unlink()
    run_split(tmp_path, monkeypatch, text, [[scene]])
    doc = json.loads(next((tmp_path / 'scenes').glob('*.json')).read_text())
    assert len(doc['shots']) == 1 and doc['shots'][0]['camera']['creative_intent']['shot_size'] is None
    assert doc['narrative_meta']['coverage']['source'] == 'one_per_action'
    assert doc['narrative_meta']['coverage']['warnings']


def test_coverage_slips_are_mended_and_the_listener_looks_back(tmp_path, monkeypatch):
    text = '1 INT. SHOP - DAY\n\nada leans on the counter.\n\nben checks the clock.\n\nADA: Pay up.\n\nBEN: Tomorrow.\n\nADA: Today.\n'
    scene = {'heading': '1 INT. SHOP - DAY', 'location_raw': 'shop', 'time_of_day': 'day',
             'characters_present': ['ada', 'ben'], 'key_actions': ['ada leans on the counter', 'ben checks the clock'],
             'action_evidence': [{'quote': 'ada leans on the counter.'}, {'quote': 'ben checks the clock.'}],
             'dialogue': [{'speaker': 'ada', 'text': 'Pay up.', 'after_action': 1},
                          {'speaker': 'ben', 'text': 'Tomorrow.', 'after_action': 1},
                          {'speaker': 'ada', 'text': 'Today.', 'after_action': 1}],
             # line 1 repeated, line 2 missing, action 1 unshown
             'coverage': [{'actions': [0], 'lines': [0, 1], 'speaker': 'ada', 'eyeline_to': 'ben'},
                          {'actions': [], 'lines': [1], 'speaker': 'ben', 'eyeline_to': 'clock'}]}
    run_split(tmp_path, monkeypatch, text, [[scene]])
    doc = json.loads(next((tmp_path / 'scenes').glob('*.json')).read_text())
    assert doc['narrative_meta']['coverage']['source'] == 'skill'
    assert [[d['text'] for d in s['events']['dialogues']] for s in doc['shots']] == [['Pay up.', 'Tomorrow.', 'Today.']]
    assert len(doc['narrative_meta']['coverage']['warnings']) == 3
    gaze = {s['character_id']: s['gaze']['target_ref'] for s in doc['shots'][0]['setup']['subjects']}
    assert gaze == {'ada': 'ben', 'ben': 'ada'}
