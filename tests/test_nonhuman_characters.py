import json
from types import SimpleNamespace

from pace_core.breakdown import extract_characters as extract
from pace_core.node import panel_modeling as modeling


def scene():
    return {"scene_id": "scene_01", "shots": [{"shot_id": "shot_01", "setup": {
        "subjects": [{"character_id": "rabbit"}, {"character_id": "butterfly"}]},
        "panels": [{"id": "scene_01_shot_01_panel_0001"}]}]}


def test_source_subjects_include_silent_animals_without_renaming():
    entries = extract.declared_characters([scene()])
    assert [e["id"] for e in entries] == ["rabbit", "butterfly"]
    assert all(e["metadata_status"] == "source_only" for e in entries)
    assert all(e["scenes"] == ["scene_01"] for e in entries)


def test_model_omissions_cannot_produce_a_successful_empty_cast(tmp_path, monkeypatch):
    from pace_core.breakdown import scenes_kb

    chars = tmp_path / "characters.json"
    monkeypatch.setattr(scenes_kb, "load_all_scenes", lambda **kw: {"scenes": [scene()]})
    monkeypatch.setattr(extract, "paths_for", lambda project: SimpleNamespace(chars_file=chars))
    monkeypatch.setattr(extract, "extract_characters_from_scenes", lambda *a, **kw: ([], 0))
    result = extract.run(project="animals", reference_strategy="prompt-only")
    assert result["extracted_ids"] == ["rabbit", "butterfly"]
    assert result["source_only_ids"] == ["rabbit", "butterfly"]
    data = json.loads(chars.read_text())
    assert data["rabbit"]["_extracted"]["metadata_status"] == "source_only"
    assert data["rabbit"]["entity_type"] == "unknown"


def test_merge_preserves_artist_fields_and_stores_nonhuman_metadata(tmp_path):
    chars = tmp_path / "characters.json"
    chars.write_text(json.dumps({"rabbit": {"anchor": "Artist description", "body_proxy_3d": "artist.obj"}}))
    extract.merge_into_characters_json([{"id": "rabbit", "anchor": "AI description", "entity_type": "animal",
                                         "species": "rabbit", "body_proxy_3d": "new.obj"}], chars)
    data = json.loads(chars.read_text())["rabbit"]
    assert data["anchor"] == "Artist description"
    assert data["body_proxy_3d"] == "artist.obj"
    assert data["entity_type"] == "animal"


def test_animal_mesh_is_independent_of_human_fallback_files(tmp_path):
    chars = tmp_path / "characters.json"
    paths = SimpleNamespace(chars_file=chars, storage=tmp_path, meshes_dir=tmp_path / "meshes")
    meshes = paths.meshes_dir / "characters"
    meshes.mkdir(parents=True)
    (meshes / "smplx_standing_175.obj").write_text("human")
    chars.write_text(json.dumps({"rabbit": {"entity_type": "animal"}}))
    selected = modeling._subject_mesh_for({"character_id": "rabbit"}, paths, "standing")
    assert selected == str(meshes / "rabbit.obj")
    chars.write_text(json.dumps({"rabbit": {"entity_type": "animal",
        "body_proxy_3d": {"mesh_file": "meshes/characters/rabbit.obj"}}}))
    assert modeling._subject_mesh_for({"character_id": "rabbit"}, paths, "standing") == selected

