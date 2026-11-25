import json
from types import SimpleNamespace

from pace_core.breakdown import extract_characters as extract
from pace_core.node import panel_modeling as modeling


def scene():
    return {"scene_id": "scene_01", "shots": [{"shot_id": "shot_01", "setup": {
        "subjects": [{"character_id": "otter"}, {"character_id": "moth"}]},
        "panels": [{"id": "scene_01_shot_01_panel_0001"}]}]}


def test_source_subjects_include_silent_animals_without_renaming():
    entries = extract.declared_characters([scene()])
    assert [e["name"] for e in entries] == ["otter", "moth"]
    assert all(e["metadata_status"] == "source_only" for e in entries)
    assert all(e["scenes"] == ["scene_01"] for e in entries)


def test_model_omissions_cannot_produce_a_successful_empty_cast(tmp_path, monkeypatch):
    from pace_core.breakdown import scenes_kb

    chars = tmp_path / "characters.json"
    monkeypatch.setattr(scenes_kb, "load_all_scenes", lambda **kw: {"scenes": [scene()]})
    monkeypatch.setattr(extract, "paths_for", lambda project: SimpleNamespace(
        chars_file=chars, props_file=tmp_path / "props.json", loc_stubs_file=tmp_path / "location_stubs.json"))
    monkeypatch.setattr(extract, "extract_characters_from_scenes", lambda *a, **kw: ([], 0))
    result = extract.run(project="animals", reference_strategy="prompt-only")
    data = json.loads(chars.read_text())
    assert [data[k]["name"] for k in result["extracted_ids"]] == ["otter", "moth"]
    assert result["source_only_ids"] == result["extracted_ids"]
    entry = data[result["extracted_ids"][0]]
    assert entry["_extracted"]["metadata_status"] == "source_only"
    assert entry["entity_type"] == "unknown"


def test_merge_preserves_artist_fields_and_stores_nonhuman_metadata(tmp_path):
    chars = tmp_path / "characters.json"
    chars.write_text(json.dumps({"otter": {"anchor": "Artist description", "body_proxy_3d": "artist.obj"}}))
    extract.merge_into_characters_json([{"name": "otter", "anchor": "AI description", "entity_type": "animal",
                                         "species": "otter", "body_proxy_3d": "new.obj"}], chars)
    data = json.loads(chars.read_text())["otter"]
    assert data["anchor"] == "Artist description"
    assert data["body_proxy_3d"] == "artist.obj"
    assert data["entity_type"] == "animal"


def test_animal_mesh_is_independent_of_human_fallback_files(tmp_path):
    chars = tmp_path / "characters.json"
    paths = SimpleNamespace(chars_file=chars, storage=tmp_path, meshes_dir=tmp_path / "meshes")
    meshes = paths.meshes_dir / "characters"
    meshes.mkdir(parents=True)
    (meshes / "smplx_standing_175.obj").write_text("human")
    chars.write_text(json.dumps({"otter": {"entity_type": "animal"}}))
    selected = modeling._subject_mesh_for({"character_id": "otter"}, paths, "standing")
    assert selected == str(meshes / "otter.obj")
    chars.write_text(json.dumps({"otter": {"entity_type": "animal",
        "body_proxy_3d": {"mesh_file": "meshes/characters/otter.obj"}}}))
    assert modeling._subject_mesh_for({"character_id": "otter"}, paths, "standing") == selected



def test_a_posed_build_stages_the_characters_own_smplx_body_over_its_registered_mesh(tmp_path):
    chars = tmp_path / "characters.json"
    paths = SimpleNamespace(chars_file=chars, storage=tmp_path, meshes_dir=tmp_path / "meshes")
    meshes = paths.meshes_dir / "characters"
    meshes.mkdir(parents=True)
    (meshes / "smplx_sitting_c1.obj").write_text("posed")
    chars.write_text(json.dumps({"c1": {"name": "Mara", "body_proxy_3d": {"mesh_file": "meshes/characters/c1.glb"}}}))
    subject = {"character_id": "c1"}
    assert modeling._subject_mesh_for(subject, paths, "sitting") == str(tmp_path / "meshes/characters/c1.glb")
    assert modeling._subject_mesh_for(subject, paths, "sitting", posed=True) == str(meshes / "smplx_sitting_c1.obj")


def test_a_merge_that_changes_nothing_leaves_the_registry_file_alone(tmp_path):
    chars = tmp_path / "characters.json"
    chars.write_text(json.dumps({"otter": {"name": "otter", "anchor": "A"}}))
    seen = [{"name": "otter", "metadata_status": "source_only"}]
    extract.merge_into_characters_json(seen, chars)
    before = chars.stat().st_mtime_ns
    extract.merge_into_characters_json(seen, chars)   # every build seeds the same cast again
    assert chars.stat().st_mtime_ns == before and not list(tmp_path.glob(".*.tmp"))
