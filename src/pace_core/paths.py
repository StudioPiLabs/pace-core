"""Project path resolution.

A project is a directory under PAI_PROJECTS_ROOT that contains `kb/`. Values
come from the environment, or from `configure()` when a host application owns
the layout.

Layout inside one project (PAI_PROJECTS_ROOT/<slug>/):

  kb/              scenes/scene_NN.json, characters.json, props.json,
                   location_stubs.json, film.json, shot_design.json
  02_assets/       project.toml, jobs.json, render_days/
  06_reference/    storyboards/ (renders, control_geometry), character_sheets/
  blender_scenes/  depth_maps/  meshes/  script/  05_output/  07_docs/
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


FILM_REPO_ROOT = _env_path("FILM_REPO_ROOT", Path.cwd())
PAI_PROJECTS_ROOT = _env_path("PAI_PROJECTS_ROOT", FILM_REPO_ROOT / "production" / "projects")
PAI_BRANCH = os.environ.get("PAI_BRANCH", "")
MODELS_FILE = _env_path("MODELS_FILE", FILM_REPO_ROOT / "assets" / "models.json")


class _Box:
    """Settings for the machine that runs Blender."""

    def __init__(self) -> None:
        self.blender_bin = os.environ.get("BLENDER_BIN") or "blender"


BOX = _Box()


def configure(*, repo_root: str | Path | None = None,
              projects_root: str | Path | None = None,
              branch: str | None = None,
              models_file: str | Path | None = None,
              blender_bin: str | None = None) -> None:
    """Point this package at a host application's layout."""
    global FILM_REPO_ROOT, PAI_PROJECTS_ROOT, PAI_BRANCH, MODELS_FILE
    if repo_root is not None:
        FILM_REPO_ROOT = Path(repo_root)
    if projects_root is not None:
        PAI_PROJECTS_ROOT = Path(projects_root)
    if branch is not None:
        PAI_BRANCH = branch
    if models_file is not None:
        MODELS_FILE = Path(models_file)
    if blender_bin is not None:
        BOX.blender_bin = blender_bin


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Every film-data path a caller needs, anchored to one project."""
    slug: str
    storage: Path
    kb_dir: Path
    scenes_dir: Path
    chars_file: Path
    props_file: Path
    loc_stubs_file: Path
    film_file: Path
    shot_design_file: Path
    voices_file: Path
    assets_subdir: Path
    project_meta_toml: Path
    comfy_workers_yaml: Path
    jobs_file: Path
    site_config_env: Path
    render_days_dir: Path
    storyboards_dir: Path
    renders_dir: Path
    greyboxes_dir: Path
    render_log_file: Path
    character_sheets_dir: Path
    lora_refs_dir: Path
    blender_scenes_dir: Path
    depth_dir: Path
    meshes_dir: Path
    dialogue_out_dir: Path
    raw_dialogue_dir: Path
    script_dir: Path
    output_dir: Path
    docs_dir: Path


def _project_root(slug: str) -> Path:
    return PAI_PROJECTS_ROOT / slug / PAI_BRANCH


# Canonical scene files are exactly `scene_<digits>.json`; enrichment and
# backup siblings share the prefix and must not be read as scenes.
_CANONICAL_SCENE_RE = re.compile(r"^scene_\d+\.json$")


def is_canonical_scene_file(path: Path | str) -> bool:
    return bool(_CANONICAL_SCENE_RE.fullmatch(Path(path).name))


def iter_canonical_scene_files(scenes_dir: Path) -> list[Path]:
    if not scenes_dir.is_dir():
        return []
    return sorted(p for p in scenes_dir.glob("scene_*.json") if is_canonical_scene_file(p))


def list_projects() -> list[str]:
    if not PAI_PROJECTS_ROOT.is_dir():
        return []
    return [d.name for d in sorted(PAI_PROJECTS_ROOT.iterdir())
            if d.is_dir() and (d / PAI_BRANCH / "kb").is_dir()]


def project_exists(slug: str) -> bool:
    return (_project_root(slug) / "kb").is_dir()


def paths_for(slug: str) -> ProjectPaths:
    """Resolve every path for one project; FileNotFoundError if it has no kb/."""
    root = _project_root(slug)
    kb = root / "kb"
    if not kb.is_dir():
        raise FileNotFoundError(
            f"project {slug!r} not found at {root} "
            f"(expected {kb} to exist; registered: {list_projects() or 'none'})")
    assets = root / "02_assets"
    refs = root / "06_reference"
    storyboards = refs / "storyboards"
    return ProjectPaths(
        slug=slug,
        storage=root,
        kb_dir=kb,
        scenes_dir=kb / "scenes",
        chars_file=kb / "characters.json",
        props_file=kb / "props.json",
        loc_stubs_file=kb / "location_stubs.json",
        film_file=kb / "film.json",
        shot_design_file=kb / "shot_design.json",
        voices_file=kb / "voices.json",
        assets_subdir=assets,
        project_meta_toml=assets / "project.toml",
        comfy_workers_yaml=assets / "comfy_workers.yaml",
        jobs_file=assets / "jobs.json",
        site_config_env=assets / "site_config.env",
        render_days_dir=assets / "render_days",
        storyboards_dir=storyboards,
        renders_dir=storyboards / "stage_b_renders",
        greyboxes_dir=storyboards / "control_geometry",
        render_log_file=storyboards / "stage_b_render_log.json",
        character_sheets_dir=refs / "character_sheets",
        lora_refs_dir=refs / "lora_refs",
        blender_scenes_dir=root / "blender_scenes",
        depth_dir=root / "depth_maps",
        meshes_dir=root / "meshes",
        dialogue_out_dir=root / "04_sfx" / "dialogue",
        raw_dialogue_dir=root / "04_sfx" / "raw_dialogue",
        script_dir=root / "script",
        output_dir=root / "05_output",
        docs_dir=root / "07_docs",
    )


__all__ = [
    "PACKAGE_DIR", "SRC_DIR", "FILM_REPO_ROOT", "PAI_PROJECTS_ROOT", "PAI_BRANCH",
    "MODELS_FILE", "BOX", "configure", "ProjectPaths", "paths_for", "list_projects",
    "project_exists", "is_canonical_scene_file", "iter_canonical_scene_files",
]
