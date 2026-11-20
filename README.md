# pace-core

The core of PACE (Precise AI Cinematic Expression). It contains:

- the typed storyboard schema;
- a script breakdown in which every extracted fact quotes the screenplay;
- solved camera geometry;
- Blender staging that renders one matte per declared subject;
- the checks that compare a delivered panel with the specification that produced it.

It accompanies the paper *PACE: Script Breakdown and Precise Cinematic Geometry as a Self-Auditing Specification* (Duan et al., 2026).

## Layout

| module | contents |
|---|---|
| `pace_core.breakdown` | screenplay parser, quote-grounded script IR, beat segmentation over a world-state timeline, breakdown verification and error injection, panel densification, tight-shot restaging |
| `pace_core.camera` | camera planner and trajectory compiler, LaMP motion DSL |
| `pace_core.setup` | composition solver (read-point aiming), location shapes |
| `pace_core.compilers` | prompt compilation (Flux 2), prompt projection |
| `pace_core.qc` | pre-render gate on the staged anchor |
| `pace_core.node` | Blender launcher, panel 3D model kernel, control-image flattening |
| `pace_core.types_v1`, `pace_core.pai_compat` | PACE schema types and field accessors |
| `pace_core.usd_export` | OpenUSD export |

## Install

```
uv add "pace-core @ git+https://github.com/StudioPiLabs/pace-core"
```

Optional extras:

- `usd`: OpenUSD export.
- `pdf`: pypdf, for PDF screenplays in `breakdown.split_script`.

`pace-scene-skills` is a git direct dependency (pinned tag), so installs need
network access to GitHub and a resolver that allows direct references.

Staging needs Blender 4.2 or later. `BlenderBox` ships the convert, passes and
panel-modeling kernels; the scene-assembly, shot-preview, depth-sequence and
GLB-export kernels (`KERNELS` in `pace_core.node.blender_box`) come from the
host application, which registers their directory with
`blender_box.register_kernel_dir(...)`. A missing kernel raises `KernelNotFound`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `PAI_PROJECTS_ROOT` | `./production/projects` | parent of project directories (`<slug>/kb/scenes/…`) |
| `MODELS_FILE` | `./assets/models.json` | LLM model registry used by the breakdown and the LaMP parser |
| `BLENDER_BIN` | `blender` | Blender executable |
| `PACE_MODEL_GENERATOR`, `PACE_MODEL_ESTIMATOR`, `PACE_MODEL_WRITER`, `PACE_MODEL_CLASSIFIER` | see `llm_client.DEFAULT_MODELS` | registry keys used when a caller names no model |

LLM provider keys are read from the environment, for example `ANTHROPIC_API_KEY`, or from `~/.config/pai/llm_tokens.json`. A host application can call `pace_core.paths.configure(...)` instead of setting variables.

Model calls go through [LiteLLM](https://github.com/BerriAI/litellm). A registry entry uses provider `anthropic`, `openai` (any OpenAI-compatible endpoint), `local`, or `codex`, or sets `litellm_model` to any LiteLLM model string (for example `gemini/gemini-2.5-pro`). Provider `codex` calls the ChatGPT subscription that the Codex CLI is signed in with: `pace_core.codex_auth` reads `$CODEX_HOME/auth.json` (default `~/.codex`) and refreshes the token under a file lock when it nears expiry. Run `codex login` once; no API key is needed.

## Knowledge base

A project's `kb/` holds `characters.json`, `props.json`, `location_stubs.json` and
`locations/<key>.bible.json`. Each entry sits under a key, and the key is opaque:

1. pace-core never prints a key into text (prompts, negatives, LLM payloads, findings,
   warnings, reasons) and never matches one against text (screenplay, action text,
   `location_raw`). Keys stay keys in lookups, file names (`<key>.bible.json`, mattes
   `body_<key>`), Blender objects and the `props.json` record keys.
2. Text names an entry by its `name` (`pai_compat.ref_name`, `location_name`). A ref
   that resolves to no entry is a label and prints as itself; an entry without a
   `name` prints its key.
3. Text matches an entry through its `name`, `name_zh` and `aliases`
   (`pai_compat.entity_names`). A ref resolves by key, then by those names
   (`resolve_character_key`, `resolve_prop_key`, `resolve_location_key`).
4. LLM payloads show entries by name (`pai_compat.named_refs`), and answers map back
   to keys through the resolvers. An entry pace-core creates, for an answer naming
   an unknown entity or for a scene's label, gets a `uuid.uuid4().hex` key and that
   name.
5. A new entry's trigger defaults to its name.

A scene's id is opaque too. A scene file is `kb/scenes/<scene_id>.json` holding that
`scene_id`, and `paths.iter_canonical_scene_files` lists them in `scene_number`
order, then by id. Other files beside the scenes (enrichment sidecars, evidence,
backups) are never read as scenes, and a file's name orders nothing.

## Tests

```
uv sync --extra dev
uv run pytest
```

Tests that need a project tree or Blender skip when it is absent.

## License

MIT
