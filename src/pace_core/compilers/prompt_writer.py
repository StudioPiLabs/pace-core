"""prompt_writer — run the prompt-writing model over one scene.

The third piece. `prompt_projection` decides what the model sees,
`prompt_meta` decides what it may do, and this calls it and keeps the result.

Where the result goes matters more than it looks: written prompts land in
`compile_hints[i].flux.prompt_override`, a sidecar the render path ALREADY
honours (compile_flux2 short-circuits on it). So nothing downstream changes,
`compile_flux2` stays the fallback for any panel that has not been written,
turning the writer off is deleting a field, and a panel can be hand-edited
afterwards without the writer having any say. A new parallel prompt channel
would have meant touching the render path and owning a second way for a panel
to acquire a prompt.

Default is DRY RUN. This is the only module in the compile path that spends
money, and a caller that has to ask for that explicitly cannot spend it by
accident — the same shape `enrich_scenes_scine` uses for the same reason.

All-or-nothing. A scene's panels are written in one call because writing them
singly makes them look like unrelated images; keeping a partial reply would
give back exactly the incoherence the single call exists to avoid. If any
panel is missing from the reply, nothing is saved.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pace_core.compilers import prompt_meta
from pace_core.compilers.prompt_projection import PROJECTION_VERSION, project_scene
from pace_core.paths import MODELS_FILE, paths_for

#: What the writer stamps on every prompt it authors. The canvas uses
#: "canvas:Compilation" in the same field, so provenance says which of the
#: two wrote a prompt — and a hand-edit that drops the stamp reads as human.
AUTHORED_BY = "llm_writer"


def load_model_cfg(model_key: str) -> dict:
    registry = json.loads(MODELS_FILE.read_text())
    if model_key not in registry:
        raise ValueError(f"model {model_key!r} not in {MODELS_FILE}; "
                         f"have {sorted(registry)}")
    return registry[model_key]


def _scene_path(project: str, scene_id: str) -> Path:
    return Path(paths_for(project).scenes_dir) / f"{scene_id}.json"


def _panel_ids(scene: dict) -> list[str]:
    return [p.get("id") for sh in scene.get("shots") or []
            for p in sh.get("panels") or [] if p.get("id")]


def _apply(scene: dict, written: dict[str, str], *, model_key: str,
           cost_usd: float) -> dict:
    """Put each panel's prose into its compile_hints sidecar, with enough
    provenance to trace a picture back to what produced its words.

    The stored prompt is the reproducible artifact even though the authoring
    was not: a render can be repeated exactly from the KB afterwards. Without
    the versions beside it, "the prompts got worse in September" has no route
    back to a cause — which of the gate, the rules, or the model changed."""
    hints = scene.setdefault("compile_hints", [])
    by_id = {h.get("panel_id"): h for h in hints if isinstance(h, dict)}
    stamp = {
        "authored_by": AUTHORED_BY,
        "model": model_key,
        "projection_version": PROJECTION_VERSION,
        "meta_version": prompt_meta.META_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cost_usd": round(cost_usd, 6),
    }
    for panel_id, prose in written.items():
        hint = by_id.get(panel_id)
        if hint is None:
            hint = {"panel_id": panel_id}
            hints.append(hint)
            by_id[panel_id] = hint
        flux = hint.setdefault("flux", {})
        # negative is left empty on purpose: every Flux workflow here runs
        # cfg=1.0, so the negative prompt is never evaluated. Writing one
        # would look like a safeguard and be inert.
        flux["prompt_override"] = {"positive": prose, "negative": "", **stamp}
    return scene


def write_scene(project: str, scene_id: str, *, model_key: str = "claude-sonnet-5-rb",
                execute: bool = False, api_key: str | None = None) -> dict:
    """Write every panel prompt for one scene.

    Returns a report either way. With execute=False nothing is called and
    nothing is saved — the report carries the exact messages that WOULD be
    sent, which is the cheapest way to review a projection or a rule change.
    """
    path = _scene_path(project, scene_id)
    scene = json.loads(path.read_text())
    payload = project_scene(scene, project=project)
    messages = prompt_meta.build_messages(payload)
    expected = _panel_ids(scene)

    report = {
        "project": project, "scene_id": scene_id, "model": model_key,
        "panels_expected": len(expected),
        "projection_version": PROJECTION_VERSION,
        "meta_version": prompt_meta.META_VERSION,
        "executed": False, "messages": messages,
    }
    if not execute:
        report["note"] = "dry run — no model call, nothing written"
        return report

    from pace_core.llm_client import call_model
    cfg = load_model_cfg(model_key)
    if cfg.get("provider") == "openai":
        cfg = {**cfg, "response_format": {"type": "json_object"}}
    text, cost = call_model(cfg, messages, api_key)
    data = prompt_meta.parse_reply(text)          # raises on a malformed reply

    common = (data.get("scene_common") or "").strip()
    got = {p["panel_id"]: p["prompt"] for p in data["panels"]}
    missing = [pid for pid in expected if pid not in got]
    if missing:
        # Nothing saved. A scene half-written is the incoherence the
        # one-call-per-scene rule exists to prevent.
        raise ValueError(f"reply covered {len(got)}/{len(expected)} panels; "
                         f"missing {missing} — nothing written")
    unknown = [pid for pid in got if pid not in expected]
    if unknown:
        raise ValueError(f"reply invented panel ids not in this scene: {unknown}")

    written = {pid: prompt_meta.compose(common, got[pid]) for pid in expected}

    # Check before keeping. The rules are instructions to a model, not a
    # guarantee from one: on the first real call every panel leaked the
    # phrase "read point" into the prose and three wrote positions the
    # greybox had already decided. A reply that does that is worse than no
    # reply, because compile_flux2 would not have done it.
    n_subjects = max((len((sh.get("setup") or {}).get("subjects") or [])
                      for sh in scene.get("shots") or []), default=0)
    # What the panel was written FROM, so transcribed staging can be told
    # apart from invented staging.
    sources = {}
    for sh in scene.get("shots") or []:
        blob = json.dumps(sh.get("setup") or {}, ensure_ascii=False)
        blob += json.dumps(sh.get("events") or {}, ensure_ascii=False)
        for pan in sh.get("panels") or []:
            sources[pan.get("id")] = blob + json.dumps(pan, ensure_ascii=False)

    errors, warnings = {}, {}
    for pid, prose in written.items():
        errs, warns = prompt_meta.validate_prose(
            prose, expect_count=n_subjects > 1, source=sources.get(pid, ""))
        if errs:
            errors[pid] = errs
        if warns:
            warnings[pid] = warns
    if errors:
        detail = "; ".join(f"{pid}: {', '.join(v)}" for pid, v in errors.items())
        raise ValueError(f"reply broke the authoring rules, nothing written — {detail}")

    _apply(scene, written, model_key=model_key, cost_usd=cost)

    # Keep one copy of what was there before. These prompts are the input to
    # every subsequent render of the scene, so an overwrite that turns out
    # worse needs somewhere to come back from.
    backup = path.with_suffix(f".pre-writer.{int(datetime.now().timestamp())}.json")
    shutil.copy2(path, backup)
    path.write_text(json.dumps(scene, ensure_ascii=False, indent=2))

    report.update(executed=True, cost_usd=round(cost, 6), warnings=warnings,
                  panels_written=len(written), backup=str(backup),
                  scene_common=common,
                  prompts={pid: written[pid] for pid in expected})
    report.pop("messages", None)     # the reply is the interesting half now
    return report
