"""Run the breakdown verifier end to end for a project: Script IR, then estimate.

The verifier is two model calls in two roles, and the roles are deliberately
given to two vendors: the GENERATOR extracts Script IR from the screenplay, and
the ESTIMATOR judges the project's breakdown against that IR. Two instances of
one model share a prior and agree on whatever is plausible, which is exactly
what a check must not do, so the split is the design and not a preference.

Both stages cost money, so the entry points are separated the way the two CLIs
already separate them: `plan()` prices a run and calls nothing, `run()` executes
one. A caller that has not shown the user `plan()`'s number first is doing the
wrong thing.

This module exists because neither stage was reachable except as a module CLI:
the studio could pick the model that SPLITS a script and had no way to pick the
model that estimates it, or to run the estimator at all.

The scene map -- which breakdown scenes answer to which screenplay scene -- was
hand-written per corpus. It is derivable whenever the breakdown came from the
same split, which is the normal case: `split_script` emits one scene file per
screenplay scene in order, so the map is the identity. A corpus that has since
been split or reordered by hand still needs an explicit map, and passing one
overrides the derivation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pace_core.breakdown import extract_script_ir as _ir
from pace_core.breakdown import verify_breakdown as _vb
from pace_core.breakdown.screenplay_parser import parse, scenes as _scenes
from pace_core.paths import paths_for

SCRIPT_SUFFIXES = (".pdf", ".md", ".txt")


def latest_script(project: str) -> Path | None:
    """The most recently stored screenplay for a project.

    Uploads are stored under `<project>/script/` with a UTC timestamp prefix,
    so the newest by name is the newest by time without stat-ing each file.
    """
    d = Path(paths_for(project).script_dir)
    if not d.is_dir():
        return None
    files = [f for f in d.iterdir()
             if f.is_file() and f.suffix.lower() in SCRIPT_SUFFIXES]
    return max(files, key=lambda f: f.name) if files else None


def screenplay_text(project: str, script: str | Path | None = None) -> tuple[str, str]:
    """(text, source label). PDFs are decoded by the same helper the split path
    uses, so the verifier reads exactly the bytes the breakdown was built from
    rather than a second decoding of them."""
    from pace_core.breakdown.split_script import script_bytes_to_text
    p = Path(script) if script else latest_script(project)
    if p is None or not p.is_file():
        raise FileNotFoundError(
            f"no screenplay found for {project!r} under "
            f"{paths_for(project).script_dir}; upload one first")
    return script_bytes_to_text(p.read_bytes(), p.name)[0], p.name


def breakdown_scene_ids(project: str) -> list[str]:
    d = Path(paths_for(project).scenes_dir)
    if not d.is_dir():
        return []
    return sorted(f.stem for f in d.glob("scene_*.json") if f.name.count(".") == 1)


def identity_scene_map(script_scenes: list[dict], scene_ids: list[str]) -> dict[str, list[str]]:
    """One screenplay scene to one breakdown scene, in order.

    Correct exactly when the breakdown came from the split of this screenplay,
    which is what `split_script` produces. It is NOT correct for a corpus whose
    scenes were later split or reordered by hand -- the Automatic Drive corpus
    in this paper is one, splitting three script scenes and reordering another
    -- so those pass an explicit map instead.
    """
    return {str(s["index"]): [sid]
            for s, sid in zip(script_scenes, scene_ids)}


def _resolve_map(project: str, script_scenes: list[dict],
                 scene_map: dict | None) -> tuple[dict, bool]:
    if scene_map:
        return {str(k): list(v) for k, v in scene_map.items()}, False
    return identity_scene_map(script_scenes, breakdown_scene_ids(project)), True


def _jobs(project: str, script_scenes: list[dict], mapping: dict) -> list[tuple]:
    out = []
    for s in script_scenes:
        ids = mapping.get(str(s["index"])) or []
        if ids:
            out.append((s, _vb.load_breakdown_actions(project, ids), ids))
    return out


def plan(project: str, *, generator_model: str, estimator_model: str,
         scene_map: dict | None = None, script: str | Path | None = None,
         out_tokens_generator: int = 900,
         out_tokens_estimator: int = 700) -> dict[str, Any]:
    """Price the run. Calls no model.

    The two stages are priced separately because they are billed separately and
    a caller may reasonably run one without the other -- and because seeing
    them apart is what makes an expensive generator against a cheap estimator,
    or the reverse, a visible choice rather than one number to accept.
    """
    text, label = screenplay_text(project, script)
    sc = _scenes(parse(text))
    mapping, derived = _resolve_map(project, sc, scene_map)
    gen_cfg, est_cfg = _ir.load_model(generator_model), _ir.load_model(estimator_model)

    gen_in = sum(_ir.estimate_tokens(_ir.build_messages(text, s)) for s in sc)
    gen_out = out_tokens_generator * len(sc)
    gen_cost = (gen_in / 1000 * gen_cfg["cost_per_1k_in"]
                + gen_out / 1000 * gen_cfg["cost_per_1k_out"])

    jobs = _jobs(project, sc, mapping)
    est_in = sum(_ir.estimate_tokens(_vb.build_messages(_stub_ir(s), acts))
                 for s, acts, _ids in jobs)
    est_out = out_tokens_estimator * len(jobs)
    est_cost = (est_in / 1000 * est_cfg["cost_per_1k_in"]
                + est_out / 1000 * est_cfg["cost_per_1k_out"])

    return {
        "project": project, "script": label,
        "script_scenes": len(sc),
        "breakdown_scenes": len(breakdown_scene_ids(project)),
        "scene_map_derived": derived,
        # A derived map assumes one screenplay scene became one breakdown
        # scene. When the counts differ, that assumption is already known to be
        # false and the map is silently mapping scene i to the wrong document
        # -- which would not fail, it would just score the wrong pair. Callers
        # must surface this rather than run past it.
        "scene_map_suspect": bool(derived and len(sc) != len(breakdown_scene_ids(project))),
        "scene_map": mapping,
        "mapped_scenes": len(jobs),
        "unmapped_scenes": [s["index"] for s in sc if not mapping.get(str(s["index"]))],
        "generator": {"model": generator_model, "resolves_to": gen_cfg["model_name"],
                      "tokens_in": gen_in, "tokens_out": gen_out,
                      "cost_usd": round(gen_cost, 4)},
        "estimator": {"model": estimator_model, "resolves_to": est_cfg["model_name"],
                      "tokens_in": est_in, "tokens_out": est_out,
                      "cost_usd": round(est_cost, 4)},
        "same_vendor": gen_cfg["model_name"].split("/")[0] == est_cfg["model_name"].split("/")[0],
        "est_cost_usd": round(gen_cost + est_cost, 4),
    }


def _stub_ir(sc: dict) -> dict:
    """A scene shaped like extracted IR but with no events.

    Pricing the estimator needs the size of its prompt, and the prompt carries
    the scene's events -- which do not exist until the generator has run. This
    under-counts by exactly the events, and says so here rather than pretending
    the estimate is tight: treat `plan()`'s estimator figure as a floor.
    """
    return {"index": sc["index"], "heading": sc["heading"], "events": []}


def run(project: str, *, generator_model: str, estimator_model: str,
        scene_map: dict | None = None, script: str | Path | None = None,
        ir_out: str | Path | None = None,
        on_progress: Callable[[str, dict], None] | None = None) -> dict[str, Any]:
    """Execute both stages. Spends money; price it with `plan()` first."""
    def _say(stage, **kw):
        if on_progress:
            on_progress(stage, kw)

    text, label = screenplay_text(project, script)
    sc = _scenes(parse(text))
    spent = 0.0

    # ── stage 1: Script IR, the ground truth the estimator judges against ──
    ir_scenes, failed_extract = [], []
    for i, s in enumerate(sc):
        rec, cost = _ir.extract_scene(text, s, generator_model)
        spent += cost
        if not rec.get("extract_ok"):
            failed_extract.append(s["index"])
        ir_scenes.append(rec)
        _say("extract", scene=s["index"], done=i + 1, total=len(sc),
             ok=bool(rec.get("extract_ok")), cost_usd=round(spent, 4))
    if ir_out:
        Path(ir_out).write_text(json.dumps(ir_scenes, ensure_ascii=False, indent=1))

    # ── stage 2: the estimator, on the breakdown the project actually holds ──
    from pace_core.llm_client import call_model, strip_fences
    mapping, derived = _resolve_map(project, sc, scene_map)
    est_cfg = _ir.load_model(estimator_model)
    aligns, failed_estimate = [], []
    jobs = [(s, _vb.load_breakdown_actions(project, mapping[str(s["index"])]),
             mapping[str(s["index"])])
            for s in ir_scenes if mapping.get(str(s["index"]))]
    for i, (s, acts, ids) in enumerate(jobs):
        raw = None
        for attempt in range(3):
            try:
                raw, c = call_model(est_cfg, _vb.build_messages(s, acts))
                spent += c
                break
            except Exception:                                  # noqa: BLE001
                if attempt == 2:
                    failed_estimate.append(s["index"])
        if raw is None:
            continue
        try:
            al = json.loads(strip_fences(raw))
        except json.JSONDecodeError:
            failed_estimate.append(s["index"])
            continue
        al["script_scene"] = s["index"]
        al["breakdown_scenes"] = ids
        aligns.append(al)
        _say("estimate", scene=s["index"], done=i + 1, total=len(jobs),
             cost_usd=round(spent, 4))

    res = _vb.score(aligns, ir_scenes)
    res.update(project=project, script=label,
               generator_model=generator_model, estimator_model=estimator_model,
               scene_map_derived=derived,
               scenes_scored=sorted(a["script_scene"] for a in aligns),
               failed_extract=failed_extract, failed_estimate=failed_estimate,
               cost_usd=round(spent, 4))
    return res
