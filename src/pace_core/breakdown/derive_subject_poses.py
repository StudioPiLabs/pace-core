"""Give each subject in a shot the posture its own beat implies.

`setup.subjects[].pose` is a per-subject field that was being written once per
shot: multi-subject shots handed every subject the identical string, and
almost none varied it. So a beat that reads "the man kneels on the road beside the woman
lying motionless" stored "kneeling or crouched grief pose" three times, and the
woman who is the point of the shot was staged kneeling like everyone else.

That matters more than a prose field usually would, because pose now selects
the proxy the greybox stages a body from (see panel_greybox.pose_key_for). A
uniform pose is three identical bodies in the control image, and the control
image is the sampler's starting latents.

Nothing else produces this field. The SCINE enricher does not mention pose, and
no other breakdown step writes it, so there is no producer to extend — this is
it.

Writes in place, and only `pose`. Enrichment output was previously written as
`scene_NN_v1_enriched.json` beside the originals, and seven call sites glob
that directory, so a sibling file silently doubles every scene in the corpus.

    uv run python -m pace_core.breakdown.derive_subject_poses --project <slug>
    uv run python -m pace_core.breakdown.derive_subject_poses --project <slug> --execute
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pace_core.paths import paths_for, MODELS_FILE          # noqa: E402
from pace_core.pai_compat import dig                    # noqa: E402
from pace_core.node.panel_greybox import pose_key_for        # noqa: E402

# The words that actually reach the geometry. A pose the resolver cannot read
# falls back to the location default silently, which is the failure this module
# exists to end, so the model is told to lead with one of these.
LEAD_WORDS = ("standing", "seated", "kneeling", "crouching", "lying")

SYSTEM = """You assign body posture to the subjects of one storyboard shot.

Return ONLY a JSON object mapping character_id to a short posture phrase. No
prose, no markdown fence, no extra keys.

Rules:
1. The beat describes what happens. Assign each subject the posture THE BEAT
   IMPLIES FOR THAT SUBJECT. Different subjects in one beat usually differ:
   "the man kneels beside the woman lying motionless" is two postures.
2. Begin every phrase with exactly one of: standing, seated, kneeling,
   crouching, lying. That first word decides which body the storyboard stages,
   so it must be the physically correct one.
3. After the first word, add a few words of specific detail — what the hands,
   torso or head are doing. Keep the whole phrase under ten words.
4. A subject the beat does not mention keeps a posture consistent with the
   scene. Do not invent action for them; describe a plausible attitude.
5. Never name emotions. Posture only — emotion is a separate field."""


def build_prompt(scene: dict, shot: dict) -> tuple[str, str]:
    act = (dig(shot, "events", "actions") or [{}])[0]
    beat = (act.get("description_en") or act.get("description_zh") or "").strip()
    subs = [{"character_id": s.get("character_id"),
             "age_state": s.get("age_state"),
             "current_pose": s.get("pose")}
            for s in (dig(shot, "setup", "subjects") or [])]
    payload = {
        "scene_heading": scene.get("scene_heading"),
        "shot_id": shot.get("shot_id"),
        "beat": beat,
        "subjects": subs,
    }
    return SYSTEM, json.dumps(payload, ensure_ascii=False, indent=1)


def parse_reply(text: str) -> dict[str, str]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        out = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j < 0:
            return {}
        out = json.loads(t[i:j + 1])
    return {str(k): str(v).strip() for k, v in out.items()
            if isinstance(v, str) and v.strip()}


def validate(poses: dict[str, str], subs: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Keep what is usable, and say what was dropped and why."""
    ids = {s.get("character_id") for s in subs}
    kept, warns = {}, []
    for cid, phrase in poses.items():
        if cid not in ids:
            warns.append(f"dropped '{cid}': not a subject of this shot")
            continue
        first = phrase.strip().lower().split()[:1]
        if not first or first[0].rstrip(",") not in LEAD_WORDS:
            warns.append(f"{cid}: '{phrase}' does not lead with a staging word")
            continue
        kept[cid] = phrase
    if len(set(kept.values())) == 1 and len(kept) > 1:
        warns.append("every subject got the same posture again")
    return kept, warns


def _load_model_cfg(model_key: str) -> dict[str, Any]:
    registry = json.loads(MODELS_FILE.read_text())
    if model_key not in registry:
        raise SystemExit(f"model '{model_key}' not in {MODELS_FILE}; "
                         f"options: {sorted(registry)}")
    return registry[model_key]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    # Same model the enrichment runs on. gpt-4o-rb was the default and is
    # two generations old; a pose is a short classification against a
    # closed vocabulary, so this does not want the expensive generator
    # either.
    ap.add_argument("--model", default="claude-sonnet-5-rb")
    ap.add_argument("--execute", action="store_true",
                    help="Call the LLM and write poses back (incurs API cost).")
    ap.add_argument("--scene", action="append", help="scene_id(s); default all")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    p = paths_for(a.project)
    files = [f for f in sorted(p.scenes_dir.glob("scene_*.json"))
             if "_v1_" not in f.name and "preview" not in f.name]
    if a.scene:
        files = [f for f in files if json.loads(f.read_text()).get("scene_id") in set(a.scene)]
    if a.limit:
        files = files[:a.limit]

    cfg = _load_model_cfg(a.model)
    in_k = float(cfg.get("cost_per_1k_in") or 0)
    out_k = float(cfg.get("cost_per_1k_out") or 0)

    shots = sum(len(json.loads(f.read_text()).get("shots") or []) for f in files)
    if not a.execute:
        est = shots * (0.5 * in_k + 0.12 * out_k)
        print(f"{len(files)} scenes / {shots} shots would be posed via {a.model}")
        print(f"estimated ~${est:.3f} total  (~{est/max(shots,1):.4f}/shot)")
        print("re-run with --execute")
        return 0

    from pace_core.llm_client import call_model
    total_cost = 0.0
    changed = warned = 0
    for f in files:
        doc = json.loads(f.read_text())
        touched = False
        for shot in doc.get("shots") or []:
            subs = dig(shot, "setup", "subjects") or []
            if not subs:
                continue
            sys_p, user_p = build_prompt(doc, shot)
            t0 = time.time()
            try:
                text, cost = call_model(cfg, [{"role": "system", "content": sys_p},
                                              {"role": "user", "content": user_p}],
                                        api_key=None)
            except Exception as exc:                      # noqa: BLE001
                print(f"[FAIL ] {f.name}/{shot.get('shot_id')}: {exc}")
                continue
            total_cost += cost
            kept, warns = validate(parse_reply(text), subs)
            for s in subs:
                new = kept.get(s.get("character_id"))
                if new and new != s.get("pose"):
                    s["pose"] = new
                    touched = True
            warned += len(warns)
            got = {s.get("character_id"): s.get("pose") for s in subs}
            distinct = len(set(got.values()))
            print(f"[ok   ] {doc.get('scene_id')}/{shot.get('shot_id'):8s} "
                  f"{distinct} distinct pose(s) t={time.time()-t0:.1f}s")
            for w in warns:
                print(f"         warn: {w}")
        if touched:
            f.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
            changed += 1
    print(f"\n=== posed {changed} scene file(s), cost ${total_cost:.4f}, "
          f"{warned} warning(s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
