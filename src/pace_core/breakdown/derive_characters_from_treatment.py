#!/usr/bin/env python3
"""Put the author's own character sheet into the character registry.

A treatment's 人物设定 names each character and describes them in the author's
words. `extract_characters` never sees it: it reads the screenplay body and
infers an appearance from how a person behaves in scenes. On Zheng the two
disagree about the protagonist, and the disagreement is not subtle --

    treatment   一头红色短发与众格格不入   (a head of RED SHORT hair, which is
                                        the point: 与众格格不入, she does not
                                        fit in)
    registry    "long black hair often loosely tied"

-- and the reference plate rendered black hair, faithfully, because the plate
is built from the registry. Forty panels would have done the same.

This parses the sheet and attaches each entry to the character it describes,
verbatim, as `treatment_note`. It does NOT rewrite the anchor. The sheet is
biography, psychology and appearance in one paragraph -- "白天是资深文案策划,
晚上是夜店DJ" is not a description of a frame -- and turning it into the
English appearance sentence the compilers quote is a translation job for a
model, not a parse. What a parse can do honestly is put the author's words
where a person and that model will both find them, and say which records now
hold an authored description their anchor was never derived from.

One contradiction it does flag, because it is the one that was found and it is
checkable without judgment: hair. Colour and length are named in both
languages with a small closed vocabulary, so a sheet saying 红 while the anchor
says black is a fact, not an opinion.

    uv run python -m pace_core.breakdown.derive_characters_from_treatment --project <slug>
    uv run python -m pace_core.breakdown.derive_characters_from_treatment --project <slug> --write
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.pai_compat import resolve_character                # noqa: E402
from pace_core.paths import paths_for                                  # noqa: E402

#: `名字：描述` — a name, optionally with a parenthetical role, then a
#: full-width colon. The name may carry a middle dot (AI·狰) and Latin letters.
_ENTRY = re.compile(
    r"(?P<name>[A-Za-z0-9·一-鿿]{1,16})"
    r"(?:（(?P<role>[^）]{0,20})）)?\s*[:：]\s*"
    r"(?P<body>.*?)(?=(?:[A-Za-z0-9·一-鿿]{1,16}(?:（[^）]{0,20}）)?\s*[:：])|$)",
    re.S)

#: A scene heading ends the sheet: the treatment's top matter runs until the
#: screenplay starts, and `_characters_raw` is cut at the head boundary, which
#: on a one-page treatment can overrun by a line.
_SCENE_START = re.compile(r"\d{1,3}\s*[、.．]\s*\S")

#: Hair, in both languages. Closed and small on purpose -- this exists to
#: report the contradiction that was actually found, not to become a
#: bilingual appearance parser.
_HAIR_ZH = {"红": "red", "黑": "black", "金": "blonde", "银": "silver",
            "白": "white", "棕": "brown", "灰": "grey"}
_HAIR_EN = ("red", "black", "blonde", "silver", "white", "brown", "grey", "gray")


def parse_sheet(raw: str) -> dict[str, str]:
    """{name: verbatim description} from the treatment's 人物设定 block."""
    text = (raw or "").strip()
    m = _SCENE_START.search(text)
    if m:
        text = text[:m.start()]
    out: dict[str, str] = {}
    for hit in _ENTRY.finditer(text):
        name = hit.group("name").strip()
        body = " ".join(hit.group("body").split()).strip()
        if name and body:
            out[name] = body
    return out


def hair_conflict(sheet_text: str, anchor: str) -> tuple[str, str] | None:
    """(treatment colour, anchor colour) when they name different hair."""
    zh = next((en for ch, en in _HAIR_ZH.items()
               if f"{ch}色短发" in sheet_text or f"{ch}色长发" in sheet_text
               or f"一头{ch}" in sheet_text), None)
    if not zh:
        return None
    low = (anchor or "").lower()
    en = next((c for c in _HAIR_EN if f"{c} hair" in low), None)
    if en and en != zh and not (en == "gray" and zh == "grey"):
        return (zh, en)
    return None


RECONCILE_MODEL = "claude-sonnet-5-rb"

RECONCILE_SYSTEM = """You correct one sentence of a character's visual description.

You are given the ANCHOR, an English appearance sentence a storyboard
generator quotes for every panel this character appears in, and the SHEET, the
author's own description of the same character in Chinese.

The anchor was inferred from how the character behaves in the screenplay. The
sheet is what the author actually wrote. Where they disagree about
APPEARANCE, the sheet is right.

Make the SMALLEST change that removes the disagreement.

  - Change only what the sheet contradicts. Every other clause of the anchor
    stays exactly as it is, in the same order and the same words.
  - Carry across only appearance. The sheet also holds biography and
    psychology -- a job, a fear, a personality -- and none of that belongs in
    a sentence a picture is drawn from.
  - Keep the anchor's language, register and length. It is an English noun
    phrase, not prose, and it is quoted verbatim into image prompts.

Return ONE JSON object: {"anchor": "...", "changed": "<what you changed, in a
few words>"}. No prose, no markdown fences."""


def _reconcile_one(anchor: str, sheet: str, model_key: str) -> tuple[str, str, float]:
    """(new anchor, what changed, cost). One call, one character."""
    from pace_core.llm_client import call_model, strip_fences
    from pace_core.paths import MODELS_FILE
    cfg = json.loads(Path(MODELS_FILE).read_text())[model_key]
    raw, cost = call_model(cfg, [
        {"role": "system", "content": RECONCILE_SYSTEM},
        {"role": "user", "content": f"ANCHOR:\n{anchor}\n\nSHEET:\n{sheet}"},
    ], api_key=None)
    data = json.loads(strip_fences(raw))
    return (data.get("anchor") or "").strip(), (data.get("changed") or "").strip(), cost


def reconcile(*, project: str, model: str = RECONCILE_MODEL,
              execute: bool = False) -> dict:
    """Rewrite a conflicted anchor from the author's sheet.

    Only characters this module flagged: a call per character would otherwise
    rewrite descriptions nobody disputed, and every rewrite is a chance to
    lose a detail that was right.
    """
    p = paths_for(project)
    found = run(project=project, write=False)
    conflicts = found["hair_conflicts"]
    if not conflicts:
        return {"project": project, "conflicts": 0, "note": "nothing to reconcile"}

    doc = json.loads(Path(p.chars_file).read_text(encoding="utf-8"))
    registry = doc.get("characters", doc)
    plan, cost = [], 0.0
    for c in conflicts:
        rec = registry.get(c["key"]) or {}
        item = {"key": c["key"], "was": rec.get("anchor") or "",
                "sheet": rec.get("treatment_note") or "", "now": None}
        if execute:
            new, changed, cst = _reconcile_one(item["was"], item["sheet"], model)
            cost += cst
            # A rewrite that lost the sentence is worse than the conflict.
            if new and len(new) > len(item["was"]) * 0.5:
                item["now"], item["changed"] = new, changed
                rec.setdefault("_provenance", {})["anchor"] = "derived"
                rec["anchor_before_treatment"] = item["was"]
                rec["anchor"] = new
            else:
                item["changed"] = f"REFUSED: reply too short ({len(new)} chars)"
        plan.append(item)

    if execute and any(i["now"] for i in plan):
        Path(p.chars_file).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"project": project, "conflicts": len(conflicts), "model": model,
            "cost_usd": round(cost, 4), "plan": plan, "written": execute}


def run(*, project: str, write: bool = False) -> dict:
    p = paths_for(project)
    film = json.loads(Path(p.film_file).read_text(encoding="utf-8")) \
        if Path(p.film_file).exists() else {}
    sheet = parse_sheet(film.get("_characters_raw") or "")
    if not sheet:
        raise ValueError(f"{project!r}: kb/film.json carries no 人物设定 to derive from")

    doc = json.loads(Path(p.chars_file).read_text(encoding="utf-8"))
    registry = doc.get("characters", doc)

    attached, unmatched, conflicts = [], [], []
    for name, body in sheet.items():
        rec = resolve_character(name, registry)
        if not rec:
            # The beast is 狰 in the panels and AI·狰 on the sheet.
            rec = next((r for n, r in registry.items()
                        if isinstance(r, dict)
                        and any(a and a in name for a in (r.get("aliases") or []))), {})
        if not rec:
            unmatched.append(name)
            continue
        key = next(k for k, v in registry.items() if v is rec)
        attached.append({"sheet_name": name, "key": key})
        clash = hair_conflict(body, rec.get("anchor") or rec.get("generic_anchor") or "")
        if clash:
            conflicts.append({"key": key, "treatment": clash[0], "anchor": clash[1]})
        if write:
            rec["treatment_note"] = body
            rec.setdefault("_provenance", {})["treatment_note"] = "authored"

    if write:
        Path(p.chars_file).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"project": project, "on_sheet": len(sheet), "attached": attached,
            "unmatched": unmatched, "hair_conflicts": conflicts, "written": write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--reconcile", action="store_true",
                    help="rewrite a conflicted anchor from the sheet (one model "
                         "call per conflicted character; needs --write to apply)")
    ap.add_argument("--model", default=RECONCILE_MODEL)
    a = ap.parse_args()
    if a.reconcile:
        out = reconcile(project=a.project, model=a.model, execute=a.write)
        for i in out.get("plan") or []:
            print(f"  {i['key']}")
            print(f"    was : {i['was'][:150]}")
            print(f"    now : {str(i['now'])[:150]}")
            print(f"    diff: {i.get('changed')}")
        print(json.dumps({k: v for k, v in out.items() if k != "plan"},
                         ensure_ascii=False, indent=1))
        return 0
    out = run(project=a.project, write=a.write)
    for x in out["attached"]:
        print(f"  {x['sheet_name']:10} -> {x['key']}")
    for c in out["hair_conflicts"]:
        print(f"  [CONFLICT] {c['key']}: the treatment says {c['treatment']} hair, "
              f"the anchor says {c['anchor']}", file=sys.stderr)
    for u in out["unmatched"]:
        print(f"  [unmatched] {u}", file=sys.stderr)
    print(json.dumps({k: v for k, v in out.items() if k != "attached"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
