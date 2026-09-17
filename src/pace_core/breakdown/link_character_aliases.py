#!/usr/bin/env python3
"""Join what the panels call a character to what the registry calls it.

`split_script` writes subjects by the name the screenplay uses -- 林小雨, 王秘书
-- and `extract_characters` keys the registry by a romanised slug --
`lin_xiaoyu`, `secretary_wang`. Nothing joins them, so in a Chinese screenplay the overlap
between the two sets of identifiers is EMPTY, and everything downstream of a
character lookup quietly does nothing:

  * `_character_hint` resolves no entry and falls back to emitting the raw
    identifier, so every compiled prompt reads "on the left, 李秘书" instead of
    the appearance the registry holds. The anchors are extracted, checked, and
    reach no image.
  * `GET /api/project/{p}/characters` filters to characters referenced by
    panels, matches nothing, and returns {} -- so the Library shows an empty
    shelf while the records and their reference plates sit on disk.

`resolve_character` already reads `aliases`, `name_zh`, `name` and `trigger`.
The mechanism was there; the registry carried none of those fields. This fills
them.

The join is on SCENE CO-OCCURRENCE, not on the names. `_extracted.scenes`
records where the extractor saw each character and the panels record where each
reference appears, so the two sets identify the same person without anyone
transliterating anything: 林小雨 and `lin_xiaoyu` appear in exactly the same
nine scenes. Names in two scripts and two languages are exactly what this
should not depend on.

Two things it refuses to guess:

  A reference that is not one person. 旁白 is narration, 众人 and 其余同事 are
  crowds. They co-occur with whoever is on screen, so a similarity score will
  always find them a partner, and every partner is wrong.

  A weak match. Below the threshold the reference is reported unresolved
  rather than attached, because a wrong alias is worse than a missing one: it
  puts one character's face on another's shot and nothing downstream can tell.

Co-occurrence resolves the principals and stops there, and no second
statistic rescues the rest. A creature's cue can match its own slug at 0.50
while the same creature cued as wearing the protagonist's face matches an
unrelated character at 0.50 with a LARGER margin over its runner-up. Ranking by margin
would take the wrong one first. Two references appearing in one scene tells
you they were on screen together and nothing else, so the remainder is a
judgment, and `--confirm ref=key` is where a person records one. Confirmations
are arguments to the tool rather than hand edits to the registry, so what was
decided and by whom is visible in the command that ran.

    uv run python -m pace_core.breakdown.link_character_aliases --project <slug>
    uv run python -m pace_core.breakdown.link_character_aliases --project <slug> --write
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pace_core.paths import iter_canonical_scene_files, paths_for      # noqa: E402

#: Below this, report rather than attach. A wrong alias puts one character's
#: face on another's shot, and no check downstream can tell.
MIN_OVERLAP = 0.6

#: References that do not denote one body. They co-occur with whoever is on
#: screen, so similarity always finds them a partner and every partner is
#: wrong. `entity_appearance` makes the same distinction for detection
#: prompts: a voice has no body in any frame.
NOT_A_PERSON = ("旁白", "众人", "其余", "画外音", "群众", "narrator", "voice over",
                "crowd", "others", "everyone")


def is_person(ref: str) -> bool:
    r = (ref or "").strip().lower()
    return bool(r) and not any(w in r for w in NOT_A_PERSON)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


#: Where each kind of entity lives in a shot's setup, and where the registry
#: records the scenes it was seen in. Props were extracted with the same
#: shape and never joined the same way.
KINDS = {
    "character": {"bucket": "subjects", "ref": "character_id",
                  "scenes": lambda rec: (rec.get("_extracted") or {}).get("scenes") or []},
    "prop":      {"bucket": "props", "ref": "prop_id",
                  "scenes": lambda rec: rec.get("linked_scenes") or []},
}


def head_noun(ref: str) -> str:
    """The noun an identifier names: the last token, singular.

    Props need a different signal from characters, and measuring says so.
    Scene co-occurrence works for a character because people enter and leave;
    a dozen office props share every office scene, so it links
    `desktop_computer` to `office_chair` and `headphones` to `potted_plant`.
    Token overlap is no better: `office_door` and `office_desk` share
    `office`, which is a category and not an identity, and score exactly what
    `potted_plant` and `desk_plant` score.

    Both sides are English slugs written by the same extractor, so the head of
    the compound is what it is actually naming. It declines far more than it
    matches -- most orphans stay orphans -- and makes no wrong match, which is
    the trade this module already takes everywhere else.
    """
    return (ref or "").strip().lower().split("_")[-1].rstrip("s")


def link_by_head_noun(registry: dict, refs: dict) -> tuple[dict, list[dict]]:
    """({key: [aliases]}, unresolved) by the noun each identifier names."""
    keys = [k for k, v in registry.items()
            if not k.startswith("_") and isinstance(v, dict)]
    linked: dict[str, list[str]] = collections.defaultdict(list)
    unresolved: list[dict] = []
    for ref in sorted(refs):
        if ref in registry:
            continue                      # already its own key
        hits = [k for k in keys if head_noun(k) == head_noun(ref)]
        if len(hits) == 1:
            linked[hits[0]].append(ref)
        else:
            why = (f"{len(hits)} registry props share the noun "
                   f"{head_noun(ref)!r}" if hits
                   else f"no registry prop names a {head_noun(ref)!r}")
            unresolved.append({"ref": ref, "reason": why,
                               "scenes": sorted(refs[ref])})
    return dict(linked), unresolved


def panel_refs(scenes: list[dict], kind: str = "character") -> dict[str, set[str]]:
    """Each reference the panels use -> the scenes it appears in."""
    spec = KINDS[kind]
    out: dict[str, set[str]] = collections.defaultdict(set)
    for doc in scenes:
        for sh in doc.get("shots") or []:
            for s in ((sh.get("setup") or {}).get(spec["bucket"]) or []):
                if not isinstance(s, dict):
                    continue
                ref = (s.get(spec["ref"]) or "").strip()
                if ref:
                    out[ref].add(doc.get("scene_id"))
    return dict(out)


def link(registry: dict, scenes: list[dict],
         threshold: float = MIN_OVERLAP,
         kind: str = "character") -> tuple[dict, list[dict]]:
    """({registry_key: [aliases]}, unresolved). Pure."""
    spec = KINDS[kind]
    if kind == "prop":
        return link_by_head_noun(registry, panel_refs(scenes, kind))
    reg_scenes = {k: set(spec["scenes"](v))
                  for k, v in registry.items()
                  if not k.startswith("_") and isinstance(v, dict)}
    refs = panel_refs(scenes, kind)

    linked: dict[str, list[str]] = collections.defaultdict(list)
    unresolved: list[dict] = []
    # Strongest first, so a qualifier like 白泽（异兽） can attach to the plain
    # 白泽 that has already been matched on its own evidence.
    for ref, rs in sorted(refs.items(), key=lambda kv: -len(kv[1])):
        if kind == "character" and not is_person(ref):
            unresolved.append({"ref": ref, "reason": "not one person",
                               "scenes": sorted(rs)})
            continue
        if not reg_scenes:
            break
        key, score = max(((k, _jaccard(rs, v)) for k, v in reg_scenes.items()),
                         key=lambda kv: kv[1])
        if score >= threshold:
            linked[key].append(ref)
            continue
        # A qualified form of a reference already linked: 白泽（异兽）, 白泽（林小雨的
        # 脸）. The base name is the evidence, not the co-occurrence.
        base = next((k for k, aliases in linked.items()
                     for a in aliases if ref.startswith(a) and ref != a), None)
        if base:
            linked[base].append(ref)
            continue
        unresolved.append({"ref": ref, "reason": f"best match {key} at {score:.2f}",
                           "scenes": sorted(rs)})
    return dict(linked), unresolved


def run(*, project: str, write: bool = False,
        threshold: float = MIN_OVERLAP,
        confirm: dict[str, str] | None = None,
        kind: str = "character") -> dict:
    p = paths_for(project)
    reg_file = Path(p.chars_file if kind == "character" else p.props_file)
    doc = json.loads(reg_file.read_text(encoding="utf-8"))
    registry = doc.get("characters" if kind == "character" else "props", doc)
    scenes = [json.loads(f.read_text(encoding="utf-8"))
              for f in iter_canonical_scene_files(Path(p.scenes_dir))]

    linked, unresolved = link(registry, scenes, threshold, kind)
    # A person's decision, recorded as an argument. Applied after the
    # automatic pass so it can only add, and refused for a key that is not in
    # the registry -- a typo would otherwise create a character.
    bad = []
    for ref, key in (confirm or {}).items():
        if key not in registry:
            bad.append(f"{ref}={key} (no such character)")
            continue
        linked.setdefault(key, []).append(ref)
        unresolved = [u for u in unresolved if u["ref"] != ref]
        # A qualified form of a confirmed reference comes with it.
        for u in list(unresolved):
            if u["ref"].startswith(ref) and u["ref"] != ref:
                linked[key].append(u["ref"])
                unresolved.remove(u)
    if write:
        for key, aliases in linked.items():
            rec = registry[key]
            have = list(rec.get("aliases") or [])
            rec["aliases"] = have + [a for a in aliases if a not in have]
            # The first alias is the name the screenplay itself uses.
            if kind == "character":
                rec.setdefault("name_zh", aliases[0])
            rec.setdefault("id", key)
        reg_file.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"project": project, "linked": linked,
            "linked_refs": sum(len(v) for v in linked.values()),
            "unresolved": unresolved, "bad_confirmations": bad,
            "written": write}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True)
    ap.add_argument("--kind", choices=sorted(KINDS), default="character")
    ap.add_argument("--threshold", type=float, default=MIN_OVERLAP)
    ap.add_argument("--confirm", action="append", default=[],
                    metavar="REF=KEY",
                    help="record a person's decision for a reference the join "
                         "could not settle; repeatable. A qualified form of a "
                         "confirmed reference comes with it.")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    conf = dict(c.split("=", 1) for c in a.confirm if "=" in c)
    out = run(project=a.project, write=a.write, threshold=a.threshold,
              confirm=conf, kind=a.kind)
    for b in out["bad_confirmations"]:
        print(f"  [refused] {b}", file=sys.stderr)
    for key, aliases in out["linked"].items():
        print(f"  {key:18} <- {', '.join(aliases)}")
    for u in out["unresolved"]:
        print(f"  [unresolved] {u['ref']:14} {u['reason']}", file=sys.stderr)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("linked", "unresolved")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
