"""rate_takes.py — director-review ratings for rendered takes (DIT-style).

Mirrors the Labels & Flags panel in Pomfort's Shooting Day Report. Stores
ratings in a sidecar JSON next to the RenderDay file.

Storage:
  production/02_assets/render_days/<render_day_id>_ratings.json

  {
    "_meta": {"render_day_id": "...", "updated_at": "..."},
    "ratings": {
      "<take_id>": {"rating": 5, "label": "OK", "review_note": "..."},
      ...
    }
  }

Three usage modes:

  # 1. Single take
  python3 pipeline/rate_takes.py \\
      --render-day UnitB_001_20260516 \\
      --take scene_01__P01__T01 --rating 5 --label OK --note "great firelight"

  # 2. Bulk: tag a list of take_ids with the same label
  python3 pipeline/rate_takes.py \\
      --render-day UnitB_001_20260516 \\
      --bulk-label NG scene_01__P01__T01 scene_01__P02__T01

  # 3. Interactive — walk every take in the day, show preview, prompt for rating
  python3 pipeline/rate_takes.py \\
      --render-day UnitB_001_20260516 --interactive
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from pace_core.paths import paths_for

VALID_LABELS = {"NO", "OK", "KP", "NG", "FLAG"}


def _ratings_path(out_base: Path, day_id: str) -> Path:
    return out_base / f"{day_id}_ratings.json"


def _day_path(out_base: Path, day_id: str) -> Path:
    return out_base / f"{day_id}.json"


def _load_sidecar(out_base: Path, day_id: str) -> dict:
    p = _ratings_path(out_base, day_id)
    if p.exists():
        return json.loads(p.read_text())
    return {"_meta": {"render_day_id": day_id, "updated_at": ""}, "ratings": {}}


def _save_sidecar(out_base: Path, day_id: str, data: dict) -> None:
    data["_meta"]["render_day_id"] = day_id
    data["_meta"]["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    p = _ratings_path(out_base, day_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _set_rating(out_base: Path, day_id: str, take_id: str, *,
                rating: int | None = None,
                label: str | None = None, note: str | None = None) -> None:
    if label and label not in VALID_LABELS:
        raise SystemExit(f"label must be one of {sorted(VALID_LABELS)}, got {label!r}")
    if rating is not None and not (0 <= rating <= 5):
        raise SystemExit(f"rating must be 0-5, got {rating}")

    data = _load_sidecar(out_base, day_id)
    entry = data["ratings"].get(take_id, {})
    if rating is not None: entry["rating"] = int(rating)
    if label is not None:  entry["label"]  = label
    if note is not None:   entry["review_note"] = note
    data["ratings"][take_id] = entry
    _save_sidecar(out_base, day_id, data)


def _list_takes(out_base: Path, day_id: str) -> list[dict]:
    p = _day_path(out_base, day_id)
    if not p.exists():
        raise SystemExit(f"render-day JSON not found: {p}.")
    return json.loads(p.read_text()).get("takes", [])


def cmd_show(out_base: Path, day_id: str, take_id: str | None = None) -> None:
    """Print current rating(s) — for verification."""
    data = _load_sidecar(out_base, day_id)
    if take_id:
        entry = data["ratings"].get(take_id, {})
        print(f"{take_id}: {entry or '(unrated)'}")
        return
    print(f"render_day_id: {day_id}  updated_at: {data['_meta'].get('updated_at','—')}")
    rows = sorted(data["ratings"].items())
    if not rows:
        print("  (no ratings yet)")
        return
    print(f"  {'take_id':<40} {'★':<6} {'label':<6} note")
    print(f"  {'-'*40} {'-'*6} {'-'*6} ----")
    for tid, e in rows:
        stars = "★" * e.get("rating", 0) + "☆" * (5 - e.get("rating", 0))
        print(f"  {tid:<40} {stars} {e.get('label','—'):<6} {e.get('review_note','')}")


def cmd_interactive(out_base: Path, day_id: str, only_unrated: bool = False) -> None:
    """Walk every take in the day; for each, show preview path + prompt for rating."""
    takes = _list_takes(out_base, day_id)
    data = _load_sidecar(out_base, day_id)
    existing = data["ratings"]

    for i, t in enumerate(takes, 1):
        tid = t["take_id"]
        cur = existing.get(tid, {})
        if only_unrated and cur:
            continue

        print(f"\n[{i}/{len(takes)}] {tid}")
        print(f"  scene: {t['scene_id']}  panel: {t['panel_number']}  "
              f"seed: {t.get('seed')}  lora: {t.get('lora_used') or '—'}")
        if t.get("output_path"):
            print(f"  open: {t['output_path']}")
        prompt = (t.get("prompt") or "")[:120]
        if prompt:
            print(f"  prompt: {prompt}…")
        if cur:
            stars = "★" * cur.get("rating", 0) + "☆" * (5 - cur.get("rating", 0))
            print(f"  current: {stars} {cur.get('label','—')} — "
                  f"{cur.get('review_note','')}")
        try:
            raw = input("  rating★ label[NO|OK|KP|NG|FLAG] note? (blank=skip / q=quit) > ")
        except EOFError:
            return
        if not raw.strip():
            continue
        if raw.strip().lower() in ("q", "quit"):
            print("  saved + exit"); return
        parts = raw.strip().split(maxsplit=2)
        rating: int | None = None
        label:  str | None = None
        note:   str | None = None
        if parts:
            try:
                rating = int(parts[0])
            except ValueError:
                # Maybe user typed just label first
                if parts[0].upper() in VALID_LABELS:
                    label = parts[0].upper()
                    parts = parts[1:]
                else:
                    print("  ✗ couldn't parse — skipping"); continue
            else:
                parts = parts[1:]
        if parts and not label:
            if parts[0].upper() in VALID_LABELS:
                label = parts[0].upper()
                parts = parts[1:]
        if parts:
            note = " ".join(parts)
        _set_rating(out_base, day_id, tid, rating=rating, label=label, note=note)
        print(f"  ✓ saved")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True,
                    help="project slug (a dir under PAI_PROJECTS_ROOT, default production/projects)")
    ap.add_argument("--render-day", required=True, help="render_day_id")
    ap.add_argument("--out-base", default=None,
                    help="dir holding RenderDay JSONs + ratings sidecars. "
                         "Default: <project>/02_assets/render_days/")

    # Single-take rating
    ap.add_argument("--take", default=None, help="take_id (e.g. scene_01__P01__T01)")
    ap.add_argument("--rating", type=int, default=None, help="0-5 stars")
    ap.add_argument("--label",  default=None, help="NO / OK / KP / NG / FLAG")
    ap.add_argument("--note",   default=None)

    # Bulk
    ap.add_argument("--bulk-label", default=None, help="label to apply to --takes list")
    ap.add_argument("--bulk-rating", type=int, default=None)
    ap.add_argument("takes", nargs="*", help="take_ids for bulk operations")

    # Other
    ap.add_argument("--interactive", action="store_true",
                    help="walk every take, prompt for rating")
    ap.add_argument("--only-unrated", action="store_true",
                    help="(interactive only) skip takes that already have a rating")
    ap.add_argument("--show", action="store_true",
                    help="print current ratings instead of modifying")
    args = ap.parse_args()

    out_base = Path(args.out_base).resolve() if args.out_base else paths_for(args.project).render_days_dir

    if args.show:
        cmd_show(out_base, args.render_day, args.take)
        return
    if args.interactive:
        cmd_interactive(out_base, args.render_day, only_unrated=args.only_unrated)
        cmd_show(out_base, args.render_day)
        return
    if args.bulk_label or args.bulk_rating is not None:
        if not args.takes:
            raise SystemExit("bulk mode requires take_ids as positional args")
        for tid in args.takes:
            _set_rating(out_base, args.render_day, tid,
                        rating=args.bulk_rating, label=args.bulk_label)
        print(f"applied {'rating={}'.format(args.bulk_rating) if args.bulk_rating is not None else ''} "
              f"{'label={}'.format(args.bulk_label) if args.bulk_label else ''} "
              f"to {len(args.takes)} take(s)")
        cmd_show(out_base, args.render_day)
        return
    if args.take:
        _set_rating(out_base, args.render_day, args.take,
                    rating=args.rating, label=args.label, note=args.note)
        cmd_show(out_base, args.render_day, args.take)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
