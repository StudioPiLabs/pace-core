"""Resolve a character registry entry to a likeness in the archetype library.

The shared archetype pool is built before any KB is authored: 90 archetypes
on three typed axes (sex x age band x region), three views each. Selecting
from it needs those three axes as *values*. A character registry that
carries them only inside a free-text anchor ("middle-aged, slender build,
Asian descent, long hair tied back") cannot index the library, which is why
characters end up with no likeness asset while the library sits fully
populated. This module supplies the typed axes and the lookup.

Two entry points, deliberately separate:

  resolve_archetype(sex, age, region)   exact lookup on typed values. This
                                        is what KB creation should call.

  infer_traits(text)                    best-effort recovery of the three
                                        axes from an existing free-text
                                        anchor, for backfilling registries
                                        authored before the fields existed.

`infer_traits` is a migration aid and reports its own confidence, because
guessing a person's attributes from prose is exactly the ambiguity the
typed fields exist to remove. It never guesses sex from a name, and it
returns None for an axis it cannot read rather than defaulting: a wrong
archetype is worse than no archetype, since a wrong one is silent.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SEXES = ("female", "male")
AGE_BANDS = ("child", "teen", "young_adult", "middle_aged", "elder")
REGIONS = ("central_asian", "east_asian", "european", "indigenous_american",
           "latin_american", "mena", "pacific_islander", "south_asian",
           "southeast_asian", "sub_saharan_african")
VIEWS = ("front", "side", "back")

# Age band boundaries, upper-inclusive; the library's own representative
# ages are 8 / 16 / 28 / 48 / 72.
_AGE_BOUNDS = ((12, "child"), (19, "teen"), (35, "young_adult"),
               (60, "middle_aged"), (200, "elder"))

_SEX_CUES = {
    "female": (r"\bwoman\b", r"\bwomen\b", r"\bgirl\b", r"\bfemale\b",
               r"\bmother\b", r"\bdaughter\b", r"\bsister\b", r"\bwife\b",
               r"\bqueen\b", r"\bprincess\b", r"\bmaid\b", r"\blady\b",
               r"\bher\b", r"\bshe\b"),
    "male": (r"\bman\b", r"\bmen\b", r"\bboy\b", r"\bmale\b", r"\bfather\b",
             r"\bson\b", r"\bbrother\b", r"\bhusband\b", r"\bking\b",
             r"\bprince\b", r"\bmonk\b", r"\bhis\b", r"\bhe\b"),
}
_AGE_CUES = {
    "child": (r"\bchild\b", r"\bboy\b", r"\bgirl\b", r"\binfant\b", r"\bkid\b"),
    "teen": (r"\bteen", r"\badolescent\b", r"\byouth\b"),
    "young_adult": (r"\byoung adult\b", r"\byoung\b", r"\btwenties\b"),
    "middle_aged": (r"\bmiddle[- ]aged\b", r"\bforties\b", r"\bfifties\b"),
    "elder": (r"\belder", r"\bold\b", r"\baged\b", r"\bsenior\b",
              r"\bseventies\b", r"\bsixties\b"),
}
# Regions the corpus names that the library has no bucket for. Recognised so
# they are reported as "recognised, unavailable" rather than silently read as
# the nearest neighbour: substituting a region is a decision, and it is
# recorded on the entry (see ArchetypeLibrary.nearest) rather than hidden.
UNSTOCKED_REGIONS: dict = {}

_REGION_CUES = {
    "central_asian": (r"\bcentral asian\b", r"\bkuchean\b", r"\btocharian\b"),
    "east_asian": (r"\beast asian\b", r"\basian descent\b", r"\bchinese\b",
                   r"\bjapanese\b", r"\bkorean\b", r"\bhan\b"),
    "south_asian": (r"\bsouth asian\b", r"\bindian\b", r"\bgandhara"),
    "southeast_asian": (r"\bsoutheast asian\b", r"\bthai\b", r"\bvietnamese\b"),
    "european": (r"\beuropean\b", r"\bwhite\b", r"\bcaucasian\b"),
    "mena": (r"\bmiddle eastern\b", r"\barab\b", r"\bpersian\b"),
    "sub_saharan_african": (r"\bafrican\b", r"\bblack\b"),
    "latin_american": (r"\blatin", r"\bhispanic\b"),
    "pacific_islander": (r"\bpacific islander\b", r"\bpolynesian\b"),
    "indigenous_american": (r"\bindigenous\b", r"\bnative american\b"),
}


_AGE_NUM = (
    re.compile(r"\b(\d{1,2})\s*[-\u2013]\s*(\d{1,2})\s*year", re.I),   # "17-18 year old"
    re.compile(r"\b(\d{1,2})\s*-?\s*year[- ]old", re.I),                # "7-year-old"
    re.compile(r"\bage[d]?\s*(\d{1,2})\b", re.I),
)
_AGE_DECADE = re.compile(r"\b(?:(early|mid|late)\s*)?(\d0)s\b", re.I)


def age_years_from_text(text: str):
    """Recover an age in years from the prose forms this corpus actually uses.

    Registries written by hand say "7-year-old", "17-18 year old", "late
    30s-40s" — none of which a keyword table matches. Returning the number
    lets the band come from `age_band_for`, so one rule decides the banding
    rather than two.
    """
    for rx in _AGE_NUM:
        m = rx.search(text)
        if m:
            return float(m.group(1))
    m = _AGE_DECADE.search(text)
    if m:
        base = float(m.group(2))
        return base + {"early": 2.0, "mid": 5.0, "late": 8.0}.get(
            (m.group(1) or "mid").lower(), 5.0)
    return None


def age_band_for(years) -> str | None:
    try:
        y = float(years)
    except (TypeError, ValueError):
        return None
    for hi, band in _AGE_BOUNDS:
        if y <= hi:
            return band
    return "elder"


@dataclass
class Traits:
    sex: str | None = None
    age_band: str | None = None
    region: str | None = None
    evidence: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all((self.sex, self.age_band, self.region))

    def missing(self) -> list[str]:
        return [k for k in ("sex", "age_band", "region") if not getattr(self, k)]


def _first_cue(text: str, table: dict) -> tuple[str | None, str | None]:
    """Return (value, matched pattern). Ambiguity yields None, not a coin flip."""
    hits = []
    for value, pats in table.items():
        for p in pats:
            if re.search(p, text):
                hits.append((value, p))
                break
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None, None
    # Prefer an explicit noun over a pronoun when both fired.
    strong = [h for h in hits if "\\bhe\\b" not in h[1] and "\\bshe\\b" not in h[1]
              and "\\bhis\\b" not in h[1] and "\\bher\\b" not in h[1]]
    return strong[0] if len(strong) == 1 else (None, None)


def infer_traits(*texts: str, age_years=None, sex_hint: str | None = None) -> Traits:
    """Best-effort read of the three axes from free text. Migration aid only."""
    blob = " ".join(t for t in texts if isinstance(t, str)).lower()
    t = Traits()
    if sex_hint in SEXES:
        t.sex, t.evidence["sex"] = sex_hint, "explicit field"
    else:
        t.sex, ev = _first_cue(blob, _SEX_CUES)
        if ev:
            t.evidence["sex"] = ev
    band = age_band_for(age_years)
    if band:
        t.age_band, t.evidence["age_band"] = band, f"age_years={age_years}"
    else:
        yrs = age_years_from_text(blob)
        band = age_band_for(yrs)
        if band:
            t.age_band, t.evidence["age_band"] = band, f"parsed age {yrs:.0f} from text"
        else:
            t.age_band, ev = _first_cue(blob, _AGE_CUES)
            if ev:
                t.evidence["age_band"] = ev
    t.region, ev = _first_cue(blob, _REGION_CUES)
    if ev:
        t.evidence["region"] = ev
    else:
        unstocked, uev = _first_cue(blob, UNSTOCKED_REGIONS)
        if unstocked:
            # Named clearly, just absent from the library. Recorded so the
            # caller can substitute deliberately instead of the axis reading
            # as unknown, which would look like a description problem.
            t.evidence["region"] = f"{unstocked} (recognised, not stocked): {uev}"
            t.region = None
    return t


class ArchetypeLibrary:
    """The shared pool, indexed by its three typed axes."""

    def __init__(self, index_path: Path):
        self.root = index_path.parent
        self.index_path = index_path
        data = json.loads(index_path.read_text())
        self.archetypes = data.get("archetypes", {})

    @classmethod
    def find(cls, *roots: Path):
        """Locate the library under any of the supplied roots, else None."""
        for r in roots:
            p = Path(r) / "shared" / "human_archetypes" / "index.json"
            if p.is_file():
                return cls(p)
            p = Path(r) / "human_archetypes" / "index.json"
            if p.is_file():
                return cls(p)
        return None

    def key_for(self, sex: str, age_band: str, region: str) -> str | None:
        k = f"{sex}_{age_band}_{region}"
        return k if k in self.archetypes else None

    def resolve(self, sex, age_band, region, views=("front", "side"),
                relative_to: Path | None = None) -> list[str]:
        """Image paths for an archetype, only for views marked done."""
        key = self.key_for(sex, age_band, region)
        if key is None:
            return []
        out = []
        for v in views:
            rec = (self.archetypes[key].get("views") or {}).get(v) or {}
            if rec.get("status") != "done" or not rec.get("image"):
                continue
            p = (self.root / rec["image"]).resolve()
            if not p.is_file():
                continue
            if relative_to is not None:
                try:
                    p = p.relative_to(Path(relative_to).resolve())
                except ValueError:
                    pass
            out.append(str(p))
        return out

    def nearest(self, sex, age_band, region) -> str | None:
        """Exact key, or the same sex+age in another region as a last resort.

        Returned separately from resolve() so a caller can record that a
        substitution happened rather than storing it as if it were exact.
        """
        k = self.key_for(sex, age_band, region)
        if k:
            return k
        if not (sex and age_band):
            return None
        for cand, rec in self.archetypes.items():
            if rec.get("sex") == sex and rec.get("age_band") == age_band:
                return cand
        return None
