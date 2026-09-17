"""Identity/focus helpers shared by every backend-specific compiler
(compile_flux2, compile_gpt_image_2, compile_nano_banana) and by
prompt_projection.

Split out of compile_flux.py when Flux 1/T5 support was retired
project-wide and that file was deleted. What lives here is genuinely
backend-agnostic — character descriptor resolution, primary-focus
ordering — or reused verbatim by a sibling that has not (yet) grown its
own tuned copy. This is deliberately NOT where a compiler's style/framing
prose tables live: those ARE backend-tuned and intentionally forked per
compiler (compile_flux2.py keeps its own copies of _SHOT_SIZE_FRAMING_EN,
_STYLE_ANCHOR, etc. rather than importing them from here, because Mistral
reacts differently to the same English than a compiler tuned for a
different text encoder would). _SHOT_SIZE_FRAMING_EN/_ANGLE_FRAMING_EN
below are the one exception: compile_gpt_image_2.py reuses them verbatim
rather than having grown its own tuned copy, so they moved here rather
than dying with compile_flux.py.
"""

from __future__ import annotations

import json
import re as _re_age
from dataclasses import dataclass, field
from pathlib import Path

from pace_core.pai_compat import costume_text, dig, resolve_character


@dataclass
class CompileContext:
    """External state injected at compile time — not part of the PAI doc."""
    characters_kb: dict = field(default_factory=dict)
    # The prop registry, so a prop's agreed appearance reaches the prompt
    # rather than being rebuilt from its identifier.
    props_kb: dict = field(default_factory=dict)
    locations_kb: dict = field(default_factory=dict)
    location_stubs: dict[str, str] = field(default_factory=dict)
    # The film's own top matter -- genre, worldview, theme -- from
    # `kb/film.json`. Only `Film.compiled` reaches a prompt; see Film.
    film: "Film | None" = None
    # Where the panels' greyboxes are, so a prop the greybox staged out of
    # its camera's view is not named (pai_compat.props_out_of_frame). None
    # names every declared prop, as before.
    greyboxes_dir: "Path | None" = None
    target_width: int = 1280
    target_height: int = 544


def lead_cap(text: str) -> str:
    """Upper-case the first letter and change nothing else.

    `str.capitalize()` lower-cases the remainder, which is wrong for any
    clause carrying an initialism: the film's own worldview came back with
    "企业智能办公 AI 系统" flattened to "ai 系统".
    """
    text = (text or "").strip()
    return text[:1].upper() + text[1:] if text else text


def sentence(text: str) -> str:
    """One clause, ready to be joined into a prompt.

    The assembler ends every part with ".", stripping one first -- but it
    strips the ASCII period only, so a Chinese clause ending in "。" came out
    as "。." with both.
    """
    return lead_cap(text).rstrip(" .。．！？")


def location_text(stub) -> str:
    """Descriptive text for a location, whatever shape the stub is.

    Stubs are a plain string in one production and a record in another
    (name / anchor / setting / era / region). Formatting the record
    directly puts Python punctuation, key names, and cross-references into
    the prompt and buries the one field that actually describes the place.
    """
    if isinstance(stub, str):
        return stub
    if isinstance(stub, dict):
        for key in ("anchor", "setting", "description", "descriptor", "text", "name"):
            v = stub.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""



@dataclass(frozen=True)
class Film:
    """The treatment's own top matter: what a film declares about itself
    before any scene exists.

    A production document states these four things above everything else --
    题材 (genre), 核心溯源 (source), 世界观设定 (worldview), 核心主题 (theme) --
    and they had no home here. The world lived as five byte-identical copies
    of `visual_constraints`, one per location stub, read by nothing; the theme
    lived as a paragraph stamped on every shot; genre and source lived
    nowhere at all.

    They are kept apart because they are spent differently, and mixing them
    is how a prompt ends up describing the wrong thing:

      genre            compiled, but only into the atmosphere slot. It is a
                       tone a model can draw, and appending it a second time
                       as a bare phrase ("science fiction, thriller") is the
                       keyword stacking Flux 2 responds worst to.
      design_language  compiled, as a constant this pipeline owns rather than
                       one the prompt writer may restate -- the treatment
                       calls this holding one aesthetic logic across every
                       frame, and it is the same reason the style anchor is
                       withheld.
      premise          NOT compiled. The world's rules, not its look. What an
                       AI system in the story can and cannot do bounds the
                       plot; it is not a description of any frame.
      invariants       NOT compiled. States something across shots that no
                       single frame can honour or violate. Checked instead.
      theme            NOT compiled. A theme is abstract by definition, and
                       prompt_projection has the measurement: writing a
                       director's intent produces a picture OF the intent --
                       "watching them like a surveillance feed" draws a
                       surveillance feed.
      source           NOT compiled. Provenance. Writing 《山海经》 into a
                       prompt draws a book.
    """
    title: str = ""
    genre: tuple[str, ...] = ()
    source: str = ""
    premise: str = ""
    design_language: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    theme: str = ""

    @property
    def compiled(self) -> list[str]:
        """The part of the film appended to every prompt as a constant.

        Genre is deliberately NOT here. It reaches the frame through the
        atmosphere slot instead, where it is the floor under a panel that
        declares no mood -- one field, one job, and no phrase written twice.
        """
        return list(self.design_language)


def _strs(v) -> tuple[str, ...]:
    if isinstance(v, str):
        v = [v]
    return tuple(s.strip() for s in (v or []) if isinstance(s, str) and s.strip())


def film_of(film_file) -> Film:
    """Read `kb/film.json`. A project without one has an empty film, not an
    error: most projects have no treatment on disk and must still compile."""
    try:
        doc = json.loads(Path(film_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return Film()
    if not isinstance(doc, dict):
        return Film()
    w = doc.get("worldview") or {}
    return Film(
        title=str(doc.get("title") or "").strip(),
        genre=_strs(doc.get("genre")),
        source=str(doc.get("source") or "").strip(),
        premise=str(w.get("premise") or "").strip(),
        design_language=_strs(w.get("design_language")),
        invariants=_strs(w.get("invariants")),
        theme=str(doc.get("theme") or "").strip(),
    )


_AGE_IN_STATE = _re_age.compile(r"(\d{1,3})")


def _age_years_from_state(age_state: str | None):
    """The age a state label carries: child_8 -> 8, elder_70_lucid -> 70.

    The label is the only place the number lives; a descriptor says "child,
    small build" and nothing about how old, so an eight-year-old renders as
    a toddler unless the state's number is carried into the prompt.
    """
    if not age_state:
        return None
    m = _AGE_IN_STATE.search(str(age_state))
    return int(m.group(1)) if m else None


def _character_hint(ref: str, characters_kb: dict) -> str:
    """LoRA-free visual descriptor for a character ref ('char' or
    'char@age_state'), read from the project's characters KB — NOT hardcoded.

    Prefers the per-age-state `generic_anchors[age]`, then the single
    `generic_anchor`, then the raw ref when the KB has no description. These
    KB fields are the trigger-free descriptors used for storyboard renders;
    the LoRA trigger (when present) is prepended separately by the caller, so
    we deliberately never fall back to `anchor` (which embeds the trigger)."""
    char, _, age = ref.partition("@")
    kb = resolve_character(char, characters_kb or {})

    _GENDER_WORDS = ("man", "woman", "boy", "girl", "male", "female",
                     "gentleman", "lady")

    def _with_sex(text: str) -> str:
        """Name the person the registry says this is.

        `sex` is authored on every character and was compiled by nothing, so a
        descriptor reached the model as "athletic build, Caucasian, with short
        greying dark hair, weathered features" — true of a man or a woman. The
        proxy does not settle it either: every subject is staged from the same
        gender="neutral" SMPL-X body, whose chest measures 1.03x its waist. So
        with a LoRA trained for none of this cast, the only sex-bearing token
        in the whole prompt was one inert trigger word, `nina_female`, emitted
        for the first-listed character and attached to no position. A male
        character on the left rendered as a woman, which is the sampler
        resolving a question nothing had answered.
        """
        sex = str(kb.get("sex") or "").strip().lower()
        if not text or sex not in ("male", "female"):
            return text
        if any(w in text.lower().split() or f"{w}," in text.lower()
               for w in _GENDER_WORDS):
            return text                     # already says who this is
        yrs = _age_years_from_state(age)
        if yrs is not None and yrs < 13:
            noun = "boy" if sex == "male" else "girl"
        elif yrs is not None and yrs < 18:
            noun = "teenage boy" if sex == "male" else "teenage girl"
        else:
            noun = "man" if sex == "male" else "woman"
        return f"{noun}, {text}"

    def _with_age(text: str) -> str:
        """Prefix the state's age when the descriptor does not state one."""
        yrs = _age_years_from_state(age)
        if not yrs or _re_age.search(r"\b\d{1,3}[- ]?(year|yr)", text or ""):
            return text
        return f"{yrs}-year-old {text}" if text else text

    ga = kb.get("generic_anchors")
    if age and isinstance(ga, dict) and age in ga:
        return _with_age(_with_sex(ga[age]))
    def _with_costume(text: str) -> str:
        """Append the registered costume when the descriptor omits clothing.

        `costumes` is a per-character registry field and was never compiled,
        so what a character wears was re-invented at every render and a shirt
        changed colour between two panels of one scene. Clothing is exactly
        the kind of detail a production fixes once and holds.
        """
        outfit = costume_text(kb, age)
        if not outfit:
            return text
        outfit = outfit.strip().rstrip(".")
        if outfit.lower() in (text or "").lower():
            return text
        return f"{text}, wearing {outfit}" if text else outfit

    generic = kb.get("generic_anchor")
    if generic:
        return _with_costume(_with_age(_with_sex(generic)))
    # Last resort before giving up: the trigger-bearing `anchor`, with the
    # trigger removed. Returning the raw ref instead sends an internal
    # identifier ("alice@adult_30") to a text encoder, which describes
    # nobody — a descriptor that merely needs cleaning is strictly better
    # than no description of the character at all.
    anchor = kb.get("anchor")
    if isinstance(anchor, str) and anchor.strip():
        trigger = kb.get("trigger")
        if isinstance(trigger, str) and trigger:
            anchor = anchor.replace(trigger, "").strip(" ,.")
        anchor = anchor.strip(" ,.")
        if anchor:
            return _with_costume(_with_age(_with_sex(anchor)))
    return ref


def _hair_phrase(descriptor: str) -> str | None:
    """The hair clause of a registry descriptor, or None.

    Descriptors read "<build>, <region>, with <hair>[, | and] <face>": the
    hair runs from the "with" before the word "hair" to the first ", " or
    " and " after it -- "short black hair flecked with grey", "long brown
    hair greying at the temples".
    """
    low = (descriptor or "").lower()
    i = low.find("hair")
    if i < 0:
        return None
    w = low.rfind("with ", 0, i)
    start = w + len("with ") if w >= 0 else low.rfind(", ", 0, i) + 2
    cuts = [j for j in (low.find(", ", i), low.find(" and ", i)) if j >= 0]
    end = min(cuts) if cuts else len(low)
    return descriptor[max(start, 0):end].strip(" ,.") or None


def _character_back_hint(ref: str, characters_kb: dict) -> str:
    """A character as a lens behind them sees them: no face.

    `_character_hint` describes the face, and a subject the camera sees from
    behind has none to show. Measured on the over-the-shoulder of one scene's
    "he swivels his chair around to face his son", with the geometry fixed:
    told the man had "weathered features and a determined expression", the sampler
    turned him to face the lens, or painted his face onto the back of his head;
    told he was seen from behind, it kept him turned toward his son. So this
    keeps what reads from behind -- sex, age, build, hair, clothing -- and says
    the face is not visible.
    """
    char, _, age = ref.partition("@")
    kb = resolve_character(char, characters_kb or {})
    ga = kb.get("generic_anchors")
    text = (ga.get(age) if age and isinstance(ga, dict) else None) \
        or kb.get("generic_anchor")
    if not text and isinstance(kb.get("anchor"), str):
        text = kb["anchor"]
        trigger = kb.get("trigger")
        if isinstance(trigger, str) and trigger:
            text = text.replace(trigger, "")
    text = (text or "").strip(" ,.")
    sex = str(kb.get("sex") or "").strip().lower()
    yrs = _age_years_from_state(age)
    if sex in ("male", "female"):
        if yrs is not None and yrs < 13:
            noun = "boy" if sex == "male" else "girl"
        elif yrs is not None and yrs < 18:
            noun = "teenage boy" if sex == "male" else "teenage girl"
        else:
            noun = "man" if sex == "male" else "woman"
    else:
        noun = "person"
    pro = {"male": "his", "female": "her"}.get(sex, "their")
    bits = [f"{yrs}-year-old {noun}" if yrs else noun,
            f"seen from behind with {pro} back to the camera and {pro} face not visible"]
    build = text.split(", with ")[0].strip(" ,.") if ", with " in text else ""
    if build:
        bits.append(build)
    hair = _hair_phrase(text)
    if hair:
        bits.append(f"the back of {pro} head with {hair}")
    outfit = costume_text(kb, age)
    if outfit:
        bits.append(f"wearing {outfit.strip().rstrip('.')}")
    return ", ".join(bits)


def seen_from_behind(shot: dict, chars: list, pf: dict) -> set:
    """Which declared characters this shot's lens sees from behind.

    `behind` sees everyone's back. An over-the-shoulder frames the focus
    subject's face past another's shoulder, so everyone else is seen from
    behind. With no focus declared an OTS names nobody: the greybox picks the
    near subject from seat depth, which the text compiler cannot see, and
    guessing would describe the wrong person's back.
    """
    pos = str(((shot.get("camera") or {}).get("extrinsics") or {})
              .get("position") or "").strip()
    if pos == "behind":
        return set(chars)
    if pos == "ots" and len(chars) > 1:
        focus = _focus_character(pf)
        if focus in chars:
            return {c for c in chars if c != focus}
    return set()


def _focus_character(pf: dict) -> str:
    """The character a primary_focus is about, or "".

    The schema keeps the focus id in `ref` and reserves `of_character` for
    type="feature" -- "which character does the feature belong to". A
    compiler that reads only `of_character` gets null for every
    character-type focus, so declaring `primary_focus: alice` would have no
    effect on anything: the trigger would not be steered to her, and
    nothing in the prompt would say she was the subject.
    """
    pf = pf or {}
    of_char = str(pf.get("of_character") or "")
    if of_char:
        return of_char.partition("@")[0]
    if str(pf.get("type") or "character") == "character":
        return str(pf.get("ref") or "").partition("@")[0]
    return ""


def _same_text(a, b) -> bool:
    """True when two authored strings say the same thing, allowing for one
    being a prefix or superset of the other after whitespace/case folding.

    A location's description is routinely copied into both a location
    stub's anchor and a shot's environment.background — the same ~700
    characters of prose authored once and referenced from two fields.
    Emitting both puts the same paragraph in a prompt twice; a compiler
    that composes a "set in {loc}" clause and a "backed by {bg}" clause
    needs this check between them before appending the second."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    na = " ".join(a.split()).strip().lower().rstrip(".")
    nb = " ".join(b.split()).strip().lower().rstrip(".")
    if not na or not nb:
        return False
    return na == nb or na.startswith(nb) or nb.startswith(na)


def _same_scope(loc, bg) -> bool:
    """True when a location anchor and a shot's environment.background both
    declare the same INT/EXT scope.

    That's the signal a shot-specific background is a refinement or
    correction of the SAME setting the location anchor already names (a
    close-up authoring an explicit exclusion of something the wide generic
    anchor still mentions), not an additional fact about a different one
    (the exterior view visible through this INT cabin's own windows) --
    the two look identical to a plain text diff, but only one of them is
    safe to simply add to the other. Neither field carrying a recognizable
    INT/EXT token is treated as "don't know" rather than "same", so an
    unmarked pair falls back to the older, more conservative behaviour of
    keeping both."""
    def _scope(s):
        s = (s or "").strip().upper()
        if s.startswith("INT"):
            return "INT"
        if s.startswith("EXT"):
            return "EXT"
        return None
    a, b = _scope(loc), _scope(bg)
    return a is not None and a == b


# Where a declared screen x falls, said the way a text encoder reads it. The
# bands are the thirteen-zone grid's own columns collapsed to the horizontal
# axis, so a subject declared `center_left` and one declared x=0.38 get the
# same words.
_SCREEN_X_PHRASE = (
    (0.20, "at the far left of the frame"),
    (0.42, "on the left"),
    (0.58, "in the centre"),
    (0.80, "on the right"),
    (1.01, "at the far right of the frame"),
)


def screen_x_phrase(x: float) -> str:
    for edge, phrase in _SCREEN_X_PHRASE:
        if x < edge:
            return phrase
    return _SCREEN_X_PHRASE[-1][1]


def subjects_left_to_right(shot: dict) -> list[dict]:
    """Declared subjects in screen order, each with the phrase for where it sits.

    The greybox stages every body from `screen_position`, and the compiled
    prompt has historically named the same people in registry order with no
    spatial anchor at all. Geometry then says one thing and text says nothing,
    so which character lands in which seat is left to the sampler: across a
    set of renders of one panel, the same three descriptions arrived in
    different seats each time while the *positions* stayed correct, because
    nothing ever tied a description to a place.

    Returning the subjects sorted by the same declared x the greybox stages
    from is what lets the prompt make that tie. A subject with no declared
    position sorts last and gets no phrase rather than an invented one: an
    unplaced subject is a KB gap, and guessing a side for it would put text and
    geometry back into disagreement in the one case where the disagreement is
    real.
    """
    out: list[dict] = []
    for i, sub in enumerate(dig(shot, "setup", "subjects") or []):
        cid = (sub or {}).get("character_id")
        if not cid:
            continue
        sp = (sub.get("screen_position") or {})
        x = sp.get("x")
        if x is None:
            x = _ZONE_X.get(sp.get("zone"))
        out.append({"character_id": cid, "age_state": sub.get("age_state"),
                    "x": None if x is None else float(x),
                    "phrase": screen_x_phrase(float(x)) if x is not None else None,
                    "_declared_order": i})
    out.sort(key=lambda s: (s["x"] is None, s["x"] if s["x"] is not None else 0.0,
                            s["_declared_order"]))
    return out


# Horizontal centre of each named zone, for subjects that declare a zone
# rather than a continuous position. Mirrors composition_solver's own table.
_ZONE_X = {
    "far_left": 0.10, "left": 0.20, "center_left": 0.38, "centre_left": 0.38,
    "center": 0.50, "centre": 0.50,
    "center_right": 0.62, "centre_right": 0.62, "right": 0.80, "far_right": 0.90,
    "top_left": 0.20, "top_center": 0.50, "top_centre": 0.50, "top_right": 0.80,
    "bottom_left": 0.20, "bottom_center": 0.50, "bottom_centre": 0.50,
    "bottom_right": 0.80,
}


def _focus_ordered_characters(chars: list[str], pf: dict) -> list[str]:
    """Put the primary-focus owner first for single-trigger selection."""
    focus_char = _focus_character(pf)
    if not focus_char:
        return chars
    ordered = [c for c in chars if str(c).partition("@")[0] == focus_char]
    ordered.extend(c for c in chars if str(c).partition("@")[0] != focus_char)
    return ordered or chars


# ─────────────────────  reused-verbatim framing tables  ─────────────────
#
# compile_gpt_image_2.py has not grown its own tuned copy of these (unlike
# compile_flux2.py, which forked its own) — reused here as-is rather than
# lost when compile_flux.py, their original home, was deleted.

_SHOT_SIZE_FRAMING_EN = {
    # Keyed on the ShotSize Literal, tightest to widest. The Chinese 景别 term
    # is kept as a comment: it is the professional vocabulary these entries
    # were written against, and it used to be the key itself.
    "extreme_close_up": (                                       # 大特写
        "an extreme close-up, a single facial feature filling 70-90% of the 2.35:1 frame, "
        "the background reduced to soft blur or a simple wall"
    ),
    "close_up": (                                               # 特写
        "a close-up of the head, the face filling 50-65% of the frame with at most "
        "a hint of shoulders, simple background"
    ),
    "medium_close_up": (                                        # 中近景
        "a medium close-up framing the subject from the chest up, head in the upper third, "
        "hands occasionally visible, the immediate setting partly visible"
    ),
    "medium": (                                                 # 中景
        "a medium shot framing the subject from the waist up, head in the upper quarter, "
        "body posture clear, the surrounding environment visible"
    ),
    "medium_full": (                                            # 中全景
        "a medium-full shot framing the subject from the knees up, "
        "stance and gesture visible, the setting legible around them"
    ),
    "full": (                                                   # 全景
        "a wide shot showing the full subject head to feet at roughly 20-30% of frame height, "
        "with the environment visible to the edges"
    ),
    "master": (                                                 # 主镜头
        "a master shot covering the whole scene in one setup, every subject "
        "visible in their staged positions"
    ),
    "wide": (                                                   # 远景
        "a distant wide shot, the figure at 10-20% of frame height in a substantial environment"
    ),
    "establishing": (                                           # 大远景 / 定场镜头
        "an extreme establishing wide shot, the figure tiny at under 5% of the frame, "
        "the environment dominating the composition with a strong sense of scale and isolation"
    ),
}


_ANGLE_FRAMING_EN = {
    "eye_level":    None,
    "high":         "shot from above looking down on the subject at a high angle",
    "low":          "shot from below looking up at the subject at a low angle",
    "aerial":       "from an aerial drone perspective far above, the ground spreading out like a map",
    "dutch":        "on a dutch angle, the horizon canted",
    "low_position": "from a very low camera position close to the ground",
    "overhead":     None,
}
