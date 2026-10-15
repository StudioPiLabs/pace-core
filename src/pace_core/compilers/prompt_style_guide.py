"""Per-model prompting conventions, in prose — single source of truth for
anything that needs to explain "what makes a good prompt for this model"
in natural language (e.g. an LLM-powered prompt-enhance action in a host UI).

This is deliberately NOT the same thing as compile_flux2.py / compile_
nano_banana.py / compile_gpt_image_2.py / movement_io.py's motion
phrasing — those are deterministic compilers over STRUCTURED PACE scene/
shot/panel data. This module distills the same underlying per-model
knowledge (documented in those files' comments) into free-text guidance
usable when there's no structured panel behind the prompt, e.g. a
hand-typed prompt.
"""
from __future__ import annotations

MODEL_PROMPT_GUIDES: dict[str, str] = {
    "flux2": (
        "Flux 2 uses a true LLM text encoder (Mistral-Small), not a token-limited "
        "CLIP/T5 encoder — it responds best to natural, fluent descriptive sentences "
        "rather than comma-separated keyword stacks. Avoid 'museum-quality', 'film "
        "grain', 'period-accurate' (over-cooked on this model, distorts more than it "
        "helps); numeric coverage percentages like '35% of frame' (no compliance "
        "payoff, just distracts); 'in the air' unless something is actually airborne "
        "(dust/smoke/mist/snow/embers); redundant 'interior'/'exterior' if the setting "
        "already states it. Keep negatives short and precise — Flux 2 handles negation "
        "well, so a few targeted exclusions beat a long redundant list. One or two "
        "clear sentences describing subject, action, setting, lighting, and camera "
        "framing outperform keyword-stuffing."
    ),
    "wan_animate": (
        "Wan2.2-Animate replaces the performer in a driving video with a reference "
        "character while preserving the original motion/pose from that video — the "
        "prompt should describe the CHARACTER's physical appearance and the general "
        "scene mood, matching whatever reference image(s) are attached. Don't describe "
        "specific poses or actions in the prompt; those come from the driving video, "
        "not the text."
    ),
    "routerbase": (
        "This targets a hosted third-party model (e.g. GPT-Image-2, Nano Banana, "
        "Seedance) via API. Most of these have NO native negative-prompt slot — state "
        "exclusions inline as a trailing clause ('Do not include: text, watermark, "
        "extra limbs' or 'AVOID INCLUDING: …') rather than a separate negative field. "
        "They favor ONE fluent, clearly-directed paragraph over repeated defensive "
        "qualifiers — with strong instruction-following models, repeating 'ONLY the "
        "subject visible, NOTHING else' several times is counterproductive; state the "
        "directive once, clearly."
    ),
}

GENERIC_PROMPT_GUIDE = (
    "Describe the subject, action, setting, lighting, and camera framing in clear, "
    "concrete language. Avoid vague qualifiers and redundant repetition. Favor plain, "
    "fluent natural-language sentences over keyword-stacking unless you know the "
    "target model specifically rewards tag-style prompting."
)


def guide_for(model_hint: str | None) -> str:
    return MODEL_PROMPT_GUIDES.get(model_hint or "", GENERIC_PROMPT_GUIDE)
