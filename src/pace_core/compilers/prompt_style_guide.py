"""Per-model prompting conventions, in prose — single source of truth for
anything that needs to explain "what makes a good prompt for this model"
in natural language (currently: the Playground/Library's LLM-powered
"✨ Enhance" prompt button, src/studio/routers/prompt_assist.py).

This is deliberately NOT the same thing as compile_flux2.py / compile_
nano_banana.py / compile_gpt_image_2.py / movement_io.py's Wan motion
phrasing — those are deterministic compilers over STRUCTURED PAI scene/
shot/panel data. This module distills the same underlying per-model
knowledge (documented in those files' comments) into free-text guidance
usable when there's no structured panel behind the prompt, e.g. a
Playground node's hand-typed textarea.
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
    "wan_vace": (
        "Wan VACE is a video model — describe both the SCENE and the MOTION (subject, "
        "action, camera movement) in fluent English motion language, e.g. 'slow camera "
        "push-in', 'tracking shot following the subject', 'gentle handheld camera "
        "shake', rather than abstract motion tokens. If a reference image or control "
        "GLB is wired into this render, describe what's already shown there — VACE's "
        "control_strength only trades structure-fidelity vs. reference-fidelity, it "
        "never adds people or objects the reference doesn't already depict. Describe "
        "ONE consistent scene held across the whole clip, not a sequence of different "
        "scenes."
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
