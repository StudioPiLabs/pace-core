"""Central, env-overridable timeouts (seconds) for long-running job operations.

Every job-bounding operation reads its budget from here, so an operator can
extend any of them via an env var without touching code — e.g.

    PAI_TIMEOUT_RECON_PIPELINE=2400  PAI_TIMEOUT_VACE=1800  pai-studio

Defaults preserve the prior hardcoded values, so behaviour is unchanged unless
an env var is set. Only the *job-bounding* timeouts live here (the ones that, if
too short, fail a job mid-flight); tiny infra calls (mkdir/rm, HTTP probes) keep
their inline values.
"""
import os


def _t(var: str, default: int) -> int:
    """Env-overridable timeout in seconds; falls back to `default` on unset/bad input."""
    try:
        v = int(os.environ.get(var, default))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


# ── Reconstruction (mesh_recon) ────────────────────────────────────────────────
RECON_PIPELINE = _t("PAI_TIMEOUT_RECON_PIPELINE", 1200)   # VGGT→Poisson on the box
RECON_FETCH    = _t("PAI_TIMEOUT_RECON_FETCH",     520)   # fetch the raw mesh (45MB+)

# ── VACE video render (playground_video) ───────────────────────────────────────
VACE_WAIT      = _t("PAI_TIMEOUT_VACE",            1200)  # wait for ComfyUI to finish a render
COMFY_WAIT     = _t("PAI_TIMEOUT_COMFY_JOB",        600)  # poll a single ComfyUI prompt
# Liveness probe before a queued job is handed to its worker. A single short
# probe decided whether the job lived, so a worker busy with a large render
# looked dead and the submission was thrown away.
COMFY_HEALTH        = _t("PAI_TIMEOUT_COMFY_HEALTH", 15)
COMFY_HEALTH_TRIES  = _t("PAI_COMFY_HEALTH_TRIES",    3)

# ── Blender (depth render is per-frame; this is the floor under n_frames×12) ────
DEPTH_RENDER_FLOOR = _t("PAI_TIMEOUT_DEPTH_RENDER",   1200)
BLENDER_CONVERT    = _t("PAI_TIMEOUT_BLENDER_CONVERT", 300)  # FBX/OBJ/.blend → GLB
BLENDER_OP         = _t("PAI_TIMEOUT_BLENDER_OP",      300)  # assemble/export/list kernels

# ── SAM3 segmentation (sam3_segment) — per clip / per chunk ─────────────────────
SAM3_SEGMENT   = _t("PAI_TIMEOUT_SAM3",             900)

# ── File staging / fetch over SSH (shared GpuBox defaults) ─────────────────────
STAGE_IN  = _t("PAI_TIMEOUT_STAGE", 180)
FETCH_OUT = _t("PAI_TIMEOUT_FETCH", 180)
