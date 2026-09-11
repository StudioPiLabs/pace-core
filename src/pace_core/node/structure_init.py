"""The sampler's starting image, with the clay render's appearance taken out.

A greybox is described everywhere in this pipeline as "shape only": subject
count, pose, and composition, carrying no appearance. The render it actually
produces is not that. Blender Workbench with STUDIO light and cavity on writes
a smoothly shaded clay pass — 144 distinct grey levels on a three-body cabin
panel — and `structure_flux2` hands that to the sampler as its starting
latents. Shading is appearance, so it survives the denoise alongside the
structure and the delivered panel comes back photoreal no matter what the
style field asked for.

Across the 33 staged panels of the evaluation corpus, a greybox's own flat-region
share — the fraction of pixels equal to both their right and lower neighbour,
which separates flat storyboard fill from photographic falloff — correlates
with the delivered panel's at r = 0.843. The control image, not the style
field, is deciding the house style.

Measured on one corpus shot, one seed, one prompt, the same ``line_art_clean``
clause every panel in the corpus compiles. *pos* is the mean error in each
subject's horizontal position against the mattes the kernel rendered, as a
fraction of frame width; *top* is where each subject's head lands as a
fraction of frame height, against the greybox's own 0.140:

=============  =======  =====  =====================  =====  ================
init           denoise  flat%  top (omar/nina/theo)  pos    delivered
=============  =======  =====  =====================  =====  ================
clay (before)  0.65      8.7   0.189 / 0.371 / 0.189  0.015  photorealism
clay           0.85      7.2   --                     --     unchanged
clay           0.95     15.3   --                     0.222  a fourth adult
flat 12        0.65     41.7   0.329 / 0.338 / 0.357  0.012  restaged smaller
**flat 12**    **0.55** 49.0   0.138 / 0.142 / 0.138  0.028  **style and staging both held**
flat 12        0.45     52.8   --                     --     near-copy of the init
flat 4         0.65     60.1   --                     0.170  positions lost
=============  =======  =====  =====================  =====  ================

Three levers were tried and only the pair works. Raising the denoise spends
the cardinality the geometry path exists to enforce: at 0.95 the three-person
cabin came back with a fourth adult in it, the failure `workflow_registry`
cites as the reason the count arrives as an image at all. Restyling a finished
panel in a second pass does not restyle it — at 0.40 and 0.55 over a delivered
photoreal frame the style stayed at 7--8%, which is the same finding again
from the other end. Flattening alone, at the 0.65 that was tuned for the clay
init, fixes the style and restages the shot: the subjects keep their
horizontal positions and lose their size, an extreme close-up delivered as a
medium.

Flattening *and* dropping the denoise is what holds both, and the two belong
together for one reason: 0.65 existed to give the sampler enough freedom to
escape the clay init's shading. A flat init has no shading to escape, so that
freedom is spent restaging instead. `GREYBOX_INIT_DENOISE` in the render
router is the other half of this module.

Same conclusion the video side already reached about the same clay pass,
recorded in `panel_greybox`: fed to VACE it "carries that appearance along with
the structure", and the fix there was to hand over a signal with no appearance
in it, because "depth has no appearance to leak". A still needs an init that
still reads as the picture, so it gets a flattened greybox rather than the
depth map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Grey levels the greybox is quantised to before it becomes the init.
#
# Both ends of this are real failures rather than a matter of taste. Too many
# levels and the gradient survives, which is the photoreal leak this module
# exists to stop: quantising to 96 levels still leaves the panel reading as a
# smooth clay render. Too few and the staging goes with it — at 4 levels the
# subjects' horizontal positions missed the mattes by 17% of frame width,
# against 1.2% at 12 — because with the tonal separation gone there is little
# left to say which body is in front of which.
DEFAULT_LEVELS = 12

# Suffix for the derived init, written next to the greybox it comes from.
SUFFIX = ".structure.png"


def flatten(img: Image.Image, *, levels: int = DEFAULT_LEVELS) -> Image.Image:
    """Quantise a greybox to `levels` grey steps, keeping its geometry.

    Posterising rather than blurring or edge-detecting is deliberate: every
    silhouette boundary in the clay pass is a step between two regions, so
    quantisation preserves the boundaries exactly while collapsing the smooth
    interior shading that carries the appearance.
    """
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    a = np.asarray(img.convert("L"), dtype=np.float32)
    q = np.round(a / 255.0 * (levels - 1)) / (levels - 1) * 255.0
    return Image.fromarray(q.clip(0, 255).astype(np.uint8)).convert("RGB")


def structure_init_for(greybox: str | Path, *, levels: int = DEFAULT_LEVELS,
                       force: bool = False) -> Path:
    """Path to the flattened init for one greybox, deriving it if needed.

    Cached next to the greybox and rebuilt whenever the greybox is newer, so a
    re-staged panel cannot be rendered from the previous staging's init. The
    greybox itself is left alone: it is what a human reviews, and a reviewer
    reading a posterised frame would be reading something the camera never
    rendered.
    """
    greybox = Path(greybox)
    if not greybox.is_file():
        raise FileNotFoundError(f"no greybox at {greybox}")
    out = greybox.with_name(greybox.name + SUFFIX)
    if (not force and out.is_file()
            and out.stat().st_mtime >= greybox.stat().st_mtime):
        return out
    with Image.open(greybox) as im:
        flatten(im, levels=levels).save(out)
    return out


def flat_region_share(img: str | Path | Image.Image) -> float:
    """Fraction of pixels equal to both their right and lower neighbour.

    The measurement the table in this module's docstring reports. Flat
    storyboard fill scores high, a smooth photographic gradient scores low,
    and it needs no reference image to compare against.
    """
    if isinstance(img, (str, Path)):
        with Image.open(img) as im:
            a = np.asarray(im.convert("L"), dtype=np.int16)
    else:
        a = np.asarray(img.convert("L"), dtype=np.int16)
    same = (a[:-1, :-1] == a[:-1, 1:]) & (a[:-1, :-1] == a[1:, :-1])
    return float(same.mean())
