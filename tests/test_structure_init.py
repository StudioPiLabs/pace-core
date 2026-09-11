"""A greybox reaching the sampler must have lost its shading first.

The greybox is a clay beauty pass, and the pipeline calls it "shape only"
everywhere while handing the sampler a smoothly shaded render. Shading is
appearance, so it survives the denoise and lands in the delivered panel:
measured on one corpus shot, one seed, one prompt, the same
`line_art_clean` style clause every panel compiles, the clay init delivered
grayscale photorealism at 8.7% flat regions where the style asked for line
art. The same panel from a flattened init delivered line art.

Raising the denoise is the lever that looks right and is not: at 0.95 the same
panel came back with a fourth adult in a three-person car, which is the
cardinality failure the geometry path exists to prevent in the first place.

These tests pin the substitution and its scope, not the flattening constants —
those are a rendering judgement and are expected to move.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pace_core.node import structure_init  # noqa: E402


def _clay(path: Path, size=(64, 32)) -> Path:
    """A stand-in for the Workbench pass: a smooth horizontal gradient with a
    darker block in it, so it has both a gradient to lose and an edge to keep."""
    w, h = size
    a = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))
    a[8:24, 16:48] = 40.0
    Image.fromarray(a.astype(np.uint8)).convert("RGB").save(path)
    return path


def test_flattening_removes_the_gradient(tmp_path):
    src = _clay(tmp_path / "gb.png")
    with Image.open(src) as im:
        flat = structure_init.flatten(im, levels=8)
    assert len(np.unique(np.asarray(flat.convert("L")))) <= 8
    # …and the clay pass it came from had many more than that.
    with Image.open(src) as im:
        assert len(np.unique(np.asarray(im.convert("L")))) > 8


def test_flattening_keeps_the_edge(tmp_path):
    """Posterising is chosen over blurring precisely because every silhouette
    in the clay pass is a step between regions, so the boundary survives."""
    src = _clay(tmp_path / "gb.png")
    with Image.open(src) as im:
        flat = np.asarray(structure_init.flatten(im, levels=8).convert("L")).astype(int)
    # the block's own boundary is still a step of real size
    assert abs(int(flat[16, 15]) - int(flat[16, 17])) > 20


def test_flat_region_share_separates_flat_from_gradient(tmp_path):
    src = _clay(tmp_path / "gb.png")
    with Image.open(src) as im:
        flat = structure_init.flatten(im, levels=6)
    assert structure_init.flat_region_share(flat) > structure_init.flat_region_share(src)


def test_the_derived_init_is_cached_next_to_the_greybox(tmp_path):
    src = _clay(tmp_path / "panel_90001_.png")
    out = structure_init.structure_init_for(src)
    assert out.parent == src.parent
    assert out.name.endswith(structure_init.SUFFIX)
    assert out.is_file()
    # second call reuses it rather than rewriting
    before = out.stat().st_mtime_ns
    assert structure_init.structure_init_for(src) == out
    assert out.stat().st_mtime_ns == before


def test_a_restaged_greybox_invalidates_the_cache(tmp_path):
    """The failure this guards is a panel re-staged and then rendered from the
    previous staging's init, which would be geometry silently one revision
    behind the document that asked for it."""
    src = _clay(tmp_path / "panel_90001_.png")
    out = structure_init.structure_init_for(src)
    first = out.read_bytes()

    a = np.zeros((32, 64), dtype=np.uint8)
    a[4:12, 4:20] = 200
    Image.fromarray(a).convert("RGB").save(src)
    import os
    os.utime(src, (out.stat().st_atime + 10, out.stat().st_mtime + 10))

    assert structure_init.structure_init_for(src).read_bytes() != first


def test_a_missing_greybox_is_an_error_not_a_blank(tmp_path):
    with pytest.raises(FileNotFoundError):
        structure_init.structure_init_for(tmp_path / "nope.png")


def test_levels_below_two_is_rejected(tmp_path):
    src = _clay(tmp_path / "gb.png")
    with Image.open(src) as im:
        with pytest.raises(ValueError):
            structure_init.flatten(im, levels=1)
