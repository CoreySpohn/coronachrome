"""SpatialChannelIR construction checks."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from conftest import sky_grid
from optixstuff.grids import PixelGrid, footprint_overlap
from optixstuff.spectral import Band

from coronachrome import LensletLayout, SpatialChannelIR


def _ir(overlap=None):
    layout = LensletLayout(2)
    grid, indices = layout.image_geometry(0.2, 0.0, None)
    return SpatialChannelIR(
        jnp.zeros((4, 1, 9), dtype=int),
        jnp.ones((4, 1, 9)) / 9,
        indices,
        (Band(650, 670),),
        sky_grid((16, 16)),
        grid,
        PixelGrid.regular((32, 32), 13e-6, "meter"),
        layout,
        (grid,),
        footprint_overlap(sky_grid((16, 16)), (grid,)) if overlap is None else overlap,
    )


def test_constructs():
    """A coherent IR carries bands and both physical grids."""
    ir = _ir()
    assert ir.n_channels == 4 and ir.n_bands == 1
    assert ir.det_shape == (32, 32)


def test_rejects_mismatched_spatial():
    """A channel-to-pixel mapping must cover every lenslet."""
    ir = _ir()
    with pytest.raises(ValueError, match="lenslet_indices"):
        replace(ir, lenslet_indices=jnp.zeros(5, dtype=int))


def test_rejects_wrong_n_channels():
    """Footprint dimensions must agree with layout and bands."""
    with pytest.raises(ValueError, match="n_channels"):
        replace(_ir(), layout=LensletLayout(3))
    assert eqx.is_array(_ir().det_vals)


@pytest.mark.parametrize("form", ["sparse", "axes"])
def test_overlap_weights_are_strong_leaves_under_x32(form):
    """Fails if the IR holds NumPy or weakly typed collection overlap weights.

    The overlap arrives from optixstuff's footprint_overlap as strong leaves:
    weights are float32 and integer indices stay int32. Truth source: the leaf
    rule; dtypes are exact, so there is no tolerance.
    """
    with jax.enable_x64(False):
        if form == "axes":
            target = PixelGrid.regular((2, 2), 0.35, "arcsec")
            overlap = footprint_overlap(sky_grid((16, 16)), target)
            assert overlap.axes is not None
        else:
            cells = (
                PixelGrid.regular((1, 2), 0.35, "arcsec"),
                PixelGrid.regular((1, 2), 0.35, "arcsec", center=(0.5, 0.0)),
            )
            overlap = footprint_overlap(sky_grid((16, 16)), cells)
            assert overlap.sparse is not None
        held = _ir(overlap).collection_overlap
        group = held.axes if form == "axes" else held.sparse
        dtypes = (
            [jnp.float32] * 2 if form == "axes" else [jnp.int32] * 2 + [jnp.float32]
        )
        for leaf, dtype in zip(group, dtypes, strict=True):
            assert isinstance(leaf, jax.Array)
            assert leaf.dtype == dtype
            assert not leaf.weak_type
