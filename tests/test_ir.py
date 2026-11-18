"""SpatialChannelIR construction checks."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import pytest
from conftest import sky_grid
from hwoutils.grids import PixelGrid, footprint_overlap
from hwoutils.spectral import Band

from coronachrome import LensletLayout, SpatialChannelIR


def _ir():
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
        footprint_overlap(sky_grid((16, 16)), (grid,)),
    )


def test_constructs():
    """A coherent IR carries bands and both physical grids."""
    ir = _ir()
    assert ir.n_channels == 4 and ir.n_wav == 1
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
