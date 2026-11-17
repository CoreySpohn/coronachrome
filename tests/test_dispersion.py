"""Tests for the dispersion model."""

import jax.numpy as jnp

from coronachrome.dispersion import lenslet_centroids
from coronachrome.grids import square_grid


def test_centroids_shape_and_center():
    """Exact algebra: opposite lenslets average to the detector centre.

    Four unit-pitch indices are [-1.5, -0.5, 0.5, 1.5]; at scale 13
    their centroid is 127.5. Removing n//2 adds 6.5 px on each axis.
    Default allclose tolerances cover arithmetic roundoff.
    """
    pos = square_grid(4)
    disp = 100.0 * jnp.log(jnp.array([650.0, 660.0, 670.0]) / 660.0)
    xc, yc = lenslet_centroids(
        pos, scale=13.0, angle_rad=0.0, dispersion_offsets=disp, det_shape=(256, 256)
    )
    assert xc.shape == (16, 3)
    assert yc.shape == (16, 3)
    assert jnp.allclose(jnp.mean(xc[:, 1]), (256 - 1) / 2)
    assert jnp.allclose(jnp.mean(yc[:, 1]), (256 - 1) / 2)
