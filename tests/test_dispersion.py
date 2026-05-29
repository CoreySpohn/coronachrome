"""Tests for the dispersion model."""

import jax.numpy as jnp

from coronachrome.dispersion import dispersion_px, lenslet_centroids
from coronachrome.grids import square_grid


def test_dispersion_zero_at_reference():
    """Dispersion offset is zero at the reference wavelength."""
    coeffs = jnp.array([100.0, 0.0])
    assert jnp.allclose(dispersion_px(coeffs, 660.0, 660.0), 0.0)


def test_dispersion_matches_crispy_log_form():
    """Linear coeffs reproduce crispy's npixperdlam*R*log(lam/lam_ref) form."""
    coeffs = jnp.array([100.0, 0.0])
    lam = 700.0
    expected = 100.0 * jnp.log(lam / 660.0)
    assert jnp.allclose(dispersion_px(coeffs, 660.0, lam), expected)


def test_centroids_shape_and_center():
    """Centroids have shape (n_lenslets, n_wav); center lenslet sits at det center."""
    pos = square_grid(4)
    disp = dispersion_px(
        jnp.array([100.0, 0.0]), 660.0, jnp.array([650.0, 660.0, 670.0])
    )
    xc, yc = lenslet_centroids(
        pos, scale=13.0, angle_rad=0.0, dispersion_offsets=disp, det_shape=(256, 256)
    )
    assert xc.shape == (16, 3)
    assert yc.shape == (16, 3)
    center_idx = int(jnp.argmin(jnp.abs(pos[:, 0]) + jnp.abs(pos[:, 1])))
    assert jnp.allclose(xc[center_idx, 1], 128.0)
    assert jnp.allclose(yc[center_idx, 1], 128.0)
