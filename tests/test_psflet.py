"""Tests for analytic PSFlet profiles and LSF smearing."""

import jax.numpy as jnp

from coronachrome.psflet import gaussian_psflet, moffat_psflet, psflet_weights


def test_gaussian_peaks_at_origin():
    """Gaussian PSFlet evaluates to unity at the origin."""
    assert jnp.allclose(gaussian_psflet(jnp.array(0.0), jnp.array(0.0), 0.7), 1.0)


def test_moffat_peaks_at_origin():
    """Moffat PSFlet evaluates to unity at the origin."""
    assert jnp.allclose(moffat_psflet(jnp.array(0.0), jnp.array(0.0), 1.0, 2.5), 1.0)


def test_psflet_weights_shape_matches_footprint():
    """psflet_weights returns one weight per footprint offset."""
    off = jnp.arange(-3, 4)
    dy, dx = jnp.meshgrid(off, off, indexing="ij")
    dy, dx = dy.reshape(-1).astype(float), dx.reshape(-1).astype(float)
    w = psflet_weights(dx, dy, "gaussian", jnp.array([0.7]), smear_px=0.0)
    assert w.shape == dx.shape


def test_lsf_smear_widens_along_x():
    """LSF smearing along x lowers the central peak relative to no smear."""
    off = jnp.arange(-5, 6)
    dy, dx = jnp.meshgrid(off, off, indexing="ij")
    dy, dx = dy.reshape(-1).astype(float), dx.reshape(-1).astype(float)
    narrow = psflet_weights(dx, dy, "gaussian", jnp.array([0.7]), smear_px=0.0)
    wide = psflet_weights(dx, dy, "gaussian", jnp.array([0.7]), smear_px=6.0)
    center = jnp.argmin(dx**2 + dy**2)
    assert wide[center] < narrow[center]
