"""Tests for analytic PSFlet profiles and LSF smearing."""

import jax
import jax.numpy as jnp

from coronachrome.psflet import gaussian_psflet, moffat_psflet, psflet_weights


def test_pixel_integration_reduces_sampling_phase_bias():
    """The integrated Gaussian's rendered width is far more stable.

    Sub-pixel centroid phase impacts point sampling much more than pixel
    integration, in the undersampled regime.
    """
    sigma = 0.7
    half = 5
    off = jnp.arange(-half, half + 1).astype(float)
    dyg, dxg = jnp.meshgrid(off, off, indexing="ij")
    dxg, dyg = dxg.reshape(-1), dyg.reshape(-1)
    phases = jnp.linspace(0.0, 0.9, 10)

    def width(weights):
        w = weights / weights.sum()
        cx = (w * dxg).sum()
        return jnp.sqrt((w * (dxg - cx) ** 2).sum())

    def point_width(phase):
        r2 = (dxg - phase) ** 2 + dyg**2
        return width(jnp.exp(-r2 / (2.0 * sigma**2)))

    def erf_width(phase):
        return width(gaussian_psflet(dxg - phase, dyg, sigma))

    point_var = jnp.var(jax.vmap(point_width)(phases))
    erf_var = jnp.var(jax.vmap(erf_width)(phases))
    assert erf_var < point_var


def test_gaussian_pixel_integral_matches_numerical():
    """ERF Gaussian matches numerical integral up to normalization.

    Compared to a fine numerical integral of the point profile.
    """
    sigma = 0.7
    off = jnp.arange(-3, 4).astype(float)
    dy, dx = jnp.meshgrid(off, off, indexing="ij")
    dx, dy = dx.reshape(-1), dy.reshape(-1)
    erf_w = gaussian_psflet(dx, dy, sigma)

    n = 51
    sub = jnp.linspace(-0.5 + 0.5 / n, 0.5 - 0.5 / n, n)
    uu, vv = jnp.meshgrid(sub, sub, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    r2 = (dx[:, None] + uu) ** 2 + (dy[:, None] + vv) ** 2
    num_w = jnp.exp(-r2 / (2.0 * sigma**2)).mean(axis=1)

    assert jnp.allclose(erf_w / erf_w.sum(), num_w / num_w.sum(), atol=1e-4)


def test_gaussian_pixel_integral_is_symmetric():
    """For a centroid centered in a pixel the integral is even in the offset."""
    sigma = 0.7
    t = jnp.array([0.0, 0.5, 1.3, 2.1])
    z = jnp.zeros_like(t)
    assert jnp.allclose(gaussian_psflet(t, z, sigma), gaussian_psflet(-t, z, sigma))


def test_gaussian_psflet_differentiable():
    """ERF Gaussian is differentiable in sigma."""
    g = jax.grad(lambda s: gaussian_psflet(jnp.array(0.3), jnp.array(0.2), s))(0.7)
    assert jnp.isfinite(g)


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
