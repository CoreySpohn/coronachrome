"""Tests for the Nyquist spectral-grid helpers and flux-conserving rebin."""

import jax
import jax.numpy as jnp
import pytest

from coronachrome.spectral import (
    channel_centers,
    channel_edges,
    n_nyquist_channels,
    rebin_channels,
    spectral_grid,
)


def test_n_nyquist_channels_formula():
    """Channel count is ceil(R ln(lam2/lam1)): R=140, 20% band -> 29 channels."""
    assert n_nyquist_channels(140.0, 855.0, 1045.0) == 29


def test_n_nyquist_channels_exact_resolution_element():
    """A band exactly one resolution element wide is one channel, not two.

    lam2 = lam1 * exp(1/R) makes R ln(lam2/lam1) exactly 1 up to float
    rounding; the ceil must not bump an epsilon overshoot to 2.
    """
    lam1 = 660.0
    lam2 = lam1 * float(jnp.exp(1.0 / 50.0))
    assert n_nyquist_channels(50.0, lam1, lam2) == 1


def test_n_nyquist_channels_validates_band():
    """A reversed (or empty) band raises instead of returning a bogus count."""
    with pytest.raises(ValueError, match="lam_max_nm"):
        n_nyquist_channels(50.0, 700.0, 600.0)


def test_channel_edges_constant_ratio():
    """Log-spaced edges keep a constant ratio, so R is constant per channel."""
    edges = channel_edges(29, 855.0, 1045.0)
    assert edges.shape == (30,)
    ratios = edges[1:] / edges[:-1]
    assert jnp.allclose(ratios, ratios[0], rtol=1e-12)
    assert jnp.allclose(edges[0], 855.0)
    assert jnp.allclose(edges[-1], 1045.0)


def test_channel_centers_are_geometric_means():
    """Centers are the geometric means of adjacent log-spaced edges."""
    edges = channel_edges(4, 500.0, 1000.0)
    centers = channel_centers(edges)
    assert centers.shape == (4,)
    assert jnp.allclose(centers, jnp.sqrt(edges[:-1] * edges[1:]))


def test_spectral_grid_composes_count_and_edges():
    """spectral_grid combines the channel count and edge/center construction."""
    centers, edges = spectral_grid(140.0, 855.0, 1045.0)
    assert centers.shape == (29,)
    assert edges.shape == (30,)


def test_spectral_grid_oversample_nests():
    """Every oversample-th hi-res edge coincides with a base channel edge."""
    _, edges = spectral_grid(140.0, 855.0, 1045.0)
    centers_hi, edges_hi = spectral_grid(140.0, 855.0, 1045.0, oversample=6)
    assert centers_hi.shape == (29 * 6,)
    assert jnp.allclose(edges_hi[::6], edges, rtol=1e-12)


def test_rebin_conserves_total_flux():
    """Total integrated flux is preserved when the grids share a support."""
    src_edges = channel_edges(24, 600.0, 700.0)
    dst_edges = channel_edges(8, 600.0, 700.0)
    values = jax.random.uniform(jax.random.PRNGKey(0), (24,)) + 0.1
    out = rebin_channels(values, src_edges, dst_edges)
    assert out.shape == (8,)
    assert jnp.allclose(out.sum(), values.sum(), rtol=1e-12)


def test_rebin_nested_grid_equals_reshape_sum():
    """On an exactly nested grid the overlap rebin reduces to reshape-sum."""
    centers, edges = spectral_grid(50.0, 600.0, 700.0)
    n = centers.shape[0]
    _, edges_hi = spectral_grid(50.0, 600.0, 700.0, oversample=4)
    vals = (jnp.arange(n * 4, dtype=float) + 1.0) ** 1.3
    out = rebin_channels(vals, edges_hi, edges)
    assert jnp.allclose(out, vals.reshape(n, 4).sum(axis=1), rtol=1e-10)


def test_rebin_along_cube_axis():
    """Rebinning applies along the chosen axis of a (n_wav, ny, nx) cube."""
    src_edges = channel_edges(12, 600.0, 700.0)
    dst_edges = channel_edges(4, 600.0, 700.0)
    cube = jax.random.uniform(jax.random.PRNGKey(0), (12, 3, 2))
    out = rebin_channels(cube, src_edges, dst_edges, axis=0)
    assert out.shape == (4, 3, 2)
    assert jnp.allclose(out.sum(axis=0), cube.sum(axis=0), rtol=1e-12)


def test_rebin_partial_overlap_takes_fraction():
    """A destination bin covering half a source bin takes half its flux."""
    out = rebin_channels(jnp.array([2.0]), jnp.array([0.0, 1.0]), jnp.array([0.0, 0.5]))
    assert jnp.allclose(out, jnp.array([1.0]))
