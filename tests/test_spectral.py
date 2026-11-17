"""Tests for the Nyquist spectral-grid helpers and flux-conserving rebin."""

from itertools import pairwise

import ineedvalidation as vv
import jax
import jax.numpy as jnp
import pytest
from hwoutils.spectral import Band, Quadrature

from coronachrome.spectral import (
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


def test_spectral_grid_composes_count_and_edges():
    """spectral_grid combines the channel count and edge/center construction."""
    edges = spectral_grid(140.0, 855.0, 1045.0)
    assert len(edges) == 29
    assert edges[0].lo_nm == 855.0 and edges[-1].hi_nm == 1045.0
    assert all(a.hi_nm == b.lo_nm for a, b in pairwise(edges))
    edges = jnp.asarray([edges[0].lo_nm] + [b.hi_nm for b in edges])


def test_spectral_grid_oversample_nests():
    """Every oversample-th hi-res edge coincides with a base channel edge."""
    edges = spectral_grid(140.0, 855.0, 1045.0)
    edges_hi = spectral_grid(140.0, 855.0, 1045.0, oversample=6)
    assert len(edges_hi) == 29 * 6
    assert all(a.lo_nm == b.lo_nm for a, b in zip(edges_hi[::6], edges, strict=True))
    assert edges_hi[-1].hi_nm == edges[-1].hi_nm


@vv.case("ifs-extraction", "code-verification")
def test_rebin_conserves_total_flux():
    """Fails if overlap weights lose flux; float64 floor of 8 eps on exact tiling."""
    src_edges = Band(600.0, 700.0).channels(24, "log")
    dst_edges = Band(600.0, 700.0).channels(8, "log")
    q = Quadrature.build(src_edges, None, 3)
    values = q.integrate(q.nodes_nm**2)
    exact = (700.0**3 - 600.0**3) / 3
    assert jnp.allclose(values.sum(), exact, rtol=2e-15, atol=0)
    out = rebin_channels(values, src_edges, dst_edges)
    assert out.shape == (8,)
    assert jnp.allclose(out.sum(), values.sum(), rtol=2e-15, atol=0)


def test_rebin_nested_grid_equals_reshape_sum():
    """On an exactly nested grid the overlap rebin reduces to reshape-sum."""
    edges = spectral_grid(50.0, 600.0, 700.0)
    n = len(edges)
    edges_hi = spectral_grid(50.0, 600.0, 700.0, oversample=4)
    vals = (jnp.arange(n * 4, dtype=float) + 1.0) ** 1.3
    out = rebin_channels(vals, edges_hi, edges)
    assert jnp.allclose(out, vals.reshape(n, 4).sum(axis=1), rtol=1e-10)


def test_rebin_along_cube_axis():
    """Rebinning applies along the chosen axis of a (n_wav, ny, nx) cube."""
    src_edges = Band(600.0, 700.0).channels(12, "log")
    dst_edges = Band(600.0, 700.0).channels(4, "log")
    cube = jax.random.uniform(jax.random.PRNGKey(0), (12, 3, 2))
    out = rebin_channels(cube, src_edges, dst_edges, axis=0)
    assert out.shape == (4, 3, 2)
    assert jnp.allclose(out.sum(axis=0), cube.sum(axis=0), rtol=2e-15, atol=0)


def test_rebin_partial_overlap_takes_fraction():
    """A destination bin covering half a source bin takes half its flux."""
    out = rebin_channels(jnp.array([2.0]), (Band(1.0, 2.0),), (Band(1.0, 1.5),))
    assert jnp.allclose(out, jnp.array([1.0]))
