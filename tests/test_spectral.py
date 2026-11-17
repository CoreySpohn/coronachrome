"""Tests for the Nyquist spectral-grid helpers."""

from itertools import pairwise

import jax.numpy as jnp
import pytest

from coronachrome.spectral import (
    n_nyquist_channels,
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
