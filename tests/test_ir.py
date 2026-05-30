"""Tests for the Spatial Channel IR data model."""

import jax.numpy as jnp
import pytest

from coronachrome.ir import SpatialChannelIR


def _ir(n_channels=4, n_wav=3, n_fp=4, n_psf=9):
    return SpatialChannelIR(
        spatial_src=jnp.zeros((n_channels, n_fp), dtype=jnp.int32),
        spatial_w=jnp.ones((n_channels, n_fp)) / n_fp,
        det_rows=jnp.zeros((n_channels, n_wav, n_psf), dtype=jnp.int32),
        det_vals=jnp.ones((n_channels, n_wav, n_psf)) / n_psf,
        n_channels=n_channels,
        n_wav=n_wav,
        fp_shape=(16, 16),
        det_shape=(32, 32),
    )


def test_constructs():
    """A well-formed IR constructs and exposes its static shape fields."""
    ir = _ir()
    assert ir.n_channels == 4
    assert ir.det_vals.shape == (4, 3, 9)


def test_rejects_mismatched_spatial():
    """Mismatched spatial_src / spatial_w shapes raise ValueError."""
    with pytest.raises(ValueError, match="spatial"):
        SpatialChannelIR(
            spatial_src=jnp.zeros((4, 4), dtype=jnp.int32),
            spatial_w=jnp.ones((4, 5)),
            det_rows=jnp.zeros((4, 3, 9), dtype=jnp.int32),
            det_vals=jnp.ones((4, 3, 9)),
            n_channels=4,
            n_wav=3,
            fp_shape=(16, 16),
            det_shape=(32, 32),
        )


def test_rejects_wrong_n_channels():
    """A det_rows leading dim that disagrees with n_channels raises ValueError."""
    ir = _ir()
    with pytest.raises(ValueError, match="n_channels"):
        SpatialChannelIR(
            spatial_src=ir.spatial_src,
            spatial_w=ir.spatial_w,
            det_rows=ir.det_rows,
            det_vals=ir.det_vals,
            n_channels=99,
            n_wav=3,
            fp_shape=(16, 16),
            det_shape=(32, 32),
        )
