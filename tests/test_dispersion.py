"""Tests for the dispersion model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps

from coronachrome.dispersion import lenslet_centroids
from coronachrome.grids import square_grid


@pytest.mark.float64_only(reason="precision_pair and jaxpr audit require x64")
def test_lenslet_centroids_precision_closure():
    """Allow 16 rounding units for trig, rotation, scale and two offsets.

    The fixture remains near the detector center, so rotation and addition
    are well conditioned relative to the peak centroid coordinate.
    """
    args = (square_grid(2), jnp.array([-0.5, 0.5]))

    def evaluate(positions, offsets):
        return lenslet_centroids(positions, 13.0, 0.25, offsets, (24, 24))

    assert_precision_closed(evaluate, *canonical(args, precision="float32"))
    precision_pair(evaluate, args, bound=float(ulps(16, np.float32(1))))


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


@pytest.mark.parametrize("offset_dtype", [np.float32, np.float64])
def test_centroids_numpy_positions_match_jit(precision, offset_dtype):
    """An unrotated half-step preserves a wide dispersion offset when available."""
    positions = np.array([[0.5, -0.25]], dtype=np.float32)
    offsets = np.array([0.25 + 2**-31], dtype=offset_dtype)
    dtype = np.float32 if precision == np.float32 else offset_dtype
    expected = 5 + offsets.astype(dtype)

    def evaluate(positions, offsets):
        return lenslet_centroids(positions, 2.0, 0.0, offsets, (9, 9))

    for evaluate_fn in (evaluate, jax.jit(evaluate)):
        xc, yc = evaluate_fn(positions, offsets)
        assert xc.dtype == yc.dtype == dtype
        np.testing.assert_array_equal(xc, [expected])
        np.testing.assert_array_equal(yc, [[3.5]])
