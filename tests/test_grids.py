"""Independent origin anchors for even and odd lenslet arrays."""

import ineedvalidation as vv
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import sum_roundoff
from hwoutils.testing import ulps

from coronachrome.grids import LensletLayout, hex_grid, square_grid


@pytest.mark.parametrize("n", [2, 3])
@vv.case("ifs-geometry", "code-verification")
def test_square_grid_count_and_centering(n):
    """Exact algebra: opposite endpoints average to zero with unit pitch."""
    axis = [i - (n - 1) / 2 for i in range(n)]
    expected = jnp.array([(x, y) for x in axis for y in axis])
    assert jnp.array_equal(square_grid(n), expected)
    assert jnp.array_equal(LensletLayout(n, "square").positions(), expected)


def test_square_grid_integer_steps():
    """Fails if the origin changes the pitch; exact algebra."""
    pos = square_grid(4)
    assert jnp.array_equal(jnp.unique(pos[:, 0]), jnp.array([-1.5, -0.5, 0.5, 1.5]))


@pytest.mark.parametrize("n", [2, 3])
@vv.case("ifs-geometry", "code-verification")
def test_hex_grid_count(n):
    """Five operations at the uncentered lattice extent bound each coordinate.

    Centering, staggering, the hex row height, and axial packing each use
    at most five scalar operations along a coordinate's computation path.
    """
    raw = [
        (col + (row % 2) / 2, row * 3**0.5 / 2) for row in range(n) for col in range(n)
    ]
    mean_x = sum(x for x, _ in raw) / n**2
    mean_y = sum(y for _, y in raw) / n**2
    expected = np.array([(x - mean_x, y - mean_y) for x, y in raw])
    layout = hex_grid(n)
    tol = float(ulps(5, layout.dtype.type(n - 1)))
    np.testing.assert_allclose(layout, expected, rtol=0, atol=tol)
    grid, indices = LensletLayout(n, "hex").image_geometry(1.0, 0.0, None)
    coords = grid.coordinates()
    packed = jnp.stack([coords.x.ravel()[indices], coords.y.ravel()[indices]], axis=1)
    np.testing.assert_allclose(packed, expected, rtol=0, atol=tol)


def test_hex_rows_offset_by_half():
    """Zero centroid allows four layout ops, a mean divide and pairwise sum."""
    layout = hex_grid(4)
    scale = jnp.max(jnp.abs(layout))
    tol = ulps(5, scale) + sum_roundoff(scale, layout.shape[0])
    np.testing.assert_allclose(layout.mean(axis=0), 0, atol=float(tol), rtol=0)
