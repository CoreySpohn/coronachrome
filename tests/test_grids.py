"""Independent origin anchors for even and odd lenslet arrays."""

import ineedvalidation as vv
import jax.numpy as jnp
import pytest

from coronachrome.grids import LensletLayout, hex_grid, square_grid


@pytest.mark.parametrize(
    ("n", "origin", "axis"),
    [
        (2, "geometric", [-0.5, 0.5]),
        (3, "geometric", [-1.0, 0.0, 1.0]),
        (2, "central_index", [-1.0, 0.0]),
        (3, "central_index", [-1.0, 0.0, 1.0]),
    ],
)
@vv.case("ifs-geometry", "code-verification")
def test_square_grid_count_and_centering(n, origin, axis):
    """Fails if even arrays shift by half a lenslet; typed centres, exact algebra."""
    expected = jnp.array([(x, y) for x in axis for y in axis])
    assert jnp.array_equal(square_grid(n, lenslet_origin=origin), expected)
    assert jnp.array_equal(LensletLayout(n, "square", origin).positions(), expected)


def test_square_grid_integer_steps():
    """Fails if the origin changes the pitch; exact algebra."""
    pos = square_grid(4)
    assert jnp.array_equal(jnp.unique(pos[:, 0]), jnp.array([-1.5, -0.5, 0.5, 1.5]))


@pytest.mark.parametrize(
    ("n", "origin", "expected"),
    [
        (
            2,
            "geometric",
            [
                [-0.75, -0.4330127018922193],
                [0.25, -0.4330127018922193],
                [-0.25, 0.4330127018922193],
                [0.75, 0.4330127018922193],
            ],
        ),
        (
            2,
            "central_index",
            [
                [-1.0, -0.8660254037844386],
                [0.0, -0.8660254037844386],
                [-0.5, 0.0],
                [0.5, 0.0],
            ],
        ),
        (
            3,
            "geometric",
            [
                [-1.1666666666666667, -0.8660254037844386],
                [-0.16666666666666666, -0.8660254037844386],
                [0.8333333333333334, -0.8660254037844386],
                [-0.6666666666666666, 0.0],
                [0.3333333333333333, 0.0],
                [1.3333333333333333, 0.0],
                [-1.1666666666666667, 0.8660254037844386],
                [-0.16666666666666666, 0.8660254037844386],
                [0.8333333333333334, 0.8660254037844386],
            ],
        ),
        (
            3,
            "central_index",
            [
                [-1.0, -0.8660254037844386],
                [0.0, -0.8660254037844386],
                [1.0, -0.8660254037844386],
                [-0.5, 0.0],
                [0.5, 0.0],
                [1.5, 0.0],
                [-1.0, 0.8660254037844386],
                [0.0, 0.8660254037844386],
                [1.0, 0.8660254037844386],
            ],
        ),
    ],
)
@vv.case("ifs-geometry", "code-verification")
def test_hex_grid_count(n, origin, expected):
    """Fails if stagger or origin shifts; typed centres, 4 ulp floating-point floor."""
    pos = hex_grid(n, lenslet_origin=origin)
    assert jnp.allclose(pos, jnp.array(expected), rtol=0, atol=5e-16)
    layout = LensletLayout(n, "hex", origin)
    grid, indices = layout.image_geometry(1.0, 0.0, None)
    coords = grid.coordinates()
    packed = jnp.stack([coords.x.ravel()[indices], coords.y.ravel()[indices]], axis=1)
    assert jnp.allclose(packed, jnp.array(expected), rtol=0, atol=5e-16)


def test_hex_rows_offset_by_half():
    """Fails if geometric centring ignores row staggering; FP floor, zero centroid."""
    assert jnp.allclose(hex_grid(4).mean(axis=0), 0, atol=2e-16, rtol=0)
