"""Tests for lenslet position grids."""

import jax.numpy as jnp

from coronachrome.grids import hex_grid, square_grid


def test_square_grid_count_and_centering():
    """square_grid(8) returns 64 points centered near zero."""
    pos = square_grid(8)
    assert pos.shape == (64, 2)
    assert jnp.allclose(pos.mean(axis=0), 0.0, atol=1.0)


def test_square_grid_integer_steps():
    """x-coordinates of square_grid(4) are unit-spaced."""
    pos = square_grid(4)
    xs = jnp.unique(pos[:, 0])
    assert jnp.allclose(jnp.diff(xs), 1.0)


def test_hex_grid_count():
    """hex_grid(8) returns 64 points."""
    pos = hex_grid(8)
    assert pos.shape == (64, 2)


def test_hex_rows_offset_by_half():
    """Row spacing in hex_grid(4) equals sqrt(3)/2."""
    pos = hex_grid(4)
    ys = jnp.unique(jnp.round(pos[:, 1], 5))
    assert jnp.allclose(jnp.diff(ys), jnp.sqrt(3.0) / 2.0)
