"""Lenslet position grids in lenslet-index coordinates, centered on zero.

These run offline (eager); ``n_lenslets`` is a small static count, so plain
Python loops are fine here.
"""

import jax.numpy as jnp


def square_grid(n_lenslets):
    """Return ``(n_lenslets**2, 2)`` (i, j) lenslet coordinates centered on zero."""
    half = n_lenslets // 2
    idx = jnp.arange(n_lenslets) - half
    ii, jj = jnp.meshgrid(idx, idx, indexing="ij")
    return jnp.stack([ii.reshape(-1), jj.reshape(-1)], axis=1).astype(float)


def hex_grid(n_lenslets):
    """Return hexagonal lenslet centers (axial offset rows), centered on zero.

    Odd rows are shifted by half a step in x; rows are spaced by sqrt(3)/2.
    """
    half = n_lenslets // 2
    rows = []
    for r in range(n_lenslets):
        x = (jnp.arange(n_lenslets) - half) + 0.5 * (r % 2)
        y = jnp.full((n_lenslets,), (r - half) * (jnp.sqrt(3.0) / 2.0))
        rows.append(jnp.stack([x, y], axis=1))
    return jnp.concatenate(rows, axis=0)
