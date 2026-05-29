"""Cross-check lenslet centroids against the crispy rotation+scale+distort form."""

import jax.numpy as jnp

from coronachrome.dispersion import dispersion_px, lenslet_centroids
from coronachrome.grids import square_grid


def test_centroids_equal_crispy_formula():
    """Our centroids equal crispy's rotation + scale + dispersion form."""
    n, scale, angle = 6, 13.3846, float(jnp.arcsin(1.0 / jnp.sqrt(5.0)))
    coeffs, lam_ref = jnp.array([100.0, 0.0]), 660.0
    lam = jnp.array([650.0, 660.0, 670.0])
    det_shape = (256, 256)
    pos = square_grid(n)
    disp = dispersion_px(coeffs, lam_ref, lam)
    xc, yc = lenslet_centroids(pos, scale, angle, disp, det_shape)

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    ii, jj = pos[:, 0], pos[:, 1]
    ex = 128.0 + disp[None, :] + (scale * (cos_a * ii - sin_a * jj))[:, None]
    ey_col = (128.0 + scale * (sin_a * ii + cos_a * jj))[:, None]
    ey = ey_col * jnp.ones_like(disp)[None, :]
    assert jnp.allclose(xc, ex)
    assert jnp.allclose(yc, ey)
