"""Cross-check lenslet centroids against the crispy rotation+scale+distort form."""

import jax.numpy as jnp

from coronachrome.dispersion import lenslet_centroids
from coronachrome.grids import square_grid


def test_centroids_equal_crispy_formula():
    """Floating-point oracle: rotate unit-pitch centres about (n-1)/2.

    For n=6, removing n//2 shifts every centroid by
    scale/2 * (cos(angle)-sin(angle), sin(angle)+cos(angle)).
    The default allclose tolerance covers arithmetic roundoff.
    """
    n, scale, angle = 6, 13.3846, float(jnp.arcsin(1.0 / jnp.sqrt(5.0)))
    coeffs, lam_ref = jnp.array([100.0, 0.0]), 660.0
    lam = jnp.array([650.0, 660.0, 670.0])
    det_shape = (256, 256)
    pos = square_grid(n)
    disp = coeffs[0] * jnp.log(lam / lam_ref) + coeffs[1]
    xc, yc = lenslet_centroids(pos, scale, angle, disp, det_shape)

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    ii = jnp.repeat(jnp.arange(n) - (n - 1) / 2, n)
    jj = jnp.tile(jnp.arange(n) - (n - 1) / 2, n)
    ex = 127.5 + disp[None, :] + (scale * (cos_a * ii - sin_a * jj))[:, None]
    ey_col = (127.5 + scale * (sin_a * ii + cos_a * jj))[:, None]
    ey = ey_col * jnp.ones_like(disp)[None, :]
    assert jnp.allclose(xc, ex)
    assert jnp.allclose(yc, ey)
