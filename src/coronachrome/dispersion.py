"""Dispersion model: lenslet index -> detector centroid versus wavelength.

Generalizes crispy's hardcoded WFIRST distort(): a polynomial in
u = log(lambda / lambda_ref) (descending-coefficient convention). The crispy
default is linear, coeffs = [npixperdlam * R, 0].
"""

import jax
import jax.numpy as jnp
from hwoutils.radial import get_center


def lenslet_centroids(positions, scale, angle_rad, dispersion_offsets, det_shape):
    """Detector centroids (xc, yc) for each (lenslet, wavelength).

    Args:
        positions: ``(n_lenslets, 2)`` lenslet (i, j) coordinates.
        scale: detector pixels per lenslet step (pitch / pixsize).
        angle_rad: interlace/rotation angle of the lenslet grid.
        dispersion_offsets: ``(n_samples,)`` spectral-axis shifts [px].
        det_shape: detector ``(ny, nx)``.

    Returns:
        Tuple ``(xc, yc)``, each ``(n_lenslets, n_samples)``.
    """
    dtype = jnp.result_type(
        positions, scale, angle_rad, *jax.tree.leaves(dispersion_offsets), 1.0
    )
    positions = jnp.asarray(positions, dtype=dtype)
    angle = jnp.asarray(angle_rad, dtype=dtype)
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    scale = jnp.asarray(scale, dtype=dtype)
    ii, jj = positions[:, 0], positions[:, 1]
    gx = scale * (cos_a * ii - sin_a * jj)
    gy = scale * (sin_a * ii + cos_a * jj)
    cy, cx = get_center(det_shape)
    disp = jnp.asarray(dispersion_offsets, dtype=dtype)
    xc = (cx + disp)[None, :] + gx[:, None]
    yc = (cy + gy)[:, None] * jnp.ones_like(disp)[None, :]
    return xc, yc
