"""Dispersion model: lenslet index -> detector centroid versus wavelength.

Generalizes crispy's hardcoded WFIRST distort(): a polynomial in
u = log(lambda / lambda_ref) (descending-coefficient convention). The crispy
default is linear, coeffs = [npixperdlam * R, 0].
"""

import jax.numpy as jnp


def dispersion_px(dispersion_coeffs, lam_ref_nm, wavelength_nm):
    """Spectral-axis detector offset [px] for the wavelength(s)."""
    u = jnp.log(jnp.asarray(wavelength_nm, dtype=float) / lam_ref_nm)
    return jnp.polyval(dispersion_coeffs, u)


def lenslet_centroids(positions, scale, angle_rad, dispersion_offsets, det_shape):
    """Detector centroids (xc, yc) for each (lenslet, wavelength).

    Args:
        positions: ``(n_lenslets, 2)`` lenslet (i, j) coordinates.
        scale: detector pixels per lenslet step (pitch / pixsize).
        angle_rad: interlace/rotation angle of the lenslet grid.
        dispersion_offsets: ``(n_wav,)`` spectral-axis shifts [px].
        det_shape: detector ``(ny, nx)``.

    Returns:
        Tuple ``(xc, yc)``, each ``(n_lenslets, n_wav)``.
    """
    cos_a, sin_a = jnp.cos(angle_rad), jnp.sin(angle_rad)
    ii, jj = positions[:, 0], positions[:, 1]
    gx = scale * (cos_a * ii - sin_a * jj)
    gy = scale * (sin_a * ii + cos_a * jj)
    ny, nx = det_shape
    disp = jnp.asarray(dispersion_offsets, dtype=float)
    xc = (nx / 2.0 + disp)[None, :] + gx[:, None]
    yc = (ny / 2.0 + gy)[:, None] * jnp.ones_like(disp)[None, :]
    return xc, yc
