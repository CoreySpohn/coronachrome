"""Analytic PSFlet profiles with line-spread-function smearing.

The PSFlet is evaluated on a detector-pixel footprint as offsets (dx, dy) from
the spectrum centroid. The LSF is approximated by averaging the profile over
``n_sub`` centroid positions spanning the bin's spectral-pixel extent along x.
"""

import math

import jax
import jax.numpy as jnp
from hwoutils.precision import like
from jax.scipy.special import erf

_SQRT2 = math.sqrt(2.0)


def _gauss_axis_integral(t, sigma_px):
    """Integral of the unit-area 1D Gaussian over the pixel [t - 0.5, t + 0.5]."""
    upper = (t + 0.5) / (_SQRT2 * sigma_px)
    lower = (t - 0.5) / (_SQRT2 * sigma_px)
    return 0.5 * (erf(upper) - erf(lower))


def gaussian_psflet(dx, dy, sigma_px):
    """2D Gaussian integrated over the unit pixel centered at (dx, dy) [px].

    Pixel-integrated (erf) form, matching CRISPY ``imgtools.gausspsf``. An overall
    normalization constant is dropped because ``build_ir`` normalizes the
    full footprint before detector clipping.
    """
    return _gauss_axis_integral(dx, sigma_px) * _gauss_axis_integral(dy, sigma_px)


def moffat_psflet(dx, dy, alpha_px, beta, n_quad=5):
    """2D Moffat integrated over the unit pixel centered at (dx, dy) [px].

    No closed form, so the pixel is integrated by a midpoint-rule average over an
    ``n_quad x n_quad`` sub-grid. An overall normalization constant is irrelevant
    because ``build_ir`` normalizes the full footprint before detector clipping.

    Args:
        dx: Spectral-axis pixel offsets from the centroid [px].
        dy: Spatial-axis pixel offsets from the centroid [px].
        alpha_px: Moffat core width [px].
        beta: Moffat power-law index.
        n_quad: Number of midpoint samples along each pixel axis.

    Returns:
        Integrated profile weights at the supplied offsets.
    """
    dtype = jnp.result_type(dx, dy, alpha_px, beta, 1.0)
    dx, dy, alpha_px, beta = (
        jnp.asarray(x, dtype=dtype) for x in (dx, dy, alpha_px, beta)
    )
    edge = like(0.5 - 0.5 / n_quad, dx)
    sub = jnp.linspace(-edge, edge, n_quad)
    uu, vv = jnp.meshgrid(sub, sub, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    r2 = (dx[..., None] + uu) ** 2 + (dy[..., None] + vv) ** 2
    prof = (1.0 + r2 / alpha_px**2) ** (-beta)
    return prof.mean(axis=-1)


def _smear_shifts(smear_px, n_sub, ref):
    half = 0.5 * like(smear_px, ref)
    return jnp.linspace(-half, half, n_sub)


def psflet_weights(dx, dy, kind, params, smear_px, n_sub=5):
    """PSFlet footprint weights with LSF smear along the spectral (x) axis.

    Args:
        dx: footprint pixel offsets along the spectral axis from the centroid [px].
        dy: footprint pixel offsets along the spatial axis from the centroid [px].
        kind: "gaussian" or "moffat" (static string).
        params: profile parameters ([sigma] or [alpha, beta]).
        smear_px: spectral extent of the wavelength bin [px].
        n_sub: number of sub-samples used to integrate across the bin.

    Returns:
        Weights with the same shape as ``dx``.
    """
    if kind == "gaussian":
        profile = gaussian_psflet
        params = (params[0],)
    else:
        profile = moffat_psflet
        params = (params[0], params[1])
    dtype = jnp.result_type(dx, dy, *params, smear_px, 1.0)
    dx, dy = (jnp.asarray(x, dtype=dtype) for x in (dx, dy))
    params = tuple(jnp.asarray(value, dtype=dtype) for value in params)
    shifts = _smear_shifts(smear_px, n_sub, dx)

    def one(shift):
        return profile(dx - shift, dy, *params)

    return jnp.mean(jax.vmap(one)(shifts), axis=0)
