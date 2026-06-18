"""Analytic PSFlet profiles with line-spread-function smearing.

The PSFlet is evaluated on a detector-pixel footprint as offsets (dx, dy) from
the spectrum centroid. The LSF is approximated by averaging the profile over
``n_sub`` centroid positions spanning the bin's spectral-pixel extent along x.
"""

import math

import jax
import jax.numpy as jnp
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
    normalization constant is dropped because ``build_ir`` renormalizes each
    footprint to unit flux.
    """
    return _gauss_axis_integral(dx, sigma_px) * _gauss_axis_integral(dy, sigma_px)


def moffat_psflet(dx, dy, alpha_px, beta, n_quad=5):
    """2D Moffat integrated over the unit pixel centered at (dx, dy) [px].

    No closed form, so the pixel is integrated by a midpoint-rule average over an
    ``n_quad x n_quad`` sub-grid. An overall normalization constant is irrelevant
    because ``build_ir`` renormalizes each footprint to unit flux.
    """
    sub = jnp.linspace(-0.5 + 0.5 / n_quad, 0.5 - 0.5 / n_quad, n_quad)
    uu, vv = jnp.meshgrid(sub, sub, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    r2 = (dx[..., None] + uu) ** 2 + (dy[..., None] + vv) ** 2
    prof = (1.0 + r2 / alpha_px**2) ** (-beta)
    return prof.mean(axis=-1)


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
    shifts = jnp.linspace(-0.5 * smear_px, 0.5 * smear_px, n_sub)

    def one(shift):
        if kind == "gaussian":
            return gaussian_psflet(dx - shift, dy, params[0])
        return moffat_psflet(dx - shift, dy, params[0], params[1])

    return jnp.mean(jax.vmap(one)(shifts), axis=0)
