"""Analytic PSFlet profiles with line-spread-function smearing.

The PSFlet is evaluated on a detector-pixel footprint as offsets (dx, dy) from
the spectrum centroid. The LSF is approximated by averaging the profile over
``n_sub`` centroid positions spanning the bin's spectral-pixel extent along x.
"""

import jax
import jax.numpy as jnp


def gaussian_psflet(dx, dy, sigma_px):
    """Unnormalized 2D Gaussian at offsets (dx, dy) [px]."""
    return jnp.exp(-(dx**2 + dy**2) / (2.0 * sigma_px**2))


def moffat_psflet(dx, dy, alpha_px, beta):
    """Unnormalized 2D Moffat at offsets (dx, dy) [px]."""
    return (1.0 + (dx**2 + dy**2) / alpha_px**2) ** (-beta)


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
