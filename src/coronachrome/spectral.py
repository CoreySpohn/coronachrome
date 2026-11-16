"""Nyquist spectral channel grids and flux-conserving rebinning.

The channel rule ``n = ceil(R ln(lam2 / lam1))`` allocates one channel per
resolution element across the band. It is a correctness contract, not a
convenience: oversampling the spectrum makes neighbouring columns of the
dispersion operator H near-duplicate, so the extraction normal equations turn
near-singular and float32 solves break down. The intended pattern is to render
the scene on an oversampled grid (``spectral_grid(..., oversample=k)``) for a
smooth truth spectrum and rebin flux-conservingly onto the Nyquist channels
that drive the forward model and extraction.

All grids are log-spaced (constant edge ratio), so the resolving power per
channel is constant across the band.
"""

import math

import jax.numpy as jnp
from hwoutils.spectral import Band

# Slack subtracted before the ceil so a band spanning an exact integer number
# of resolution elements (up to float rounding) is not bumped to the next
# channel count.
_CEIL_TOL = 1e-9


def n_nyquist_channels(resolving_power, lam_min_nm, lam_max_nm):
    """Number of Nyquist spectral channels: ``ceil(R ln(lam2 / lam1))``.

    One channel per resolution element of a spectrograph with resolving power
    ``R = lambda / dlambda``; in log-wavelength each resolution element has
    constant width ``1 / R``, so the band ``ln(lam2 / lam1)`` holds
    ``R ln(lam2 / lam1)`` of them.

    Args:
        resolving_power: Resolving power ``R`` (constant across the band).
        lam_min_nm: Band minimum wavelength [nm].
        lam_max_nm: Band maximum wavelength [nm].

    Returns:
        Channel count as a Python int (>= 1).
    """
    if resolving_power <= 0.0:
        raise ValueError("resolving_power must be positive")
    if lam_max_nm <= lam_min_nm:
        raise ValueError("lam_max_nm must exceed lam_min_nm")
    n_reselts = resolving_power * math.log(lam_max_nm / lam_min_nm)
    return max(1, math.ceil(n_reselts - _CEIL_TOL))


def channel_edges(n_chan, lam_min_nm, lam_max_nm):
    """Partition the exact band into log-spaced Band intervals.

    Endpoints retain their nm meaning; the return is a tuple of Bands.
    The reference wavelength of each channel is its geometric mean, exposed
    by channel_centers, distinct from Band.center_nm (arithmetic).
    """
    return Band(lam_min_nm, lam_max_nm).channels(n_chan, spacing="log")


def channel_centers(bands):
    """Geometric-mean reference wavelengths [nm] of a sequence of Bands."""
    return jnp.asarray([math.sqrt(b.lo_nm * b.hi_nm) for b in bands])


def spectral_grid(resolving_power, lam_min_nm, lam_max_nm, oversample=1):
    """Return one Band per log-spaced spectral channel.

    oversample subdivides every Nyquist band, preserving shared edges exactly.
    Use channel_centers(bands) for geometric reference wavelengths [nm].
    """
    from operator import index

    oversample = index(oversample)
    if oversample < 1:
        raise ValueError("oversample must be positive")
    n = n_nyquist_channels(resolving_power, lam_min_nm, lam_max_nm)
    base = channel_edges(n, lam_min_nm, lam_max_nm)
    return tuple(sub for band in base for sub in band.channels(oversample, "log"))


def rebin_channels(values, src_bands, dst_bands, axis=0):
    """Rebin channel integrals using exact Band overlaps, preserving units.

    Assumes constant spectral density within each source Band. An Image
    retains its spatial grid, measure, quantity and reconstruction; axis must
    name a leading (non-spatial) dimension in that case.
    """
    from dataclasses import replace

    from hwoutils.grids import Image

    image = values if isinstance(values, Image) else None
    data = image.data if image is not None else jnp.asarray(values)
    axis = axis % data.ndim
    if image is not None and axis >= data.ndim - 2:
        raise ValueError("spectral axis must be a leading Image axis")
    if data.shape[axis] != len(src_bands):
        raise ValueError("source bands must match the spectral axis")
    src_lo = jnp.asarray([b.lo_nm for b in src_bands])
    src_hi = jnp.asarray([b.hi_nm for b in src_bands])
    dst_lo = jnp.asarray([b.lo_nm for b in dst_bands])
    dst_hi = jnp.asarray([b.hi_nm for b in dst_bands])
    lo = jnp.maximum(dst_lo[:, None], src_lo[None])
    hi = jnp.minimum(dst_hi[:, None], src_hi[None])
    frac = jnp.maximum(hi - lo, 0) / (src_hi - src_lo)[None]
    out = jnp.moveaxis(
        jnp.tensordot(frac, jnp.moveaxis(data, axis, 0), axes=1), 0, axis
    )
    return replace(image, data=out) if image is not None else out
