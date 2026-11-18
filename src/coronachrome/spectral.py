"""Nyquist spectral channel grids.

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

from optixstuff.spectral import Band

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


def spectral_grid(resolving_power, lam_min_nm, lam_max_nm, oversample=1):
    """Return one Band per log-spaced spectral channel.

    oversample subdivides every Nyquist band, preserving shared edges exactly.
    Use each Band.geometric_center_nm for reference wavelengths [nm].
    """
    from operator import index

    oversample = index(oversample)
    if oversample < 1:
        raise ValueError("oversample must be positive")
    n = n_nyquist_channels(resolving_power, lam_min_nm, lam_max_nm)
    base = Band(lam_min_nm, lam_max_nm).channels(n, "log")
    return tuple(sub for band in base for sub in band.channels(oversample, "log"))
