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
    """Log-spaced channel edges: ``(n_chan + 1,)`` with a constant ratio."""
    return jnp.geomspace(lam_min_nm, lam_max_nm, int(n_chan) + 1)


def channel_centers(edges):
    """Geometric-mean channel centers of log-spaced edges: ``(n_chan,)``."""
    e = jnp.asarray(edges)
    return jnp.sqrt(e[:-1] * e[1:])


def spectral_grid(resolving_power, lam_min_nm, lam_max_nm, oversample=1):
    """Nyquist channel grid for a band: ``(centers, edges)``.

    ``oversample > 1`` returns the hi-res render grid with that many
    sub-channels per Nyquist channel; its bins nest exactly in the base grid,
    so ``rebin_channels`` between the two reduces to a plain sum.

    Args:
        resolving_power: Resolving power ``R`` (e.g. from
            ``disperser.spectral_resolution`` at band center).
        lam_min_nm: Band minimum wavelength [nm].
        lam_max_nm: Band maximum wavelength [nm].
        oversample: Sub-channels per Nyquist channel.

    Returns:
        Tuple ``(centers, edges)`` of shapes ``(n,)`` and ``(n + 1,)`` where
        ``n = n_nyquist_channels(...) * oversample``.
    """
    n_chan = n_nyquist_channels(resolving_power, lam_min_nm, lam_max_nm)
    edges = channel_edges(n_chan * int(oversample), lam_min_nm, lam_max_nm)
    return channel_centers(edges), edges


def rebin_channels(values, src_edges, dst_edges, axis=0):
    """Flux-conserving rebin of bin-integrated values onto new channel edges.

    ``values`` are integrals over the source bins (e.g. photon rates per
    channel), not spectral densities: each destination bin receives the
    fraction of every source bin's integral proportional to their overlap, so
    total flux is conserved wherever the destination grid covers the source
    grid. Both edge arrays must be ascending.

    Args:
        values: Bin-integrated values; the spectral axis is ``axis``.
        src_edges: ``(n_src + 1,)`` source bin edges.
        dst_edges: ``(n_dst + 1,)`` destination bin edges.
        axis: The spectral axis of ``values``.

    Returns:
        Values on the destination bins, with ``axis`` resized to ``n_dst``.
    """
    src = jnp.asarray(src_edges, dtype=float)
    dst = jnp.asarray(dst_edges, dtype=float)
    lo = jnp.maximum(dst[:-1, None], src[None, :-1])
    hi = jnp.minimum(dst[1:, None], src[None, 1:])
    frac = jnp.clip(hi - lo, 0.0, None) / (src[None, 1:] - src[None, :-1])
    vals = jnp.moveaxis(jnp.asarray(values), axis, 0)
    out = jnp.tensordot(frac, vals, axes=1)
    return jnp.moveaxis(out, 0, axis)
