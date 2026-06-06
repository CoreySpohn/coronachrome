"""Compile a disperser descriptor into a SpatialChannelIR.

Runs offline (eager) but fully vectorized: footprints for all (channel,
wavelength) pairs are computed by broadcasting, not Python loops. build_ir
dispatches on the descriptor type so new IFS geometries register their own
builders without touching optixstuff.
"""

import functools
import warnings

import jax
import jax.numpy as jnp
from optixstuff.disperser import LensletDisperser

from coronachrome.dispersion import dispersion_px, lenslet_centroids
from coronachrome.grids import hex_grid, square_grid
from coronachrome.ir import SpatialChannelIR
from coronachrome.psflet import psflet_weights


@functools.singledispatch
def build_ir(disperser, wavelengths_nm, fp_shape, fp_px_per_lenslet=1.0, half=3):
    """Build a SpatialChannelIR from a disperser descriptor."""
    raise NotImplementedError(
        f"build_ir not implemented for {type(disperser).__name__}"
    )


def _bilinear_footprints(cx, cy, fp_shape):
    """Bilinear 4-pixel focal-plane footprints for every channel.

    Args:
        cx: ``(n_channels,)`` lenslet x-centers in focal-plane pixels.
        cy: ``(n_channels,)`` lenslet y-centers in focal-plane pixels.
        fp_shape: focal-plane ``(ny, nx)``.

    Returns:
        Tuple ``(idx, weights)``, each ``(n_channels, 4)``: flat focal-plane
        indices and partition-of-unity weights with out-of-bounds corners
        masked to zero.
    """
    ny, nx = fp_shape
    x0 = jnp.floor(cx)
    y0 = jnp.floor(cy)
    fx = cx - x0
    fy = cy - y0
    xs = jnp.stack([x0, x0 + 1, x0, x0 + 1], axis=1)
    ys = jnp.stack([y0, y0, y0 + 1, y0 + 1], axis=1)
    ws = jnp.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    valid = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    ws = jnp.where(valid, ws, 0.0)
    idx = jnp.clip(ys * nx + xs, 0, ny * nx - 1).astype(jnp.int32)
    return idx, ws


def _resampling_footprints(cx, cy, angle_rad, cell_px, fp_shape, supersample=4):
    """Flux-conserving, rotated focal-plane footprints for every lenslet.

    Each lenslet integrates the focal-plane flux over its square cell of side
    ``cell_px`` (focal-plane pixels), rotated by ``angle_rad`` about the lenslet
    center. The cell is supersampled on an ``S x S`` grid; each subpoint is
    rotated, binned to its nearest focal-plane pixel, and contributes area
    ``(cell_px / S) ** 2``. The footprint is a fixed window of ``K = (2R + 1) ** 2``
    source pixels per lenslet, with zero weight where the cell does not reach or
    the pixel is off the focal plane. Weights sum to ``cell_px ** 2`` for a fully
    interior lenslet, so a uniform input integrates to the cell area (flux
    conservation), unlike the partition-of-unity bilinear footprint.

    Args:
        cx: ``(n_channels,)`` lenslet x-centers in focal-plane pixels.
        cy: ``(n_channels,)`` lenslet y-centers in focal-plane pixels.
        angle_rad: lenslet-grid rotation angle (radians).
        cell_px: lenslet cell side in focal-plane pixels (the lenslet pitch).
        fp_shape: focal-plane ``(ny, nx)``.
        supersample: subpoints per cell axis (``S``); accuracy grows with ``S``.

    Returns:
        Tuple ``(idx, weights)``, each ``(n_channels, K)``: flat focal-plane
        indices and flux-conserving weights, out-of-bounds masked to zero weight.
    """
    ny, nx = fp_shape
    s = int(supersample)
    r = int(jnp.ceil(cell_px * (2.0**0.5 / 2.0))) + 1
    span = 2 * r + 1

    wo = jnp.arange(-r, r + 1)
    wdy, wdx = jnp.meshgrid(wo, wo, indexing="ij")
    wdx = wdx.reshape(-1)
    wdy = wdy.reshape(-1)

    u = (jnp.arange(s) + 0.5) / s * cell_px - cell_px / 2.0
    lx, ly = jnp.meshgrid(u, u, indexing="ij")
    lx = lx.reshape(-1)
    ly = ly.reshape(-1)
    ca, sa = jnp.cos(angle_rad), jnp.sin(angle_rad)
    rx = ca * lx - sa * ly
    ry = sa * lx + ca * ly
    sub_w = (cell_px / s) ** 2

    def one_lenslet(cxi, cyi):
        bx, by = jnp.round(cxi), jnp.round(cyi)
        sdx = (jnp.round(cxi + rx) - bx).astype(jnp.int32)
        sdy = (jnp.round(cyi + ry) - by).astype(jnp.int32)
        in_win = (jnp.abs(sdx) <= r) & (jnp.abs(sdy) <= r)
        widx = jnp.where(in_win, (sdy + r) * span + (sdx + r), 0)
        win = jnp.zeros(span * span).at[widx].add(jnp.where(in_win, sub_w, 0.0))
        sx = (bx + wdx).astype(jnp.int32)
        sy = (by + wdy).astype(jnp.int32)
        valid = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
        win = jnp.where(valid, win, 0.0)
        src = jnp.clip(sy * nx + sx, 0, ny * nx - 1).astype(jnp.int32)
        return src, win

    return jax.vmap(one_lenslet)(cx, cy)


@build_ir.register
def _(
    disperser: LensletDisperser,
    wavelengths_nm,
    fp_shape,
    fp_px_per_lenslet=1.0,
    half=3,
    supersample=4,
):
    """Build a SpatialChannelIR from a LensletDisperser.

    The PSFlet core width (``disperser.psflet_params[0]``, in detector pixels)
    scales linearly with wavelength about ``disperser.psflet_ref_nm`` -- a
    diffraction-limited spot grows as ``lambda f / D``, so at fixed pixel scale
    its pixel width is proportional to wavelength. The fixed ``half`` footprint
    (``(2 * half + 1)`` per side) must therefore be wide enough for the widest
    PSFlet in the band; otherwise the long-wavelength spot is truncated (its
    footprint still renormalizes to unit flux, but the wings are clipped).
    """
    lam = jnp.asarray(wavelengths_nm, dtype=float)
    n_wav = int(lam.shape[0])
    positions = (
        square_grid(disperser.n_lenslets)
        if disperser.grid_kind == "square"
        else hex_grid(disperser.n_lenslets)
    )
    n_channels = int(positions.shape[0])
    scale = disperser.pitch_m / disperser.pixsize_m
    ny, nx = disperser.detector_shape

    # Spatial sampling: flux-conserving footprints over each lenslet's cell. The
    # lenslet grid is rotated by the lenslet angle in the focal plane, matching
    # the rotation applied to the detector centroids below.
    fx0, fy0 = fp_shape[1] / 2.0, fp_shape[0] / 2.0
    ca, sa = jnp.cos(disperser.angle_rad), jnp.sin(disperser.angle_rad)
    px, py = positions[:, 0], positions[:, 1]
    cx = fx0 + fp_px_per_lenslet * (ca * px - sa * py)
    cy = fy0 + fp_px_per_lenslet * (sa * px + ca * py)
    spatial_src, spatial_w = _resampling_footprints(
        cx, cy, disperser.angle_rad, fp_px_per_lenslet, fp_shape, supersample
    )

    # Detector centroids (n_channels, n_wav) and PSFlet footprint offsets (n_psf,).
    coeffs, lam_ref = disperser.dispersion_coeffs, disperser.lam_ref_nm
    disp = dispersion_px(coeffs, lam_ref, lam)
    xc, yc = lenslet_centroids(
        positions, scale, disperser.angle_rad, disp, disperser.detector_shape
    )
    off = jnp.arange(-half, half + 1)
    ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddy = ddy.reshape(-1).astype(float)
    ddx = ddx.reshape(-1).astype(float)

    # Broadcast footprints to (n_channels, n_wav, n_psf).
    px = jnp.round(xc)[..., None] + ddx
    py = jnp.round(yc)[..., None] + ddy
    dx = px - xc[..., None]
    dy = py - yc[..., None]

    # Per-wavelength LSF smear width [px].
    dlam = jnp.gradient(lam) if n_wav > 1 else jnp.array([1.0])
    smear = jnp.abs(
        dispersion_px(coeffs, lam_ref, lam + 0.5 * dlam)
        - dispersion_px(coeffs, lam_ref, lam - 0.5 * dlam)
    )

    # Diffraction scaling: the PSFlet core width (params[0], px) scales linearly
    # with wavelength at a fixed detector pixel scale (spot size ~ lambda f / D),
    # referenced to psflet_ref_nm. Trailing shape params (e.g. Moffat beta) are
    # dimensionless and do not scale.
    width_scale = lam / disperser.psflet_ref_nm
    n_params = disperser.psflet_params.shape[0]
    params_w = jnp.broadcast_to(disperser.psflet_params, (n_wav, n_params))
    params_w = params_w.at[:, 0].multiply(width_scale)

    # PSFlet weights for every (channel, wavelength, footprint pixel). vmap over
    # the wavelength axis so each wavelength's scaled width and smear apply.
    def psf_one_wav(dx_w, dy_w, smear_w, params_one):
        return psflet_weights(dx_w, dy_w, disperser.psflet_kind, params_one, smear_w)

    g = jax.vmap(psf_one_wav, in_axes=(1, 1, 0, 0), out_axes=1)(dx, dy, smear, params_w)

    valid = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    g = jnp.where(valid, g, 0.0)
    total = g.sum(axis=2, keepdims=True)
    det_vals = g / jnp.clip(total, 1e-12, None)
    det_rows = jnp.clip((py * nx + px).astype(jnp.int32), 0, ny * nx - 1)

    n_off = int((total[..., 0] <= 1e-12).sum())
    if n_off:
        warnings.warn(
            f"{n_off} (channel, wavelength) footprints fell off the detector",
            stacklevel=2,
        )

    return SpatialChannelIR(
        spatial_src=spatial_src,
        spatial_w=spatial_w,
        det_rows=det_rows,
        det_vals=det_vals,
        n_channels=n_channels,
        n_wav=n_wav,
        fp_shape=tuple(fp_shape),
        det_shape=disperser.detector_shape,
    )
