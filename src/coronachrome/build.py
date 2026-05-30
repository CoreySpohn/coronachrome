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


@build_ir.register
def _(
    disperser: LensletDisperser,
    wavelengths_nm,
    fp_shape,
    fp_px_per_lenslet=1.0,
    half=3,
):
    """Build a SpatialChannelIR from a LensletDisperser."""
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

    # Spatial sampling: bilinear footprints for all channels at once.
    fx0, fy0 = fp_shape[1] / 2.0, fp_shape[0] / 2.0
    cx = fx0 + positions[:, 0] * fp_px_per_lenslet
    cy = fy0 + positions[:, 1] * fp_px_per_lenslet
    spatial_src, spatial_w = _bilinear_footprints(cx, cy, fp_shape)

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

    # PSFlet weights for every (channel, wavelength, footprint pixel). vmap over
    # the wavelength axis so each wavelength's smear width applies.
    def psf_one_wav(dx_w, dy_w, smear_w):
        return psflet_weights(
            dx_w, dy_w, disperser.psflet_kind, disperser.psflet_params, smear_w
        )

    g = jax.vmap(psf_one_wav, in_axes=(1, 1, 0), out_axes=1)(dx, dy, smear)

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
