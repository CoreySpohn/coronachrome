"""Compile a disperser descriptor into a SpatialChannelIR.

Runs offline (eager). build_ir dispatches on the descriptor type so new IFS
geometries register their own builders without touching optixstuff.
"""

import functools
import warnings

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


def _bilinear_footprint(cx, cy, fp_shape):
    """Return (4,) flat focal-plane indices and partition-of-unity weights."""
    ny, nx = fp_shape
    x0 = jnp.floor(cx)
    y0 = jnp.floor(cy)
    fx = cx - x0
    fy = cy - y0
    xs = jnp.array([x0, x0 + 1, x0, x0 + 1])
    ys = jnp.array([y0, y0, y0 + 1, y0 + 1])
    ws = jnp.array([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
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
    lam = jnp.asarray(wavelengths_nm, dtype=float)
    n_wav = int(lam.shape[0])
    positions = (
        square_grid(disperser.n_lenslets)
        if disperser.grid_kind == "square"
        else hex_grid(disperser.n_lenslets)
    )
    n_channels = int(positions.shape[0])
    scale = disperser.pitch_m / disperser.pixsize_m
    det_shape = disperser.detector_shape
    ny, nx = det_shape

    fy0, fx0 = fp_shape[0] / 2.0, fp_shape[1] / 2.0
    cxs = fx0 + positions[:, 0] * fp_px_per_lenslet
    cys = fy0 + positions[:, 1] * fp_px_per_lenslet
    spatial_src = []
    spatial_w = []
    for c in range(n_channels):
        idx, ws = _bilinear_footprint(cxs[c], cys[c], fp_shape)
        spatial_src.append(idx)
        spatial_w.append(ws)
    spatial_src = jnp.stack(spatial_src)
    spatial_w = jnp.stack(spatial_w)

    disp = dispersion_px(disperser.dispersion_coeffs, disperser.lam_ref_nm, lam)
    xc, yc = lenslet_centroids(positions, scale, disperser.angle_rad, disp, det_shape)

    off = jnp.arange(-half, half + 1)
    ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddy = ddy.reshape(-1).astype(float)
    ddx = ddx.reshape(-1).astype(float)

    dlam = jnp.gradient(lam) if n_wav > 1 else jnp.array([1.0])
    coeffs = disperser.dispersion_coeffs
    lam_ref = disperser.lam_ref_nm
    smear = jnp.abs(
        dispersion_px(coeffs, lam_ref, lam + 0.5 * dlam)
        - dispersion_px(coeffs, lam_ref, lam - 0.5 * dlam)
    )

    det_rows = []
    det_vals = []
    n_off = 0
    for c in range(n_channels):
        rows_w = []
        vals_w = []
        for w in range(n_wav):
            px = jnp.round(xc[c, w]) + ddx
            py = jnp.round(yc[c, w]) + ddy
            dx = px - xc[c, w]
            dy = py - yc[c, w]
            g = psflet_weights(
                dx, dy, disperser.psflet_kind, disperser.psflet_params, float(smear[w])
            )
            valid = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
            g = jnp.where(valid, g, 0.0)
            total = g.sum()
            if total <= 1e-12:
                n_off += 1
            g = g / jnp.clip(total, 1e-12, None)
            rows = jnp.clip((py * nx + px).astype(jnp.int32), 0, ny * nx - 1)
            rows_w.append(rows)
            vals_w.append(g)
        det_rows.append(jnp.stack(rows_w))
        det_vals.append(jnp.stack(vals_w))
    det_rows = jnp.stack(det_rows)
    det_vals = jnp.stack(det_vals)

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
        det_shape=det_shape,
    )
