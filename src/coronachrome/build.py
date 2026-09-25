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
from coronachrome.templates import (
    load_psflet_pack,
    nearest_field_idx,
    template_weights,
)


@functools.singledispatch
def build_ir(disperser, wavelengths_nm, fp_shape, **kwargs):
    """Build a SpatialChannelIR from a disperser descriptor."""
    raise NotImplementedError(
        f"build_ir not implemented for {type(disperser).__name__}"
    )


def _resolve_fp_px_per_lenslet(disperser, fp_px_per_lenslet, fp_pixel_scale_arcsec):
    """Resolve the spatial sampling (focal-plane pixels per lenslet cell).

    The physical route derives it from the descriptor: ``sky_pitch_arcsec``
    (the lenslet pitch on sky) over ``fp_pixel_scale_arcsec`` (the cube's
    angular plate scale). Both are plain angles in arcseconds -- the cube
    lives on one fixed angular grid for every wavelength plane, so no
    reference wavelength enters. ``fp_px_per_lenslet`` remains as an explicit
    override for reference-implementation parity work; passing both is an
    error, and so is passing neither -- the sampling is a derived quantity,
    not a default.
    """
    if fp_pixel_scale_arcsec is not None:
        if fp_px_per_lenslet is not None:
            raise ValueError(
                "pass fp_pixel_scale_arcsec (derived sampling) or fp_px_per_lenslet "
                "(explicit override), not both"
            )
        if disperser.sky_pitch_arcsec is None:
            raise ValueError(
                "deriving the sampling from fp_pixel_scale_arcsec needs "
                "sky_pitch_arcsec on the disperser descriptor"
            )
        fp_px_per_lenslet = float(disperser.sky_pitch_arcsec) / float(
            fp_pixel_scale_arcsec
        )
    elif fp_px_per_lenslet is None:
        raise ValueError(
            "spatial sampling is underdetermined: pass fp_pixel_scale_arcsec (the "
            "cube plate scale; fp px per lenslet is then derived from the "
            "descriptor's sky_pitch_arcsec) or an explicit fp_px_per_lenslet"
        )
    if fp_px_per_lenslet < 2.0:
        warnings.warn(
            f"the focal-plane cube undersamples the lenslet pitch "
            f"(fp_px_per_lenslet = {fp_px_per_lenslet:.3g} < 2): the "
            f"flux-conserving cell integral degrades; use a finer cube grid",
            stacklevel=3,
        )
    return fp_px_per_lenslet


def _resolve_psflet_pack(disperser, psflet_pack):
    """Resolve the template pack: the explicit argument, else the descriptor path."""
    if psflet_pack is not None:
        return psflet_pack
    if disperser.psflet_pack_path is not None:
        return load_psflet_pack(disperser.psflet_pack_path)
    raise ValueError(
        'psflet_kind="template" needs a psflet_pack argument or a '
        "psflet_pack_path on the disperser descriptor"
    )


def _validate_pack_for_band(pack, lam, half):
    """Reject template extrapolation; warn when the footprint outruns the pack."""
    lam_lo = float(pack.wavelengths_nm[0])
    lam_hi = float(pack.wavelengths_nm[-1])
    if float(lam.min()) < lam_lo or float(lam.max()) > lam_hi:
        raise ValueError(
            f"requested band [{float(lam.min()):.6g}, {float(lam.max()):.6g}] nm "
            f"extends beyond the pack's tabulated range [{lam_lo:.6g}, "
            f"{lam_hi:.6g}] nm; template extrapolation is not supported"
        )
    extent = float(pack.offsets[-1])
    if extent < half:
        warnings.warn(
            f"PSFlet footprint (half = {half}) extends past the template extent "
            f"({extent:.3g} px); outer footprint pixels get zero weight",
            stacklevel=3,
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


def lenslet_positions(disperser):
    """Lenslet-index coordinates of the descriptor's grid, ``(n_channels, 2)``.

    Row ``k`` is channel ``k`` (the ordering of every per-channel array in the
    IR). The coordinates are the :func:`~coronachrome.grids.square_grid` or
    :func:`~coronachrome.grids.hex_grid` convention: one unit per lenslet
    pitch, with the lenslet-index origin ``(0, 0)`` at the grid center.
    """
    if disperser.grid_kind == "square":
        return square_grid(disperser.n_lenslets)
    return hex_grid(disperser.n_lenslets)


def lenslet_cell_centers(disperser, fp_shape, fp_px_per_lenslet, positions=None):
    """Focal-plane centers of the lenslet collection cells, in cube pixels.

    Coordinates are cube pixel-center coordinates: pixel ``(row, column)`` of
    a cube plane is centered on ``(x, y) = (column, row)``. The lenslet-index
    origin sits at ``(fp_shape[1] / 2, fp_shape[0] / 2)``, which for an even
    cube dimension is half a pixel above the geometric array center
    ``(n - 1) / 2``. Each cell is a square of side ``fp_px_per_lenslet``
    rotated by ``disperser.angle_rad`` about its center; these are the cells
    :func:`build_ir` integrates the cube over.

    Args:
        disperser: A ``LensletDisperser``.
        fp_shape: Focal-plane cube ``(ny, nx)``.
        fp_px_per_lenslet: Cube pixels per lenslet pitch.
        positions: Optional ``(n, 2)`` lenslet-index coordinates to place
            (default: the descriptor's whole grid).

    Returns:
        Tuple ``(cx, cy)``, each ``(n,)``.
    """
    if positions is None:
        positions = lenslet_positions(disperser)
    positions = jnp.asarray(positions, dtype=float)
    fx0, fy0 = fp_shape[1] / 2.0, fp_shape[0] / 2.0
    ca, sa = jnp.cos(disperser.angle_rad), jnp.sin(disperser.angle_rad)
    px, py = positions[:, 0], positions[:, 1]
    cx = fx0 + fp_px_per_lenslet * (ca * px - sa * py)
    cy = fy0 + fp_px_per_lenslet * (sa * px + ca * py)
    return cx, cy


def _placement_centroids(disperser, positions, lam, pack):
    """Detector centroids the footprints are placed at, plus the anchor index.

    The geometric dispersion trace (:func:`lenslet_centroids`), shifted by a
    template pack's per-anchor wavecal correction when ``pack`` carries
    ``centroids``. Returns ``(xc, yc, field_idx)``; ``field_idx`` is None
    without a pack.
    """
    scale = disperser.pitch_m / disperser.pixsize_m
    disp = dispersion_px(disperser.dispersion_coeffs, disperser.lam_ref_nm, lam)
    xc, yc = lenslet_centroids(
        positions, scale, disperser.angle_rad, disp, disperser.detector_shape
    )
    if pack is None:
        return xc, yc, None
    field_idx = nearest_field_idx(pack, positions)
    if pack.centroids is not None:
        # Per-anchor (dx, dy) wavecal corrections, interpolated to the
        # requested wavelengths, applied before footprint placement.
        def interp_corr(anchor_corr):
            return jnp.stack(
                [
                    jnp.interp(lam, pack.wavelengths_nm, anchor_corr[:, 0]),
                    jnp.interp(lam, pack.wavelengths_nm, anchor_corr[:, 1]),
                ],
                axis=-1,
            )

        corr = jax.vmap(interp_corr)(pack.centroids)  # (n_field, n_wav, 2)
        xc = xc + corr[field_idx, :, 0]
        yc = yc + corr[field_idx, :, 1]
    return xc, yc, field_idx


def detector_centroids(
    disperser, wavelengths_nm, *, psflet_pack=None, positions=None, corrected=True
):
    """PSFlet centroids on the detector, as :func:`build_ir` places them.

    The centroid of channel ``k`` at wavelength ``lambda`` is the detector
    point the PSFlet footprint offsets ``(dx, dy)`` are measured from, in
    detector pixel-center coordinates (pixel ``(row, column)`` is centered on
    ``(x, y) = (column, row)``). Dispersion runs along detector x: the offset
    is ``polyval(dispersion_coeffs, log(lambda / lam_ref_nm))``, so a positive
    leading coefficient moves longer wavelengths toward larger x.

    Args:
        disperser: A ``LensletDisperser``.
        wavelengths_nm: ``(n_wav,)`` wavelengths.
        psflet_pack: Template pack for ``psflet_kind="template"`` (default:
            the descriptor's ``psflet_pack_path``, as in :func:`build_ir`).
        positions: Optional ``(n, 2)`` lenslet-index coordinates (default:
            the descriptor's whole grid, in channel order).
        corrected: Apply a template pack's per-anchor centroid correction.
            False returns the geometric dispersion trace alone.

    Returns:
        Tuple ``(xc, yc)``, each ``(n, n_wav)``.
    """
    lam = jnp.atleast_1d(jnp.asarray(wavelengths_nm, dtype=float))
    if positions is None:
        positions = lenslet_positions(disperser)
    positions = jnp.asarray(positions, dtype=float)
    pack = None
    if corrected and disperser.psflet_kind == "template":
        pack = _resolve_psflet_pack(disperser, psflet_pack)
    xc, yc, _ = _placement_centroids(disperser, positions, lam, pack)
    return xc, yc


def detector_trace_origin(disperser):
    """Detector point the trace geometry is anchored to, ``(x, y)`` pixels.

    Where the lenslet-index origin lands at zero dispersion offset:
    ``(detector_shape[1] / 2, detector_shape[0] / 2)`` in detector
    pixel-center coordinates. Every trace is this point plus the rotated,
    scaled lenslet offset plus the dispersion offset along x. It coincides
    with the reference-wavelength centroid of lenslet ``(0, 0)`` only when
    the dispersion polynomial has no constant term.
    """
    scale = disperser.pitch_m / disperser.pixsize_m
    xc, yc = lenslet_centroids(
        jnp.zeros((1, 2)),
        scale,
        disperser.angle_rad,
        jnp.zeros(1),
        disperser.detector_shape,
    )
    return float(xc[0, 0]), float(yc[0, 0])


@build_ir.register
def _(
    disperser: LensletDisperser,
    wavelengths_nm,
    fp_shape,
    fp_px_per_lenslet=None,
    half=3,
    supersample=4,
    fp_pixel_scale_arcsec=None,
    psflet_pack=None,
    wavelength_edges=None,
):
    """Build a SpatialChannelIR from a LensletDisperser.

    Spatial sampling is a derived quantity: pass ``fp_pixel_scale_arcsec``
    (the focal-plane cube's angular plate scale, arcsec per pixel -- for a
    cube rendered onto a detector grid this is
    ``optical_path.detector.pixel_scale_arcsec``) and the pixels-per-lenslet
    ratio follows from the descriptor's ``sky_pitch_arcsec``.
    ``fp_px_per_lenslet`` is an explicit override for parity work; exactly one
    of the two must be given. Build-time diagnostics warn when the cube
    undersamples the lenslet pitch (< 2 px per cell) and when lenslet cells
    extend past the cube bounds (those spaxels silently lose flux).

    The PSFlet core width (``disperser.psflet_params[0]``, in detector pixels)
    scales linearly with wavelength about ``disperser.psflet_ref_nm`` -- a
    diffraction-limited spot grows as ``lambda f / D``, so at fixed pixel scale
    its pixel width is proportional to wavelength. The fixed ``half`` footprint
    (``(2 * half + 1)`` per side) must therefore be wide enough for the widest
    PSFlet in the band; otherwise the long-wavelength spot is truncated (its
    footprint still renormalizes to unit flux, but the wings are clipped).

    PSFlets are pixel-integrated (the Gaussian via an erf pixel integral, the
    Moffat via sub-pixel quadrature), not point-sampled at pixel centers.

    ``wavelength_edges`` (``n_wav + 1`` ascending values) give each channel's
    bin extent exactly, so the LSF smear is the detector span of the real bin.
    Without them the bin widths are approximated from the center spacing
    (``jnp.gradient``), and a single-wavelength build assumes a 1 nm bin --
    fine for quick looks, wrong for wide single-channel bins.

    With ``psflet_kind="template"`` the PSFlet comes from a frozen template
    pack (``psflet_pack`` argument, else the descriptor's
    ``psflet_pack_path``): planes are blended in wavelength and
    bilinear-sampled at the footprint offsets, with the same LSF smear. No
    linear width scaling is applied and ``psflet_params`` / ``psflet_ref_nm``
    are ignored -- chromatic morphology is the pack's job (a physical
    micro-pupil does not scale linearly with wavelength). A pack with
    ``centroids`` also corrects each field anchor's trace centroids (the
    per-field wavecal residual). The requested band must lie inside the
    pack's tabulated range.

    Disperser throughput is baked into the operator: each wavelength's footprint
    is scaled by ``disperser.throughput(lambda)`` after unit-flux renormalization,
    so the forward and the extraction that inverts H stay consistent.
    """
    fp_px_per_lenslet = _resolve_fp_px_per_lenslet(
        disperser, fp_px_per_lenslet, fp_pixel_scale_arcsec
    )
    lam = jnp.asarray(wavelengths_nm, dtype=float)
    n_wav = int(lam.shape[0])
    positions = lenslet_positions(disperser)
    n_channels = int(positions.shape[0])
    ny, nx = disperser.detector_shape

    # Spatial sampling: flux-conserving footprints over each lenslet's cell. The
    # lenslet grid is rotated by the lenslet angle in the focal plane, matching
    # the rotation applied to the detector centroids below.
    ca, sa = jnp.cos(disperser.angle_rad), jnp.sin(disperser.angle_rad)
    cx, cy = lenslet_cell_centers(disperser, fp_shape, fp_px_per_lenslet, positions)
    spatial_src, spatial_w = _resampling_footprints(
        cx, cy, disperser.angle_rad, fp_px_per_lenslet, fp_shape, supersample
    )

    # Coverage diagnostic: a rotated square cell of side fp_px_per_lenslet spans
    # 0.5 * cell * (|cos a| + |sin a|) per axis; cells reaching past the cube
    # get zero weight there, so those spaxels lose flux.
    cell_ext = 0.5 * fp_px_per_lenslet * (jnp.abs(ca) + jnp.abs(sa))
    n_outside = int(
        (
            (cx - cell_ext < -0.5)
            | (cx + cell_ext > fp_shape[1] - 0.5)
            | (cy - cell_ext < -0.5)
            | (cy + cell_ext > fp_shape[0] - 0.5)
        ).sum()
    )
    if n_outside:
        warnings.warn(
            f"{n_outside} lenslet cells extend past the focal-plane cube bounds; "
            f"those spaxels lose flux",
            stacklevel=2,
        )

    # Detector centroids (n_channels, n_wav) and PSFlet footprint offsets (n_psf,).
    coeffs, lam_ref = disperser.dispersion_coeffs, disperser.lam_ref_nm
    pack = None
    if disperser.psflet_kind == "template":
        pack = _resolve_psflet_pack(disperser, psflet_pack)
        _validate_pack_for_band(pack, lam, half)
    xc, yc, field_idx = _placement_centroids(disperser, positions, lam, pack)

    off = jnp.arange(-half, half + 1)
    ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddy = ddy.reshape(-1).astype(float)
    ddx = ddx.reshape(-1).astype(float)

    # Broadcast footprints to (n_channels, n_wav, n_psf).
    px = jnp.round(xc)[..., None] + ddx
    py = jnp.round(yc)[..., None] + ddy
    dx = px - xc[..., None]
    dy = py - yc[..., None]

    # Per-wavelength LSF smear width [px]: the detector extent of each
    # wavelength bin -- exact from the bin edges when given, else approximated
    # from the center spacing (with a 1 nm bin assumed for a single channel).
    if wavelength_edges is not None:
        edges = jnp.asarray(wavelength_edges, dtype=float)
        if edges.shape != (n_wav + 1,):
            raise ValueError("wavelength_edges must have n_wav + 1 entries")
        smear = jnp.abs(
            dispersion_px(coeffs, lam_ref, edges[1:])
            - dispersion_px(coeffs, lam_ref, edges[:-1])
        )
    else:
        dlam = jnp.gradient(lam) if n_wav > 1 else jnp.array([1.0])
        smear = jnp.abs(
            dispersion_px(coeffs, lam_ref, lam + 0.5 * dlam)
            - dispersion_px(coeffs, lam_ref, lam - 0.5 * dlam)
        )

    # PSFlet weights for every (channel, wavelength, footprint pixel). vmap
    # over the wavelength axis so each wavelength's shape and smear apply.
    if pack is not None:
        # Template mode: no width scaling -- the pack's per-wavelength planes
        # carry the chromatic morphology (a micro-pupil is not ~ lambda).
        def psf_one_wav(dx_w, dy_w, smear_w, lam_w):
            return template_weights(pack, dx_w, dy_w, lam_w, field_idx, smear_w)

        g = jax.vmap(psf_one_wav, in_axes=(1, 1, 0, 0), out_axes=1)(dx, dy, smear, lam)
    else:
        # Diffraction scaling: the PSFlet core width (params[0], px) scales
        # linearly with wavelength at a fixed detector pixel scale (spot size
        # ~ lambda f / D), referenced to psflet_ref_nm. Trailing shape params
        # (e.g. Moffat beta) are dimensionless and do not scale.
        width_scale = lam / disperser.psflet_ref_nm
        n_params = disperser.psflet_params.shape[0]
        params_w = jnp.broadcast_to(disperser.psflet_params, (n_wav, n_params))
        params_w = params_w.at[:, 0].multiply(width_scale)

        def psf_one_wav(dx_w, dy_w, smear_w, params_one):
            return psflet_weights(
                dx_w, dy_w, disperser.psflet_kind, params_one, smear_w
            )

        g = jax.vmap(psf_one_wav, in_axes=(1, 1, 0, 0), out_axes=1)(
            dx, dy, smear, params_w
        )

    valid = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    g = jnp.where(valid, g, 0.0)
    total = g.sum(axis=2, keepdims=True)
    det_vals = g / jnp.clip(total, 1e-12, None)
    # Bake disperser throughput into H: scale each wavelength's footprint by the
    # fraction of that wavelength's photons that survive the disperser. Applied
    # after the unit-flux renorm, so a zero-throughput wavelength gives an
    # all-zero footprint with no division by zero.
    tput = disperser.throughput(lam)  # (n_wav,)
    det_vals = det_vals * tput[None, :, None]
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
