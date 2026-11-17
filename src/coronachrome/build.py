"""Compile a disperser descriptor into a SpatialChannelIR.

Runs offline (eager): detector footprints are vectorized over channels and
wavelengths; cell overlaps are clipped on the host from static grid geometry. build_ir
dispatches on the descriptor type so new IFS geometries register their own
builders without touching optixstuff.
"""

import functools
import warnings

import jax
import jax.numpy as jnp
from hwoutils.grids import PixelGrid
from hwoutils.spectral import Quadrature
from optixstuff.disperser import LensletDisperser
from optixstuff.optical_elements import SpectralThroughput

from coronachrome._collection import cell_overlap
from coronachrome.dispersion import lenslet_centroids
from coronachrome.grids import LensletLayout
from coronachrome.ir import SpatialChannelIR
from coronachrome.psflet import psflet_weights
from coronachrome.templates import (
    load_psflet_pack,
    nearest_field_idx,
    template_weights,
)


@functools.singledispatch
def build_ir(disperser, bands, fp_grid, **kwargs):
    """Build a SpatialChannelIR from a disperser descriptor."""
    raise NotImplementedError(
        f"build_ir not implemented for {type(disperser).__name__}"
    )


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


def lenslet_positions(disperser, *, layout=None):
    """Centres in lenslet pitches; layout owns the origin convention."""
    return _layout(disperser, layout).positions()


def _layout(disperser, layout):
    if layout is None:
        layout = LensletLayout(disperser.n_lenslets, disperser.grid_kind)
    if (layout.n_lenslets, layout.grid_kind) != (
        disperser.n_lenslets,
        disperser.grid_kind,
    ):
        raise ValueError("layout must match disperser n_lenslets and grid_kind")
    return layout


def lenslet_cell_centers(disperser, fp_grid, *, layout=None, positions=None):
    """Return focal-plane pixel (x,y) centres using the declared arcsec grid.

    The layout is rotated by angle_rad in the input grid's oriented unit
    plane; the sky origin is fp_grid.center_x, fp_grid.center_y.
    """
    if fp_grid.unit != "arcsec":
        raise ValueError("fp_grid must use arcsec; convert lod through its Frame")
    if disperser.sky_pitch_arcsec is None:
        raise ValueError("sky_pitch_arcsec is required")
    positions = (
        lenslet_positions(disperser, layout=layout) if positions is None else positions
    )
    ca, sa = jnp.cos(disperser.angle_rad), jnp.sin(disperser.angle_rad)
    x, y = positions[:, 0], positions[:, 1]
    return fp_grid.unit_to_pixel(
        disperser.sky_pitch_arcsec * (ca * x - sa * y),
        disperser.sky_pitch_arcsec * (sa * x + ca * y),
    )


def _placement_centroids(disperser, positions, lam, pack):
    """Detector centroids the footprints are placed at, plus the anchor index.

    The geometric dispersion trace (:func:`lenslet_centroids`), shifted by a
    template pack's per-anchor wavecal correction when ``pack`` carries
    ``centroids``. Returns ``(xc, yc, field_idx)``; ``field_idx`` is None
    without a pack.
    """
    scale = disperser.pitch_m / disperser.pixsize_m
    disp = disperser.dispersion_px(lam)
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
    disperser,
    wavelengths_nm,
    *,
    psflet_pack=None,
    positions=None,
    corrected=True,
    layout=None,
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
        layout: LensletLayout defining the origin convention.
        corrected: Apply a template pack's per-anchor centroid correction.
            False returns the geometric dispersion trace alone.

    Returns:
        Tuple ``(xc, yc)``, each ``(n, n_wav)``.
    """
    lam = jnp.atleast_1d(jnp.asarray(wavelengths_nm, dtype=float))
    if positions is None:
        positions = lenslet_positions(disperser, layout=layout)
    positions = jnp.asarray(positions, dtype=float)
    pack = None
    if corrected and disperser.psflet_kind == "template":
        pack = _resolve_psflet_pack(disperser, psflet_pack)
    xc, yc, _ = _placement_centroids(disperser, positions, lam, pack)
    return xc, yc


def detector_trace_origin(disperser):
    """Detector point the trace geometry is anchored to, ``(x, y)`` pixels.

    Where the lenslet-index origin lands at zero dispersion offset:
    ``((detector_shape[1] - 1) / 2, (detector_shape[0] - 1) / 2)`` in detector
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
    bands,
    fp_grid,
    *,
    layout=None,
    half=3,
    psflet_pack=None,
    quadrature_order=5,
):
    """Build an operator from channel Bands and a focal-plane PixelGrid.

    Input Images hold band-integrated quantities on an arcsec grid. Each
    band's constant spectral density is integrated with a Quadrature; the
    geometric mean is its trace reference wavelength. Throughput, chromatic
    shape, and dispersion are evaluated at nodes before integration.
    Detector pixels have a meter grid with geometric centre. Focal sampling
    uses a host-built sparse matrix of exact pixel/cell overlap areas.
    This integrates piecewise-constant density; smooth reconstructions are
    approximated by that density (see spatial_sample).
    """
    if fp_grid.unit != "arcsec":
        raise ValueError("fp_grid must use arcsec; convert lod through its Frame")
    if disperser.sky_pitch_arcsec is None or disperser.sky_pitch_arcsec <= 0:
        raise ValueError("sky_pitch_arcsec must be positive")
    layout = _layout(disperser, layout)
    bands = tuple(bands)
    lam = jnp.asarray([band.geometric_center_nm for band in bands])
    pack = None
    if disperser.psflet_kind == "template":
        pack = _resolve_psflet_pack(disperser, psflet_pack)
        _validate_pack_for_band(
            pack, jnp.asarray([b.lo_nm for b in bands] + [b.hi_nm for b in bands]), half
        )
    knots = []
    if isinstance(disperser.throughput_element, SpectralThroughput):
        knots.extend(disperser.throughput_element.wavelengths_nm.tolist())
    if pack is not None:
        knots.extend(pack.wavelengths_nm.tolist())
    quadrature = Quadrature.build(bands, knots, quadrature_order)
    lenslet_grid, lenslet_indices = layout.image_geometry(
        disperser.sky_pitch_arcsec, float(disperser.angle_rad), fp_grid.orientation
    )
    collection_grids = layout.collection_grids(
        disperser.sky_pitch_arcsec, float(disperser.angle_rad), fp_grid.orientation
    )
    channel_order = (
        tuple(map(int, lenslet_indices.tolist()))
        if layout.grid_kind == "square"
        else tuple(range(layout.n_lenslets**2))
    )
    collection_overlap = cell_overlap(fp_grid, collection_grids, channel_order)
    fp_px_per_lenslet = disperser.sky_pitch_arcsec / fp_grid.pixel_area**0.5
    if fp_px_per_lenslet < 2:
        warnings.warn(
            "the focal-plane cube undersamples the lenslet pitch (< 2)", stacklevel=2
        )
    cx, cy = lenslet_cell_centers(disperser, fp_grid, layout=layout)
    # Transform one cell's affine half-extents into source pixel coordinates.
    (a, b), (c, d) = fp_grid.basis
    det = a * d - b * c
    (e, f), (g, h) = collection_grids[0].basis
    ext_x = (abs((d * e - b * g) / det) + abs((d * f - b * h) / det)) / 2
    ext_y = (abs((a * g - c * e) / det) + abs((a * h - c * f) / det)) / 2
    ny_fp, nx_fp = fp_grid.shape
    if bool(
        jnp.any(
            (cx - ext_x < -0.5)
            | (cx + ext_x > nx_fp - 0.5)
            | (cy - ext_y < -0.5)
            | (cy + ext_y > ny_fp - 0.5)
        )
    ):
        warnings.warn(
            "lenslet cells extend past the focal-plane cube bounds; "
            "those spaxels lose flux",
            stacklevel=2,
        )
    positions = layout.positions()
    ny, nx = disperser.detector_shape
    xc, yc, _ = _placement_centroids(disperser, positions, lam, pack)
    off = jnp.arange(-half, half + 1)
    dy, dx = jnp.meshgrid(off, off, indexing="ij")
    px = jnp.round(xc)[..., None] + dx.ravel()
    py = jnp.round(yc)[..., None] + dy.ravel()
    nodes = quadrature.nodes_nm.ravel()
    xq, yq, field_idx = _placement_centroids(disperser, positions, nodes, pack)
    xq = xq.reshape(len(positions), len(bands), -1)
    yq = yq.reshape(len(positions), len(bands), -1)
    dxq, dyq = px[:, :, None, :] - xq[..., None], py[:, :, None, :] - yq[..., None]

    def profile(dx_w, dy_w, wave):
        if pack is not None:
            return template_weights(pack, dx_w, dy_w, wave, field_idx, 0.0)
        params = disperser.psflet_params.at[0].multiply(wave / disperser.psflet_ref_nm)
        return psflet_weights(dx_w, dy_w, disperser.psflet_kind, params, 0.0)

    nc, nw, nq, npix = dxq.shape
    g = jax.vmap(profile, in_axes=(1, 1, 0), out_axes=1)(
        dxq.reshape(nc, nw * nq, npix), dyq.reshape(nc, nw * nq, npix), nodes
    ).reshape(nc, nw, nq, npix)
    valid = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    g = jnp.where(valid[:, :, None, :], g, 0.0)
    total = g.sum(axis=-1, keepdims=True)
    g = g / jnp.clip(total, 1e-12, None)
    throughput = disperser.throughput(nodes).reshape(quadrature.nodes_nm.shape)
    # Integrate a unit bin-integrated spectrum: density = 1 / band width.
    weights = quadrature.weights_nm / jnp.asarray([b.width_nm for b in bands])[:, None]
    det_vals = jnp.einsum("cwqp,wq->cwp", g, weights * throughput)
    det_rows = jnp.clip((py * nx + px).astype(jnp.int32), 0, ny * nx - 1)
    if bool(jnp.any((total[..., 0] <= 1e-12) & (quadrature.weights_nm > 0)[None])):
        warnings.warn(
            "(channel, wavelength) footprints fell off the detector", stacklevel=2
        )
    return SpatialChannelIR(
        det_rows=det_rows,
        det_vals=det_vals,
        bands=bands,
        fp_grid=fp_grid,
        lenslet_grid=lenslet_grid,
        lenslet_indices=lenslet_indices,
        layout=layout,
        collection_grids=collection_grids,
        collection_overlap=collection_overlap,
        detector_grid=PixelGrid.regular(
            disperser.detector_shape, disperser.pixsize_m, "meter"
        ),
    )
