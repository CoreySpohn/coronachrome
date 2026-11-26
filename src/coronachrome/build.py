"""Compile a disperser descriptor into a SpatialChannelIR.

Runs offline (eager): detector footprints are vectorized over channels and
wavelengths; cell overlaps are clipped on the host from static grid geometry. build_ir
dispatches on the descriptor type so new IFS geometries register their own
builders without touching optixstuff.
"""

import functools
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import canonical, is_traced
from hwoutils.precision import like
from hwoutils.radial import get_center
from optixstuff.disperser import LensletDisperser
from optixstuff.grids import PixelGrid, footprint_overlap
from optixstuff.psflet_pack import load_psflet_pack
from optixstuff.spectral import Quadrature, same_wavelength

from coronachrome.dispersion import lenslet_centroids
from coronachrome.grids import LensletLayout
from coronachrome.ir import SpatialChannelIR
from coronachrome.psflet import psflet_weights
from coronachrome.templates import (
    _template_weights,
    _wavelength_blend,
    check_template_origin,
    nearest_field_idx,
)


@functools.singledispatch
def build_ir(disperser, bands, fp_grid, **kwargs) -> SpatialChannelIR:
    """Build a SpatialChannelIR from a disperser descriptor.

    Args:
        disperser: Descriptor defining the instrument geometry.
        bands: Ordered spectral bands of the input cube.
        fp_grid: Focal-plane pixel grid.
        **kwargs: Options for the registered disperser builder.

    Returns:
        The compiled operator for the supplied geometry and bands.

    Raises:
        NotImplementedError: The descriptor has no registered builder.
    """
    raise NotImplementedError(
        f"build_ir not implemented for {type(disperser).__name__}"
    )


def _resolve_psflet_pack(disperser, psflet_pack):
    """Resolve a centroid-relative PSFlet pack for footprint placement.

    Args:
        disperser: Descriptor carrying the optional pack path.
        psflet_pack: Explicit in-memory pack, taking precedence over the path.

    Returns:
        The validated in-memory pack or the pack loaded from the descriptor.

    Raises:
        ValueError: The origin convention is unsupported or no pack is supplied.
    """
    if psflet_pack is None:
        if disperser.psflet_pack_path is None:
            raise ValueError(
                'psflet_kind="template" needs a psflet_pack argument or a '
                "psflet_pack_path on the disperser descriptor"
            )
        psflet_pack = load_psflet_pack(disperser.psflet_pack_path)
    check_template_origin(psflet_pack)
    return psflet_pack


def _validate_pack_for_band(pack, bands, half):
    """Reject template extrapolation; warn when the footprint outruns the pack."""
    pack_nm = pack.wavelengths_nm.values
    lam_lo = float(pack_nm[0])
    lam_hi = float(pack_nm[-1])
    band_lo = min(b.low_nm for b in bands)
    band_hi = max(b.high_nm for b in bands)
    if pack_nm.size == 1:
        # A one-wavelength pack is a monochromatic template set: valid only for
        # a band whose edges are at its own wavelength.
        if not (same_wavelength(band_lo, lam_lo) and same_wavelength(band_hi, lam_lo)):
            raise ValueError(
                f"the pack is tabulated at one wavelength, {lam_lo:.6g} nm, so "
                "it is monochromatic; the requested band "
                f"[{band_lo:.6g}, {band_hi:.6g}] nm is "
                "not at that wavelength. Use a pack with several wavelengths "
                "for a wider band"
            )
    elif (band_lo < lam_lo and not same_wavelength(band_lo, lam_lo)) or (
        band_hi > lam_hi and not same_wavelength(band_hi, lam_hi)
    ):
        raise ValueError(
            f"requested band [{band_lo:.6g}, {band_hi:.6g}] nm "
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
    """Return lenslet centers relative to the layout origin.

    Args:
        disperser: Descriptor supplying the layout and working precision.
        layout: Optional layout matching the disperser.

    Returns:
        Lenslet centers in lenslet pitches, in channel order.
    """
    dtype = jnp.result_type(disperser.dispersion_coeffs, 1.0)
    return _layout_positions(_layout(disperser, layout), dtype)


def _layout_positions(layout, dtype):
    if type(layout).positions is LensletLayout.positions:
        return layout.positions(dtype=dtype)
    # The public override signature did not require a dtype keyword.
    return canonical(jnp.asarray(layout.positions()), precision=dtype)


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
    """Return focal-plane pixel (x, y) centers using the declared arcsec grid.

    The layout is rotated by angle_rad in the input grid's oriented unit
    plane; the sky origin is fp_grid.center_x, fp_grid.center_y.

    Args:
        disperser: Descriptor supplying sky pitch and rotation.
        fp_grid: Focal-plane pixel grid in arcsec.
        layout: Optional layout matching the disperser.
        positions: Optional lenslet centers in lenslet pitches.

    Returns:
        Focal-plane x and y pixel coordinates in channel order.
    """
    if fp_grid.unit != "arcsec":
        raise ValueError("fp_grid must use arcsec; convert lod through its Frame")
    if disperser.sky_pitch_arcsec is None:
        raise ValueError("sky_pitch_arcsec is required")
    dtype = jnp.result_type(
        disperser.dispersion_coeffs,
        disperser.angle_rad,
        disperser.sky_pitch_arcsec,
        *jax.tree.leaves(positions),
        1.0,
    )
    if positions is None:
        positions = _layout_positions(_layout(disperser, layout), dtype)
    positions = jnp.asarray(positions, dtype=dtype)
    angle = jnp.asarray(disperser.angle_rad, dtype=dtype)
    ca, sa = jnp.cos(angle), jnp.sin(angle)
    pitch = jnp.asarray(disperser.sky_pitch_arcsec, dtype=dtype)
    x, y = positions[:, 0], positions[:, 1]
    return fp_grid.unit_to_pixel(
        pitch * (ca * x - sa * y),
        pitch * (sa * x + ca * y),
    )


def _placement_centroids(disperser, positions, wavelengths_nm, pack, dtype):
    """Detector centroids the footprints are placed at, plus the anchor index.

    The geometric dispersion trace (:func:`lenslet_centroids`), shifted by a
    template pack's per-anchor wavecal correction when ``pack`` carries
    ``centroids``. Returns ``(xc, yc, field_idx)``; ``field_idx`` is None
    without a pack.
    """
    lam = jnp.atleast_1d(jnp.asarray(wavelengths_nm, dtype=dtype))
    pitch, pixsize, angle = (
        jnp.asarray(value, dtype=dtype)
        for value in (disperser.pitch_m, disperser.pixsize_m, disperser.angle_rad)
    )
    disp = canonical(disperser.dispersion_px(lam), precision=dtype)
    xc, yc = lenslet_centroids(
        positions, pitch / pixsize, angle, disp, disperser.detector_shape
    )
    if pack is None:
        return xc, yc, None
    field_idx = nearest_field_idx(pack, positions)
    if pack.centroids is not None:
        if len(pack.wavelengths_nm) == 1:
            corr = pack.centroids[field_idx]
        else:
            lower, frac = _wavelength_blend(
                pack.wavelengths_nm.values, wavelengths_nm, dtype
            )
            lower = jnp.atleast_1d(lower)
            frac = jnp.atleast_1d(frac)[:, None]
            left = pack.centroids[field_idx[:, None], lower]
            right = pack.centroids[field_idx[:, None], lower + 1]
            corr = (1.0 - frac) * left + frac * right
        xc = xc + corr[..., 0]
        yc = yc + corr[..., 1]
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
        wavelengths_nm: ``(n_samples,)`` wavelengths.
        psflet_pack: Template pack for ``psflet_kind="template"`` (default:
            the descriptor's ``psflet_pack_path``, as in :func:`build_ir`).
        positions: Optional ``(n, 2)`` lenslet-index coordinates (default:
            the descriptor's whole grid, in channel order).
        corrected: Apply a template pack's per-anchor centroid correction.
            False returns the geometric dispersion trace alone.
        layout: LensletLayout defining the origin convention.

    Returns:
        Tuple ``(xc, yc)``, each ``(n, n_samples)``.
    """
    pack = None
    if corrected and disperser.psflet_kind == "template":
        pack = _resolve_psflet_pack(disperser, psflet_pack)
    wavecal = (
        ()
        if pack is None or pack.centroids is None
        else (pack.centroids, pack.field_xy)
    )
    dtype = jnp.result_type(
        disperser.dispersion_coeffs,
        disperser.pitch_m,
        disperser.pixsize_m,
        disperser.angle_rad,
        disperser.lam_ref_nm,
        *jax.tree.leaves((positions, wavelengths_nm, wavecal)),
        1.0,
    )
    if positions is None:
        positions = _layout_positions(_layout(disperser, layout), dtype)
    positions = jnp.asarray(positions, dtype=dtype)
    if any(is_traced(leaf) for leaf in jax.tree.leaves(wavelengths_nm)):
        wavelengths_nm = jnp.asarray(wavelengths_nm)
    xc, yc, _ = _placement_centroids(disperser, positions, wavelengths_nm, pack, dtype)
    return xc, yc


def detector_trace_origin(disperser):
    """Detector point the trace geometry is anchored to, ``(x, y)`` pixels.

    Where the lenslet-index origin lands at zero dispersion offset:
    ``((detector_shape[1] - 1) / 2, (detector_shape[0] - 1) / 2)`` in detector
    pixel-center coordinates. Every trace is this point plus the rotated,
    scaled lenslet offset plus the dispersion offset along x. It coincides
    with the reference-wavelength centroid of lenslet ``(0, 0)`` only when
    the dispersion polynomial has no constant term.

    Args:
        disperser: Descriptor specifying the detector shape.

    Returns:
        Detector center coordinates in x, y order.
    """
    cy, cx = get_center(disperser.detector_shape)
    return cx, cy


def _band_response(quadrature, profiles, valid, throughput):
    # Normalize the modeled footprint before discarding off-detector light.
    total = profiles.sum(axis=-1, keepdims=True)
    profiles = profiles / jnp.clip(total, like(1e-12, total), None)
    profiles = jnp.where(valid[:, :, None, :], profiles, like(0.0, profiles))
    response = profiles.transpose(1, 2, 0, 3) * throughput[:, :, None, None]
    det_vals = quadrature.mean(response).transpose(1, 0, 2)
    # A dark template can have zero weight before detector clipping.
    outside = ~valid.any(axis=-1, keepdims=True)
    lost = (profiles.sum(axis=-1) == 0) & ((total[..., 0] > 0) | outside)
    return det_vals, lost & quadrature.active[None]


def _footprint_axis(centroids, taps, n_bands, dtype):
    centroids = np.asarray(centroids)
    nominal = np.rint(centroids[:, :n_bands]).astype(np.int64)
    pixels = nominal[..., None] + taps.ravel()
    samples = centroids[:, n_bands:].reshape(centroids.shape[0], n_bands, -1)
    offsets = canonical(pixels[:, :, None, :] - samples[..., None], precision=dtype)
    return pixels, offsets


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

    Args:
        disperser: Lenslet geometry, dispersion, PSFlet and throughput model.
        bands: Ordered spectral intervals for the input flux integrals.
        fp_grid: Focal-plane pixel grid in arcseconds.
        layout: Optional lenslet layout matching the disperser.
        half: Half-width of the modeled detector footprint in pixels.
        psflet_pack: Optional centroid-relative template pack.
        quadrature_order: Number of nodes per spectral integration interval.

    Returns:
        The spatial collection and detector dispersion operator at the
        working precision of the disperser.

    Raises:
        TypeError: The builder received traced inputs.
        ValueError: The geometry or template pack is incompatible with the
            requested grid, wavelength bands or footprint.
    """
    if fp_grid.unit != "arcsec":
        raise ValueError("fp_grid must use arcsec; convert lod through its Frame")
    if any(is_traced(leaf) for leaf in jax.tree.leaves((disperser, psflet_pack))):
        raise TypeError(
            "build_ir requires concrete inputs; build it outside a JAX trace"
        )
    if disperser.sky_pitch_arcsec is None or disperser.sky_pitch_arcsec <= 0:
        raise ValueError("sky_pitch_arcsec must be positive")
    layout = _layout(disperser, layout)
    bands = tuple(bands)
    analytic = (
        ()
        if disperser.psflet_kind == "template"
        else (disperser.psflet_params, disperser.psflet_ref_nm)
    )
    dtype = jnp.result_type(
        disperser.dispersion_coeffs,
        disperser.pitch_m,
        disperser.pixsize_m,
        disperser.angle_rad,
        disperser.lam_ref_nm,
        disperser.sky_pitch_arcsec,
        *analytic,
        1.0,
    )
    sky_pitch, angle = float(disperser.sky_pitch_arcsec), float(disperser.angle_rad)
    pack = None
    # The upstream builders and disperser evaluators return JAX arrays. Read
    # them at host precision before choosing support or differencing offsets.
    with jax.enable_x64():  # precision-ok: concrete host-only IR setup
        models = [disperser.throughput_element]
        if disperser.psflet_kind == "template":
            pack = _resolve_psflet_pack(disperser, psflet_pack)
            _validate_pack_for_band(pack, bands, half)
            models.append(pack)
        quadrature = Quadrature.build(bands, *models, order=quadrature_order)
        nodes = np.asarray(quadrature.nodes_nm).ravel()
        centers = [band.geometric_center_nm for band in bands]
        waves = np.concatenate((centers, nodes))
        host_dtype = np.dtype(np.float64)
        positions = _layout_positions(layout, host_dtype)
        xc, yc, field_idx = _placement_centroids(
            disperser, positions, waves, pack, host_dtype
        )
        throughput = np.asarray(disperser.throughput(nodes), dtype=host_dtype).reshape(
            quadrature.nodes_nm.shape
        )
        cx, cy = lenslet_cell_centers(disperser, fp_grid, positions=positions)
        lenslet_grid, lenslet_indices = layout.image_geometry(
            sky_pitch, angle, fp_grid.orientation
        )
        collection_grids = layout.collection_grids(
            sky_pitch, angle, fp_grid.orientation
        )
        overlap = footprint_overlap(fp_grid, collection_grids)

    collection_overlap = canonical(overlap, precision=dtype)
    fp_px_per_lenslet = sky_pitch / fp_grid.pixel_area**0.5
    if fp_px_per_lenslet < 2:
        warnings.warn(
            "the focal-plane cube undersamples the lenslet pitch (< 2)", stacklevel=2
        )
    cx, cy = np.asarray(cx), np.asarray(cy)
    # Transform one cell's affine half-extents into source pixel coordinates.
    (a, b), (c, d) = fp_grid.basis
    det = a * d - b * c
    (e, f), (g, h) = collection_grids[0].basis
    ext_x = (abs((d * e - b * g) / det) + abs((d * f - b * h) / det)) / 2
    ext_y = (abs((a * g - c * e) / det) + abs((a * h - c * f) / det)) / 2
    ny_fp, nx_fp = fp_grid.shape
    if bool(
        np.any(
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
    ny, nx = disperser.detector_shape
    off = np.arange(-half, half + 1, dtype=np.int32)
    dy, dx = np.meshgrid(off, off, indexing="ij")
    px, dxq = _footprint_axis(xc, dx, len(bands), dtype)
    py, dyq = _footprint_axis(yc, dy, len(bands), dtype)
    throughput = canonical(throughput, precision=dtype)
    # The numerical quadrature consumes only its weights and node shape.
    weights = canonical(np.asarray(quadrature.weights_nm), precision=dtype)
    quadrature = eqx.tree_at(lambda rule: rule.weights_nm, quadrature, weights)
    if pack is not None:
        profile_args = _wavelength_blend(pack.wavelengths_nm.values, nodes, dtype)
        pack = canonical(pack, precision=dtype)

        def profile(dx_w, dy_w, blend):
            return _template_weights(pack, dx_w, dy_w, field_idx, 0.0, 5, blend)
    else:
        profile_args = canonical(
            nodes / float(disperser.psflet_ref_nm), precision=dtype
        )
        params = canonical(disperser.psflet_params, precision=dtype)

        def profile(dx_w, dy_w, scale):
            scaled = params.at[0].multiply(scale)
            return psflet_weights(dx_w, dy_w, disperser.psflet_kind, scaled, 0.0)

    nc, nw, nq, npix = dxq.shape
    profiles = jax.vmap(profile, in_axes=(1, 1, 0), out_axes=1)(
        dxq.reshape(nc, nw * nq, npix), dyq.reshape(nc, nw * nq, npix), profile_args
    ).reshape(nc, nw, nq, npix)
    valid = jnp.asarray((px >= 0) & (px < nx) & (py >= 0) & (py < ny))
    # Mean response per detector pixel for constant spectral density in a band.
    det_vals, lost = _band_response(quadrature, profiles, valid, throughput)
    det_rows = jnp.asarray(np.clip(py * nx + px, 0, ny * nx - 1), dtype=jnp.int32)
    if bool(jnp.any(lost)):
        warnings.warn(
            "(channel, wavelength) footprints fell off the detector", stacklevel=2
        )
    return SpatialChannelIR(
        det_rows=det_rows,
        det_vals=det_vals,
        bands=bands,
        fp_grid=fp_grid,
        lenslet_grid=lenslet_grid,
        lenslet_indices=jnp.asarray(lenslet_indices, dtype=jnp.int32),
        layout=layout,
        collection_grids=collection_grids,
        collection_overlap=collection_overlap,
        detector_grid=PixelGrid.regular(
            disperser.detector_shape, disperser.pixsize_m, "meter"
        ),
    )
