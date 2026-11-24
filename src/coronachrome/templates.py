"""Frozen PSFlet template packs and template-mode PSFlet evaluation.

A pack (:class:`optixstuff.psflet_pack.PsfletPack`) tabulates the
pixel-integrated PSFlet response on a fine sub-pixel offset grid. The absolute
(and per-wavelength) normalization is irrelevant: the IR build renormalizes
each footprint to unit flux and the disperser throughput curve owns chromatic
transmission, so templates carry shape only.

Packs are the array waist between a PSFlet generator and this library: an
external wave-optics code (or the analytic reference emitter here) writes the
npz format documented in optixstuff, and ``build_ir`` consumes it via
``psflet_kind="template"``. Unlike the analytic profiles, template planes are
NOT width-scaled with wavelength -- chromatic morphology (e.g. a
geometrically-sized micro-pupil with diffractive sinc^2 wings) comes from the
per-wavelength tabulation.
"""

import jax
import jax.numpy as jnp
from optixstuff.psflet_pack import TEMPLATE_ORIGIN, PsfletPack

from coronachrome.psflet import gaussian_psflet, moffat_psflet


def analytic_psflet_pack(
    kind,
    params,
    wavelengths_nm,
    *,
    psflet_ref_nm=None,
    half_extent=6.0,
    step=0.125,
    field_xy=None,
    centroids=None,
    meta_json="",
):
    """Tabulate an analytic PSFlet profile into a pack (the reference emitter).

    Physical packs come from an external wave-optics generator; this emitter
    exists for tests, documentation examples, and validation bridges. With
    ``psflet_ref_nm`` set, the core width scales linearly (lambda / ref) per
    tabulated plane, exactly like the analytic build path, so template mode
    reproduces the gaussian/moffat modes to interpolation tolerance. Multiple
    ``field_xy`` anchors all receive the same analytic template (field
    dependence is what real generators provide).

    Args:
        kind: "gaussian" or "moffat".
        params: Profile parameters, as ``LensletDisperser.psflet_params``
            ([sigma] or [alpha, beta], core width in detector px).
        wavelengths_nm: Ascending tabulation wavelengths.
        psflet_ref_nm: Wavelength at which ``params[0]`` is specified; None
            keeps the width constant across planes.
        half_extent: Tabulated offset extent [px]; the plane spans
            [-half_extent, half_extent] on both axes.
        step: Offset grid spacing [px].
        field_xy: ``(n_field, 2)`` anchor positions in lenslet-index
            coordinates (default: one anchor at the origin).
        centroids: Optional ``(n_field, n_lam, 2)`` centroid corrections.
        meta_json: Optional provenance string stored verbatim.

    Returns:
        A validated :class:`PsfletPack`.
    """
    lam = jnp.asarray(wavelengths_nm, dtype=float)
    n_off = round(2.0 * half_extent / step) + 1
    offsets = jnp.linspace(-half_extent, half_extent, n_off)
    vv, uu = jnp.meshgrid(offsets, offsets, indexing="ij")

    def plane(lam_w):
        scale = float(lam_w) / psflet_ref_nm if psflet_ref_nm is not None else 1.0
        width = params[0] * scale
        if kind == "gaussian":
            return gaussian_psflet(uu, vv, width)
        return moffat_psflet(uu, vv, width, params[1])

    planes = jnp.stack([plane(lam_w) for lam_w in lam])
    anchors = jnp.array([[0.0, 0.0]]) if field_xy is None else jnp.asarray(field_xy)
    n_field = anchors.shape[0]
    templates = jnp.broadcast_to(planes[None], (n_field, *planes.shape))
    return PsfletPack(
        templates=templates,
        offsets=offsets,
        wavelengths_nm=lam,
        field_xy=anchors,
        template_origin=TEMPLATE_ORIGIN,
        centroids=centroids,
        meta_json=meta_json,
    )


def nearest_field_idx(pack, positions):
    """Index of each lenslet's nearest field anchor, shape ``(n_channels,)``.

    ``positions`` are lenslet-index coordinates (the ``square_grid`` /
    ``hex_grid`` convention), matching ``PsfletPack.field_xy``.
    """
    d2 = ((positions[:, None, :] - pack.field_xy[None, :, :]) ** 2).sum(axis=-1)
    return jnp.argmin(d2, axis=1).astype(jnp.int32)


def _blend_planes(pack, wavelength_nm):
    """Linearly blend the two tabulated planes bracketing ``wavelength_nm``.

    Returns ``(n_field, n_off, n_off)``. The caller guarantees the wavelength
    lies inside the tabulated range; the blend weight is clipped so exact
    endpoints stay exact.
    """
    if len(pack.wavelengths_nm) == 1:
        return pack.templates[:, 0]
    lam_grid = pack.wavelengths_nm.array()
    i = jnp.clip(
        jnp.searchsorted(lam_grid, wavelength_nm, side="right") - 1,
        0,
        lam_grid.shape[0] - 2,
    )
    frac = jnp.clip(
        (wavelength_nm - lam_grid[i]) / (lam_grid[i + 1] - lam_grid[i]), 0.0, 1.0
    )
    return (1.0 - frac) * pack.templates[:, i] + frac * pack.templates[:, i + 1]


def _bilinear(templates, field_idx, dx, dy, offsets):
    """Bilinear-sample per-channel template planes at pixel offsets.

    ``templates`` is ``(n_field, n_off, n_off)`` (one blended plane per
    anchor); ``dx, dy`` are ``(n_channels, n_psf)`` offsets; ``field_idx``
    selects each channel's plane. Implemented as four corner gathers so no
    per-channel template array is ever materialized. Offsets outside the
    tabulated extent contribute zero (the pack defines the wings).
    """
    origin = offsets[0]
    step = offsets[1] - offsets[0]
    n_off = offsets.shape[0]
    gu = (dx - origin) / step
    gv = (dy - origin) / step
    iu0 = jnp.floor(gu)
    iv0 = jnp.floor(gv)
    fu = gu - iu0
    fv = gv - iv0

    def corner(iv, iu, w):
        in_bounds = (iu >= 0) & (iu < n_off) & (iv >= 0) & (iv < n_off)
        iuc = jnp.clip(iu, 0, n_off - 1).astype(jnp.int32)
        ivc = jnp.clip(iv, 0, n_off - 1).astype(jnp.int32)
        vals = templates[field_idx[:, None], ivc, iuc]
        return jnp.where(in_bounds, vals * w, 0.0)

    return (
        corner(iv0, iu0, (1.0 - fu) * (1.0 - fv))
        + corner(iv0, iu0 + 1.0, fu * (1.0 - fv))
        + corner(iv0 + 1.0, iu0, (1.0 - fu) * fv)
        + corner(iv0 + 1.0, iu0 + 1.0, fu * fv)
    )


def template_weights(pack, dx, dy, wavelength_nm, field_idx, smear_px, n_sub=5):
    """Template-mode PSFlet footprint weights with LSF smear along x.

    The template plane is blended in wavelength, then bilinear-sampled at the
    footprint offsets, averaged over ``n_sub`` centroid shifts spanning the
    wavelength bin's spectral-pixel extent -- the same smear model as the
    analytic :func:`coronachrome.psflet.psflet_weights`.

    Args:
        pack: A :class:`PsfletPack`.
        dx: ``(n_channels, n_psf)`` spectral-axis offsets from the centroid [px].
        dy: ``(n_channels, n_psf)`` spatial-axis offsets from the centroid [px].
        wavelength_nm: Scalar wavelength.
        field_idx: ``(n_channels,)`` field-anchor index per channel.
        smear_px: Spectral extent of the wavelength bin [px].
        n_sub: Number of sub-samples used to integrate across the bin.

    Returns:
        Weights with the same shape as ``dx``.
    """
    templates = _blend_planes(pack, wavelength_nm)
    shifts = jnp.linspace(-0.5 * smear_px, 0.5 * smear_px, n_sub)

    def one(shift):
        return _bilinear(templates, field_idx, dx - shift, dy, pack.offsets)

    return jnp.mean(jax.vmap(one)(shifts), axis=0)
