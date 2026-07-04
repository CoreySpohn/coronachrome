"""Tests for PSFlet template packs and the template-mode IR build."""

import jax.numpy as jnp
import numpy as np
import pytest
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.templates import (
    PsfletPack,
    analytic_psflet_pack,
    load_psflet_pack,
    save_psflet_pack,
    template_weights,
)


def _disp(n=4, **overrides):
    kwargs = dict(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=n,
        psflet_kind="gaussian",
        detector_shape=(256, 256),
    )
    kwargs.update(overrides)
    return LensletDisperser(**kwargs)


def _gaussian_pack(lam, sigma=0.7, ref_nm=660.0, **kwargs):
    return analytic_psflet_pack(
        "gaussian", jnp.array([sigma]), lam, psflet_ref_nm=ref_nm, **kwargs
    )


def test_pack_roundtrip(tmp_path):
    """save/load round-trips every pack field, including provenance."""
    lam = jnp.array([640.0, 660.0, 680.0])
    centroids = jnp.zeros((1, 3, 2)).at[:, :, 0].set(1.5)
    pack = _gaussian_pack(lam, centroids=centroids, meta_json='{"generator": "test"}')
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    loaded = load_psflet_pack(path)
    assert jnp.allclose(loaded.templates, pack.templates)
    assert jnp.allclose(loaded.offsets, pack.offsets)
    assert jnp.allclose(loaded.wavelengths_nm, pack.wavelengths_nm)
    assert jnp.allclose(loaded.field_xy, pack.field_xy)
    assert jnp.allclose(loaded.centroids, pack.centroids)
    assert loaded.meta_json == pack.meta_json


def test_pack_roundtrip_without_centroids(tmp_path):
    """A pack without centroid corrections loads with centroids=None."""
    pack = _gaussian_pack(jnp.array([640.0, 680.0]))
    assert pack.centroids is None
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    assert load_psflet_pack(path).centroids is None


def test_pack_validation_rejects_bad_shapes():
    """Malformed packs fail at construction, not deep inside a build."""
    lam = jnp.array([640.0, 680.0])
    good = _gaussian_pack(lam)
    with pytest.raises(ValueError, match="templates"):
        PsfletPack(
            templates=good.templates[0],
            offsets=good.offsets,
            wavelengths_nm=lam,
            field_xy=good.field_xy,
        )
    with pytest.raises(ValueError, match="uniform"):
        PsfletPack(
            templates=good.templates[:, :, :3, :3],
            offsets=jnp.array([0.0, 1.0, 3.0]),
            wavelengths_nm=lam,
            field_xy=good.field_xy,
        )
    with pytest.raises(ValueError, match="ascending"):
        PsfletPack(
            templates=good.templates,
            offsets=good.offsets,
            wavelengths_nm=lam[::-1],
            field_xy=good.field_xy,
        )
    with pytest.raises(ValueError, match="centroids"):
        PsfletPack(
            templates=good.templates,
            offsets=good.offsets,
            wavelengths_nm=lam,
            field_xy=good.field_xy,
            centroids=jnp.zeros((2, 2, 2)),
        )


def test_template_mode_parity_with_gaussian():
    """An analytic Gaussian pack reproduces the analytic gaussian build.

    The pack tabulates the same erf pixel-integrated Gaussian (with the same
    linear chromatic width scaling) on a fine offset grid, so the template-mode
    IR must match the gaussian-mode IR to bilinear-interpolation tolerance.
    """
    lam = jnp.linspace(640.0, 680.0, 3)
    disp_g = _disp(4)
    disp_t = _disp(4, psflet_kind="template")
    pack = _gaussian_pack(lam, step=0.0625)
    ir_g = build_ir(disp_g, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0)
    ir_t = build_ir(
        disp_t, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0, psflet_pack=pack
    )
    assert jnp.array_equal(ir_g.det_rows, ir_t.det_rows)
    assert jnp.allclose(ir_g.det_vals, ir_t.det_vals, atol=2e-3)
    assert jnp.array_equal(ir_g.spatial_src, ir_t.spatial_src)


def test_template_mode_requires_pack():
    """Template mode without a pack argument or descriptor path is an error."""
    with pytest.raises(ValueError, match="psflet_pack"):
        build_ir(
            _disp(4, psflet_kind="template"),
            jnp.array([660.0]),
            fp_shape=(32, 32),
            fp_px_per_lenslet=2.0,
        )


def test_template_mode_loads_descriptor_path(tmp_path):
    """The descriptor's psflet_pack_path is loaded when no pack is passed."""
    lam = jnp.array([640.0, 660.0, 680.0])
    pack = _gaussian_pack(lam)
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    disp = _disp(4, psflet_kind="template", psflet_pack_path=str(path))
    ir_from_path = build_ir(disp, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0)
    ir_from_arg = build_ir(
        disp, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0, psflet_pack=pack
    )
    assert jnp.allclose(ir_from_path.det_vals, ir_from_arg.det_vals)


def test_band_outside_pack_raises():
    """Wavelengths beyond the tabulated range must not extrapolate silently."""
    pack = _gaussian_pack(jnp.array([640.0, 680.0]))
    with pytest.raises(ValueError, match="tabulated"):
        build_ir(
            _disp(4, psflet_kind="template"),
            jnp.array([700.0]),
            fp_shape=(32, 32),
            fp_px_per_lenslet=2.0,
            psflet_pack=pack,
        )


def test_template_weights_interpolate_between_planes():
    """Between tabulated planes the weights are the linear blend of the planes.

    The reference wavelength makes the two planes genuinely different (widths
    1.0 and 1.33 px), so the assertion discriminates plane blending from
    evaluating some intermediate-width profile.
    """
    pack = analytic_psflet_pack(
        "gaussian", jnp.array([1.0]), jnp.array([600.0, 800.0]), psflet_ref_nm=600.0
    )
    off = jnp.arange(-3.0, 4.0)
    dx, dy = jnp.meshgrid(off, off, indexing="xy")
    dx = dx.reshape(1, -1)
    dy = dy.reshape(1, -1)
    fi = jnp.zeros((1,), dtype=jnp.int32)
    w600 = template_weights(pack, dx, dy, 600.0, fi, 0.0)
    w800 = template_weights(pack, dx, dy, 800.0, fi, 0.0)
    w700 = template_weights(pack, dx, dy, 700.0, fi, 0.0)
    assert jnp.allclose(w700, 0.5 * (w600 + w800), rtol=1e-10)


def test_centroid_corrections_shift_footprints():
    """A constant +2 px x centroid correction moves the traces by two pixels."""
    lam = jnp.array([640.0, 660.0, 680.0])
    pack_plain = _gaussian_pack(lam)
    centroids = jnp.zeros((1, 3, 2)).at[:, :, 0].set(2.0)
    pack_corr = _gaussian_pack(lam, centroids=centroids)
    disp = _disp(4, psflet_kind="template")
    ir0 = build_ir(
        disp, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0, psflet_pack=pack_plain
    )
    ir2 = build_ir(
        disp, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0, psflet_pack=pack_corr
    )
    nx = disp.detector_shape[1]
    x0 = (ir0.det_rows % nx) * ir0.det_vals
    x2 = (ir2.det_rows % nx) * ir2.det_vals
    mean_shift = (x2.sum(axis=2) - x0.sum(axis=2)) / ir0.det_vals.sum(axis=2)
    assert jnp.allclose(mean_shift, 2.0, atol=0.05)


def test_nearest_field_anchor_selects_template():
    """Each lenslet uses its nearest field anchor's template.

    Anchor 0 (left half) tabulates a narrow PSFlet, anchor 1 (right half) a
    twice-wider one, so left-side channels must render narrower footprints.
    """
    lam = jnp.array([660.0])
    narrow = analytic_psflet_pack("gaussian", jnp.array([0.7]), lam, psflet_ref_nm=None)
    wide = analytic_psflet_pack("gaussian", jnp.array([1.4]), lam, psflet_ref_nm=None)
    pack = PsfletPack(
        templates=jnp.concatenate([narrow.templates, wide.templates], axis=0),
        offsets=narrow.offsets,
        wavelengths_nm=lam,
        field_xy=jnp.array([[-1.5, 0.0], [1.5, 0.0]]),
    )
    disp = _disp(4, psflet_kind="template", angle_rad=0.0)
    ir = build_ir(
        disp, lam, fp_shape=(32, 32), fp_px_per_lenslet=2.0, psflet_pack=pack, half=6
    )
    from coronachrome.grids import square_grid

    positions = square_grid(4)
    off = jnp.arange(-6.0, 7.0)
    _ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddx = ddx.reshape(-1)

    def rms_x(ch):
        w = ir.det_vals[ch, 0]
        m = (w * ddx).sum() / w.sum()
        return jnp.sqrt((w * (ddx - m) ** 2).sum() / w.sum())

    left = jnp.array([rms_x(c) for c in range(16) if positions[c, 0] < 0])
    right = jnp.array([rms_x(c) for c in range(16) if positions[c, 0] > 0])
    assert float(left.max()) < float(right.min())


def test_footprint_past_template_extent_warns():
    """A footprint reaching past the tabulated offsets warns about clipping."""
    pack = _gaussian_pack(jnp.array([640.0, 680.0]), half_extent=3.0)
    with pytest.warns(UserWarning, match="template extent"):
        build_ir(
            _disp(4, psflet_kind="template"),
            jnp.array([660.0]),
            fp_shape=(32, 32),
            fp_px_per_lenslet=2.0,
            psflet_pack=pack,
            half=6,
        )


def test_loader_rejects_unknown_format(tmp_path):
    """A pack file with an unknown format version fails loudly."""
    pack = _gaussian_pack(jnp.array([640.0, 680.0]))
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    data = dict(np.load(path, allow_pickle=False))
    data["format_version"] = np.int64(99)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="format"):
        load_psflet_pack(path)
