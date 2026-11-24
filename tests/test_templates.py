"""Tests for PSFlet template packs and the template-mode IR build."""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, sky_grid
from optixstuff import AbstractUniformElement
from optixstuff.disperser import LensletDisperser
from optixstuff.psflet_pack import PsfletPack, save_psflet_pack
from optixstuff.spectral import Band

from coronachrome.build import build_ir
from coronachrome.templates import (
    analytic_psflet_pack,
    template_weights,
)


def _disp(n=4, **overrides):
    kwargs = dict(
        pitch_m=174e-6,
        sky_pitch_arcsec=0.2,
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
    lam = lam.at[0].add(-1e-3).at[-1].add(1e-3)
    return analytic_psflet_pack(
        "gaussian", jnp.array([sigma]), lam, psflet_ref_nm=ref_nm, **kwargs
    )


def test_build_splits_at_template_and_throughput_breaks():
    """Fails if template or throughput breaks are omitted from the two-node rule.

    The right pixel has a unit-height triangular response peaking at 625 nm.
    Throughput is one up to 650 nm, then falls linearly to zero at 700 nm.
    In normalized band coordinates x = (lambda - 600) / 100, the red rolloff
    removes integral(4/3 * (1-x) * (2*x-1), x=.5..1) = 1/18 from the
    triangle area 1/2. The two nonzero pixels sum to the mean throughput 3/4.
    """

    class Filter(AbstractUniformElement):
        def spectral_breaks_nm(self) -> np.ndarray:
            return np.array([650.0])

        def get_throughput(self, wavelength_nm):
            return jnp.minimum(1.0, (700.0 - wavelength_nm) / 50.0)

    templates = jnp.zeros((1, 3, 3, 3))
    templates = templates.at[0, (0, 2), 1, 1].set(1).at[0, 1, 1, 2].set(1)
    pack = PsfletPack(
        templates,
        jnp.arange(-1, 2),
        jnp.array([600.0, 625.0, 700.0]),
        jnp.zeros((1, 2)),
    )
    ir = build_ir(
        _disp(
            1,
            psflet_kind="template",
            dispersion_coeffs=jnp.zeros(1),
            throughput_element=Filter(),
            detector_shape=(9, 9),
        ),
        (Band(600, 700),),
        sky_grid((16, 16)),
        half=1,
        psflet_pack=pack,
        quadrature_order=2,
    )
    expected = np.zeros(9)
    expected[4:6] = [11 / 36, 4 / 9]
    np.testing.assert_allclose(ir.det_vals[0, 0], expected, rtol=0, atol=2e-15)


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
    ir_g = build_ir(
        disp_g, fp_grid=sky_grid((32, 32), 0.2 / (2.0)), bands=bands_at(lam)
    )
    ir_t = build_ir(
        disp_t,
        psflet_pack=pack,
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(lam),
    )
    assert jnp.array_equal(ir_g.det_rows, ir_t.det_rows)
    assert jnp.allclose(ir_g.det_vals, ir_t.det_vals, atol=2e-3)
    assert ir_g.lenslet_grid == ir_t.lenslet_grid


def test_template_mode_requires_pack():
    """Template mode without a pack argument or descriptor path is an error."""
    with pytest.raises(ValueError, match="psflet_pack"):
        build_ir(
            _disp(4, psflet_kind="template"),
            fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
            bands=bands_at(jnp.array([660.0])),
        )


def test_template_mode_loads_descriptor_path(tmp_path):
    """The descriptor's psflet_pack_path is loaded when no pack is passed."""
    lam = jnp.array([640.0, 660.0, 680.0])
    pack = _gaussian_pack(lam)
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    disp = _disp(4, psflet_kind="template", psflet_pack_path=str(path))
    ir_from_path = build_ir(
        disp, fp_grid=sky_grid((32, 32), 0.2 / (2.0)), bands=bands_at(lam)
    )
    ir_from_arg = build_ir(
        disp,
        psflet_pack=pack,
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(lam),
    )
    assert jnp.allclose(ir_from_path.det_vals, ir_from_arg.det_vals)


def test_band_outside_pack_raises():
    """Wavelengths beyond the tabulated range must not extrapolate silently."""
    pack = _gaussian_pack(jnp.array([640.0, 680.0]))
    with pytest.raises(ValueError, match="tabulated"):
        build_ir(
            _disp(4, psflet_kind="template"),
            psflet_pack=pack,
            fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
            bands=bands_at(jnp.array([700.0])),
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
        disp,
        psflet_pack=pack_plain,
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(lam),
    )
    ir2 = build_ir(
        disp,
        psflet_pack=pack_corr,
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(lam),
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
    narrow = analytic_psflet_pack(
        "gaussian", jnp.array([0.7]), jnp.array([659.0, 661.0]), psflet_ref_nm=None
    )
    wide = analytic_psflet_pack(
        "gaussian", jnp.array([1.4]), jnp.array([659.0, 661.0]), psflet_ref_nm=None
    )
    pack = PsfletPack(
        templates=jnp.concatenate([narrow.templates, wide.templates], axis=0),
        offsets=narrow.offsets,
        wavelengths_nm=narrow.wavelengths_nm,
        field_xy=jnp.array([[-1.5, 0.0], [1.5, 0.0]]),
    )
    disp = _disp(4, psflet_kind="template", angle_rad=0.0)
    ir = build_ir(
        disp,
        psflet_pack=pack,
        half=6,
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(lam),
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
            psflet_pack=pack,
            half=6,
            fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
            bands=bands_at(jnp.array([660.0])),
        )


def test_one_wavelength_pack_builds_with_and_without_centroids():
    """A pack tabulated at one wavelength is constant in wavelength.

    Truth: its IR equals the IR of a two-wavelength pack whose planes are the
    same, to float round-off, and a +2 px centroid correction shifts the
    traces by two pixels, so the single-point grid is read through its array
    accessor like a multi-wavelength one.
    """
    single = _gaussian_pack(jnp.array([660.0]))
    assert len(single.wavelengths_nm) == 1
    double = PsfletPack(
        jnp.broadcast_to(single.templates, (1, 2, *single.templates.shape[2:])),
        single.offsets,
        jnp.array([659.0, 661.0]),
        single.field_xy,
    )
    disp = _disp(4, psflet_kind="template")
    kwargs = dict(
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)), bands=bands_at(jnp.array([660.0]))
    )
    ir_single = build_ir(disp, psflet_pack=single, **kwargs)
    ir_double = build_ir(disp, psflet_pack=double, **kwargs)
    assert jnp.allclose(ir_single.det_vals, ir_double.det_vals, rtol=0, atol=1e-12)
    shifted = PsfletPack(
        single.templates,
        single.offsets,
        single.wavelengths_nm,
        single.field_xy,
        centroids=jnp.zeros((1, 1, 2)).at[:, :, 0].set(2.0),
    )
    ir_shift = build_ir(disp, psflet_pack=shifted, **kwargs)
    nx = disp.detector_shape[1]
    x0 = (ir_single.det_rows % nx) * ir_single.det_vals
    x2 = (ir_shift.det_rows % nx) * ir_shift.det_vals
    mean_shift = (x2.sum(axis=2) - x0.sum(axis=2)) / ir_single.det_vals.sum(axis=2)
    assert jnp.allclose(mean_shift, 2.0, atol=0.05)
