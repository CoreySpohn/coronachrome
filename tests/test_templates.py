"""Tests for PSFlet template packs and the template-mode IR build."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, sky_grid
from hwoutils.leaves import canonical
from hwoutils.numerics import sum_roundoff
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff import AbstractUniformElement
from optixstuff.disperser import LensletDisperser
from optixstuff.psflet_pack import TEMPLATE_ORIGIN, PsfletPack, save_psflet_pack
from optixstuff.spectral import Band, merge_tolerance_nm

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


@pytest.mark.float64_only(reason="precision_pair and jaxpr audit require x64")
@pytest.mark.parametrize(
    ("n_waves", "host_wavelength"), [(1, False), (2, False), (2, True)]
)
def test_template_weights_precision_closure(n_waves, host_wavelength):
    """Allow 40 rounding units for a conditioned uniform template grid.

    The wavelength knots and query are exactly representable. Allow eight
    units for wavelength interpolation, eight for the shift and offset grid
    coordinates, 16 for the four bilinear products and their sum, and eight
    for the five-sample smear reduction. The scale is the peak sampled weight.
    """
    pack = analytic_psflet_pack(
        "gaussian",
        [0.75],
        [660.0] if n_waves == 1 else [640.0, 680.0],
        half_extent=2.0,
        step=0.25,
        psflet_ref_nm=660.0,
    )
    dx = jnp.array([[-0.75, -0.25, 0.125, 0.5]])
    dy = jnp.array([[0.5, 0.125, -0.25, -0.75]])

    def evaluate(pack, dx, dy, wave):
        wave = 660.0 if host_wavelength else wave
        return template_weights(pack, dx, dy, wave, jnp.zeros(1, dtype=jnp.int32), 0.25)

    args = (pack, dx, dy, jnp.array(660.0))
    assert_precision_closed(evaluate, *canonical(args, precision="float32"))
    precision_pair(evaluate, args, bound=float(ulps(40, np.float32(1))))


def test_close_template_wavelengths_use_local_differences():
    """Constant zero and one planes give the blend fraction at every pixel.

    The decimal wavelengths span 0.001 nm and the query is 0.000338 nm above
    the first knot. Allow 16 units for the blend, bilinear taps and smear.
    """
    pack = PsfletPack(
        jnp.broadcast_to(jnp.array([0.0, 1.0])[None, :, None, None], (1, 2, 2, 2)),
        jnp.array([-1.0, 1.0]),
        [650.0, 650.001],
        jnp.zeros((1, 2)),
        TEMPLATE_ORIGIN,
    )
    dx = jnp.zeros((1, 1))
    field = jnp.zeros(1, dtype=jnp.int32)

    def evaluate(pack, dx):
        return template_weights(pack, dx, dx, 650.000338, field, 0.125)

    args = (pack, dx)
    narrow = canonical(args, precision="float32")
    bound = float(ulps(16, np.float32(1)))
    for evaluate_fn in (evaluate, jax.jit(evaluate)):
        np.testing.assert_allclose(evaluate_fn(*narrow), 0.338, atol=0, rtol=bound)
    if dx.dtype == np.float64:
        assert_precision_closed(evaluate, *narrow)
        precision_pair(evaluate, args, bound=bound)


@pytest.mark.parametrize("wave_dtype", [np.float32, np.float64])
def test_traced_template_query_preserves_local_spacing(precision, wave_dtype):
    """A slope of 1000/nm exposes rounding of either decimal knot.

    The reference uses the actual query supplied to JAX in each lane. Allow
    eight rounding units for the fractional distance and four bilinear taps.
    """
    lo, hi = 650.00001, 650.00101
    wave = wave_dtype(650.000338)
    if precision == np.float32:
        wave = np.float32(wave)
    pack = PsfletPack(
        jnp.broadcast_to(
            jnp.array([0.0, 1.0], dtype=jnp.float32)[None, :, None, None],
            (1, 2, 2, 2),
        ),
        jnp.array([-1.0, 1.0], dtype=jnp.float32),
        [lo, hi],
        jnp.zeros((1, 2), dtype=jnp.float32),
        TEMPLATE_ORIGIN,
    )
    dx = jnp.zeros((1, 1), dtype=jnp.float32)
    field = jnp.zeros(1, dtype=jnp.int32)

    def evaluate(wave):
        return template_weights(pack, dx, dx, wave, field, 0.0, n_sub=1)

    expected = (float(wave) - lo) / (hi - lo)
    bound = float(ulps(8, wave.dtype.type(1)))
    for evaluate_fn in (evaluate, jax.jit(evaluate)):
        actual = evaluate_fn(wave)
        assert actual.dtype == wave.dtype
        np.testing.assert_allclose(actual, expected, rtol=bound, atol=0)


def test_template_numpy_offsets_match_jit():
    """A constant unit plane gives exact unit taps in either execution mode."""
    pack = PsfletPack(
        jnp.ones((1, 1, 2, 2), dtype=jnp.float32),
        jnp.array([-1.0, 1.0], dtype=jnp.float32),
        [650.0],
        jnp.zeros((1, 2), dtype=jnp.float32),
        TEMPLATE_ORIGIN,
    )
    dx = np.zeros((1, 1), dtype=np.float32)

    def evaluate(dx):
        return template_weights(pack, dx, dx, 650.0, jnp.zeros(1, dtype=jnp.int32), 0.0)

    for evaluate_fn in (evaluate, jax.jit(evaluate)):
        result = evaluate_fn(dx)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [[1.0]])


def test_build_splits_at_template_and_throughput_breaks():
    """Fails if template or throughput breaks are omitted from the two-node rule.

    The right pixel has a unit-height triangular response peaking at 625 nm.
    Throughput is one up to 650 nm, then falls linearly to zero at 700 nm.
    In normalized band coordinates x = (lambda - 600) / 100, the red rolloff
    removes integral(4/3 * (1-x) * (2*x-1), x=.5..1) = 1/18 from the
    triangle area 1/2. The two nonzero pixels sum to the mean throughput 3/4.
    The wavelength cast contributes one rounding at 700 nm times the sum
    of the template and throughput slope bounds. Add two six-node sums and
    twelve operations for interpolation, footprint normalization and weighting.
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
        template_origin=TEMPLATE_ORIGIN,
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
    one = jnp.ones((), dtype=ir.det_vals.dtype)
    tol = ulps(1, 700 * one) * (1 / 25 + 1 / 50)
    tol += 2 * sum_roundoff(one, 6) + ulps(12, one)
    actual = ir.det_vals[0, 0]
    np.testing.assert_array_equal(actual[expected == 0], 0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=float(tol))


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
    """The same calibrated pack gives identical rows and taps by either route."""
    lam = jnp.array([640.0, 660.0, 680.0])
    pack = _gaussian_pack(
        lam, centroids=jnp.array([[[0.2, 0.1], [0.3, 0.0], [0.4, -0.1]]])
    )
    path = tmp_path / "pack.npz"
    save_psflet_pack(path, pack)
    disp = canonical(
        _disp(4, psflet_kind="template", psflet_pack_path=str(path)),
        precision="float32",
    )
    ir_from_path = build_ir(
        disp, fp_grid=sky_grid((32, 32), 0.2 / (2.0)), bands=bands_at(lam)
    )
    ir_from_arg = build_ir(
        disp,
        psflet_pack=pack,
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(lam),
    )
    assert ir_from_path.det_vals.dtype == ir_from_arg.det_vals.dtype == np.float32
    np.testing.assert_array_equal(ir_from_path.det_rows, ir_from_arg.det_rows)
    np.testing.assert_array_equal(ir_from_path.det_vals, ir_from_arg.det_vals)


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


@pytest.mark.parametrize("edge_dtype", [np.float64, np.float32])
def test_band_accepts_its_own_decimal_pack_edges(edge_dtype):
    """Fails if pack metadata narrows or its own decimal band edges are refused."""
    edges = [600.3, 700.2]
    pack = analytic_psflet_pack(
        "gaussian", jnp.array([0.7]), edges, half_extent=1, step=0.5
    )
    np.testing.assert_array_equal(pack.wavelengths_nm.values, edges)
    ir = build_ir(
        _disp(1, psflet_kind="template", dispersion_coeffs=jnp.zeros(1)),
        (Band(*np.asarray(edges, dtype=edge_dtype)),),
        sky_grid((16, 16)),
        psflet_pack=pack,
        half=1,
    )
    assert ir.n_bands == 1


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


def test_one_wavelength_pack_is_monochromatic():
    """A one-wavelength pack is valid only for a band at its own wavelength.

    Truth: the merge tolerance at 660 nm is four float32 ULPs, 2.4e-4 nm. A
    band 1.5 tolerances wide is at the pack wavelength (both edges within the
    tolerance of it) and builds; a band of 0.1 nm, hundreds of tolerances
    wide, or one centred 1 nm away, is refused, naming the pack wavelength.
    """
    single = _gaussian_pack(jnp.array([660.0]))
    disp = _disp(4, psflet_kind="template")
    grid = sky_grid((32, 32), 0.2 / (2.0))
    tol = float(merge_tolerance_nm(660.0))
    build_ir(disp, psflet_pack=single, fp_grid=grid, bands=bands_at([660.0], 1.5 * tol))
    for center, width in ((660.0, 0.1), (661.0, 1.5 * tol)):
        with pytest.raises(ValueError, match="one wavelength, 660 nm"):
            build_ir(
                disp,
                psflet_pack=single,
                fp_grid=grid,
                bands=bands_at([center], width),
            )


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
        template_origin=TEMPLATE_ORIGIN,
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
        template_origin=TEMPLATE_ORIGIN,
    )
    disp = _disp(4, psflet_kind="template")
    kwargs = dict(
        fp_grid=sky_grid((32, 32), 0.2 / (2.0)),
        bands=bands_at(
            jnp.array([660.0]), width_nm=1.5 * float(merge_tolerance_nm(660.0))
        ),
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
        template_origin=TEMPLATE_ORIGIN,
    )
    ir_shift = build_ir(disp, psflet_pack=shifted, **kwargs)
    nx = disp.detector_shape[1]
    x0 = (ir_single.det_rows % nx) * ir_single.det_vals
    x2 = (ir_shift.det_rows % nx) * ir_shift.det_vals
    mean_shift = (x2.sum(axis=2) - x0.sum(axis=2)) / ir_single.det_vals.sum(axis=2)
    assert jnp.allclose(mean_shift, 2.0, atol=0.05)


def test_zero_template_on_detector_is_not_clipping():
    """A dark template fully on the detector causes no clipping warning."""
    pack = PsfletPack(
        templates=jnp.zeros((1, 2, 3, 3)),
        offsets=jnp.arange(-1, 2),
        wavelengths_nm=[650, 670],
        field_xy=jnp.zeros((1, 2)),
        template_origin=TEMPLATE_ORIGIN,
    )
    ir = build_ir(
        _disp(1, psflet_kind="template"),
        (Band(650, 670),),
        sky_grid((4, 4)),
        half=1,
        psflet_pack=pack,
        quadrature_order=1,
    )
    assert not np.any(ir.det_vals)
