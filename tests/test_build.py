"""Build-time Band, grid and layout contracts."""

import re
from contextlib import nullcontext

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, focal_image, sky_grid
from hwoutils.leaves import canonical
from hwoutils.numerics import sum_roundoff
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from optixstuff import TransmissiveElement
from optixstuff.disperser import LensletDisperser
from optixstuff.grids import Frame, Measure, PixelGrid, SkyOrientation
from optixstuff.psflet_pack import TEMPLATE_ORIGIN, PsfletPack
from optixstuff.spectral import Band, Quadrature, Transmission

from coronachrome import IFSRenderer, LensletLayout, build_ir, spatial_sample
from coronachrome.build import (
    _band_response,
    detector_centroids,
    lenslet_cell_centers,
    lenslet_positions,
)


def _disp(n=6, **overrides):
    kwargs = dict(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=0.0,
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=n,
        psflet_kind="gaussian",
        detector_shape=(128, 128),
        sky_pitch_arcsec=0.2,
    )
    kwargs.update(overrides)
    return LensletDisperser(**kwargs)


def _build(d=None, **kw):
    return build_ir(
        _disp() if d is None else d, bands_at([640, 660, 680]), sky_grid((32, 32)), **kw
    )


@pytest.mark.float64_only(reason="precision_pair and jaxpr audit require x64")
@pytest.mark.parametrize("grid_kind", ["square", "hex"])
@pytest.mark.parametrize("entry", ["lenslet_positions", "lenslet_cell_centers"])
def test_lenslet_geometry_precision_closure(grid_kind, entry):
    """Allow 16 rounding units for stagger, rotation and pixel conversion.

    The grid is axis-aligned with a 2:1 pitch ratio. The hex stagger adds
    one square root and a multiply. For the focal centers allow one sine
    and cosine, two products and a sum for rotation, scaling and the grid
    inversion, all at the peak coordinate scale.
    """
    args = (_disp(2, grid_kind=grid_kind, angle_rad=0.25),)
    grid = sky_grid((6, 6))

    def evaluate(model):
        if entry == "lenslet_positions":
            return lenslet_positions(model)
        return lenslet_cell_centers(model, grid)

    assert_precision_closed(evaluate, *canonical(args, precision="float32"))
    precision_pair(evaluate, args, bound=float(ulps(16, np.float32(1))))


@pytest.mark.float64_only(reason="precision_pair and jaxpr audit require x64")
def test_band_response_precision_closure():
    """Allow 24 rounding units for the normalized band-constant response.

    The two bands have six and three active nodes, respectively. Allow
    six units for three positive footprint weights and their normalization,
    six for throughput and Gauss weights, and twelve for two six-term sums
    and the final division. One tap in each footprint is off the detector.
    """
    bands = (Band(650, 660), Band(660, 670))
    rule = Quadrature.build(bands, breaks_nm=(656.0,), order=3)
    profiles = jnp.linspace(0.25, 1.0, 36).reshape(1, 2, 6, 3)
    valid = jnp.array([[[True, False, True], [False, True, True]]])
    throughput = jnp.linspace(0.5, 1.0, 12).reshape(2, 6)

    args = (rule, profiles, valid, throughput)
    assert_precision_closed(_band_response, *canonical(args, precision="float32"))
    precision_pair(_band_response, args, bound=float(ulps(24, np.float32(1))))


@pytest.mark.float64_only(reason="build a float32 IR inside the x64 lane")
def test_ir_build_preserves_model_precision():
    """Store narrow operator leaves and retain central-footprint agreement.

    On this one-lenslet, three-node fixture the trace stays within one
    central detector pixel. Allow eight units for wavelength-to-centroid
    and width scaling, 32 for the erf profile and 24 for normalizing nine
    taps and averaging them over the band, at the peak tap value.
    """
    model = _disp(
        1,
        angle_rad=0.25,
        detector_shape=(9, 9),
        dispersion_coeffs=jnp.array([10.0, 0.0]),
    )
    bands = (Band(650, 670),)
    grid = sky_grid((4, 4))

    def build(model):
        return build_ir(model, bands, grid, half=1, quadrature_order=3)

    narrow = build(canonical(model, precision="float32"))
    wide = build(model)
    assert narrow.det_vals.dtype == np.float32
    assert narrow.collection_overlap.sparse[2].dtype == np.float32
    np.testing.assert_array_equal(narrow.det_rows, wide.det_rows)
    precision_pair(
        lambda model: build(model).det_vals,
        (model,),
        bound=float(ulps(64, np.float32(1))),
    )


def test_ir_build_keeps_nodes_below_filter_support_dark():
    """Both bands have exact integrated responses because the filter is flat.

    Every Gauss node in the first band is below 650 nm, even the highest
    node, which rounds to 650 nm in float32. All nodes in the second band
    have unit transmission. Allow 16 units for normalizing and summing taps.
    """
    model = canonical(
        _disp(
            1,
            detector_shape=(9, 9),
            throughput_element=TransmissiveElement(
                Transmission([650.0, 660.0], [1.0, 1.0])
            ),
        ),
        precision="float32",
    )
    bands = (Band(649.9992, 650.0), Band(650.0, 650.0008))
    ir = build_ir(model, bands, sky_grid((5, 5)), half=1, quadrature_order=6)
    assert ir.det_vals.dtype == np.float32
    np.testing.assert_array_equal(ir.det_vals[0, 0], 0)
    np.testing.assert_allclose(
        ir.det_vals[0, 1].sum(), 1, rtol=0, atol=float(ulps(16, np.float32(1)))
    )


@pytest.mark.parametrize("above_tie", [False, True])
def test_ir_build_rounds_host_centroid(above_tie):
    """Round 4.5 to even, but move one column at the next float32 shift.

    The 9x9 detector is centered at 4. A half-width of one includes three
    columns centered at 4 or 5 depending on the exact host sum.
    """
    shift = np.float32(0.5)
    if above_tie:
        shift = np.nextafter(shift, np.float32(1))
    model = canonical(
        _disp(1, detector_shape=(9, 9), dispersion_coeffs=jnp.array([shift])),
        precision="float32",
    )
    ir = build_ir(model, (Band(659, 661),), sky_grid((5, 5)), half=1)
    cols = np.arange(3, 6) + above_tie
    np.testing.assert_array_equal(ir.det_rows[0, 0] % 9, np.tile(cols, 3))


def test_ir_build_accepts_legacy_layout_override():
    """A custom quarter-lenslet shift places the middle tap one pixel right."""

    class CustomLayout(LensletLayout):
        def positions(self):
            return super().positions() + jnp.array([[0.25, 0.0]])

    model = _disp(
        1, pitch_m=52e-6, detector_shape=(9, 9), dispersion_coeffs=jnp.zeros(1)
    )
    layout = CustomLayout(1)
    np.testing.assert_array_equal(
        lenslet_positions(model, layout=layout), [[0.25, 0.0]]
    )
    ir = build_ir(model, (Band(659, 661),), sky_grid((5, 5)), half=1, layout=layout)
    assert int(ir.det_rows[0, 0, 4]) == 4 * 9 + 5


class _OptixstuffDispersionWidening(AssertionError):
    """The known widening at the optixstuff dispersion input cast."""


@pytest.mark.float64_only(reason="precision_pair and jaxpr audit require x64")
@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize(
    "closed_dispersion",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=_OptixstuffDispersionWidening,
                reason="optixstuff dispersion_px widens the input to float64",
            ),
        ),
    ],
    ids=["closed-dispersion", "optixstuff-dispersion"],
)
def test_detector_centroids_precision_closure(
    corrected, closed_dispersion, monkeypatch
):
    """Allow 16 units for the trace geometry and two-plane centroid blend.

    A constant dispersion shifts the central lenslet to (4.25, 4).
    The binary32 query is exactly 2**-11 nm above the first knot in a
    0.001 nm interval. Rounding the upper knot would move the blend to 1/2.
    """
    if closed_dispersion:
        monkeypatch.setattr(
            LensletDisperser,
            "dispersion_px",
            lambda self, wave: jnp.full_like(wave, self.dispersion_coeffs[0]),
        )
    model = _disp(
        1,
        detector_shape=(9, 9),
        psflet_kind="template",
        dispersion_coeffs=jnp.array([0.25]),
    )
    pack = PsfletPack(
        jnp.ones((1, 2, 2, 2)),
        jnp.array([-1.0, 1.0]),
        [650.0, 650.001],
        jnp.zeros((1, 2)),
        TEMPLATE_ORIGIN,
        centroids=jnp.array([[[0.25, -0.5], [0.75, 0.0]]]),
    )
    args = (model, jnp.array([650.0 + 2**-11]), pack)

    def evaluate(model, wave, pack):
        return detector_centroids(model, wave, psflet_pack=pack, corrected=corrected)

    frac = 2**-11 / 0.001
    expected = (
        [[[4.5 + 0.5 * frac]], [[3.5 + 0.5 * frac]]]
        if corrected
        else [[[4.25]], [[4.0]]]
    )
    bound = float(ulps(16, np.float32(1)))
    np.testing.assert_allclose(evaluate(*args), expected, rtol=bound, atol=0)
    precision_pair(evaluate, args, bound=bound)
    try:
        assert_precision_closed(evaluate, *canonical(args, precision="float32"))
    except AssertionError as error:
        if not closed_dispersion and re.fullmatch(
            r".*/optixstuff/disperser\.py:155:20 \(LensletDisperser\.dispersion_px\)"
            r": precision widened to float64",
            str(error),
        ):
            raise _OptixstuffDispersionWidening(str(error)) from error
        raise


def _rms(ir, w):
    x = (ir.det_rows[0, w] % ir.det_shape[1]).astype(float)
    g = ir.det_vals[0, w]
    mean = (x * g).sum() / g.sum()
    return jnp.sqrt(((x - mean) ** 2 * g).sum() / g.sum())


def test_psflet_width_scales_linearly_with_wavelength():
    """Fails if diffraction scaling is lost.

    Pixel integration allows a 2% discretization
    band.
    """
    ir = build_ir(
        _disp(
            1,
            psflet_params=jnp.array([2.0]),
            psflet_ref_nm=500,
            dispersion_coeffs=jnp.array([0.0]),
        ),
        bands_at([500, 1000]),
        sky_grid((16, 16)),
        half=20,
    )
    assert jnp.allclose(_rms(ir, 1) / _rms(ir, 0), 2, rtol=0.02)


def test_moffat_scales_core_width_not_shape():
    """Fails if beta scales; finite-wing truncation and pixel variance allow 3%."""
    ir = build_ir(
        _disp(
            1,
            psflet_kind="moffat",
            psflet_params=jnp.array([2.0, 3.0]),
            psflet_ref_nm=500,
            dispersion_coeffs=jnp.array([0.0]),
        ),
        bands_at([500, 1000]),
        sky_grid((16, 16)),
        half=30,
    )
    assert jnp.allclose(_rms(ir, 1) / _rms(ir, 0), 2, rtol=0.03)


def test_build_ir_returns_ir_with_expected_shapes():
    """The sparse operator dimensions match the supplied bands and layout."""
    ir = _build()
    assert ir.det_rows.shape[:2] == (36, 3)
    assert ir.fp_grid == sky_grid((32, 32))
    assert ir.detector_grid.unit == "meter"


def _footprint_sum_roundoff(ir, expected):
    """Budget two stamp sums, two five-node sums, and five scalar roundings."""
    unit = jnp.asarray(expected, dtype=ir.det_vals.dtype)
    return float(
        2 * sum_roundoff(unit, ir.det_vals.shape[-1])
        + 2 * sum_roundoff(unit, 5)
        + ulps(5, unit)
    )


def test_det_vals_normalized_per_channel_wavelength():
    """Unit flux within two stamp sums, two node sums, and five roundings.

    Scalar operations are profile division, weight storage, throughput and
    quadrature products, and division by the integrated band width.
    """
    ir = _build()
    np.testing.assert_allclose(
        ir.det_vals.sum(axis=2), 1, atol=_footprint_sum_roundoff(ir, 1), rtol=0
    )


def test_indices_in_bounds():
    """Masked footprints never index outside the detector."""
    ir = _build()
    assert int(ir.det_rows.min()) >= 0
    assert int(ir.det_rows.max()) < 128**2


def test_unregistered_type_raises():
    """Unregistered descriptors fail explicitly."""
    with pytest.raises(NotImplementedError):
        build_ir(object(), (Band(650, 670),), sky_grid((8, 8)))


def test_moffat_psflet_build_normalizes():
    """Moffat unit flux allows two stamp sums, two node sums, and five roundings."""
    ir = _build(_disp(psflet_kind="moffat", psflet_params=jnp.array([1.5, 2.5])))
    np.testing.assert_allclose(
        ir.det_vals.sum(axis=2), 1, atol=_footprint_sum_roundoff(ir, 1), rtol=0
    )


def test_off_detector_footprints_warn():
    """Lost traces produce a diagnostic."""
    with pytest.warns(UserWarning, match="off the detector"):
        _build(_disp(detector_shape=(8, 8)))


def test_derived_sampling_matches_explicit_ratio():
    """Fails if pitch/scale units drift; 0.2 arcsec / 0.1 arcsec = 2, exact algebra."""
    ir = _build()
    assert ir.fp_pixel_scale_arcsec == 0.1
    assert ir.fp_px_per_lenslet == 2


def test_sampling_knobs_mutually_exclusive():
    """Removed independent sampling keywords must not survive as aliases."""
    with pytest.raises(TypeError):
        _build(fp_px_per_lenslet=2)
    with pytest.raises(TypeError):
        _build(fp_pixel_scale_arcsec=0.1)


def test_sampling_requires_one_knob():
    """A focal plane has to supply physical grid geometry."""
    with pytest.raises(AttributeError):
        build_ir(_disp(), (Band(650, 670),), (32, 32))


def test_derived_sampling_requires_descriptor_pitch():
    """An unspecified sky pitch cannot define a physical layout."""
    with pytest.raises(ValueError, match="sky_pitch_arcsec"):
        _build(_disp(sky_pitch_arcsec=None))


def test_undersampled_cube_warns():
    """A pitch below two source pixels produces a diagnostic."""
    with pytest.warns(UserWarning, match="undersamples"):
        build_ir(_disp(), (Band(650, 670),), sky_grid((32, 32), 0.2))


def test_lenslet_cells_beyond_cube_warn():
    """Cropping produces a diagnostic."""
    with pytest.warns(UserWarning, match="past the focal-plane"):
        build_ir(_disp(), (Band(650, 670),), sky_grid((4, 4)))


def test_wavelength_edges_control_lsf_smear():
    """Fails if Band widths are ignored; 40 nm trace smear exceeds 1 nm by 2x."""
    d = _disp(1)
    narrow = build_ir(d, (Band(659.5, 660.5),), sky_grid((16, 16)), half=10)
    wide = build_ir(d, (Band(640, 680),), sky_grid((16, 16)), half=10)
    assert _rms(wide, 0) > 2 * _rms(narrow, 0)


def test_wavelength_edges_match_gradient_on_uniform_grid():
    """Constant throughput allows its cast, two stamp and node sums, and five ops."""
    from optixstuff import ConstantThroughput

    ir = build_ir(
        _disp(1, throughput_element=ConstantThroughput(0.3)),
        (Band(600, 700),),
        sky_grid((16, 16)),
        half=12,
    )
    tol = _footprint_sum_roundoff(ir, 0.3) + float(ulps(1, ir.det_vals.dtype.type(0.3)))
    np.testing.assert_allclose(ir.det_vals.sum(), 0.3, atol=tol, rtol=0)


def test_wavelength_edges_shape_validated():
    """Bare wavelength arrays are rejected instead of assuming a bin width."""
    with pytest.raises(AttributeError):
        build_ir(_disp(), jnp.array([660.0]), sky_grid((32, 32)))


def test_resampling_footprint_conserves_flux():
    """Fails if the input measure is ignored.

    Flat 2px cells collect 4, exact
    algebra.
    """
    ir = _build()
    z = spatial_sample(focal_image(jnp.ones((3, 32, 32)), ir.fp_grid), ir)
    assert jnp.allclose(z, 4, atol=2e-14, rtol=0)


def test_resampling_footprint_axis_aligned_box():
    """Fails if deposition splits a pixel wholly contained in one aperture.

    Piecewise-constant collection is used for smooth declarations too.
    """
    ir = _build(_disp(2))
    data = jnp.zeros((3, 32, 32)).at[:, 15, 15].set(1)
    near = spatial_sample(focal_image(data, ir.fp_grid), ir)
    linear = spatial_sample(focal_image(data, ir.fp_grid, reconstruction="linear"), ir)
    assert jnp.allclose(near.sum(axis=0), 1, atol=1e-15, rtol=0)
    assert jnp.allclose(
        near,
        jnp.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        atol=1e-15,
        rtol=0,
    )
    assert jnp.array_equal(near, linear)


def test_resampling_footprint_multi_lenslet():
    """Fails if orientation is dropped; declared sky orientation is retained exactly."""
    grid = sky_grid((32, 32), orientation=SkyOrientation(37, -1))
    ir = build_ir(_disp(), (Band(650, 670),), grid)
    assert ir.lenslet_grid.orientation == grid.orientation
    assert (
        IFSRenderer(ir).channel_cube(jnp.ones((36, 1)), "e").grid.orientation
        == grid.orientation
    )


def test_resampling_footprint_rotation_moves_corners():
    """Fails if the layout rotation is dropped.

    Typed quarter-turn coordinates at the floating-point floor.
    """
    cx, cy = lenslet_cell_centers(
        _disp(2, angle_rad=float(jnp.pi / 2)), sky_grid((16, 16))
    )
    assert jnp.allclose(cx, jnp.array([8.5, 6.5, 8.5, 6.5]), atol=2e-15, rtol=0)
    assert jnp.allclose(cy, jnp.array([6.5, 6.5, 8.5, 8.5]), atol=2e-15, rtol=0)


def test_resampling_footprint_masks_off_grid():
    """Fails if cropping invents flux; a source outside the lenslet field gives zero."""
    ir = _build(_disp(2))
    z = spatial_sample(
        focal_image(jnp.zeros((3, 32, 32)).at[:, 0, 0].set(1), ir.fp_grid), ir
    )
    assert jnp.array_equal(z, jnp.zeros_like(z))


def test_throughput_scales_footprint_sums():
    """Fails if throughput is sampled only at reference.

    Knot-split triangle integral at the floating-point floor.
    """
    curve = TransmissiveElement(
        Transmission(jnp.array([600.0, 650.0, 700.0]), jnp.array([0.2, 0.8, 0.2]))
    )
    ir = build_ir(
        _disp(1, throughput_element=curve),
        (Band(600, 700),),
        sky_grid((16, 16)),
        half=12,
    )
    assert jnp.allclose(ir.det_vals.sum(), 0.5, atol=3e-15, rtol=0)


def test_build_ir_spatial_footprint_is_flux_conserving():
    """Anchor the per-steradian conversion to float64 geometry.

    The cell allows nine overlapping pixel contributions plus six roundings
    for weight storage, products and angle/area conversion. The detector total
    adds PSFlet normalization, one product, one contribution per channel-band
    per detector pixel, and the detector reduction depth.
    """
    ir = _build()
    image = focal_image(
        jnp.full((3, 32, 32), 100.0), ir.fp_grid, measure=Measure.density("sr")
    )
    # One arcsecond spans pi / 648000 radians. Each lenslet is 0.2" square.
    sr_per_arcsec2 = (np.pi / 648000.0) ** 2
    z = spatial_sample(image, ir)
    unit = jnp.ones((), dtype=z.dtype)
    cell_rtol = float(ulps(9 + 6, unit))
    np.testing.assert_allclose(z, 4 * sr_per_arcsec2, rtol=cell_rtol, atol=0)
    renderer = IFSRenderer(ir)
    detector = renderer.forward_spmv(image)
    assert detector.quantity == "e/s" and detector.measure == Measure.integral()
    detector_rtol = (
        cell_rtol
        + _footprint_sum_roundoff(ir, 1)
        + float(ulps(1 + ir.n_channels * ir.n_bands, unit))
        + float(sum_roundoff(unit, detector.data.size))
    )
    np.testing.assert_allclose(
        detector.data.sum(), 432 * sr_per_arcsec2, rtol=detector_rtol, atol=0
    )
    # lambda/D cannot cross the seam without an explicit Frame conversion.
    lod = PixelGrid.focal(32, 0.1, Frame(660, 6, "inscribed"))
    with pytest.raises(ValueError, match="Frame"):
        build_ir(_disp(), (Band(650, 670),), lod)
    converted = lod.to_arcsec(660)
    assert converted.frame == lod.frame
    assert eqx.is_array(z)


@pytest.mark.parametrize(("offset", "fraction"), [(-16, 0.5), (0, 1.0), (-24, 0.0)])
def test_gaussian_detector_edge(offset, fraction):
    """Symmetry gives half the flux at x=-0.5, within 1 ppm for finite wings."""
    disp = _disp(1, detector_shape=(32, 32), dispersion_coeffs=jnp.array([offset]))
    warning = pytest.warns(UserWarning, match="fell off the detector")
    with warning if fraction == 0 else nullcontext():
        ir = build_ir(disp, bands_at([660]), sky_grid((4, 4)), half=6)
    np.testing.assert_allclose(ir.det_vals.sum(axis=-1), fraction, rtol=1e-6, atol=0)


def test_uniform_array_clipped_flux():
    """Two of four lenslets straddle the edge, leaving three lenslets of flux.

    Relative roundoff covers stamp and detector sums, four lenslet deposits
    per pixel, and up to nine cell-overlap terms with four scalar roundings.
    The analytic half-flux check also includes PSFlet normalization.
    """
    disp = _disp(
        2,
        pitch_m=8 * 13e-6,
        detector_shape=(32, 32),
        dispersion_coeffs=jnp.array([-12.0]),
    )
    ir = build_ir(disp, bands_at([660]), sky_grid((8, 8)), half=6)
    image = focal_image(jnp.ones((1, 8, 8)), ir.fp_grid)
    counts = IFSRenderer(ir).forward_spmv(image).data.sum()
    flux_per_lenslet = 4.0  # Each square cell covers four unit-flux focal pixels.
    expected = (ir.det_vals.sum(axis=-1) * flux_per_lenslet).sum()
    unit = jnp.ones((), dtype=counts.dtype)
    tol = float(
        sum_roundoff(unit, np.prod(ir.det_shape))
        + sum_roundoff(unit, ir.det_vals.shape[-1])
        + sum_roundoff(unit, ir.n_channels)
        + ulps(ir.n_channels + 9 + 4, unit)
    )
    np.testing.assert_allclose(counts, expected, rtol=tol, atol=0)
    np.testing.assert_allclose(
        counts, 3 * flux_per_lenslet, rtol=tol + _footprint_sum_roundoff(ir, 1), atol=0
    )
