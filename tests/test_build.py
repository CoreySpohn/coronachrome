"""Build-time Band, grid and layout contracts."""

from contextlib import nullcontext

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, focal_image, sky_grid
from optixstuff import TransmissiveElement
from optixstuff.disperser import LensletDisperser
from optixstuff.grids import Frame, Measure, PixelGrid, SkyOrientation
from optixstuff.spectral import Band, Transmission

from coronachrome import IFSRenderer, build_ir, spatial_sample
from coronachrome.build import lenslet_cell_centers


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


def test_det_vals_normalized_per_channel_wavelength():
    """Fails if bin normalization changes integrated flux; floating-point floor."""
    assert jnp.allclose(_build().det_vals.sum(axis=2), 1, atol=2e-15, rtol=0)


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
    """Fails if Moffat quadrature loses bin flux; floating-point floor."""
    ir = _build(_disp(psflet_kind="moffat", psflet_params=jnp.array([1.5, 2.5])))
    assert jnp.allclose(ir.det_vals.sum(axis=2), 1, atol=2e-15, rtol=0)


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
    """Fails if spectral integration aliases: constant throughput integrates exactly."""
    from optixstuff import ConstantThroughput

    ir = build_ir(
        _disp(1, throughput_element=ConstantThroughput(0.3)),
        (Band(600, 700),),
        sky_grid((16, 16)),
        half=12,
    )
    assert jnp.allclose(ir.det_vals.sum(), 0.3, atol=2e-16, rtol=0)


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
    """Fails if an angular footprint treats per-steradian data as per arcsec squared."""
    ir = _build()
    image = focal_image(
        jnp.full((3, 32, 32), 100.0), ir.fp_grid, measure=Measure.density("sr")
    )
    # One arcsecond spans pi / 648000 radians. Each lenslet is 0.2" square.
    sr_per_arcsec2 = (jnp.pi / 648000.0) ** 2
    z = spatial_sample(image, ir)
    assert jnp.allclose(z, 4 * sr_per_arcsec2, rtol=7e-15, atol=0)
    renderer = IFSRenderer(ir)
    detector = renderer.forward_spmv(image)
    assert detector.quantity == "e/s" and detector.measure == Measure.integral()
    assert jnp.allclose(detector.data.sum(), 432 * sr_per_arcsec2, rtol=3e-15, atol=0)
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
    """Two of four lenslets straddle the edge, leaving three lenslets of flux."""
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
    np.testing.assert_allclose(counts, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(counts, 3 * flux_per_lenslet, rtol=1e-12, atol=0)
