"""Tests for build_ir."""

import jax.numpy as jnp
import pytest
from optixstuff.disperser import LensletDisperser

from coronachrome.build import _resampling_footprints, build_ir
from coronachrome.ir import SpatialChannelIR


def _disp(n=6, **overrides):
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


def test_psflet_width_scales_linearly_with_wavelength():
    """The analytic PSFlet core width grows linearly with wavelength.

    A diffraction-limited spot scales as lambda f / D, so at a fixed detector
    pixel scale a single PSFlet at 1000 nm is twice as wide as at 500 nm in both
    axes. Zero dispersion is used so the LSF smear vanishes and both axes show
    pure diffraction width.
    """
    disp = LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=0.0,
        lam_ref_nm=500.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([0.0]),  # no dispersion -> zero LSF smear
        psflet_params=jnp.array([2.0]),
        psflet_ref_nm=500.0,
        grid_kind="square",
        n_lenslets=1,
        psflet_kind="gaussian",
        detector_shape=(128, 128),
    )
    half = 20
    lam = jnp.array([500.0, 1000.0])
    ir = build_ir(disp, lam, fp_shape=(16, 16), fp_px_per_lenslet=2.0, half=half)

    off = jnp.arange(-half, half + 1).astype(float)
    ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddx, ddy = ddx.reshape(-1), ddy.reshape(-1)

    def rms_widths(w):
        weights = ir.det_vals[0, w]
        wsum = weights.sum()
        mx = (weights * ddx).sum() / wsum
        my = (weights * ddy).sum() / wsum
        sx = jnp.sqrt((weights * (ddx - mx) ** 2).sum() / wsum)
        sy = jnp.sqrt((weights * (ddy - my) ** 2).sum() / wsum)
        return sx, sy

    sx500, sy500 = rms_widths(0)
    sx1000, sy1000 = rms_widths(1)
    assert jnp.allclose(sx1000 / sx500, 2.0, rtol=0.02)
    assert jnp.allclose(sy1000 / sy500, 2.0, rtol=0.02)


def test_moffat_scales_core_width_not_shape():
    """Moffat alpha (params[0]) scales with wavelength; beta (params[1]) does not.

    The second-moment width of a Moffat depends on both alpha and beta. If beta
    were (wrongly) scaled too, the width ratio would not equal the wavelength
    ratio; observing a 2x width ratio confirms only the core width scales.
    """
    disp = LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=0.0,
        lam_ref_nm=500.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([0.0]),
        psflet_params=jnp.array([2.0, 3.0]),  # alpha=2 px, beta=3
        psflet_ref_nm=500.0,
        grid_kind="square",
        n_lenslets=1,
        psflet_kind="moffat",
        detector_shape=(160, 160),
    )
    half = 40
    ir = build_ir(
        disp,
        jnp.array([500.0, 1000.0]),
        fp_shape=(16, 16),
        fp_px_per_lenslet=2.0,
        half=half,
    )
    off = jnp.arange(-half, half + 1).astype(float)
    ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddx, ddy = ddx.reshape(-1), ddy.reshape(-1)

    def rms(w):
        weights = ir.det_vals[0, w]
        wsum = weights.sum()
        sx = jnp.sqrt((weights * ddx**2).sum() / wsum)
        sy = jnp.sqrt((weights * ddy**2).sum() / wsum)
        return sx, sy

    sx500, sy500 = rms(0)
    sx1000, sy1000 = rms(1)
    assert jnp.allclose(sx1000 / sx500, 2.0, rtol=0.03)
    assert jnp.allclose(sy1000 / sy500, 2.0, rtol=0.03)


def test_build_ir_returns_ir_with_expected_shapes():
    """build_ir returns a SpatialChannelIR with expected channel/wav/detector shapes."""
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(_disp(6), lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)
    assert isinstance(ir, SpatialChannelIR)
    assert ir.n_channels == 36
    assert ir.n_wav == 5
    assert ir.det_rows.shape[:2] == (36, 5)
    assert ir.det_shape == (256, 256)


def test_det_vals_normalized_per_channel_wavelength():
    """On-detector PSFlet footprints are flux-normalized; off-detector ones are zero."""
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(_disp(6), lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)
    sums = ir.det_vals.sum(axis=2)
    nonzero = sums[sums > 1e-8]
    assert jnp.allclose(nonzero, 1.0, atol=1e-5)


def test_indices_in_bounds():
    """All detector row indices stay within the detector bounds."""
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(_disp(6), lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)
    ny, nx = ir.det_shape
    assert int(ir.det_rows.min()) >= 0
    assert int(ir.det_rows.max()) < ny * nx


def test_unregistered_type_raises():
    """build_ir raises NotImplementedError for an unregistered disperser type."""
    with pytest.raises(NotImplementedError):
        build_ir(object(), jnp.array([660.0]), fp_shape=(8, 8))


def test_moffat_psflet_build_normalizes():
    """A Moffat-PSFlet disperser builds an IR with flux-normalized footprints."""
    disp = LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([1.5, 2.5]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=6,
        psflet_kind="moffat",
        detector_shape=(256, 256),
    )
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(disp, lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)
    sums = ir.det_vals.sum(axis=2)
    nonzero = sums[sums > 1e-8]
    assert jnp.allclose(nonzero, 1.0, atol=1e-5)


def test_off_detector_footprints_warn():
    """Lenslets whose traces fall off a small detector trigger a warning."""
    disp = LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=8,
        psflet_kind="gaussian",
        detector_shape=(40, 40),
    )
    lam = jnp.linspace(640.0, 680.0, 5)
    with pytest.warns(UserWarning, match="off the detector"):
        build_ir(disp, lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)


def test_derived_sampling_matches_explicit_ratio():
    """fp_pixel_scale_arcsec derives the sampling from the descriptor pitch.

    A 0.5 arcsec on-sky pitch over a 0.1 arcsec per pixel cube grid is 5
    focal-plane pixels per lenslet, so the derived IR must be identical to the
    explicit fp_px_per_lenslet=5 build.
    """
    lam = jnp.linspace(640.0, 680.0, 3)
    d = _disp(6, sky_pitch_arcsec=0.5)
    ir_derived = build_ir(d, lam, fp_shape=(64, 64), fp_pixel_scale_arcsec=0.1)
    ir_explicit = build_ir(d, lam, fp_shape=(64, 64), fp_px_per_lenslet=5.0)
    assert jnp.array_equal(ir_derived.spatial_src, ir_explicit.spatial_src)
    assert jnp.allclose(ir_derived.spatial_w, ir_explicit.spatial_w)
    assert jnp.allclose(ir_derived.det_vals, ir_explicit.det_vals)


def test_sampling_knobs_mutually_exclusive():
    """Passing both the derived and the explicit sampling knob is an error."""
    d = _disp(6, sky_pitch_arcsec=0.5)
    with pytest.raises(ValueError, match="not both"):
        build_ir(
            d,
            jnp.array([660.0]),
            fp_shape=(64, 64),
            fp_px_per_lenslet=5.0,
            fp_pixel_scale_arcsec=0.1,
        )


def test_sampling_requires_one_knob():
    """Omitting both sampling inputs is an error, not a silent default."""
    with pytest.raises(ValueError, match="fp_pixel_scale_arcsec"):
        build_ir(_disp(6), jnp.array([660.0]), fp_shape=(64, 64))


def test_derived_sampling_requires_descriptor_pitch():
    """fp_pixel_scale_arcsec without a descriptor sky pitch is an error."""
    with pytest.raises(ValueError, match="sky_pitch_arcsec"):
        build_ir(
            _disp(6), jnp.array([660.0]), fp_shape=(64, 64), fp_pixel_scale_arcsec=0.1
        )


def test_undersampled_cube_warns():
    """Fewer than two cube pixels per lenslet degrades the cell flux integral."""
    with pytest.warns(UserWarning, match="undersamples"):
        build_ir(_disp(6), jnp.array([660.0]), fp_shape=(64, 64), fp_px_per_lenslet=1.0)


def test_lenslet_cells_beyond_cube_warn():
    """Lenslet cells reaching past the cube bounds lose flux and must warn."""
    with pytest.warns(UserWarning, match="past the focal-plane"):
        build_ir(_disp(6), jnp.array([660.0]), fp_shape=(16, 16), fp_px_per_lenslet=8.0)


def _rms_x(ir, half):
    """Weighted x-rms of channel 0's footprint at wavelength 0."""
    off = jnp.arange(-half, half + 1).astype(float)
    _ddy, ddx = jnp.meshgrid(off, off, indexing="ij")
    ddx = ddx.reshape(-1)
    w = ir.det_vals[0, 0]
    m = (w * ddx).sum() / w.sum()
    return float(jnp.sqrt((w * (ddx - m) ** 2).sum() / w.sum()))


def test_wavelength_edges_control_lsf_smear():
    """Explicit bin edges set the LSF smear exactly.

    A single-wavelength build without edges assumes a 1 nm bin; passing the
    real 40 nm bin edges must smear the PSFlet along the dispersion axis by a
    clearly larger amount (the boxcar over a 6 px bin dominates the 0.7 px
    core).
    """
    disp = _disp(1, angle_rad=0.0, detector_shape=(160, 160))
    half = 10
    lam = jnp.array([660.0])
    ir_default = build_ir(
        disp, lam, fp_shape=(16, 16), fp_px_per_lenslet=2.0, half=half
    )
    ir_edges = build_ir(
        disp,
        lam,
        fp_shape=(16, 16),
        fp_px_per_lenslet=2.0,
        half=half,
        wavelength_edges=jnp.array([640.0, 680.0]),
    )
    assert _rms_x(ir_edges, half) > 2.0 * _rms_x(ir_default, half)


def test_wavelength_edges_match_gradient_on_uniform_grid():
    """On a uniform center grid the edges path equals the gradient fallback."""
    lam = jnp.linspace(640.0, 680.0, 5)
    edges = jnp.linspace(635.0, 685.0, 6)
    ir_g = build_ir(_disp(6), lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)
    ir_e = build_ir(
        _disp(6),
        lam,
        fp_shape=(64, 64),
        fp_px_per_lenslet=2.0,
        wavelength_edges=edges,
    )
    assert jnp.allclose(ir_g.det_vals, ir_e.det_vals, atol=1e-12)


def test_wavelength_edges_shape_validated():
    """Edges must bracket every wavelength: n_wav + 1 entries required."""
    with pytest.raises(ValueError, match="wavelength_edges"):
        build_ir(
            _disp(6),
            jnp.array([650.0, 660.0]),
            fp_shape=(64, 64),
            fp_px_per_lenslet=2.0,
            wavelength_edges=jnp.array([640.0, 655.0, 670.0, 680.0]),
        )


def test_resampling_footprint_conserves_flux():
    """A fully interior lenslet integrates its cell area (weights sum to cell^2)."""
    fp_shape = (64, 64)
    cell = 6.0
    idx, w = _resampling_footprints(
        jnp.array([32.0]), jnp.array([32.0]), 0.0, cell, fp_shape, supersample=8
    )
    assert idx.shape == w.shape
    assert idx.shape[0] == 1
    assert jnp.allclose(w.sum(), cell**2, rtol=1e-6)
    cube = jnp.ones(fp_shape).reshape(-1)
    z = (cube[idx[0]] * w[0]).sum()
    assert jnp.allclose(z, cell**2, rtol=1e-6)


def test_resampling_footprint_axis_aligned_box():
    """Cover exactly the cell block and integrate to the cell area at angle 0.

    Uses an integer cell on a pixel-centered lenslet, with supersample=7 (not a
    divisor of the cell) so the per-pixel weights are deliberately non-uniform;
    only the robust invariants are asserted.
    """
    fp_shape = (32, 32)
    cell = 5.0
    _idx, w = _resampling_footprints(
        jnp.array([16.0]), jnp.array([16.0]), 0.0, cell, fp_shape, supersample=7
    )
    nz = w[0][w[0] > 1e-6]
    assert nz.size == 25  # 5 x 5 axis-aligned block
    assert jnp.allclose(nz.sum(), cell**2, rtol=1e-6)  # flux-conserving


def test_resampling_footprint_multi_lenslet():
    """Vmap handles several lenslets independently (shapes and per-lenslet flux)."""
    fp_shape = (48, 48)
    cell = 5.0
    cx = jnp.array([24.0, 0.0])  # one interior, one at the focal-plane edge
    cy = jnp.array([24.0, 24.0])
    idx, w = _resampling_footprints(cx, cy, 0.0, cell, fp_shape, supersample=8)
    assert idx.shape == w.shape
    assert idx.shape[0] == 2
    # interior conserves the cell area; the edge lenslet loses its off-grid part
    assert jnp.allclose(w[0].sum(), cell**2, rtol=1e-6)
    assert float(w[1].sum()) < cell**2


def test_resampling_footprint_rotation_moves_corners():
    """A 45 deg rotation widens the footprint bounding box vs axis-aligned."""
    fp_shape = (64, 64)
    cell = 8.0
    _, w0 = _resampling_footprints(
        jnp.array([32.0]), jnp.array([32.0]), 0.0, cell, fp_shape, supersample=8
    )
    _, w45 = _resampling_footprints(
        jnp.array([32.0]),
        jnp.array([32.0]),
        float(jnp.pi / 4),
        cell,
        fp_shape,
        supersample=8,
    )
    assert jnp.allclose(w0.sum(), w45.sum(), rtol=1e-6)
    assert int((w45[0] > 1e-6).sum()) > int((w0[0] > 1e-6).sum())


def test_resampling_footprint_masks_off_grid():
    """A lenslet straddling the edge loses the off-grid part of its cell."""
    fp_shape = (32, 32)
    cell = 6.0
    _, w_in = _resampling_footprints(
        jnp.array([16.0]), jnp.array([16.0]), 0.0, cell, fp_shape, supersample=8
    )
    _, w_edge = _resampling_footprints(
        jnp.array([1.0]), jnp.array([16.0]), 0.0, cell, fp_shape, supersample=8
    )
    assert float(w_edge.sum()) < float(w_in.sum())


def test_throughput_scales_footprint_sums():
    """build_ir bakes throughput into H: footprints sum to throughput not 1."""
    from optixstuff import SpectralThroughput

    curve = SpectralThroughput(
        jnp.array([600.0, 660.0, 720.0]), jnp.array([0.3, 0.9, 0.6])
    )
    disp = LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=6,
        psflet_kind="gaussian",
        detector_shape=(256, 256),
        throughput_element=curve,
    )
    lam = jnp.array([600.0, 660.0, 720.0])
    ir = build_ir(disp, lam, fp_shape=(64, 64), fp_px_per_lenslet=2.0)
    sums = ir.det_vals.sum(axis=2)  # (n_channels, n_wav)
    expected = disp.throughput(lam)  # (n_wav,) = [0.3, 0.9, 0.6]
    for j in range(int(lam.shape[0])):
        col = sums[:, j]
        nonzero = col[col > 1e-8]
        assert nonzero.size > 0
        assert jnp.allclose(nonzero, expected[j], atol=1e-5)


def test_build_ir_spatial_footprint_is_flux_conserving():
    """Every lenslet's spatial weights sum to the cell area (fp_px^2).

    For this small grid on a 64x64 focal plane every lenslet is interior, so
    each integrates a full cell. fp_px != 1 is what makes the test
    discriminating: the old bilinear point-sample summed to 1 for any cell size.
    """
    fp_px = 5.0
    ir = build_ir(
        _disp(),
        jnp.linspace(600.0, 700.0, 4),
        fp_shape=(64, 64),
        fp_px_per_lenslet=fp_px,
    )
    sums = ir.spatial_w.sum(axis=1)
    assert jnp.allclose(sums, fp_px**2, rtol=1e-3)
    assert bool(jnp.all(sums <= fp_px**2 + 1e-6))
