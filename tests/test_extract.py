"""Tests for spectral extraction from a dispersed IFS detector image."""

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import bands_at, sky_grid
from hwoutils.numerics import COARSEST, sum_roundoff
from hwoutils.precision import PrecisionError
from hwoutils.testing import precision_pair, ulps
from optixstuff.detector import Detector
from optixstuff.disperser import LensletDisperser
from optixstuff.exposure import FramePlan
from optixstuff.grids import Measure
from optixstuff.rates import Rates
from optixstuff.spectral import Band, Quadrature

from coronachrome.build import build_ir, detector_centroids
from coronachrome.extract import (
    detector_statistics,
    lstsq,
    matched_filter,
    spectrum_covariance,
    spectrum_errorbars,
)
from coronachrome.render import IFSRenderer


def _dense_normal(r, weights):
    """Dense weighted normal matrix A = H^T W H (ncw, ncw) for reference."""
    hd = np.asarray(r.H_mono.todense())
    w = np.ones(hd.shape[0]) if weights is None else np.asarray(weights).reshape(-1)
    return hd.T @ (w[:, None] * hd)


def _renderer(n=2, n_bands=5, fp=(32, 32)):
    disp = LensletDisperser(
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
        detector_shape=(64, 64),
    )
    # Use linspace over a wide range so adjacent PSFlets are well-separated
    # in pixel space; 9 channels over 620-700 nm packs them too tightly for
    # a matched-filter test (off-diagonal H^T H ~ 50% of diagonal).
    lam = jnp.linspace(580.0, 740.0, n_bands)
    ir = build_ir(disp, fp_grid=sky_grid(fp, 0.2 / (2.0)), bands=bands_at(lam))
    return IFSRenderer(ir), n_bands


def test_matched_filter_correlates_with_truth():
    """The matched filter is strongly correlated with the injected spectra."""
    r, n_bands = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(1), (r.ir.n_channels, n_bands))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z_hat = r.channel_spectra(
        matched_filter(
            r,
            r.detector_image(detector, "e"),
        )
    )
    assert z_hat.shape == (r.ir.n_channels, n_bands)
    c = jnp.corrcoef(z_true.reshape(-1), z_hat.reshape(-1))[0, 1]
    assert float(c) > 0.9


def test_lstsq_recovers_injected_spectra():
    """Noiseless lstsq recovers an injected per-lenslet spectrum near-exactly."""
    r, n_bands = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(0), (r.ir.n_channels, n_bands))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z_hat = r.channel_spectra(
        lstsq(
            r,
            r.detector_image(detector, "e"),
        )
    )
    assert z_hat.shape == (r.ir.n_channels, n_bands)
    assert jnp.allclose(z_hat, z_true, atol=1e-4)


def test_weighted_lstsq_recovers_spectra():
    """Per-detector-pixel weighting still recovers the noiseless spectrum."""
    r, n_bands = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(2), (r.ir.n_channels, n_bands))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    weights = 1.0 / jnp.clip(detector.reshape(-1), 1e-3, None)
    z_hat = r.channel_spectra(
        lstsq(r, r.detector_image(detector, "e"), weights=weights)
    )
    assert jnp.allclose(z_hat, z_true, atol=1e-3)


def test_lstsq_is_differentiable():
    """Verify lstsq has finite gradients w.r.t. the detector (lineax stable grads)."""
    r, _n_bands = _renderer()
    detector = jnp.ones(r.ir.det_shape)

    def loss(d):
        return jnp.sum(
            r.channel_spectra(
                lstsq(
                    r,
                    r.detector_image(d, "e"),
                )
            )
            ** 2
        )

    g = jax.grad(loss)(detector)
    assert g.shape == detector.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_lstsq_robust_to_operator_scale():
    """Recover spectra regardless of the dispersion operator's absolute scale.

    NormalCG injects NaN when its breakdown safeguard (proportional to the array
    dtype's rcond) exceeds the normal operator's smallest eigenvalue. A real IFS
    operator has det_vals of O(0.1), so in float32 that safeguard tripped and the
    solve returned NaN even though the operator is well-conditioned. Scaling the
    operator down here reproduces that small-eigenvalue regime in float64; column
    equilibration inside lstsq keeps the solve well-scaled and scale-invariant.
    """
    r, n_bands = _renderer()
    s = 1e-6  # shrink the operator so the un-equilibrated normal eigenvalues ~ s^2
    rs = eqx.tree_at(
        lambda m: (m.ir.det_vals, m.H_mono.data),
        r,
        (r.ir.det_vals * s, r.H_mono.data * s),
    )
    z_true = jax.random.uniform(jax.random.PRNGKey(7), (rs.ir.n_channels, n_bands))
    detector = (rs.H_mono @ z_true.reshape(-1)).reshape(rs.ir.det_shape)
    # Data is ~s here, so converge on the relative tolerance only (atol=0).
    z_hat = rs.channel_spectra(
        lstsq(rs, rs.detector_image(detector, "e"), rtol=1e-8, atol=0.0)
    )
    assert bool(jnp.all(jnp.isfinite(z_hat)))
    assert jnp.allclose(z_hat, z_true, atol=1e-6)


def test_lstsq_damping_matches_dense_tikhonov():
    """Damped lstsq equals the dense equilibrated Tikhonov solution.

    Solves z = D (D H^T W H D + damping I)^-1 D H^T W y, with D the column
    equilibration; the closed form is the reference.
    """
    r, n_bands = _renderer()
    ncw = r.ir.n_channels * n_bands
    z_true = jax.random.uniform(jax.random.PRNGKey(5), (r.ir.n_channels, n_bands))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    y = np.asarray(detector).reshape(-1)
    damping = 1e-2

    hd = np.asarray(r.H_mono.todense())
    wdiag = (hd**2).sum(axis=0)  # diag(H^T H), W = I here
    d = 1.0 / np.sqrt(wdiag)
    dmat = d[:, None] * (hd.T @ hd) * d[None, :]  # D H^T H D (unit diagonal)
    rhs = d * (hd.T @ y)
    zp = np.linalg.solve(dmat + damping * np.eye(ncw), rhs)
    z_ref = (d * zp).reshape(r.ir.n_channels, n_bands)

    z_hat = r.channel_spectra(
        lstsq(r, r.detector_image(detector, "e"), damping=damping)
    )
    assert jnp.allclose(z_hat, z_ref, rtol=1e-4, atol=1e-6)


def test_lstsq_damping_shrinks_solution():
    """Positive damping biases the recovered spectrum toward zero (Tikhonov)."""
    r, n_bands = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(6), (r.ir.n_channels, n_bands))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z0 = r.channel_spectra(lstsq(r, r.detector_image(detector, "e"), damping=0.0))
    z1 = r.channel_spectra(lstsq(r, r.detector_image(detector, "e"), damping=1e-1))
    assert float(jnp.linalg.norm(z1)) < float(jnp.linalg.norm(z0))
    assert jnp.allclose(z0, z_true, atol=1e-4)  # undamped still recovers truth


def test_lstsq_recovers_sharp_dip_spaxel():
    """A sharp absorption dip (O2-like) at one spaxel is recovered by lstsq."""
    r, n_bands = _renderer(n_bands=15)
    ch = r.ir.n_channels // 2
    spec = jnp.ones(n_bands).at[11].set(0.05)  # deep dip at wavelength index 11
    z_true = jnp.ones((r.ir.n_channels, n_bands)).at[ch].set(spec)
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z_hat = r.channel_spectra(
        lstsq(
            r,
            r.detector_image(detector, "e"),
        )
    )
    assert jnp.allclose(z_hat[ch], spec, atol=1e-3)
    assert float(z_hat[ch][11]) < 0.2


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_spectrum_covariance_matches_dense_block():
    """Per-channel covariance equals the dense (H^T W H)^-1 channel block."""
    r, n_bands = _renderer()
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    weights = jax.random.uniform(
        jax.random.PRNGKey(3), (n_det,), minval=0.1, maxval=2.0
    )
    ainv = np.linalg.inv(_dense_normal(r, weights))
    chans = jnp.array([r.ir.n_channels // 2, 0])
    cov = np.asarray(
        spectrum_covariance(r, weights=weights, channels=chans, rtol=1e-11, atol=1e-11)
    )
    for i, ch in enumerate(np.asarray(chans)):
        base = int(ch) * n_bands
        ref = ainv[base : base + n_bands, base : base + n_bands]
        assert np.allclose(cov[i], ref, rtol=1e-3, atol=1e-8)


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_spectrum_covariance_many_channels_matches_dense():
    """The lax.map path matches dense (H^T W H)^-1 blocks across many channels."""
    r, n_bands = _renderer()
    chans = jnp.array([0, 1, 2, r.ir.n_channels - 1])
    ainv = np.linalg.inv(_dense_normal(r, None))
    cov = np.asarray(spectrum_covariance(r, channels=chans, rtol=1e-11, atol=1e-11))
    assert cov.shape == (len(chans), n_bands, n_bands)
    for i, ch in enumerate(np.asarray(chans)):
        base = int(ch) * n_bands
        ref = ainv[base : base + n_bands, base : base + n_bands]
        assert np.allclose(cov[i], ref, rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("entry", [spectrum_covariance, spectrum_errorbars])
def test_extraction_uncertainty_refuses_float32(entry, precision):
    """A float32 dispersion operator is refused even when x64 is enabled."""
    renderer, _ = _renderer(n=1, n_bands=1)
    renderer = IFSRenderer(
        eqx.tree_at(
            lambda ir: ir.det_vals, renderer.ir, renderer.ir.det_vals.astype(COARSEST)
        )
    )
    remedy = "canonical" if precision == np.float64 else "enable_x64"
    with pytest.raises(PrecisionError, match=rf"{entry.__name__}.*{remedy}"):
        entry(renderer, channels=[0])


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_spectrum_errorbars_is_sqrt_diag():
    """Error bars equal sqrt of the covariance block diagonal."""
    r, _n_bands = _renderer()
    chans = jnp.array([r.ir.n_channels // 2])
    cov = spectrum_covariance(r, channels=chans, rtol=1e-11, atol=1e-11)
    err = spectrum_errorbars(r, channels=chans)
    assert jnp.allclose(err, jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1)))


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_errorbars_scale_as_sqrt_noise():
    """Doubling the per-pixel noise variance scales error bars by sqrt(2)."""
    r, _n_bands = _renderer()
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    chans = jnp.array([r.ir.n_channels // 2])
    err_n1 = spectrum_errorbars(r, weights=None, channels=chans)  # N = 1
    err_n2 = spectrum_errorbars(
        r, weights=jnp.full(n_det, 0.5), channels=chans
    )  # 1/N = 0.5 -> N = 2
    assert jnp.allclose(err_n2 / err_n1, jnp.sqrt(2.0), rtol=1e-3)


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_covariance_montecarlo_coverage():
    """Predicted covariance matches the empirical covariance of the GLS estimator."""
    r, n_bands = _renderer()
    ch = r.ir.n_channels // 2
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    hd = np.asarray(r.H_mono.todense())  # W = I here
    ainv = np.linalg.inv(hd.T @ hd)
    gls = ainv @ hd.T  # z_hat = G y
    base = ch * n_bands
    rng = np.random.default_rng(0)
    clean = np.zeros(n_det)  # true spectra are zero; covariance is signal-independent
    draws = np.stack(
        [
            (gls @ (clean + rng.standard_normal(n_det)))[base : base + n_bands]
            for _ in range(4000)
        ]
    )
    emp = np.cov(draws.T)
    pred = np.asarray(
        spectrum_covariance(r, weights=jnp.ones(n_det), channels=jnp.array([ch]))
    )[0]
    # Variances (diagonal) are the error bars; check those tightly, full block loosely.
    assert np.allclose(np.diag(emp), np.diag(pred), rtol=0.1)
    assert np.allclose(emp, pred, atol=0.15 * float(np.max(np.diag(pred))))


@vv.case("ifs-noise-seam", "code-verification")
@vv.seam("optixstuff", "coronachrome")
@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_detector_variance_one_lenslet_one_band():
    """Fails if detector noise is duplicated or count variance becomes a rate.

    Exact algebra: half=0 puts H=1 in one pixel. Source 3+2 e/s, two 4s
    frames, dark .5 e/pix/s, CIC .25 e/pix/frame, F2=2, read=3 e RMS:
    mean=44.5 e; variance=2*44.5+2*9=107 e^2. GLS variance is 107.
    Tolerance is 32 eps for one scalar solve, not a noise realization.
    """
    from coronachrome import extraction_weights

    disperser = LensletDisperser(
        pitch_m=130e-6,
        pixsize_m=13e-6,
        angle_rad=0.0,
        sky_pitch_arcsec=0.2,
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([0.0]),
        psflet_params=jnp.array([0.2]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=1,
        psflet_kind="gaussian",
        detector_shape=(3, 3),
    )
    r = IFSRenderer(build_ir(disperser, (Band(650, 670),), sky_grid((5, 5)), half=0))
    det = Detector(
        pixel_scale_arcsec=0.1,
        shape=(3, 3),
        dark_current_e_per_pixel_s=0.5,
        clock_induced_charge_e_per_pixel_frame=0.25,
        read_noise_e_rms=3.0,
        excess_noise_factor_sq=2.0,
    )
    # Two source-band contributions illuminate the same physical pixel.
    rates = Rates(
        (Band(650, 660), Band(660, 670)),
        "electron",
        planet=jnp.zeros((2, 3, 3)).at[0, 1, 1].set(3.0).at[1, 1, 1].set(2.0),
    )
    plan = FramePlan(4.0, 2)
    mean, variance = detector_statistics(r, det, rates, plan)
    weights = extraction_weights(r, det, rates, plan)
    assert mean.quantity == "e" and variance.quantity == "e^2"
    assert mean.data[1, 1] == 44.5 and variance.data[1, 1] == 107.0
    cov = spectrum_covariance(r, weights=weights, channels=jnp.array([0]))
    assert jnp.allclose(cov[0, 0, 0], 107.0, rtol=7e-15, atol=0)
    signal = r.detector_image(jnp.zeros((3, 3)).at[1, 1].set(40.0), "e")
    fitted = lstsq(r, signal, weights=weights)
    assert fitted.grid == r.ir.lenslet_grid and fitted.measure == Measure.integral()
    assert fitted.quantity == "e"
    assert jnp.allclose(fitted.data[0, 0, 0], 40.0, rtol=7e-15, atol=0)


@pytest.fixture(scope="module")
def detector_setup():
    """A shared detector and exposure for testing per-pixel statistics."""
    renderer, _ = _renderer(n=1, n_bands=2, fp=(4, 4))
    detector = Detector(
        pixel_scale_arcsec=0.1,
        shape=renderer.ir.det_shape,
        dark_current_e_per_pixel_s=0.5,
        read_noise_e_rms=3.0,
    )
    return renderer, detector, FramePlan(4.0, 2)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("photon", "photon"),
        ("spectral", "integrate over the bands first"),
        ("aperture", "n_pix"),
        ("variance_aperture", "n_pix"),
    ],
)
def test_detector_statistics_refuses_invalid_rates(detector_setup, kind, message):
    """The detector accepts per-pixel band-integrated electrons."""
    renderer, detector, plan = detector_setup
    bands = renderer.ir.bands
    sampling = Quadrature.build(bands) if kind == "spectral" else bands
    leading = sampling.nodes_nm.shape if kind == "spectral" else (len(bands),)
    rates = Rates(
        sampling,
        "photon" if kind == "photon" else "electron",
        planet=jnp.ones((*leading, *detector.shape)),
        n_pix=1 if kind == "aperture" else None,
        variance=Rates(bands, "electron", n_pix=1)
        if kind == "variance_aperture"
        else None,
    )
    with pytest.raises(ValueError, match=message):
        detector_statistics(renderer, detector, rates, plan)


def test_detector_statistics_sums_variance_components(detector_setup):
    """Detector moments equal hand-collapsed float64 moments bit for bit."""
    renderer, detector, plan = detector_setup
    bands = renderer.ir.bands
    signal = np.arange(2 * np.prod(detector.shape), dtype=np.float64).reshape(
        2, *detector.shape
    )
    variance_planet = np.array([0.25, 1.5])[:, None, None]
    rates = Rates(
        bands,
        "electron",
        planet=signal,
        variance=Rates(bands, "electron", planet=variance_planet, thermal=0.5),
    )
    passband = (Band(bands[0].low_nm, bands[-1].high_nm),)
    variance_sum = np.broadcast_to(variance_planet + 0.5, signal.shape).sum(
        0, keepdims=True
    )
    hand = Rates(
        passband,
        "electron",
        planet=signal.sum(0, keepdims=True),
        variance=Rates(passband, "electron", planet=variance_sum),
    )
    expected_mean, expected_variance = detector.mean_and_variance(hand, plan)
    mean, variance = detector_statistics(renderer, detector, rates, plan)
    np.testing.assert_array_equal(mean.data, expected_mean[0], strict=True)
    np.testing.assert_array_equal(variance.data, expected_variance[0], strict=True)


def _edge_renderer(xc, half, sigma=0.7):
    """One lenslet at 700 nm whose Gaussian PSFlet is centered at detector x = xc."""
    shape = (32, 32)
    offset = xc - (shape[1] - 1) / 2
    disp = LensletDisperser(
        pitch_m=174e-6,
        sky_pitch_arcsec=0.2,
        pixsize_m=13e-6,
        angle_rad=0.0,
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([offset]),
        psflet_params=jnp.array([sigma]),
        psflet_ref_nm=660.0,
        grid_kind="square",
        n_lenslets=1,
        psflet_kind="gaussian",
        detector_shape=shape,
    )
    lam = [700.0]
    assert float(detector_centroids(disp, lam)[0][0, 0]) == xc
    ir = build_ir(disp, fp_grid=sky_grid((2, 2), 0.1), bands=bands_at(lam), half=half)
    return IFSRenderer(ir)


def test_matched_filter_recovers_flux_of_a_deeply_clipped_edge_column():
    """Recover the input flux from a clipped tail in both lanes.

    The reference is 10 electrons. Allow two stamp reductions and six scalar
    roundings for normalization, products and divisions.
    """
    r = _edge_renderer(-4.0, half=6)
    column = float((r.ir.det_vals**2).sum())
    assert 0 < column < 1e-12  # below the former absolute floor
    detector = (r.H_mono @ jnp.array([10.0])).reshape(r.ir.det_shape)
    cube = matched_filter(r, r.detector_image(detector, "e"))
    spectra = r.channel_spectra(cube)
    expected = spectra.dtype.type(10)
    tol = 2 * sum_roundoff(expected, r.ir.det_vals.shape[-1]) + ulps(6, expected)
    np.testing.assert_allclose(spectra, 10.0, rtol=0, atol=float(tol))


def _sparse_column_renderer(renderer, weight, n_taps=1):
    vals = jnp.zeros_like(renderer.ir.det_vals, dtype=jnp.asarray(weight).dtype)
    vals = vals.at[0, 0, :n_taps].set(weight)
    return IFSRenderer(eqx.tree_at(lambda ir: ir.det_vals, renderer.ir, vals))


@pytest.mark.parametrize("signal", [1e-10, 1e-30])
def test_matched_filter_faint_column_derivative(signal):
    """Fails if the faint-column quotient or its derivative underflows.

    The float64 analytic y/a and -y/a**2 allow eight rounding operations:
    two casts and six scalar operations for scaling and differentiating. In
    the float64 process the same relative bound also checks the precision pair.
    """
    renderer, _ = _renderer(n=1, n_bands=1)
    weight = 1e-10

    def estimate(a, y):
        r = _sparse_column_renderer(renderer, a)
        detector = r.detector_image(jnp.broadcast_to(y, r.ir.det_shape), "e")
        return matched_filter(r, detector).data[0, 0, 0]

    evaluate = jax.jit(jax.value_and_grad(estimate))
    inputs = (jnp.asarray(weight), jnp.asarray(signal))
    actual = jnp.asarray(evaluate(*inputs))
    expected = np.array([signal / weight, -signal / weight**2])
    tol = ulps(8, jnp.asarray(expected, dtype=actual.dtype))
    np.testing.assert_array_equal(
        np.abs(np.asarray(actual) - expected) <= np.asarray(tol), True
    )
    if actual.dtype == np.float64:
        precision_pair(evaluate, inputs, bound=float(ulps(8, np.float32(1))), scale=0)


_FLOOR = float(np.sqrt(np.finfo(COARSEST).tiny))


@pytest.mark.parametrize(
    ("tap", "n_taps", "inv_variance", "supported"),
    [
        (1e-20, 1, None, False),
        (_FLOOR, 1, None, True),
        (_FLOOR / 2, 4, None, True),
        (8.852473344610577e-20, 1, 1.5, True),
        (7.666466952210875e-20, 1, 2.0, False),
    ],
)
def test_extraction_support_at_float32_floor(tap, n_taps, inv_variance, supported):
    """Support includes equality after float32 weighting in either lane.

    Equal taps give 1/tap within one cast and one division for matched
    filtering, or eight roundings for equilibration and the scalar solve.
    """
    renderer, _ = _renderer(n=1, n_bands=1)
    renderer = _sparse_column_renderer(renderer, tap, n_taps)
    detector = renderer.detector_image(jnp.ones(renderer.ir.det_shape), "e")
    weights = (
        None if inv_variance is None else jnp.full(renderer.ir.det_shape, inv_variance)
    )
    expected = 1 / tap if supported else 0.0
    for solve in (lstsq, eqx.filter_jit(lstsq)):
        flux = solve(renderer, detector, weights).data.item()
        assert abs(flux - expected) <= ulps(8, detector.data.dtype.type(expected))
    if weights is None:
        for extract in (matched_filter, eqx.filter_jit(matched_filter)):
            flux = extract(renderer, detector).data.item()
            assert abs(flux - expected) <= ulps(2, detector.data.dtype.type(expected))


@pytest.mark.parametrize("signal", [1e38, float(np.finfo(COARSEST).tiny)])
def test_matched_filter_four_half_strength_taps(signal):
    """Four half-strength taps recover twice the stored signal in either lane.

    The tap normalization and final factor of two are exact. Allow three
    additions, one division by the accumulation scale and one rescaling.
    """
    renderer, _ = _renderer(n=1, n_bands=1)
    renderer = _sparse_column_renderer(renderer, 0.5, 4)
    detector = renderer.detector_image(jnp.full(renderer.ir.det_shape, signal), "e")
    expected = 2 * detector.data[0, 0]
    for extract in (matched_filter, eqx.filter_jit(matched_filter)):
        flux = extract(renderer, detector).data.item()
        assert abs(flux - float(expected)) <= ulps(5, expected)


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_covariance_derivative_at_zero_pixel_weight():
    """Two distinct quarter-strength taps give variance 16/(1+w).

    At zero weight the scalar solve and its derivative use exact dyadic values.
    """
    renderer, _ = _renderer(n=1, n_bands=1)
    renderer = _sparse_column_renderer(renderer, 0.25, 2)
    first, second = renderer.ir.det_rows[0, 0, :2]
    assert first != second

    def variance(w):
        weights = jnp.ones(renderer.H_mono.shape[0]).at[first].set(w)
        return spectrum_covariance(renderer, weights, channels=[0])[0, 0, 0]

    evaluate = jax.value_and_grad(variance)
    for run in (evaluate, jax.jit(evaluate)):
        np.testing.assert_array_equal(run(jnp.asarray(0.0)), [16.0, -16.0])


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
def test_covariance_is_finite_for_a_faint_supported_weight():
    """One faint supported coefficient has variance 1/a**2 within eight ops."""
    r, _ = _renderer(n=1, n_bands=1)
    weight = np.float64(1e-18)
    r = _sparse_column_renderer(r, weight)
    variance = spectrum_covariance(r, channels=[0])
    expected = 1 / weight**2
    np.testing.assert_allclose(
        variance[0, 0, 0], expected, rtol=0, atol=float(ulps(8, expected))
    )


def test_a_column_with_no_support_is_removed_not_nan():
    """Fails if a column with no support returns nonzero or nonfinite flux."""
    r = _edge_renderer(-1.0, half=1)
    dead = jnp.zeros_like(r.ir.det_vals)
    r = IFSRenderer(eqx.tree_at(lambda ir: ir.det_vals, r.ir, dead))
    detector = r.detector_image(jnp.ones(r.ir.det_shape), "e")
    assert float(matched_filter(r, detector).data.sum()) == 0.0
    assert float(lstsq(r, detector).data.sum()) == 0.0


@pytest.mark.float64_only(reason="GLS spectrum covariance requires float64")
@pytest.mark.parametrize("weight", [0.0, 1e-20])
def test_covariance_of_column_with_no_support_is_infinite(weight):
    """Fails if a dead or below-floor column has finite or NaN variance."""
    r, _ = _renderer(n=1, n_bands=1)
    r = _sparse_column_renderer(r, weight)
    cov = spectrum_covariance(r, channels=[0])
    assert np.isinf(cov[0, 0, 0]) and not np.any(np.isnan(cov))
