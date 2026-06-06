"""Tests for spectral extraction from a dispersed IFS detector image."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.extract import (
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


def _renderer(n=6, n_wav=9, fp=(64, 64)):
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
        n_lenslets=n,
        psflet_kind="gaussian",
        detector_shape=(256, 256),
    )
    # Use linspace over a wide range so adjacent PSFlets are well-separated
    # in pixel space; 9 channels over 620-700 nm packs them too tightly for
    # a matched-filter test (off-diagonal H^T H ~ 50% of diagonal).
    lam = jnp.linspace(580.0, 740.0, n_wav)
    return IFSRenderer(build_ir(disp, lam, fp_shape=fp)), n_wav


def test_matched_filter_correlates_with_truth():
    """The matched filter is strongly correlated with the injected spectra."""
    r, n_wav = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(1), (r.ir.n_channels, n_wav))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z_hat = matched_filter(r, detector)
    assert z_hat.shape == (r.ir.n_channels, n_wav)
    c = jnp.corrcoef(z_true.reshape(-1), z_hat.reshape(-1))[0, 1]
    assert float(c) > 0.9


def test_lstsq_recovers_injected_spectra():
    """Noiseless lstsq recovers an injected per-lenslet spectrum near-exactly."""
    r, n_wav = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(0), (r.ir.n_channels, n_wav))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z_hat = lstsq(r, detector)
    assert z_hat.shape == (r.ir.n_channels, n_wav)
    assert jnp.allclose(z_hat, z_true, atol=1e-4)


def test_weighted_lstsq_recovers_spectra():
    """Per-detector-pixel weighting still recovers the noiseless spectrum."""
    r, n_wav = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(2), (r.ir.n_channels, n_wav))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    weights = 1.0 / jnp.clip(detector.reshape(-1), 1e-3, None)
    z_hat = lstsq(r, detector, weights=weights)
    assert jnp.allclose(z_hat, z_true, atol=1e-3)


def test_lstsq_is_differentiable():
    """Verify lstsq has finite gradients w.r.t. the detector (lineax stable grads)."""
    r, _n_wav = _renderer()
    detector = jnp.ones(r.ir.det_shape)

    def loss(d):
        return jnp.sum(lstsq(r, d) ** 2)

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
    r, n_wav = _renderer()
    s = 1e-6  # shrink the operator so the un-equilibrated normal eigenvalues ~ s^2
    rs = eqx.tree_at(
        lambda m: (m.ir.det_vals, m.H_mono.data),
        r,
        (r.ir.det_vals * s, r.H_mono.data * s),
    )
    z_true = jax.random.uniform(jax.random.PRNGKey(7), (rs.ir.n_channels, n_wav))
    detector = (rs.H_mono @ z_true.reshape(-1)).reshape(rs.ir.det_shape)
    # Data is ~s here, so converge on the relative tolerance only (atol=0).
    z_hat = lstsq(rs, detector, rtol=1e-8, atol=0.0)
    assert bool(jnp.all(jnp.isfinite(z_hat)))
    assert jnp.allclose(z_hat, z_true, atol=1e-6)


def test_lstsq_damping_matches_dense_tikhonov():
    """Damped lstsq equals the dense equilibrated Tikhonov solution.

    Solves z = D (D H^T W H D + damping I)^-1 D H^T W y, with D the column
    equilibration; the closed form is the reference.
    """
    r, n_wav = _renderer()
    ncw = r.ir.n_channels * n_wav
    z_true = jax.random.uniform(jax.random.PRNGKey(5), (r.ir.n_channels, n_wav))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    y = np.asarray(detector).reshape(-1)
    damping = 1e-2

    hd = np.asarray(r.H_mono.todense())
    wdiag = (hd**2).sum(axis=0)  # diag(H^T H), W = I here
    d = 1.0 / np.sqrt(wdiag)
    dmat = d[:, None] * (hd.T @ hd) * d[None, :]  # D H^T H D (unit diagonal)
    rhs = d * (hd.T @ y)
    zp = np.linalg.solve(dmat + damping * np.eye(ncw), rhs)
    z_ref = (d * zp).reshape(r.ir.n_channels, n_wav)

    z_hat = lstsq(r, detector, damping=damping)
    assert jnp.allclose(z_hat, z_ref, rtol=1e-4, atol=1e-6)


def test_lstsq_damping_shrinks_solution():
    """Positive damping biases the recovered spectrum toward zero (Tikhonov)."""
    r, n_wav = _renderer()
    z_true = jax.random.uniform(jax.random.PRNGKey(6), (r.ir.n_channels, n_wav))
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z0 = lstsq(r, detector, damping=0.0)
    z1 = lstsq(r, detector, damping=1e-1)
    assert float(jnp.linalg.norm(z1)) < float(jnp.linalg.norm(z0))
    assert jnp.allclose(z0, z_true, atol=1e-4)  # undamped still recovers truth


def test_lstsq_recovers_sharp_dip_spaxel():
    """A sharp absorption dip (O2-like) at one spaxel is recovered by lstsq."""
    r, n_wav = _renderer(n_wav=15)
    ch = r.ir.n_channels // 2
    spec = jnp.ones(n_wav).at[11].set(0.05)  # deep dip at wavelength index 11
    z_true = jnp.ones((r.ir.n_channels, n_wav)).at[ch].set(spec)
    detector = (r.H_mono @ z_true.reshape(-1)).reshape(r.ir.det_shape)
    z_hat = lstsq(r, detector)
    assert jnp.allclose(z_hat[ch], spec, atol=1e-3)
    assert float(z_hat[ch][11]) < 0.2


def test_spectrum_covariance_matches_dense_block():
    """Per-channel covariance equals the dense (H^T W H)^-1 channel block."""
    r, n_wav = _renderer()
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    weights = jax.random.uniform(
        jax.random.PRNGKey(3), (n_det,), minval=0.1, maxval=2.0
    )
    ainv = np.linalg.inv(_dense_normal(r, weights))
    chans = jnp.array([r.ir.n_channels // 2, 0])
    cov = np.asarray(spectrum_covariance(r, weights=weights, channels=chans))
    for i, ch in enumerate(np.asarray(chans)):
        base = int(ch) * n_wav
        ref = ainv[base : base + n_wav, base : base + n_wav]
        assert np.allclose(cov[i], ref, rtol=1e-3, atol=1e-8)


def test_spectrum_covariance_many_channels_matches_dense():
    """The lax.map path matches dense (H^T W H)^-1 blocks across many channels."""
    r, n_wav = _renderer()
    chans = jnp.array([0, 5, 11, 18, r.ir.n_channels - 1])
    ainv = np.linalg.inv(_dense_normal(r, None))
    cov = np.asarray(spectrum_covariance(r, channels=chans))
    assert cov.shape == (len(chans), n_wav, n_wav)
    for i, ch in enumerate(np.asarray(chans)):
        base = int(ch) * n_wav
        ref = ainv[base : base + n_wav, base : base + n_wav]
        assert np.allclose(cov[i], ref, rtol=1e-3, atol=1e-8)


def test_spectrum_errorbars_is_sqrt_diag():
    """Error bars equal sqrt of the covariance block diagonal."""
    r, _n_wav = _renderer()
    chans = jnp.array([r.ir.n_channels // 2])
    cov = spectrum_covariance(r, channels=chans)
    err = spectrum_errorbars(r, channels=chans)
    assert jnp.allclose(err, jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1)))


def test_errorbars_scale_as_sqrt_noise():
    """Doubling the per-pixel noise variance scales error bars by sqrt(2)."""
    r, _n_wav = _renderer()
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    chans = jnp.array([r.ir.n_channels // 2])
    err_n1 = spectrum_errorbars(r, weights=None, channels=chans)  # N = 1
    err_n2 = spectrum_errorbars(
        r, weights=jnp.full(n_det, 0.5), channels=chans
    )  # 1/N = 0.5 -> N = 2
    assert jnp.allclose(err_n2 / err_n1, jnp.sqrt(2.0), rtol=1e-3)


def test_covariance_montecarlo_coverage():
    """Predicted covariance matches the empirical covariance of the GLS estimator."""
    r, n_wav = _renderer()
    ch = r.ir.n_channels // 2
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    hd = np.asarray(r.H_mono.todense())  # W = I here
    ainv = np.linalg.inv(hd.T @ hd)
    gls = ainv @ hd.T  # z_hat = G y
    base = ch * n_wav
    rng = np.random.default_rng(0)
    clean = np.zeros(n_det)  # true spectra are zero; covariance is signal-independent
    draws = np.stack(
        [
            (gls @ (clean + rng.standard_normal(n_det)))[base : base + n_wav]
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
