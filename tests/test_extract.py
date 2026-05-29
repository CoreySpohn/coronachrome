"""Tests for spectral extraction from a dispersed IFS detector image."""

import jax
import jax.numpy as jnp
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.extract import lstsq, matched_filter
from coronachrome.render import IFSRenderer


def _renderer(n=6, n_wav=9, fp=(64, 64)):
    disp = LensletDisperser(
        pitch_m=174e-6,
        pixsize_m=13e-6,
        angle_rad=float(jnp.arcsin(1.0 / jnp.sqrt(5.0))),
        lam_ref_nm=660.0,
        pix_per_reselt=2.0,
        dispersion_coeffs=jnp.array([100.0, 0.0]),
        psflet_params=jnp.array([0.7]),
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
