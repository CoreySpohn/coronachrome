"""Tests for spectral extraction from a dispersed IFS detector image."""

import jax
import jax.numpy as jnp
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.extract import matched_filter
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
