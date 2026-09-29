"""Tests for the nuisance-marginal measurement covariance."""

import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.extract import (
    amplitude_basis,
    marginal_covariance,
    unmodeled_covariance,
)
from coronachrome.render import IFSRenderer, spatial_sample


def _renderer(n=4, n_wav=5, fp=(32, 32)):
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
        detector_shape=(200, 200),
    )
    lam = jnp.linspace(580.0, 740.0, n_wav)
    ir = build_ir(disp, lam, fp_shape=fp, fp_px_per_lenslet=2.0)
    return IFSRenderer(ir), n_wav


def _blob(fp_shape, n_wav, center, sigma=1.2, key=None):
    """A smooth unit-sum focal-plane blob replicated over wavelength."""
    ny, nx = fp_shape
    yy, xx = jnp.mgrid[0:ny, 0:nx]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    blob = jnp.exp(-0.5 * r2 / sigma**2)
    blob = blob / blob.sum()
    if key is not None:
        scale = 1.0 + 0.3 * jax.random.normal(key, (n_wav,))
        return blob[None] * scale[:, None, None]
    return jnp.repeat(blob[None], n_wav, axis=0)


def _nuisance_basis(r, n_wav, n_modes=3, key=None):
    """Channel-space nuisance columns from smooth focal-plane mode cubes."""
    key = jax.random.PRNGKey(7) if key is None else key
    keys = jax.random.split(key, n_modes)
    cubes = jnp.stack(
        [
            _blob(r.ir.fp_shape, n_wav, (8.0 + 3.0 * i, 9.0 + 2.0 * i), 2.5, k)
            for i, k in enumerate(keys)
        ]
    )
    return jax.vmap(spatial_sample, in_axes=(0, None))(cubes, r.ir)


def _dense_fisher(r, cols, weights):
    """Dense F = C^T W C for channel-space basis columns (n_basis, n_ch, n_wav)."""
    hd = np.asarray(r.H_mono.todense())
    z = np.asarray(cols).reshape(cols.shape[0], -1)
    det_cols = hd @ z.T  # (n_det, n_basis)
    w = np.ones(hd.shape[0]) if weights is None else np.asarray(weights).reshape(-1)
    return det_cols.T @ (w[:, None] * det_cols)


def test_amplitude_basis_is_block_diagonal_in_wavelength():
    """One free amplitude per wavelength: column j touches only wavelength j."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    basis = amplitude_basis(template, r.ir)
    assert basis.shape == (n_wav, r.ir.n_channels, n_wav)
    z = spatial_sample(template, r.ir)
    for j in range(n_wav):
        off = jnp.delete(basis[j], j, axis=1)
        assert float(jnp.abs(off).max()) == 0.0
        assert jnp.allclose(basis[j][:, j], z[:, j])


def test_amplitude_basis_contracts_to_a_scaled_template():
    """Summing the basis against a spectrum reproduces the scaled template."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    basis = amplitude_basis(template, r.ir)
    spectrum = jnp.linspace(0.5, 2.0, n_wav)
    contracted = jnp.tensordot(spectrum, basis, axes=(0, 0))
    direct = spatial_sample(template * spectrum[:, None, None], r.ir)
    assert jnp.allclose(contracted, direct, rtol=1e-10, atol=1e-12)


def test_marginal_without_nuisance_is_the_conditional_fisher_inverse():
    """No nuisance block means the plain (C^T W C)^-1 of the science block."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    basis = amplitude_basis(template, r.ir)
    cov = marginal_covariance(r, basis)
    ref = np.linalg.inv(_dense_fisher(r, basis, None))
    assert cov.shape == (n_wav, n_wav)
    assert np.allclose(np.asarray(cov), ref, rtol=1e-6, atol=1e-12)


def test_marginal_equals_the_joint_inverse_science_block():
    """The Schur complement matches the science block of the full joint inverse."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    science = amplitude_basis(template, r.ir)
    nuisance = _nuisance_basis(r, n_wav)
    n_mode = nuisance.shape[0]
    prior = jnp.diag(jnp.linspace(4.0, 1.0, n_mode))

    cov = marginal_covariance(r, science, nuisance, nuisance_cov=prior)

    stacked = np.concatenate([np.asarray(science), np.asarray(nuisance)], axis=0)
    fisher = _dense_fisher(r, jnp.asarray(stacked), None)
    prior_prec = np.zeros_like(fisher)
    prior_prec[n_wav:, n_wav:] = np.linalg.inv(np.asarray(prior))
    joint = np.linalg.inv(fisher + prior_prec)
    assert np.allclose(np.asarray(cov), joint[:n_wav, :n_wav], rtol=1e-6, atol=1e-12)


def test_marginalizing_costs_information_but_less_than_ignoring():
    """Conditional <= marginal <= unmodeled, per wavelength."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    science = amplitude_basis(template, r.ir)
    nuisance = _nuisance_basis(r, n_wav)

    conditional = jnp.diag(marginal_covariance(r, science))
    # Nuisance prior comparable to the conditional error: the regime where the
    # speckle and photon floors are of the same order (the RI-10 regime).
    prior = jnp.eye(nuisance.shape[0]) * float(jnp.mean(conditional))
    marginal = jnp.diag(marginal_covariance(r, science, nuisance, nuisance_cov=prior))
    unmodeled = jnp.diag(unmodeled_covariance(r, science, nuisance, nuisance_cov=prior))

    assert bool(jnp.all(marginal >= conditional * (1.0 - 1e-9)))
    assert bool(jnp.all(unmodeled >= marginal * (1.0 - 1e-9)))
    # A nuisance that genuinely overlaps the science template costs something.
    assert float(jnp.max(unmodeled / conditional)) > 1.05


def test_weights_scale_the_covariance_inversely():
    """Doubling the inverse-variance weights halves the covariance."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    science = amplitude_basis(template, r.ir)
    nuisance = _nuisance_basis(r, n_wav)
    n_det = r.ir.det_shape[0] * r.ir.det_shape[1]
    w1 = jnp.ones(n_det)

    # An improper flat nuisance prior leaves the covariance purely Fisher, so
    # it scales exactly with the weights; any proper prior would not.
    cov1 = marginal_covariance(r, science, nuisance, weights=w1)
    cov2 = marginal_covariance(r, science, nuisance, weights=2 * w1)
    assert jnp.allclose(cov2, 0.5 * cov1, rtol=1e-6)


def test_degenerate_nuisance_destroys_the_science_information():
    """A nuisance equal to the science block leaves nothing identifiable."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    science = amplitude_basis(template, r.ir)
    conditional = jnp.diag(marginal_covariance(r, science))
    # A prior far wider than the science error leaves the identical nuisance
    # basis effectively unconstrained.
    prior = jnp.eye(n_wav) * float(jnp.mean(conditional)) * 1e8
    degenerate = jnp.diag(marginal_covariance(r, science, science, nuisance_cov=prior))
    assert float(jnp.min(degenerate / conditional)) > 1e6


def test_marginal_covariance_matches_monte_carlo():
    """The closed form reproduces the joint estimator's sampling covariance."""
    r, n_wav = _renderer()
    template = _blob(r.ir.fp_shape, n_wav, (7.0, 9.0))
    science = amplitude_basis(template, r.ir)
    nuisance = _nuisance_basis(r, n_wav)
    n_mode = nuisance.shape[0]
    prior = jnp.diag(jnp.linspace(2.0, 0.5, n_mode))

    hd = np.asarray(r.H_mono.todense())
    sci_det = hd @ np.asarray(science).reshape(n_wav, -1).T
    nui_det = hd @ np.asarray(nuisance).reshape(n_mode, -1).T
    n_det = hd.shape[0]

    sigma = 0.05
    w = np.full(n_det, 1.0 / sigma**2)
    fisher = _dense_fisher(
        r, jnp.concatenate([science, nuisance], axis=0), jnp.asarray(w)
    )
    prior_prec = np.zeros_like(fisher)
    prior_prec[n_wav:, n_wav:] = np.linalg.inv(np.asarray(prior))
    joint_inv = np.linalg.inv(fisher + prior_prec)

    rng = np.random.default_rng(3)
    n_mc = 1500
    hats = np.empty((n_mc, n_wav))
    truth = np.linspace(1.0, 2.0, n_wav)
    prior_sd = np.sqrt(np.diag(np.asarray(prior)))
    for i in range(n_mc):
        nu = prior_sd * rng.standard_normal(n_mode)
        mean = sci_det @ truth + nui_det @ nu
        y = mean + sigma * rng.standard_normal(n_det)
        rhs = np.concatenate([sci_det.T @ (w * y), nui_det.T @ (w * y)])
        hats[i] = (joint_inv @ rhs)[:n_wav]

    empirical = np.diag(np.cov(hats.T))
    closed = np.asarray(
        jnp.diag(
            marginal_covariance(
                r, science, nuisance, weights=jnp.asarray(w), nuisance_cov=prior
            )
        )
    )
    mc_tol = 4.0 / np.sqrt(2.0 * (n_mc - 1))  # ~7% on a variance at 4 sigma
    assert np.allclose(empirical / closed, 1.0, atol=mc_tol)
