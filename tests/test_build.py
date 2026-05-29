"""Tests for build_ir."""

import jax.numpy as jnp
import pytest
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.ir import SpatialChannelIR


def _disp(n=6):
    return LensletDisperser(
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


def test_build_ir_returns_ir_with_expected_shapes():
    """build_ir returns a SpatialChannelIR with expected channel/wav/detector shapes."""
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(_disp(6), lam, fp_shape=(64, 64))
    assert isinstance(ir, SpatialChannelIR)
    assert ir.n_channels == 36
    assert ir.n_wav == 5
    assert ir.det_rows.shape[:2] == (36, 5)
    assert ir.det_shape == (256, 256)


def test_det_vals_normalized_per_channel_wavelength():
    """On-detector PSFlet footprints are flux-normalized; off-detector ones are zero."""
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(_disp(6), lam, fp_shape=(64, 64))
    sums = ir.det_vals.sum(axis=2)
    nonzero = sums[sums > 1e-8]
    assert jnp.allclose(nonzero, 1.0, atol=1e-5)


def test_indices_in_bounds():
    """All detector row indices stay within the detector bounds."""
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(_disp(6), lam, fp_shape=(64, 64))
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
        grid_kind="square",
        n_lenslets=6,
        psflet_kind="moffat",
        detector_shape=(256, 256),
    )
    lam = jnp.linspace(640.0, 680.0, 5)
    ir = build_ir(disp, lam, fp_shape=(64, 64))
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
        grid_kind="square",
        n_lenslets=8,
        psflet_kind="gaussian",
        detector_shape=(40, 40),
    )
    lam = jnp.linspace(640.0, 680.0, 5)
    with pytest.warns(UserWarning, match="off the detector"):
        build_ir(disp, lam, fp_shape=(64, 64))
