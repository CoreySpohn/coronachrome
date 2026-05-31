"""Tests for build_ir."""

import jax.numpy as jnp
import pytest
from optixstuff.disperser import LensletDisperser

from coronachrome.build import _resampling_footprints, build_ir
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
