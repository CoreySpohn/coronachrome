"""Tests for the IFS renderer forward model."""

import jax.numpy as jnp
from optixstuff.disperser import LensletDisperser

from coronachrome.build import build_ir
from coronachrome.render import IFSRenderer


def _renderer(n=6, n_wav=5, fp=(64, 64)):
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
    lam = jnp.linspace(640.0, 680.0, n_wav)
    ir = build_ir(disp, lam, fp_shape=fp)
    return IFSRenderer(ir), ir, fp, n_wav


def test_forward_spmv_shape_and_flux():
    """forward_spmv returns a detector-shaped map with positive total flux."""
    r, ir, fp, n_wav = _renderer()
    cube = jnp.ones((n_wav, fp[0], fp[1]))
    det = r.forward_spmv(cube)
    assert det.shape == ir.det_shape
    assert float(det.sum()) > 0.0


def test_h_mono_matches_spmv():
    """forward_spmv equals the spatial step followed by an H_mono matvec."""
    r, ir, fp, n_wav = _renderer()
    cube = jnp.ones((n_wav, fp[0], fp[1]))
    from coronachrome.render import spatial_sample

    z = spatial_sample(cube, ir)
    direct = (r.H_mono @ z.reshape(-1)).reshape(ir.det_shape)
    assert jnp.allclose(direct, r.forward_spmv(cube))


def test_streaming_matches_spmv():
    """Streaming scatter-add forward equals the BCOO spmv forward."""
    r, _ir, fp, n_wav = _renderer()
    key_cube = jnp.arange(n_wav * fp[0] * fp[1], dtype=float).reshape(
        n_wav, fp[0], fp[1]
    )
    cube = jnp.sin(key_cube)
    assert jnp.allclose(r.forward_streaming(cube), r.forward_spmv(cube), atol=1e-9)
