"""Tests for the IFS renderer forward model."""

import jax
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


def test_adjoint_dot_product_identity():
    """The adjoint satisfies the inner-product identity <H z, y> == <z, H^T y>."""
    r, ir, _fp, _n_wav = _renderer()
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    z = jax.random.normal(k1, (ir.n_channels, ir.n_wav))
    y = jax.random.normal(k2, ir.det_shape)
    hz = r.H_mono @ z.reshape(-1)
    hty = r.adjoint(y)
    assert jnp.allclose(
        jnp.dot(hz, y.reshape(-1)),
        jnp.dot(z.reshape(-1), hty.reshape(-1)),
        atol=1e-8,
    )


def test_forward_is_differentiable():
    """forward_spmv has a finite, nonzero gradient w.r.t. the input cube."""
    r, ir, fp, n_wav = _renderer()
    target = jnp.ones(ir.det_shape)

    def misfit(cube):
        return jnp.sum((r.forward_spmv(cube) - target) ** 2)

    cube0 = jnp.zeros((n_wav, fp[0], fp[1]))
    g = jax.grad(misfit)(cube0)
    assert g.shape == cube0.shape
    assert float(jnp.linalg.norm(g)) > 0.0


def test_end_to_end_point_source_lands_on_detector():
    """A single bright focal-plane pixel produces nonzero detector flux."""
    r, ir, fp, n_wav = _renderer()
    cube = jnp.zeros((n_wav, fp[0], fp[1]))
    cube = cube.at[:, fp[0] // 2, fp[1] // 2].set(1.0)
    det = r.forward_spmv(cube)
    assert det.shape == ir.det_shape
    assert float(det.max()) > 0.0
    assert float(det.sum()) > 0.0
