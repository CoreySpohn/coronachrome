"""IFS renderer: compile the IR into a sparse operator and run the forward model.

forward_spmv applies the dispersion as one BCOO matvec; forward_streaming
(next task) does the same as a per-wavelength scatter-add. The two are
numerically identical.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from coronachrome.ir import SpatialChannelIR


def spatial_sample(cube, ir):
    """Contract a focal-plane cube to per-lenslet flux z (n_channels, n_wav)."""
    cube_flat = cube.reshape(ir.n_wav, -1)
    gathered = cube_flat[:, ir.spatial_src]
    z = jnp.sum(gathered * ir.spatial_w[None], axis=2)
    return z.T


class IFSRenderer(eqx.Module):
    """Backend renderer: holds the IR and a compiled BCOO dispersion operator."""

    ir: SpatialChannelIR
    H_mono: BCOO

    def __init__(self, ir):
        """Compile the dispersion footprints into a BCOO H_mono."""
        self.ir = ir
        ny, nx = ir.det_shape
        ncw = ir.n_channels * ir.n_wav
        n_psf = ir.det_rows.shape[2]
        rows = ir.det_rows.reshape(ncw, n_psf).reshape(-1)
        vals = ir.det_vals.reshape(ncw, n_psf).reshape(-1)
        cols = jnp.repeat(jnp.arange(ncw), n_psf)
        indices = jnp.stack([rows, cols], axis=1)
        self.H_mono = BCOO((vals, indices), shape=(ny * nx, ncw))

    @property
    def extraction_operator(self):
        """The forward operator H_mono, exported for downstream extraction."""
        return self.H_mono

    @eqx.filter_jit
    def adjoint(self, detector):
        """Map a detector image back to channel space: z = H_mono^T y."""
        zt = self.H_mono.T @ detector.reshape(-1)
        return zt.reshape(self.ir.n_channels, self.ir.n_wav)

    @eqx.filter_jit
    def forward_spmv(self, cube):
        """Forward via one BCOO matvec: spatial sample, then H_mono @ z."""
        z = spatial_sample(cube, self.ir)
        y = self.H_mono @ z.reshape(-1)
        return y.reshape(self.ir.det_shape)

    @eqx.filter_jit
    def forward_streaming(self, cube):
        """Forward via per-wavelength scatter-add accumulation into the detector."""
        ir = self.ir
        cube_flat = cube.reshape(ir.n_wav, -1)
        ny, nx = ir.det_shape

        def body(detector, w):
            sampled = cube_flat[w][ir.spatial_src]
            zc = jnp.sum(sampled * ir.spatial_w, axis=1)
            rows_w = ir.det_rows[:, w, :]
            vals_w = ir.det_vals[:, w, :]
            contrib = (vals_w * zc[:, None]).reshape(-1)
            detector = detector.at[rows_w.reshape(-1)].add(contrib)
            return detector, None

        det0 = jnp.zeros(ny * nx)
        detector, _ = jax.lax.scan(body, det0, jnp.arange(ir.n_wav))
        return detector.reshape(ir.det_shape)
