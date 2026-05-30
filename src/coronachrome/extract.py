"""Spectral extraction from a dispersed IFS detector image.

Inverts the dispersion operator H_mono (detector -> per-lenslet spectra
z, shape (n_channels, n_wav)). Linear tier: a matched filter and a
noise-weighted least-squares solve (lineax NormalCG, matrix-free, with
numerically stable gradients). The regularized positivity + total-variation
extractor is a later addition (Plan 3).
"""

import jax.numpy as jnp
import lineax as lx
from jax import eval_shape


def matched_filter(renderer, detector):
    """Matched-filter spectra estimate, shape (n_channels, n_wav).

    Returns ``(H^T y)`` normalized by the per-(channel, wavelength) column
    sum-of-squares, which for this operator equals
    ``(ir.det_vals ** 2).sum(axis=2)``. Fast and differentiable, but biased
    by cross-talk between overlapping traces.
    """
    num = renderer.adjoint(detector)  # (n_channels, n_wav) = (H^T y) reshaped
    colnorm2 = (renderer.ir.det_vals**2).sum(axis=2)  # (n_channels, n_wav)
    return num / jnp.clip(colnorm2, 1e-12, None)


def lstsq(renderer, detector, weights=None, rtol=1e-6, atol=1e-6):
    """Noise-weighted least-squares spectra via lineax NormalCG (matrix-free).

    Solves ``min_z || sqrt(w) * (H_mono z - y) ||^2`` where ``z`` is the
    flattened (n_channels, n_wav) spectra and ``y`` is the flattened detector.
    ``weights`` is a per-detector-pixel weight (default uniform); pass
    ``1 / N`` (N the per-pixel noise variance) for noise-weighted extraction.
    Returns z_hat of shape (n_channels, n_wav). Differentiable through the
    solve with numerically stable gradients.
    """
    ir = renderer.ir
    ncw = ir.n_channels * ir.n_wav
    h_mono = renderer.H_mono
    y = detector.reshape(-1)
    if weights is None:
        sw = jnp.ones_like(y)
    else:
        sw = jnp.sqrt(jnp.asarray(weights).reshape(-1))
    # Column-equilibrate (Jacobi precondition): solve for unit-column-norm
    # variables zp with z = d * zp, so the normal operator has a unit diagonal
    # and O(1) eigenvalues. The det_vals are O(0.1), so without this the normal
    # operator's smallest eigenvalue sits below NormalCG's float32 breakdown
    # safeguard (~100 * size * eps) and the solver returns NaN, even though the
    # operator is well-conditioned. The change of variables leaves z unchanged.
    colnorm = jnp.sqrt((ir.det_vals**2).sum(axis=2)).reshape(-1)
    d = 1.0 / jnp.clip(colnorm, 1e-12, None)
    z_struct = eval_shape(lambda: jnp.zeros(ncw, dtype=y.dtype))
    operator = lx.FunctionLinearOperator(lambda zp: sw * (h_mono @ (d * zp)), z_struct)
    sol = lx.linear_solve(operator, sw * y, solver=lx.NormalCG(rtol=rtol, atol=atol))
    return (d * sol.value).reshape(ir.n_channels, ir.n_wav)
