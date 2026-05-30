"""Spectral extraction from a dispersed IFS detector image.

Inverts the dispersion operator H_mono (detector -> per-lenslet spectra
z, shape (n_channels, n_wav)). Linear tier: a matched filter, a noise-weighted
least-squares solve (lineax NormalCG, matrix-free, stable gradients), and the
GLS covariance of that solve for per-wavelength error bars. The regularized
positivity + total-variation extractor is a later addition (Plan 3).
"""

import jax
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


def _equilibration(renderer, weights):
    """Per-pixel weights ``w`` and weight-aware Jacobi equilibration ``d``.

    ``w`` is the flattened per-detector-pixel weight (inverse noise variance
    ``1/N``; default uniform). ``d = 1 / sqrt(diag(H^T W H))`` is the column
    equilibration that gives the weighted normal operator a unit diagonal, so
    the NormalCG / CG solves stay well-scaled and float32-safe (the det_vals
    are O(0.1), so without it the normal operator's smallest eigenvalue sits
    below lineax's float32 breakdown safeguard and the solver returns NaN).
    """
    ir = renderer.ir
    n_det = renderer.H_mono.shape[0]
    if weights is None:
        w = jnp.ones(n_det, dtype=renderer.H_mono.data.dtype)
    else:
        w = jnp.asarray(weights).reshape(-1)
    # diag(H^T W H)_(ch,wav) = sum_k det_vals[ch,wav,k]^2 * w[det_rows[ch,wav,k]]
    wdiag = (ir.det_vals**2 * w[ir.det_rows]).sum(axis=2).reshape(-1)  # (ncw,)
    d = 1.0 / jnp.sqrt(jnp.clip(wdiag, 1e-30, None))
    return w, d


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
    w, d = _equilibration(renderer, weights)
    sw = jnp.sqrt(w)
    z_struct = eval_shape(lambda: jnp.zeros(ncw, dtype=y.dtype))
    operator = lx.FunctionLinearOperator(lambda zp: sw * (h_mono @ (d * zp)), z_struct)
    sol = lx.linear_solve(operator, sw * y, solver=lx.NormalCG(rtol=rtol, atol=atol))
    return (d * sol.value).reshape(ir.n_channels, ir.n_wav)


def spectrum_covariance(renderer, weights=None, channels=None, rtol=1e-6, atol=1e-6):
    """Per-channel GLS covariance blocks of the extracted spectrum.

    For each requested lenslet ``channel``, returns the ``(n_wav, n_wav)``
    covariance block ``[(H^T W H)^-1]_kk`` of the noise-weighted least-squares
    spectrum (the same estimator as :func:`lstsq`), with ``W = diag(weights)``.
    This is the Gauss-Markov / Cramer-Rao covariance; it captures intra-spectrum
    wavelength correlations from trace cross-talk, and its diagonal gives the
    per-wavelength error bars (see :func:`spectrum_errorbars`).

    Matrix-free: for each channel it runs ``n_wav`` symmetric-positive-definite
    CG solves against the same weight-equilibrated normal operator as
    :func:`lstsq`, so it never materializes ``H^T W H`` and scales to large IFS
    grids when only a few spaxels are of interest.

    Args:
        renderer: An ``IFSRenderer``.
        weights: Per-detector-pixel inverse noise variance ``1/N`` (default
            uniform). For a true detector-noise covariance whose error bars
            scale with wavelength, pass ``1 / detector.noise_variance(rate, t)``
            evaluated on the noiseless dispersed rate map.
        channels: 1-D array of lenslet channel indices (the spaxels of interest).
        rtol: CG relative tolerance.
        atol: CG absolute tolerance.

    Returns:
        Covariance blocks of shape ``(len(channels), n_wav, n_wav)``.
    """
    ir = renderer.ir
    n_wav = ir.n_wav
    ncw = ir.n_channels * n_wav
    h_mono = renderer.H_mono
    dtype = h_mono.data.dtype
    w, d = _equilibration(renderer, weights)

    def normal_mv(v):
        # D H^T W H D v -- the equilibrated SPD normal operator.
        return d * (h_mono.T @ (w * (h_mono @ (d * v))))

    operator = lx.FunctionLinearOperator(
        normal_mv,
        eval_shape(lambda: jnp.zeros(ncw, dtype=dtype)),
        tags=frozenset({lx.positive_semidefinite_tag, lx.symmetric_tag}),
    )

    def solve_unit(col):
        e = jnp.zeros(ncw, dtype=dtype).at[col].set(1.0)
        return lx.linear_solve(operator, e, solver=lx.CG(rtol=rtol, atol=atol)).value

    def block_for_channel(ch):
        cols = ch * n_wav + jnp.arange(n_wav)
        # xp[j] = (D H^T W H D)^-1 e_{cols[j]} ; (n_wav, ncw)
        xp = jax.vmap(solve_unit)(cols)
        # (equilibrated inverse) block [i, j] = xp[j][cols[i]]
        block_p = xp[:, cols].T
        dch = d[cols]
        # undo equilibration: Cov_z = D (.)^-1 D
        return dch[:, None] * block_p * dch[None, :]

    return jax.vmap(block_for_channel)(jnp.asarray(channels))


def spectrum_errorbars(renderer, weights=None, channels=None, rtol=1e-6, atol=1e-6):
    """Per-wavelength 1-sigma error bars of the extracted spectrum.

    ``sqrt`` of the diagonal of :func:`spectrum_covariance`, shape
    ``(len(channels), n_wav)``. With ``weights = 1 / N`` these scale with
    wavelength through the detector noise: the shot term of ``N`` is
    proportional to the dispersed source rate at each wavelength.
    """
    cov = spectrum_covariance(
        renderer, weights=weights, channels=channels, rtol=rtol, atol=atol
    )
    return jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
