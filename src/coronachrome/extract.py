"""Spectral extraction from a dispersed IFS detector image.

Inverts the dispersion operator H_mono (detector -> per-lenslet spectra
z, shape (n_channels, n_wav)). Linear tier: a matched filter, a noise-weighted
least-squares solve (a lineax conjugate-gradient solve on the normal
equations, matrix-free, with stable gradients), and the
GLS covariance of that solve for per-wavelength error bars. The regularized
positivity + total-variation extractor is a later addition (Plan 3).

The module also carries the parametric face of the same operator: when the
scene is described by a few basis functions (a planet template with one free
amplitude per wavelength, plus a residual-speckle mode basis) rather than by a
free spectrum per spaxel, the measurement covariance follows from the Fisher
matrix of those basis functions, with the nuisance block marginalized. See
:func:`marginal_covariance`.
"""

import jax
import jax.numpy as jnp
import lineax as lx
from jax import eval_shape

from coronachrome.render import spatial_sample


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
    the CG solves stay well-scaled and float32-safe (the det_vals
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


def lstsq(renderer, detector, weights=None, damping=0.0, rtol=1e-6, atol=1e-6):
    """Noise-weighted least-squares spectra via a lineax CG solve (matrix-free).

    Solves ``min_z || sqrt(w) * (H_mono z - y) ||^2 + damping * || z_eq ||^2``
    where ``z`` is the flattened (n_channels, n_wav) spectra, ``y`` is the
    flattened detector, and ``z_eq`` is ``z`` in the column-equilibrated
    coordinates (so ``damping`` is relative to the unit-diagonal normal
    operator). ``weights`` is a per-detector-pixel weight (default uniform);
    pass ``1 / N`` (N the per-pixel noise variance) for noise-weighted
    extraction. Returns z_hat of shape (n_channels, n_wav). Differentiable
    through the solve with numerically stable gradients.

    Precision and conditioning: the forward model and this solve run in the
    stack's native precision (float32 by default). A well-sampled extraction
    (the number of wavelengths matched to the micro-spectrum's resolving power)
    is float32-safe even for large lenslet grids. An over-sampled spectrum makes
    neighbouring columns of ``H`` near-duplicate, so the normal equations become
    near-singular and the float32 solve can break down (non-finite); recover by
    reducing the wavelength count, enabling x64 (the global ``jax_enable_x64``
    flag), or raising ``damping``. The covariance path
    (:func:`spectrum_covariance`) squares the conditioning and in practice needs
    x64.

    Args:
        renderer: An ``IFSRenderer`` holding the dispersion operator.
        detector: The dispersed detector image.
        weights: Per-detector-pixel weight (inverse noise variance), default
            uniform.
        damping: Tikhonov regularization on the equilibrated spectra, relative
            to the unit-diagonal normal operator. ``0`` (default) is the plain
            least-squares estimate; a small positive value trades a little bias
            for stability on ill-conditioned (over-sampled) extractions.
        rtol: CG relative tolerance.
        atol: CG absolute tolerance.

    Returns:
        Extracted spectra of shape ``(n_channels, n_wav)``.
    """
    ir = renderer.ir
    ncw = ir.n_channels * ir.n_wav
    h_mono = renderer.H_mono
    y = detector.reshape(-1)
    w, d = _equilibration(renderer, weights)
    sw = jnp.sqrt(w)
    z_struct = eval_shape(lambda: jnp.zeros(ncw, dtype=y.dtype))
    if damping > 0.0:
        # Augment A -> [A; sqrt(damping) I], b -> [b; 0] so the solver handles the
        # Tikhonov problem while keeping its (square-root) conditioning advantage
        # over forming H^T W H explicitly.
        sd = jnp.sqrt(jnp.asarray(damping, dtype=y.dtype))
        operator = lx.FunctionLinearOperator(
            lambda zp: jnp.concatenate([sw * (h_mono @ (d * zp)), sd * zp]), z_struct
        )
        rhs = jnp.concatenate([sw * y, jnp.zeros(ncw, dtype=y.dtype)])
    else:
        operator = lx.FunctionLinearOperator(
            lambda zp: sw * (h_mono @ (d * zp)), z_struct
        )
        rhs = sw * y
    solver = lx.Normal(lx.CG(rtol=rtol, atol=atol))
    sol = lx.linear_solve(operator, rhs, solver=solver)
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
    grids when only a few spaxels are of interest. The channels and their unit
    columns are looped with ``lax.map`` (not ``vmap``): vmapping an iterative
    solver batches its whole while-loop into one program whose compile time and
    memory grow with the batch, so ``lax.map`` keeps compilation bounded.

    Precision: this forms the (equilibrated) normal operator explicitly, which
    squares the condition number, so the CG solves are markedly less
    float32-stable than :func:`lstsq`. Run the covariance under x64 (the global
    ``jax_enable_x64`` flag) for reliable results.

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
        xp = jax.lax.map(solve_unit, cols)
        # (equilibrated inverse) block [i, j] = xp[j][cols[i]]
        block_p = xp[:, cols].T
        dch = d[cols]
        # undo equilibration: Cov_z = D (.)^-1 D
        return dch[:, None] * block_p * dch[None, :]

    # lax.map (not vmap) over channels and unit columns: each block is an
    # iterative CG solve, and vmapping a solver batches its whole while-loop into
    # one program whose compile time and memory grow with the batch (minutes even
    # for one spaxel). lax.map compiles a single solve body and loops it, keeping
    # compilation O(1) and memory bounded, at the cost of running the solves
    # sequentially -- the right trade for an expensive, iterative body.
    return jax.lax.map(block_for_channel, jnp.asarray(channels))


def amplitude_basis(template, ir):
    """Channel-space basis for one free amplitude per wavelength.

    The characterization science block: a source whose spatial distribution is
    known at every wavelength (an off-axis PSF at the planet position, say) but
    whose brightness at each wavelength is the unknown. Column ``j`` carries the
    template's channel weights in wavelength slot ``j`` and zero elsewhere, so
    contracting the basis against a spectrum reproduces the spatially sampled,
    spectrally scaled template.

    Args:
        template: Focal-plane cube ``(n_wav, ny, nx)`` of the source's spatial
            distribution per wavelength, in the units the amplitudes are
            wanted in (normalize each plane to unit sum for amplitudes that
            mean "total rate at this wavelength").
        ir: The ``SpatialChannelIR`` whose spatial sampling defines the
            contraction.

    Returns:
        Basis of shape ``(n_wav, n_channels, n_wav)``.
    """
    z = spatial_sample(template, ir)  # (n_channels, n_wav)
    eye = jnp.eye(ir.n_wav, dtype=z.dtype)
    return z[None] * eye[:, None, :]


def _prior_precision(nuisance_cov, n_nuisance, dtype):
    """Prior precision block from a covariance, a diagonal, or None (flat)."""
    if nuisance_cov is None:
        return jnp.zeros((n_nuisance, n_nuisance), dtype=dtype)
    cov = jnp.asarray(nuisance_cov, dtype=dtype)
    if cov.ndim == 1:
        return jnp.diag(1.0 / cov)
    return jnp.linalg.inv(cov)


def _prior_covariance(nuisance_cov, dtype):
    """Dense prior covariance from a covariance or a diagonal."""
    if nuisance_cov is None:
        raise ValueError(
            "unmodeled_covariance needs a nuisance_cov: the inflation it "
            "reports is the prior spread of the unmodeled term, which is "
            "unbounded under a flat prior"
        )
    cov = jnp.asarray(nuisance_cov, dtype=dtype)
    return jnp.diag(cov) if cov.ndim == 1 else cov


def _basis_fisher(renderer, cols, weights):
    """Fisher matrix ``C^T H^T W H C`` for channel-space basis columns.

    Matrix-free in the detector dimension: each basis column is pushed through
    the weighted normal operator ``H^T W H`` (one forward and one adjoint spmv),
    so the largest array ever formed is ``(n_basis, n_channels * n_wav)`` rather
    than ``(n_detector_pixels, n_basis)``. The columns are looped with
    ``lax.map`` for the same bounded-memory reason as
    :func:`spectrum_covariance`.
    """
    h_mono = renderer.H_mono
    n_det = h_mono.shape[0]
    if weights is None:
        w = jnp.ones(n_det, dtype=h_mono.data.dtype)
    else:
        w = jnp.asarray(weights).reshape(-1)
    flat = jnp.asarray(cols).reshape(cols.shape[0], -1)  # (n_basis, ncw)

    def normal_mv(v):
        return h_mono.T @ (w * (h_mono @ v))

    pushed = jax.lax.map(normal_mv, flat)  # (n_basis, ncw)
    fisher = flat @ pushed.T
    return 0.5 * (fisher + fisher.T)


def _split_fisher(renderer, science, nuisance, weights):
    """Fisher blocks of the stacked (science, nuisance) basis."""
    n_sci = science.shape[0]
    stacked = jnp.concatenate([jnp.asarray(science), jnp.asarray(nuisance)], axis=0)
    fisher = _basis_fisher(renderer, stacked, weights)
    return (
        fisher[:n_sci, :n_sci],
        fisher[:n_sci, n_sci:],
        fisher[n_sci:, n_sci:],
    )


def marginal_covariance(
    renderer, science, nuisance=None, weights=None, nuisance_cov=None
):
    """Covariance of the science amplitudes with the nuisance block marginalized.

    The honest measurement covariance when the scene is parameterized rather
    than extracted spaxel by spaxel. With a science basis ``S`` and a nuisance
    basis ``N`` (both in channel space), the joint Fisher matrix of the
    detector likelihood is ``F = [S N]^T H^T W H [S N]``, and marginalizing the
    nuisance leaves the Schur complement

    ``Cov = (F_ss - F_sn (F_nn + Sigma_nu^-1)^-1 F_ns)^-1``.

    This is what :func:`spectrum_covariance` cannot express. ``R_spec =
    (H^T W H)^-1`` is the covariance *conditional on a known residual speckle
    field*; pass a speckle mode basis here and the field's uncertainty enters
    the science error bars by construction. The two limits bracket it: with
    ``nuisance=None`` this returns the conditional covariance (speckle known
    exactly), and :func:`unmodeled_covariance` returns the covariance of an
    estimator that ignores the nuisance entirely. Conditional <= marginal <=
    unmodeled always, since ``F_nn - F_ns F_ss^-1 F_sn`` is positive
    semidefinite.

    Precision: this forms the basis Fisher matrix explicitly, squaring the
    conditioning of ``H``, so run it under x64 (the global ``jax_enable_x64``
    flag) as for :func:`spectrum_covariance`.

    Args:
        renderer: An ``IFSRenderer`` holding the dispersion operator.
        science: Channel-space science basis ``(n_science, n_channels, n_wav)``,
            for example from :func:`amplitude_basis`.
        nuisance: Channel-space nuisance basis ``(n_nuisance, n_channels,
            n_wav)``, for example ``jax.vmap(spatial_sample, in_axes=(0, None))``
            over a speckle mode cube stack. ``None`` (default) returns the
            conditional covariance.
        weights: Per-detector-pixel inverse noise variance ``1 / N`` (default
            uniform), as for :func:`lstsq`.
        nuisance_cov: Prior covariance of the nuisance coefficients: a full
            ``(n_nuisance, n_nuisance)`` matrix, a 1-D per-mode variance (the
            generator PSD), or ``None`` for an improper flat prior, which needs
            a nonsingular ``F_nn``.

    Returns:
        Covariance of shape ``(n_science, n_science)``.
    """
    if nuisance is None:
        return jnp.linalg.inv(_basis_fisher(renderer, jnp.asarray(science), weights))
    f_ss, f_sn, f_nn = _split_fisher(renderer, science, nuisance, weights)
    prec = f_nn + _prior_precision(nuisance_cov, f_nn.shape[0], f_nn.dtype)
    return jnp.linalg.inv(f_ss - f_sn @ jnp.linalg.solve(prec, f_sn.T))


def unmodeled_covariance(renderer, science, nuisance, weights=None, nuisance_cov=None):
    """Covariance of a science estimator that ignores the nuisance block.

    The practiced pipeline: fit (or extract) the science amplitudes as if the
    residual speckle field were absent, then live with the fact that it is not.
    The estimator is unbiased only in the mean over the nuisance prior, and its
    covariance picks up a nuisance-projection term,

    ``Cov = C + S Sigma_nu S^T``,  ``C = F_ss^-1``,  ``S = C F_sn``,

    which is non-diagonal, correlated across wavelength, and therefore not
    recoverable by inflating per-wavelength error bars. The ratio of this to
    :func:`marginal_covariance` is what a joint fit buys; the ratio of this to
    the conditional covariance (``nuisance=None``) is how much a pipeline
    quoting ``R_spec`` underquotes its own error.

    Args:
        renderer: An ``IFSRenderer`` holding the dispersion operator.
        science: Channel-space science basis ``(n_science, n_channels, n_wav)``.
        nuisance: Channel-space nuisance basis ``(n_nuisance, n_channels,
            n_wav)``.
        weights: Per-detector-pixel inverse noise variance (default uniform).
        nuisance_cov: Prior covariance of the nuisance coefficients (matrix or
            1-D per-mode variance). Required: the inflation is unbounded under
            a flat prior.

    Returns:
        Covariance of shape ``(n_science, n_science)``.
    """
    f_ss, f_sn, _ = _split_fisher(renderer, science, nuisance, weights)
    prior_cov = _prior_covariance(nuisance_cov, f_ss.dtype)
    conditional = jnp.linalg.inv(f_ss)
    sens = conditional @ f_sn
    return conditional + sens @ prior_cov @ sens.T


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
