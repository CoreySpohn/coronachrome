# Measurement covariance and nuisance marginalization

An IFS measurement is only as good as the error bar attached to it, and the
error bar depends on what the analysis is willing to assume. This page states
what coronachrome's extraction covariance means, why it is optimistic whenever
a residual speckle field is present, and how to compute the covariance that
accounts for that field instead of conditioning on it.

## Two faces of the same operator

The forward model is linear, $y = H x$, where $x$ is the spatially sampled
focal-plane cube in channel space and $y$ is the dispersed detector image. That
single operator supports two different questions.

The **reduction** question is: given a detector image, what is the spectrum in
every spaxel? The unknown is a free spectrum per lenslet, the estimator is the
noise-weighted least squares of {func}`~coronachrome.lstsq`, and its covariance
is the Gauss-Markov matrix computed by
{func}`~coronachrome.spectrum_covariance`,

$$
R_\text{spec} = \left(H^\top W H\right)^{-1}, \qquad W = \operatorname{diag}(1 / N).
$$

The **characterization** question is: given a detector image, how bright is the
planet at each wavelength? Here the unknown is not a free spectrum per spaxel.
The planet's spatial distribution is known at every wavelength, from the
off-axis PSF at its position, and only its brightness per wavelength is
unknown. The scene is parameterized, not extracted.

The distinction matters because the two questions have different covariances,
and because the second one has somewhere to put the residual speckle field.

## What $R_\text{spec}$ assumes

$R_\text{spec}$ is the covariance of the extracted spectrum **conditional on a
known residual speckle field**. It propagates detector noise through the
inverse operator and nothing else. In an observation where the speckle field is
uncertain, and it always is, the honest covariance of the same extraction picks
up a second term,

$$
\operatorname{Cov}[\hat{s}] = R_\text{spec} + S \, \Sigma_\nu \, S^\top,
\qquad S = \frac{\partial \hat{s}}{\partial \nu},
$$

where $\nu$ are the coefficients of the residual field in some mode basis and
$\Sigma_\nu$ is their prior covariance. The added term is not diagonal, is
correlated over long wavelength baselines, and is correlated with the planet's
own extraction. Inflating per-wavelength error bars cannot reproduce it.

## The parametric covariance

Write the scene as a science basis $S$ and a nuisance basis $N$, both in
channel space, so the detector mean is $H (S a + N \nu)$ for science amplitudes
$a$ and nuisance coefficients $\nu$. The joint Fisher matrix of the detector
likelihood is

$$
F = \begin{bmatrix} S & N \end{bmatrix}^\top H^\top W H
    \begin{bmatrix} S & N \end{bmatrix}
  = \begin{bmatrix} F_{ss} & F_{s\nu} \\ F_{\nu s} & F_{\nu\nu} \end{bmatrix},
$$

and three covariances of the science amplitudes follow, each corresponding to a
different treatment of the nuisance block:

| treatment | covariance | meaning |
|---|---|---|
| conditional | $F_{ss}^{-1}$ | the speckle field is known exactly |
| marginal | $\left(F_{ss} - F_{s\nu}\left(F_{\nu\nu} + \Sigma_\nu^{-1}\right)^{-1} F_{\nu s}\right)^{-1}$ | the field is fit jointly with the science |
| unmodeled | $F_{ss}^{-1} + S \Sigma_\nu S^\top$, $\;S = F_{ss}^{-1} F_{s\nu}$ | the field is present but ignored |

The ordering conditional $\preceq$ marginal $\preceq$ unmodeled always holds,
because the Schur complement $F_{\nu\nu} - F_{\nu s} F_{ss}^{-1} F_{s\nu}$ is
positive semidefinite. The conditional covariance is the parametric analogue of
$R_\text{spec}$ and is equally optimistic. The gap between the last two is what
a joint fit buys over a pipeline that extracts first and subtracts a background
estimate afterwards, and the gap between the last and the first is how much
such a pipeline underquotes its own error.

## The API

{func}`~coronachrome.marginal_covariance` returns the marginal covariance, or
the conditional one when no nuisance basis is given.
{func}`~coronachrome.unmodeled_covariance` returns the third.
{func}`~coronachrome.amplitude_basis` builds the science basis for the common
case of one free amplitude per wavelength, and a nuisance basis is a plain
`vmap` of {func}`~coronachrome.spatial_sample` over a stack of mode cubes.

```python
import jax
import jax.numpy as jnp

from coronachrome import (
    amplitude_basis,
    marginal_covariance,
    spatial_sample,
    unmodeled_covariance,
)

# planet_template: (n_wav, ny, nx), unit sum per plane, so amplitudes are rates.
# speckle_modes:   (n_modes, n_wav, ny, nx) from a speckle generator.
# mode_variance:   (n_modes,) prior variance per mode, the generator PSD.
science = amplitude_basis(planet_template, ir)
nuisance = jax.vmap(spatial_sample, in_axes=(0, None))(speckle_modes, ir)

weights = 1.0 / detector.noise_variance(rate_map, exposure_time).reshape(-1)
conditional = marginal_covariance(renderer, science, weights=weights)
marginal = marginal_covariance(
    renderer, science, nuisance, weights=weights, nuisance_cov=mode_variance
)
unmodeled = unmodeled_covariance(
    renderer, science, nuisance, weights=weights, nuisance_cov=mode_variance
)

sigma = jnp.sqrt(jnp.diag(marginal))  # honest per-wavelength error bars
gap = 0.5 * jnp.linalg.slogdet(unmodeled)[1] - 0.5 * jnp.linalg.slogdet(marginal)[1]
```

The last line is the information gap in nats: how much is lost by ignoring the
nuisance rather than marginalizing it.

## Cost and precision

The Fisher matrix is built matrix-free in the detector dimension. Each basis
column is pushed through the weighted normal operator $H^\top W H$ with one
forward and one adjoint sparse product, looped with `lax.map`, so the largest
array formed is `(n_basis, n_channels * n_wav)` rather than one of
`(n_detector_pixels, n_basis)`. The remaining linear algebra is on matrices of
the size of the basis, which is tens to hundreds of columns.

Forming the Fisher matrix squares the conditioning of $H$, exactly as
{func}`~coronachrome.spectrum_covariance` does, so run these under `x64` by
setting the global `jax_enable_x64` flag.

## Choosing the nuisance basis

Nothing here constrains where the modes come from, and coronachrome does not
generate them: a mode basis is an array contract, like the PSFlet template
pack. Principal components of a speckle time series, the response modes of a
wavefront-control basis, and a generator's own draw basis all work. What
matters is that $\Sigma_\nu$ honestly describes the residual field's spread
over the observation, since the marginal covariance interpolates between the
conditional limit (a tight prior, meaning a well-known field) and a much wider
one (a loose prior on modes that overlap the planet template).

When the mode count grows past what a dense $F_{\nu\nu}$ can hold, the standard
remedy in the differential-imaging literature is to factor the spatio-spectral
correlation into separate spatial and spectral parts rather than to model the
joint covariance directly.
