"""Spectral extraction from a dispersed IFS detector image.

Inverts the dispersion operator H_mono (detector -> per-lenslet spectra
z, shape (n_channels, n_wav)). Linear tier: a matched filter and a
noise-weighted least-squares solve (lineax NormalCG, matrix-free, with
numerically stable gradients). The regularized positivity + total-variation
extractor is a later addition (Plan 3).
"""

import jax.numpy as jnp


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
