# coronachrome

**coronachrome** is a JAX lenslet integral-field-spectrograph (IFS) forward
model and spectral extractor for high-contrast imaging, built for the
Habitable Worlds Observatory (HWO) simulation suite.

Given a multi-wavelength focal-plane datacube (for example from
`coronagraphoto`) and a hardware descriptor (an `optixstuff.LensletDisperser`),
coronachrome compiles the instrument geometry into a single sparse linear
operator, renders the dispersed IFS detector image of per-lenslet
micro-spectra, and inverts the same operator to recover per-spaxel spectra
with calibrated uncertainties. Everything is JAX-native: JIT-compilable,
GPU-capable, and differentiable end to end.

## Features

- **Compiled forward model**: the lenslet grid, dispersion polynomial, and
  pixel-integrated PSFlets (Gaussian, Moffat, or tabulated templates) become
  one sparse matrix-vector product, with no per-lenslet loops at runtime.
- **Derived spatial sampling**: the focal-plane pixels per lenslet follow from
  the descriptor's on-sky pitch and the cube plate scale, with build-time
  coverage and Nyquist diagnostics, rather than from a free knob.
- **Nyquist spectral grids**: helpers implement the
  `n = ceil(R ln(lam2 / lam1))` channel rule, log-spaced channel grids, and
  flux-conserving rebinning from an oversampled render grid.
- **PSFlet template packs**: a documented npz format carries pixel-integrated
  PSFlet templates (per field anchor and wavelength, with optional wavecal
  centroid corrections) from any generator, such as a wave-optics propagation
  of the lenslet train, into the forward model.
- **Linear extraction with error bars**: a matched filter, a matrix-free
  noise-weighted least-squares solve, and the per-spaxel Gauss-Markov
  covariance whose diagonal gives per-wavelength uncertainties.
- **One operator everywhere**: the extraction exports the same `H` the
  simulation used, so simulation studies and data reduction share a single
  forward model.

## Installation

```bash
pip install coronachrome
```

## Quick example

```python
import jax.numpy as jnp
from optixstuff import LensletDisperser

from coronachrome import IFSRenderer, build_ir, lstsq, spectral_grid

disperser = LensletDisperser(
    pitch_m=174e-6,
    pixsize_m=13e-6,
    angle_rad=0.4636,
    lam_ref_nm=660.0,
    pix_per_reselt=2.0,
    dispersion_coeffs=jnp.array([100.0, 0.0]),
    psflet_params=jnp.array([0.7]),
    psflet_ref_nm=660.0,
    grid_kind="square",
    n_lenslets=8,
    psflet_kind="gaussian",
    detector_shape=(256, 256),
    sky_pitch_lod=0.5,
)

lam, _ = spectral_grid(resolving_power=50.0, lam_min_nm=620.0, lam_max_nm=700.0)
ir = build_ir(disperser, lam, fp_shape=(64, 64), fp_pixel_scale_lod=0.25)
renderer = IFSRenderer(ir)

cube = jnp.ones((lam.shape[0], 64, 64))  # focal-plane rate maps
detector = renderer.forward_spmv(cube)   # dispersed micro-spectra
spectra = lstsq(renderer, detector)      # per-spaxel extraction
```

## Documentation

Full documentation, including the mathematical formulation, the sampling
contracts, the PSFlet template pack format, and the CRISPY heritage mapping,
lives at [coronachrome.readthedocs.io](https://coronachrome.readthedocs.io).

## Heritage and roadmap

coronachrome is a JAX reimplementation of the forward model in
[CRISPY](https://github.com/mjrfringes/crispy), the WFIRST/Roman CGI IFS
simulator, generalized to configurable dispersion laws and extended with
differentiability and an analytic extraction covariance. Planned directions
include an image-slicer geometry (a second `build_ir` registration), a
regularized positivity and total-variation extractor, and physically
propagated PSFlet template packs. See the documentation roadmap page for
detail.
