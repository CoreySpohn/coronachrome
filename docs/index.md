# coronachrome

A JAX lenslet integral-field-spectrograph (IFS) forward model and spectral
extractor for the Habitable Worlds Observatory.

`coronachrome` takes a multi-wavelength focal-plane datacube (for example from
{mod}`coronagraphoto`) together with a hardware description (an
{class}`optixstuff.OpticalPath` carrying a lenslet disperser), and produces the
dispersed IFS detector image: a field of per-lenslet micro-spectra, where each
lenslet's light is spread into a short spectrum on the detector. It also inverts
that forward model to recover per-lenslet spectra from a detector image, with
calibrated uncertainties. The whole forward and inverse path is JAX-native,
so it is JIT-compilable, GPU-capable, and differentiable end to end.

## What it does

- **Forward model.** A lenslet array samples the focal plane, and each lenslet
  disperses its light into a micro-spectrum placed on the detector. coronachrome
  compiles this geometry into a single sparse operator and applies it as one
  matrix-vector product.
- **Extraction.** Given a detector image and a per-pixel noise model, coronachrome
  recovers the per-lenslet spectra by inverting the same operator: a matched
  filter, a noise-weighted least-squares solve, and the least-squares covariance
  for per-wavelength error bars.
- **Differentiable.** The forward pass and the extraction both carry gradients, so
  coronachrome can sit inside a fitting or spectral-retrieval loop.
- **Sampling contracts.** The focal-plane pixels per lenslet are derived from the
  descriptor's on-sky pitch and the cube plate scale (with build-time coverage and
  Nyquist diagnostics), and spectral channel grids follow the one-channel-per-
  resolution-element rule with flux-conserving rebinning. See
  [Sampling contracts](explanation/sampling_contracts).
- **PSFlet templates.** Beyond the analytic Gaussian and Moffat profiles, a
  documented template-pack format carries tabulated, field- and
  wavelength-dependent PSFlets (with optional wavecal centroid corrections) from
  any generator into the forward model. See
  [PSFlet template packs](explanation/psflet_templates).

## Where it sits

coronachrome is a forward model. It does not generate scenes, compute PSFs, or
apply detector noise:

- Scenes (stars, planets, disks) come from {mod}`skyscapes`, rendered to a
  focal-plane datacube by {mod}`coronagraphoto`.
- Hardware (primary, coronagraph, detector, and the lenslet disperser) is described
  by {mod}`optixstuff`. The detector noise model, including the per-pixel variance
  used to weight the extraction, lives on the optixstuff detector.
- Post-processing and spectral retrieval consume coronachrome's extracted spectra
  and their covariance.

## Heritage

coronachrome is a JAX reimplementation of the forward model in
[CRISPY](https://github.com/mjrfringes/crispy), the WFIRST/Roman CGI IFS simulator
from NASA Goddard. It keeps CRISPY's physics and inverse-problem structure,
generalizes the WFIRST-specific dispersion to a configurable model, replaces the
nested per-lenslet loops with a single sparse matrix-vector product, and adds
differentiability and an analytic extraction covariance. See
[CRISPY heritage](explanation/crispy_heritage) for the full mapping.

## Roadmap

coronachrome's linear-operator core is in place; the library is growing along
three axes:

- **Geometries.** The lenslet array is the primary IFS geometry. An
  image-slicer builder is planned as a second `build_ir` registration on its
  own descriptor type, targeting ultraviolet channels where lenslets are not
  practical; the singledispatch seam exists for exactly this.
- **Extraction.** The linear tier (matched filter, least squares, covariance)
  is complete. A regularized extractor with positivity and total-variation
  penalties is planned for crowded fields and low signal-to-noise regimes
  where the unconstrained least squares rings.
- **PSFlet fidelity.** The template-pack seam accepts physically propagated
  PSFlets today (see [PSFlet template packs](explanation/psflet_templates));
  a wave-optics generator that emits packs from a propagated lenslet train
  (micro-pupil, sinc-squared wings, pinhole masks, field-dependent
  aberrations, per-lenslet wavelength solutions) is planned in the
  [physicaloptix](https://pypi.org/project/physicaloptix/) package, alongside
  an end-to-end validation of the forward model against CRISPY output for an
  identical scene.

```{toctree}
:maxdepth: 1
:caption: Explanation
:hidden:

explanation/model
explanation/sampling_contracts
explanation/psflet_templates
explanation/mathematical_formulation
explanation/crispy_heritage
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/coronachrome/index
```
