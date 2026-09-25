---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# PSFlet template packs

coronachrome's analytic PSFlets (pixel-integrated Gaussian and Moffat
profiles) are fast and adequate for many studies, but a real lenslet PSFlet is
not an analytic blob. It is the telescope **micro-pupil**, an image of the
pupil whose size is set geometrically and is roughly wavelength-independent,
convolved with the square-lenslet **sinc-squared** diffraction pattern, whose
slow $r^{-2}$ wings drive the cross-talk between neighbouring micro-spectra.
Aberrations make the shape vary across the field, and the true trace centroids
deviate from any global dispersion polynomial (the wavelength calibration
problem). None of that is analytic, so coronachrome accepts **tabulated
PSFlet templates** through a frozen file format, the template pack, and a
third PSFlet mode, `psflet_kind="template"`.

The pack is deliberately a file, not an import: any generator that can write
the format below can feed the forward model, whether it is a wave-optics
propagation of the lenslet train, a calibration fit to lab or on-sky data, or
the analytic reference emitter shipped here. This keeps the forward model and
the PSFlet physics decoupled, in the same way coronagraph performance data
arrives through files rather than a live diffraction code.

## The chromatic caveat, made explicit

The analytic modes scale the PSFlet core width linearly with wavelength
(`psflet_params[0]` times $\lambda / \texttt{psflet\_ref\_nm}$), the
diffraction-limited approximation. In the micro-pupil-dominated regime that
scaling is wrong: the pupil image size is geometric, and only the sinc-squared
blur grows with wavelength. Template mode therefore applies **no width
scaling** and ignores `psflet_params` and `psflet_ref_nm` entirely. Chromatic
morphology comes from tabulating templates at multiple wavelengths; the build
interpolates linearly between the bracketing planes and refuses to
extrapolate outside the tabulated range.

## Pack format (npz, `format_version = 1`)

| field | shape | meaning |
|---|---|---|
| `format_version` | scalar int | must be 1 |
| `templates` | `(n_field, n_lam, n_off, n_off)` | pixel-integrated PSFlet response: entry `[f, l, iy, ix]` is the flux falling in a unit detector pixel whose center sits at offset `(dy, dx) = (offsets[iy], offsets[ix])` px from the PSFlet centroid |
| `offsets` | `(n_off,)` | uniform, ascending pixel-offset axis, shared by both plane axes |
| `wavelengths_nm` | `(n_lam,)` | ascending tabulation wavelengths |
| `field_xy` | `(n_field, 2)` | field-anchor positions in lenslet-index coordinates; each lenslet uses its nearest anchor's templates |
| `centroids` | `(n_field, n_lam, 2)`, optional | `(dx, dy)` pixel corrections to the dispersion-model centroid per anchor and wavelength, the wavecal residual |
| `meta_json` | string, optional | provenance (generator, configuration); stored verbatim, never interpreted |

Two normalization facts make packs easy to emit. Templates carry **shape
only**: the build renormalizes every footprint to unit flux, so the absolute
scale of a plane is irrelevant, and per-wavelength throughput belongs on the
descriptor's throughput element, not in the template amplitudes. Offsets are
pixel-center offsets of the *sampling pixel*, so a generator produces a plane
by integrating its high-resolution PSFlet over a unit-pixel window centered at
each grid offset.

## Using template mode

{class}`~coronachrome.PsfletPack` holds the arrays and validates the
conventions at construction; {func}`~coronachrome.save_psflet_pack` and
{func}`~coronachrome.load_psflet_pack` are the writer/reader pair. The pack
reaches {func}`~coronachrome.build_ir` either as an explicit argument or as a
file reference on the descriptor (`LensletDisperser.psflet_pack_path`), so a
pure-config hardware description stays possible:

```python
from coronachrome import analytic_psflet_pack, build_ir, save_psflet_pack

pack = analytic_psflet_pack(
    "gaussian", jnp.array([0.7]), lam, psflet_ref_nm=660.0
)
save_psflet_pack("psflets.npz", pack)

disperser = LensletDisperser(..., psflet_kind="template",
                             psflet_pack_path="psflets.npz")
ir = build_ir(disperser, lam, fp_shape=fp_shape, fp_px_per_lenslet=4.0)
```

{func}`~coronachrome.analytic_psflet_pack` is the reference emitter: it
tabulates the same erf pixel-integrated profiles the analytic modes use, which
makes it the parity bridge (template mode reproduces the gaussian build to
interpolation tolerance) and a template-mode entry point that needs no
external generator.

At build time, each lenslet is assigned its nearest `field_xy` anchor, the
anchor's planes are blended linearly in wavelength, and the blended plane is
bilinear-sampled at the footprint pixel offsets with the same
line-spread-function smear as the analytic path. When `centroids` are present
the interpolated corrections shift each anchor's trace centroids before the
footprints are placed, so a field-dependent wavelength solution rides in the
same pack. Offsets beyond the tabulated extent contribute zero, and a build
whose footprint half-width exceeds the tabulated extent warns about the
clipped wings.

The correction is applied once, to the centroid, never to the template: the
template stays centered on its own centroid. The figure below builds a
template-mode IR from a pack whose wavelength solution drifts along $y$ by
$\pm 0.6$ px across the band. The thin line is the geometric trace from the
dispersion model alone; the markers are the corrected centroids that
{func}`coronachrome.build.detector_centroids` returns and the footprints are
placed at, so the calibrated correction shows as the markers leaving the line.
It is drawn with {func}`coronachrome.viz.plot_traces` (the `viz` extra).

```{code-cell} ipython3
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from optixstuff.disperser import LensletDisperser

from coronachrome import analytic_psflet_pack, build_ir
from coronachrome import viz

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")

lam = jnp.linspace(600.0, 720.0, 7)
drift = np.zeros((1, lam.shape[0], 2))
drift[0, :, 1] = np.linspace(-0.6, 0.6, lam.shape[0])  # (dx, dy) per wavelength
pack = analytic_psflet_pack(
    "gaussian", jnp.array([0.9]), lam, psflet_ref_nm=660.0, centroids=drift
)
disperser = LensletDisperser(
    pitch_m=174e-6,
    pixsize_m=13e-6,
    angle_rad=float(np.arctan(0.5)),
    lam_ref_nm=660.0,
    pix_per_reselt=2.0,
    dispersion_coeffs=jnp.array([140.0, 0.0]),
    psflet_params=jnp.array([0.9]),
    psflet_ref_nm=660.0,
    grid_kind="square",
    n_lenslets=5,
    psflet_kind="template",
    detector_shape=(96, 96),
)
ir = build_ir(disperser, lam, (40, 40), fp_px_per_lenslet=5.0, psflet_pack=pack)
fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
viz.plot_traces(
    ir, disperser, lam, channels=(12,), psflet_pack=pack, colorbar="figure", ax=ax
)
plt.show()
```

## Emitting packs from a physical-optics generator

A wave-optics library (for example
[physicaloptix](https://pypi.org/project/physicaloptix/), the HWO suite's
physical-optics package) emits a pack by propagating the lenslet train once
per field anchor and wavelength: form the micro-pupil from the local complex
field over the lenslet aperture, apply any pinhole mask, propagate through the
spectrograph relay to the detector plane, integrate the resulting intensity
over unit-pixel windows on the offset grid, and write the plane. Cross-talk
enters the forward model through the tabulated wings, and field-dependent
aberrations enter through multiple anchors. The pack freezes that expensive
propagation into arrays, so the serving path (building and applying $H$)
never runs wave optics.
