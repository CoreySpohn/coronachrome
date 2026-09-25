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

# Sampling contracts

Two quantities in an IFS simulation look like free parameters but are not: how
many focal-plane cube pixels feed one lenslet, and how many spectral channels
sample the band. Choosing either by hand invites silent inconsistency between
the hardware description and the simulation grid, so coronachrome derives both
from first principles and validates them at build time. This page states the
two contracts and the API that implements them.

## Spatial sampling is derived, not chosen

The lenslet pitch on sky is an instrument property, and the focal-plane cube's
plate scale is a simulation property. Their ratio, the number of cube pixels
per lenslet cell, is therefore a derived quantity:

$$
\texttt{fp\_px\_per\_lenslet}
  = \frac{\texttt{sky\_pitch\_arcsec}}{\texttt{fp\_pixel\_scale\_arcsec}}.
$$

The descriptor carries the pitch (`LensletDisperser.sky_pitch_arcsec`, the
lenslet pitch projected on sky) and the build call supplies the cube's
angular plate scale ({func}`~coronachrome.build_ir` keyword
`fp_pixel_scale_arcsec`). Both are plain angles in arcseconds. The unit
choice is deliberate: upstream simulators render every wavelength onto one
fixed angular grid (the imaging detector's), so the cube's plate scale is a
wavelength-independent angle, and quoting either quantity in $\lambda / D$
would smuggle a reference wavelength into a relation that has none. When the
cube comes from an optical path, the plate scale to pass is exactly the
detector's:

```python
disperser = LensletDisperser(..., sky_pitch_arcsec=0.014)
ir = build_ir(
    disperser,
    lam,
    fp_shape=cube.shape[1:],
    fp_pixel_scale_arcsec=path.detector.pixel_scale_arcsec,
)
```

An explicit `fp_px_per_lenslet` override remains available for parity work
against reference implementations that fix the ratio directly. Exactly one of
the two inputs must be given: passing both is an error, and passing neither is
an error rather than a silent default.

`build_ir` also runs two diagnostics at build time, when a mistake is cheap to
see:

- **Nyquist**: if the cube provides fewer than two pixels per lenslet cell,
  the flux-conserving cell integral degrades (the cube cannot resolve the
  cells it is being integrated over), and the build warns. Use a finer cube
  grid.
- **Coverage**: any lenslet cell that extends past the cube bounds receives
  zero weight there, so that spaxel silently loses flux. The build counts the
  affected lenslets and warns.

The figure shows both quantities on one plane. Each outline is a lenslet
collection cell, the square that {func}`~coronachrome.build_ir` integrates the
cube over, drawn by {func}`coronachrome.viz.plot_lenslet_cells` (the `viz`
extra) over the cube's own pixels. A 0.0175 arcsec pitch on a 0.005 arcsec
cube grid gives 3.5 cube pixels per cell, above the Nyquist floor of two. The
cube is too small for this grid, though: the corner cells hang past its edge,
and the build's coverage warning counts them.

```{code-cell} ipython3
import hwostyle
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from optixstuff.disperser import LensletDisperser

from coronachrome import build_ir
from coronachrome import viz

hwostyle.use("dark")

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
    n_lenslets=7,
    psflet_kind="gaussian",
    detector_shape=(128, 128),
    sky_pitch_arcsec=0.0175,
)
fp_shape = (32, 32)
lam = jnp.array([650.0, 660.0, 670.0])
ir = build_ir(disperser, lam, fp_shape, fp_pixel_scale_arcsec=0.005)

yy, xx = np.mgrid[: fp_shape[0], : fp_shape[1]]
scene = np.exp(-((xx - 21.0) ** 2 + (yy - 12.0) ** 2) / (2 * 1.2**2)) + 1e-3
viz.plot_lenslet_cells(
    disperser,
    fp_shape,
    fp_pixel_scale_arcsec=0.005,
    image=scene,
    window=(-6.0, 37.0, -6.0, 37.0),
)
plt.show()
```

## Spectral channels follow the Nyquist rule

A spectrograph with resolving power $R = \lambda / \Delta\lambda$ has
resolution elements of constant width $1 / R$ in log wavelength, so a band
from $\lambda_1$ to $\lambda_2$ holds

$$
n_\mathrm{chan} = \left\lceil R \ln(\lambda_2 / \lambda_1) \right\rceil
$$

of them, one channel per resolution element. This is a correctness contract,
not a convenience. Oversampling the spectrum makes neighbouring columns of the
dispersion operator $H$ near-duplicate, the extraction normal equations turn
near-singular, and the float32 solve breaks down (see the precision discussion
in [the model page](model)). Undersampling wastes resolution the instrument
paid for.

{func}`~coronachrome.n_nyquist_channels` implements the rule,
{func}`~coronachrome.channel_edges` and {func}`~coronachrome.channel_centers`
build the log-spaced grid (constant edge ratio, so $R$ is constant per
channel), and {func}`~coronachrome.spectral_grid` combines them.

The intended usage pattern renders the scene on an oversampled grid for a
smooth underlying spectrum and rebins onto the Nyquist channels that drive the
forward model:

```python
centers_hi, edges_hi = spectral_grid(R, lam1, lam2, oversample=6)
cube_hi = render(scene, centers_hi, jnp.diff(edges_hi))  # bin-integrated rates
edges = edges_hi[::6]
cube = rebin_channels(cube_hi, edges_hi, edges, axis=0)
lam = channel_centers(edges)
```

{func}`~coronachrome.rebin_channels` is flux-conserving for bin-integrated
values (each destination bin takes the overlap fraction of every source bin),
and on an exactly nested grid like the one above it reduces to a plain sum of
sub-channels. Passing the same `edges` to `build_ir(wavelength_edges=edges)`
gives the line-spread-function smear the exact extent of each bin instead of
an approximation from the center spacing.
