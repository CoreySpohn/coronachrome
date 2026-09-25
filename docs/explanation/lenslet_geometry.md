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

# Lenslet geometry and detector traces

This page follows light through a lenslet IFS as coronachrome models it: from
the entrance cube, through the lenslet collection cells, onto the detector as
overlapping micro-spectra, and back out as extracted lenslet-bin fluxes whose
uncertainties are correlated. Every position drawn here comes from the same
functions {func}`~coronachrome.build_ir` uses to place its footprints
({func}`coronachrome.build.lenslet_cell_centers`,
{func}`coronachrome.build.detector_centroids`,
{func}`coronachrome.build.detector_trace_origin`), and every detector image is
built from the footprints stored in the IR, so the figures show the operator
the forward model applies. The plots come from {mod}`coronachrome.viz`, which
needs the `viz` extra (`pip install 'coronachrome[viz]'`).

The instrument below is synthetic: a 7 by 7 square lenslet grid rotated by
$\arctan(1/2)$, linear dispersion in $\log\lambda$, Moffat PSFlets, and ten
Nyquist channels from 600 to 720 nm.

```{code-cell} ipython3
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from optixstuff.disperser import LensletDisperser

import eyepiece as ep
from coronachrome import IFSRenderer, build_ir, spectral_grid, spectrum_covariance
from coronachrome import viz

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")

disperser = LensletDisperser(
    pitch_m=174e-6,
    pixsize_m=13e-6,
    angle_rad=float(np.arctan(0.5)),
    lam_ref_nm=660.0,
    pix_per_reselt=2.0,
    dispersion_coeffs=jnp.array([140.0, 0.0]),
    psflet_params=jnp.array([1.3, 2.5]),
    psflet_ref_nm=660.0,
    grid_kind="square",
    n_lenslets=7,
    psflet_kind="moffat",
    detector_shape=(120, 120),
)
lam, edges = spectral_grid(50.0, 600.0, 720.0)
fp_shape = (64, 64)
ir = build_ir(
    disperser, lam, fp_shape, fp_px_per_lenslet=6.0, wavelength_edges=edges, half=4
)

# Two lenslets that are neighbors on the grid: lenslet-index (0, 0) and (1, 0).
a, b = 24, 31
styles = ep.SourceStyles([f"lenslet {a}", f"lenslet {b}"])
```

The build warns that two (lenslet, wavelength) footprints fall entirely off
the detector: the shortest-wavelength bin of lenslet 6 and the
longest-wavelength bin of lenslet 42, at opposite edges. The clipping section
below returns to the detector edge.

## Collection cells on the entrance plane

Each lenslet integrates the cube over its square cell, one lenslet pitch on a
side and rotated by the lenslet angle. Three reference points are easy to
conflate and are kept separate. The **optical center** belongs to whoever made
the cube; here the scene is centered on the geometric array center,
$((n_x - 1)/2, (n_y - 1)/2) = (31.5, 31.5)$. The **lenslet-grid origin** is
where lenslet-index $(0, 0)$ sits, which coronachrome fixes at
$(n_x/2, n_y/2) = (32, 32)$ in cube pixel-center coordinates. For an even cube
the two differ by half a pixel on each axis, which the zoomed panel shows. The
third point, the detector trace origin, lives on the detector and appears in
the next section.

```{code-cell} ipython3
yy, xx = np.mgrid[: fp_shape[0], : fp_shape[1]]
optical_center = (31.5, 31.5)
scene = np.exp(
    -((xx - optical_center[0]) ** 2 + (yy - optical_center[1]) ** 2) / (2 * 3.0**2)
)
scene += 0.05 * np.exp(-((xx - 44.0) ** 2 + (yy - 38.0) ** 2) / (2 * 1.5**2))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
for ax, window in zip(axes, (None, (27.0, 37.0, 27.0, 37.0))):
    viz.plot_lenslet_cells(
        disperser,
        fp_shape,
        fp_px_per_lenslet=6.0,
        image=scene,
        channels=(a, b),
        optical_center_px=optical_center,
        styles=styles,
        window=window,
        colorbar=ax is axes[1],
        ax=ax,
    )
axes[1].set_title("zoom: two origins half a pixel apart")
plt.show()
```

## PSFlets, traces, and overlap on the detector

On the detector, each (lenslet, wavelength) pair lands as one PSFlet whose
**centroid** is the point its footprint offsets are measured from. Dispersion
runs along detector $x$: the offset is a polynomial in
$\log(\lambda / \lambda_\mathrm{ref})$, so with a positive leading coefficient
longer wavelengths sit at larger $x$, and one lenslet's PSFlets across the band
form its **trace**. The **detector trace origin** (hollow circle) is where the
lenslet-grid origin lands at zero dispersion offset, $(n_x/2, n_y/2)$ of the
detector. Here the dispersion polynomial has no constant term, so it coincides
with the 660 nm centroid of lenslet $(0, 0)$.

The image is the operator's response to a unit flux in every wavelength bin of
the two lenslets, on a log stretch. The traces of the two grid neighbors run
parallel and six pixels apart, so the Moffat wings of one footprint fall inside
the footprint of the other: that overlap is the cross-talk extraction has to
undo. The box outlines the detector pixels the operator assigns to a single
PSFlet at 651 nm.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8.0, 3.6), layout="constrained")
result = viz.plot_traces(
    ir,
    disperser,
    lam,
    channels=(a, b),
    scan_index=4,
    styles=styles,
    colorbar="figure",
    ax=ax,
)
plt.show()
```

Scanning the highlight along the trace moves only the highlighted centroid,
its footprint box, and the readout. The image and its color scale stay fixed,
because the scan is a change of the wavelength being pointed at, not photons
arriving in sequence: every bin is on the detector at once.

```{code-cell} ipython3
def draw(fig, k):
    result.update(k)


anim = ep.animate(result.fig, draw, range(len(lam)), fps=3)
HTML(anim.jshtml(dpi=80))
```

## Clipping at the detector edge

A trace near the edge of the detector loses footprint pixels. The dashed line
is the detector boundary; pixels beyond it receive nothing. coronachrome
renormalizes each surviving footprint to unit flux, so the pixels that remain
carry the whole bin flux rather than the fraction that physically reached the
detector.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
viz.plot_traces(
    ir,
    disperser,
    lam,
    channels=(43,),
    window=(98.0, 124.0, 46.0, 62.0),
    colorbar="figure",
    ax=ax,
)
plt.show()
```

## Extracted fluxes share uncertainty

Extraction inverts the operator. With per-pixel weights $W$ the least-squares
spectrum has covariance $(H^\top W H)^{-1}$, and
{func}`~coronachrome.spectrum_covariance` returns its within-lenslet blocks.
Neighboring wavelength bins of one trace overlap on the detector, so their
extracted fluxes are anticorrelated.

```{code-cell} ipython3
renderer = IFSRenderer(ir)
block = spectrum_covariance(renderer, channels=jnp.array([a]))
viz.plot_channel_covariance(block, block=0, wavelengths_nm=lam)
plt.show()
```

The within-lenslet blocks leave out the covariance between lenslets. For this
small grid the full matrix fits in memory, so it can be formed densely (with
uniform weights, as above) and the two neighbors' blocks read off together.
The two (lenslet, wavelength) columns whose footprints fell off the detector
are all zero and would make the normal matrix singular, so they are dropped
first. The diagonal blocks match the within-lenslet result above. The
off-diagonal blocks hold the covariance that overlapping traces put between
different lenslets; in this layout the neighboring traces are six pixels apart
and only the Moffat wings overlap, so those correlations stay near a part in a
thousand. They grow as PSFlet wings widen or traces crowd, and a selected
within-lenslet block is exact only where they are negligible.

```{code-cell} ipython3
h = np.asarray(renderer.H_mono.todense())
live = np.flatnonzero(np.abs(h).sum(axis=0) > 0)  # columns that reach the detector
full = np.linalg.inv(h[:, live].T @ h[:, live])
n_wav = len(lam)
idx = np.r_[a * n_wav : (a + 1) * n_wav, b * n_wav : (b + 1) * n_wav]
where = np.searchsorted(live, idx)
assert np.array_equal(live[where], idx)
pair = full[np.ix_(where, where)]
assert np.allclose(pair[:n_wav, :n_wav], np.asarray(block[0]), rtol=1e-4, atol=1e-8)
sd = np.sqrt(np.diagonal(pair))
cross = pair[:n_wav, n_wav:] / np.outer(sd[:n_wav], sd[n_wav:])
print(f"largest cross-lenslet correlation: {np.abs(cross).max():.1e}")

fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
viz.plot_channel_covariance(
    pair,
    wavelengths_nm=lam,
    channel_labels=[f"lenslet {a}", f"lenslet {b}"],
    max_ticks=3,
    ax=ax,
)
plt.show()
```
