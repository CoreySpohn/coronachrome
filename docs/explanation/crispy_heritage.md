# CRISPY heritage

coronachrome is a JAX reimplementation of the forward model in
[CRISPY](https://github.com/mjrfringes/crispy) (Coronagraph and Rapid Imaging
Spectrograph in Python), the WFIRST/Roman CGI IFS simulator developed at NASA
Goddard. This page writes out CRISPY's model, shows that coronachrome's sparse
operator is the same model expressed differently, and then lists where the two
genuinely diverge. It is written for readers who already know CRISPY.

## CRISPY's forward model

CRISPY builds the detector image with an explicit three-stage pipeline.

**1. `processImagePlane` (`tools/lenslet.py`).** The input focal-plane image is
padded, rotated by the lenslet angle `par.philens`, and flux-conservatively rebinned
to the lenslet sampling `par.pixperlenslet`. The result is an image in which each
pixel holds the flux collected by one lenslet. Call that per-lenslet flux
$z_{\ell w}$, for lenslet $\ell$ in wavelength sub-bin $w$.

**2. `propagateLenslets` (`tools/lenslet.py`).** For each wavelength sub-bin $w$
across the bandpass, CRISPY:

- interpolates the high-resolution PSFlet template from the `hires_arrs` library,
  linearly in wavelength (the `hires_psflets_lam???.fits` files);
- computes each lenslet centroid $c_\ell(\lambda_w)$ on the detector, either from a
  fitted wavelength solution (`PSFLetPositions`) or from the analytic dispersion
  model `distort()` (lenslet pitch, pixel size, resolving power, and `philens`);
- loops over lenslets, and for each one looks up its flux $z_{\ell w}$, bilinearly
  interpolates the four nearest high-resolution PSFlets, and places that weighted
  PSFlet on the detector subarray at $c_\ell(\lambda_w)$ via `map_coordinates`.

Written as one expression, the detector image is the sum over every channel of its
placed PSFlet:

$$ d(x) = \sum_\ell \sum_w z_{\ell w}\; \phi_{\ell w}\!\left(x - c_\ell(\lambda_w)\right), $$

where $d(x)$ is the detector intensity at detector position $x$, $z_{\ell w}$ is the
per-lenslet flux from step (1), $c_\ell(\lambda_w)$ is the dispersed centroid from
step (2), and $\phi_{\ell w}$ is the interpolated PSFlet for that lenslet and
wavelength.

**3. `rebinDetector` (`tools/detector.py`).** The oversampled detector is
flux-conservatively rebinned to the physical detector pixel scale. Noise is then
applied by `readDetector`, which coronachrome does not port.

The inverse model has two methods: `lstsqExtract`, which fits each micro-spectrum as
a weighted sum of the known PSFlet templates, and `intOptimalExtract`, a matched
filter that uses the inverse wavelength-calibration map.

## The same model as a single operator

The accumulation in `propagateLenslets`,
$d(x) = \sum_{\ell,w} z_{\ell w}\,\phi_{\ell w}(x - c_\ell(\lambda_w))$, is exactly a
sparse matrix-vector product. Index the detector pixels by $i$, with position $x_i$,
and the channels by $(\ell, w)$, and define

$$ H_{i,\,(\ell w)} = \phi_{\ell w}\!\left(x_i - c_\ell(\lambda_w)\right). $$

Then CRISPY's loop computes $y = H z$, where $y$ collects the detector values
$d(x_i)$ over pixels. Each column of $H$ is the placed PSFlet of one
channel, which is precisely what one pass of the inner loop deposits on the detector.
(See [mathematical formulation](mathematical_formulation) for $H$ and the inverse
problem in full.)

The two codes therefore compute the same operator. CRISPY assembles and applies $H$
implicitly, rebuilding the placed PSFlets inside the per-lenslet loop on every call.
coronachrome assembles $H$ once, offline, into the Spatial Channel IR and a sparse
`H_mono` matrix (`build_ir`), and then applies it as a single matvec (`forward_spmv`).
The `rebinDetector` step is not separate in coronachrome; the detector-pixel weights
are baked into the footprint of $H$. Extraction matches the same way: CRISPY's
`lstsqExtract` is least squares against the PSFlet basis, which is the least-squares
inverse of the same $H$ (`lstsq`), and `intOptimalExtract` is the matched filter
(`matched_filter`).

| CRISPY | coronachrome | relation |
|---|---|---|
| `processImagePlane` (rotate + rebin to lenslets) | spatial sampling $z = Sf$ | same role, $f \to z$; different sampling kernel |
| `propagateLenslets` loop | $y = Hz$ (`forward_spmv`) | identical operator; the loop is the matvec |
| `hires_arrs` PSFlet templates | analytic Gaussian / Moffat PSFlet | same role, the column shape; different source |
| `distort` / wavecal centroids | dispersion-polynomial centroids | identical for the analytic case (cross-checked) |
| `rebinDetector` | folded into the $H$ footprint weights | absorbed |
| `lstsqExtract` | `lstsq` | least squares on the same $H$ |
| `intOptimalExtract` | `matched_filter` | matched filter on the same $H$ |

## Where the models differ

The structure above is shared. The columns of $H$, and the map that forms $z$, are
where the physics differs.

- **PSFlet shape ($\phi_{\ell w}$).** CRISPY reads data-driven, spatially varying
  high-resolution templates from FITS (Zemax spot diagrams, so the PSFlet carries
  field-dependent aberrations). coronachrome uses an analytic Gaussian or Moffat
  PSFlet with a line-spread-function smear, the same shape at every field point. This
  changes the columns of $H$, not the structure of the model. Reading FITS templates
  is planned.
- **Spatial sampling ($S$).** CRISPY rotates the whole focal plane by `philens` and
  flux-conservatively rebins it to lenslet sampling. coronachrome samples the focal
  plane at each lenslet center with a bilinear footprint.
- **Centroids ($c_\ell(\lambda_w)$).** CRISPY can use a fitted wavelength solution or
  the analytic `distort` model. coronachrome uses the analytic dispersion polynomial
  in $\log(\lambda / \lambda_\mathrm{ref})$, whose centroids reproduce CRISPY's
  `distort` form exactly. CRISPY's hardcoded twenty-coefficient WFIRST polynomial is
  the special case of a configurable one.
- **Detector noise.** CRISPY models an EMCCD (electron-multiplying register
  statistics, photon counting, charge traps). coronachrome emits a noiseless rate map
  and lets the optixstuff detector apply noise.
- **Wavelength calibration.** CRISPY fits the wavelength solution from monochromatic
  calibration data. coronachrome treats the dispersion as a known analytic model.
- **Execution.** CRISPY runs the per-lenslet loops on the CPU with numpy, scipy, and
  multiprocessing. coronachrome compiles $H$ and applies it as one sparse matvec,
  JIT-compiled, GPU-capable, and differentiable end to end. This is the reason for the
  port.

## What coronachrome adds

- **Differentiability.** Gradients flow through both $y = Hz$ and the extraction.
- **Uncertainties.** The least-squares extraction comes with the analytic covariance
  $(H^\top W H)^{-1}$ and per-wavelength error bars, which CRISPY does not provide.
- **Generality.** Hexagonal as well as square grids, Moffat as well as Gaussian
  PSFlets, and a configurable dispersion polynomial.

## Validation

coronachrome's lenslet centroids are cross-checked against CRISPY's rotation, scale,
and `distort` form directly, and match to numerical precision. The forward and
inverse pair is checked internally by round-trip recovery: inject a spectrum, run the
forward model, extract, and compare. A full end-to-end comparison against CRISPY
output for an identical scene is not yet done. It is the main remaining parity check,
and it is governed by the PSFlet-shape and spatial-sampling differences above rather
than by the dispersion geometry.
