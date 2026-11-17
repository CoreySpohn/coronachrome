# Sampling contracts

Inputs are `hwoutils.grids.Image` cubes with data `(band, y, x)`, leading axes
`("wavelength",)`, an arcsec `PixelGrid`, a spatial `measure`, a `quantity`, and
an explicit reconstruction. Pixel `(j, i)` is centred at `(x, y)=(i, j)`.
Generated grids use geometric centres `(n-1)/2`.

## Spatial sampling

`build_ir(disperser, bands, fp_grid, *, layout=None, half=3,
psflet_pack=None, quadrature_order=5)` stores physical geometry. No independent
sampling floats are accepted. The sky pitch comes from the disperser; the
focal-plane spacing comes from the grid. For isotropic axis-aligned grids,
`ir.fp_pixel_scale_arcsec = fp_grid.step` and `ir.fp_px_per_lenslet` is the
lenslet pitch divided by that spacing. Scalar scale inspection raises for
anisotropic or rotated source bases; rendering supports their full bases.

`build_ir` clips each square lenslet cell against focal pixels by convex-polygon
clipping on the host. `ir.collection_overlap` stores the resulting sparse COO
matrix (cells by focal pixels) as static host tuples of indices and areas in
arcsec squared. `spatial_sample(image, ir)` applies it with JAX: one sparse
product per wavelength, batched across bands. No geometry is rebuilt during
rendering, and no dense cell-by-focal-pixel matrix is allocated.

Per-pixel values are divided by focal pixel area to obtain density;
per-unit-area values are already densities. The overlap integral is exact up
to floating-point roundoff for a piecewise-constant image (nearest
reconstruction), including rotated square cells and clipped field edges.
There is no edge renormalization; the density is zero outside the focal grid.
Square cells that tile a field conserve its total flux to roundoff.

For `linear`, `cubic`, and `band_limited` declarations, collection approximates
the declared continuous reconstruction by constant density in each pixel.
It does not integrate those interpolants exactly. This local approximation
keeps collection sparse, including for rotated cells; Fourier reconstruction
would in general couple every cell to every focal pixel. Refine the focal
sampling to assess the error for the actual scene. The band-limited error
measurement below is a smooth-scene example, not a universal bound near Nyquist.

`test_band_limited_cell_integral_error` integrates the zero-padded Fourier
interpolant of a Gaussian with sigma 0.2 arcsec analytically, using sinc factors
for each rotated square aperture (side 0.2 arcsec, angle arcsin(1/sqrt(5))).
On a 3.2 arcsec field with pad factor 2, maximum relative cell errors are
1.579101%, 0.346238%, and 0.094159% at spacings 0.1, 0.05, and 0.025 arcsec.
The measured convergence orders are 2.19 and 1.88. The finest grid achieves
about 0.1% error while retaining sparse local collection; this motivates the
approximation, but sharp or poorly sampled scenes require their own check.

For density inputs, `forward_spmv` and `forward_streaming` require an explicit
`output_quantity` for the integrated unit, for example `"Jy"` when the input
quantity is `"Jy/arcsec2"`. Per-pixel input quantities are preserved.
Builds warn when cells extend beyond the source.

A lambda/D grid must first call `grid.to_arcsec(wavelength_nm)` through its
`Frame`, which declares reference wavelength, diameter and diameter convention.
Each cube has one fixed angular grid and retains its `SkyOrientation`.

## Spectral channels

`Band(lo_nm, hi_nm).channels(n, "log")` returns `Band` intervals with shared
log-spaced edges, including both outer endpoints. Each channel's
`Band.geometric_center_nm` is its reference wavelength in nm;
`Band.center_nm` is the arithmetic centre.

`spectral_grid(R, lo_nm, hi_nm, oversample=1)` returns Bands, with
`n = ceil(R*log(hi_nm/lo_nm))`. Oversampling subdivides each base Band, preserving
its edges exactly.
`hwoutils.spectral.rebin_bands(values, src_bands, dst_bands, axis=0)` apportions
bin integrals by overlap. For an Image it preserves spatial metadata. Full
coverage conserves the integral to the floating-point floor; partial coverage
loses the uncovered fraction. A source bin is assumed to have constant density.

Every bin response is integrated by a `Quadrature` built from the Bands, split
at tabulated throughput and template wavelengths. Nodes evaluate throughput,
dispersion, chromatic PSFlet shape and wavecal before the bin integral. The
operator assumes constant spectral density inside each input band. Increase
quadrature_order or refine Bands to study numerical convergence.

Hex layouts retain square collection cells at staggered centres, integrated by
the same sparse overlap matrix. The square cells overlap; this is not a physical
hexagonal-aperture model. Extracted hex Images pack the same centres onto an
axial grid with unused pixels zero. Packing does not redefine collection area.
