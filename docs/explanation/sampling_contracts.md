# Sampling contracts

Inputs are `optixstuff.grids.Image` cubes with data `(band, y, x)`, leading axes
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

The renderer prepares `optixstuff.grids.footprint_overlap` from the focal grid
and the lenslet collection grids. Aligned cells use separable interval
overlaps. Rotated cells use exact convex-polygon intersections. Rendering
applies the stored weights to all wavelengths with JAX.

Cell integrals contribute in proportion to the covered fraction of each
focal pixel. Densities contribute in proportion to the covered area in
their declared unit (grid units squared or steradians). Clipping at the
field edge loses the uncovered flux. A complete tiling conserves total
flux to floating-point roundoff.

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

Both forward methods preserve the numerator unit in `Image.quantity`.
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
`optixstuff.spectral.rebin_bands(values, src_bands, dst_bands, axis=0)` apportions
bin integrals by overlap. For an Image it preserves spatial metadata. Full
coverage conserves the integral to the floating-point floor; partial coverage
loses the uncovered fraction. A source bin is assumed to have constant density.

The throughput element and optional PSFlet pack each declare their interpolation
breakpoints through `spectral_breaks_nm()`. The `Quadrature` splits the channel
Bands at the union of those wavelengths. Its nodes evaluate throughput,
dispersion, chromatic PSFlet shape and wavecal before the bin integral. The
operator assumes constant spectral density inside each input band. Increase
quadrature_order or refine Bands to study numerical convergence.

Hex layouts retain square collection cells at staggered centres, integrated by
the same sparse overlap matrix. The square cells overlap; this is not a physical
hexagonal-aperture model. Extracted hex Images pack the same centres onto an
axial grid with unused pixels zero. Packing does not redefine collection area.
