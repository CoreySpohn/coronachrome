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
