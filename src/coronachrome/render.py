"""Integrate focal-plane Images and apply the sparse dispersion operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from optixstuff.grids import Image, Measure, integrate_footprints

from coronachrome.ir import SpatialChannelIR


def _check_band_axis(cube, ir):
    """Validate the leading axis of a band-integrated cube.

    Args:
        cube: Focal-plane or lenslet image.
        ir: Operator defining the ordered bands.

    Raises:
        ValueError: The axis name or length differs from the band specification.
    """
    if cube.leading_axes != ("band",):
        raise ValueError(
            "The cube axis indexes wavelength bands (each a Band from the IR), "
            'not monochromatic wavelengths. Relabel band integrals to ("band",).'
        )
    if cube.data.shape[0] != ir.n_bands:
        raise ValueError("cube band axis length must match the IR bands")


def spatial_sample(cube: Image, ir: SpatialChannelIR):
    """Return band-integrated lenslet flux (n_channels,n_bands).

    Integrate piecewise-constant focal pixels over each collection cell.
    Outside the focal grid the density is zero. Square layouts enumerate
    channels x-major, while hex layouts enumerate cells in row order.

    Args:
        cube: Focal-plane Image with one integrated plane per IR band.
        ir: Operator defining the collection cells and bands.

    Returns:
        Lenslet fluxes in channel and band order.

    Raises:
        ValueError: The focal grid, leading axis or band count differs from the IR.
    """
    if cube.grid != ir.fp_grid:
        raise ValueError("cube grid must match the IR focal-plane grid")
    _check_band_axis(cube, ir)
    flux = integrate_footprints(cube, ir.collection_overlap)
    if ir.layout.grid_kind == "square":
        flux = flux[:, ir.lenslet_indices]
    return flux.T


class IFSRenderer(eqx.Module):
    """Sparse band-integrated renderer; detector and cube outputs are Images."""

    ir: SpatialChannelIR
    H_mono: BCOO

    def __init__(self, ir):
        """Compile dispersion footprints into H_mono."""
        self.ir = ir
        ncw = ir.n_channels * ir.n_bands
        n_psf = ir.det_rows.shape[2]
        cols = jnp.repeat(jnp.arange(ncw), n_psf)
        indices = jnp.stack([ir.det_rows.ravel(), cols], axis=1)
        self.H_mono = BCOO(
            (ir.det_vals.ravel(), indices),
            shape=(ir.det_shape[0] * ir.det_shape[1], ncw),
        )

    @property
    def extraction_operator(self):
        """Forward dispersion operator H_mono."""
        return self.H_mono

    def detector_image(self, data, quantity):
        """Wrap detector values with physical pixel geometry and explicit units."""
        return Image(
            data.reshape(self.ir.det_shape),
            self.ir.detector_grid,
            (),
            Measure.integral(),
            quantity,
            "nearest",
        )

    def channel_cube(self, spectra, quantity):
        """Pack (lenslet,band) integrals into a sky Image (band,y,x).

        Unoccupied hex axial pixels are zero; lenslet_indices identifies the
        occupied pixels. Square channel order remains x-major.

        Args:
            spectra: Flux integrals of shape (n_channels, n_bands).
            quantity: Numerator unit of the flux integrals.

        Returns:
            A lenslet-grid Image with leading axis ("band",).
        """
        ir = self.ir
        data = jnp.zeros(
            (ir.n_bands, ir.lenslet_grid.shape[0] * ir.lenslet_grid.shape[1]),
            dtype=spectra.dtype,
        )
        data = data.at[:, ir.lenslet_indices].set(spectra.T)
        return Image(
            data.reshape(ir.n_bands, *ir.lenslet_grid.shape),
            ir.lenslet_grid,
            ("band",),
            Measure.integral(),
            quantity,
            "nearest",
        )

    def channel_spectra(self, cube: Image):
        """Unpack a lenslet cube into (lenslet, band) integrals.

        Args:
            cube: Band-integrated Image on the lenslet grid.

        Returns:
            Flux array of shape (n_channels, n_bands) in IR channel order.

        Raises:
            ValueError: The grid, measure, band axis name, or band count is invalid.
        """
        if cube.grid != self.ir.lenslet_grid or cube.measure != Measure.integral():
            raise ValueError("expected an integral Image on the lenslet grid")
        _check_band_axis(cube, self.ir)
        return cube.data.reshape(self.ir.n_bands, -1)[:, self.ir.lenslet_indices].T

    def _detector_data(self, detector):
        if (
            detector.grid != self.ir.detector_grid
            or detector.measure != Measure.integral()
            or detector.leading_axes
        ):
            raise ValueError("expected an integral detector Image without leading axes")
        return detector.data

    @eqx.filter_jit
    def adjoint(self, detector: Image):
        """Return H_mono^T y as a lenslet-grid Image (band,y,x)."""
        z = self.H_mono.T @ self._detector_data(detector).ravel()
        return self.channel_cube(
            z.reshape(self.ir.n_channels, self.ir.n_bands), detector.quantity
        )

    @eqx.filter_jit
    def forward_spmv(self, cube: Image):
        """Integrate cells and disperse their flux onto the detector."""
        z = spatial_sample(cube, self.ir)
        return self.detector_image(self.H_mono @ z.ravel(), cube.quantity)

    @eqx.filter_jit
    def forward_streaming(self, cube: Image):
        """Integrate cells and disperse with a per-band scatter-add scan."""
        ir = self.ir
        z = spatial_sample(cube, ir)

        def body(detector, w):
            contrib = ir.det_vals[:, w, :] * z[:, w, None]
            return detector.at[ir.det_rows[:, w, :].ravel()].add(contrib.ravel()), None

        det0 = jnp.zeros(ir.det_shape[0] * ir.det_shape[1], dtype=z.dtype)
        detector, _ = jax.lax.scan(body, det0, jnp.arange(ir.n_bands))
        return self.detector_image(detector, cube.quantity)
