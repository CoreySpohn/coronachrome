"""Integrate focal-plane Images and apply the sparse dispersion operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.grids import Image
from jax.experimental.sparse import BCOO

from coronachrome.ir import SpatialChannelIR


def spatial_sample(cube: Image, ir: SpatialChannelIR):
    """Return band-integrated lenslet flux (n_channels,n_wav).

    The stored sparse overlap matrix exactly integrates piecewise-constant
    density: per_pixel values divided by focal pixel area, or per_unit_area
    values as supplied. This is exact for nearest reconstruction. For linear,
    cubic and band_limited declarations it approximates the smooth density
    by constant pixel values; it does not integrate their interpolants.
    Refine the focal grid to control that approximation (see sampling docs).
    Outside the focal grid density is zero, without edge renormalization.
    Hex layouts retain overlapping square apertures at staggered centres.
    """
    if cube.grid != ir.fp_grid:
        raise ValueError("cube grid must match the IR focal-plane grid")
    if cube.leading_axes != ("wavelength",) or cube.data.shape[0] != ir.n_wav:
        raise ValueError("cube must have one wavelength axis matching the IR bands")
    overlap = ir.collection_overlap
    matrix = BCOO(
        (
            jnp.asarray(overlap.areas_arcsec2),
            jnp.asarray(overlap.indices, dtype=jnp.int32).reshape(-1, 2),
        ),
        shape=overlap.shape,
        indices_sorted=True,
        unique_indices=True,
    )
    density = cube.data.reshape(ir.n_wav, -1)
    if cube.measure == "per_pixel":
        density = density / ir.fp_grid.pixel_area
    return matrix @ density.T


class IFSRenderer(eqx.Module):
    """Sparse band-integrated renderer; detector and cube outputs are Images."""

    ir: SpatialChannelIR
    H_mono: BCOO

    def __init__(self, ir):
        """Compile dispersion footprints into H_mono."""
        self.ir = ir
        ncw = ir.n_channels * ir.n_wav
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
            "per_pixel",
            quantity,
            "nearest",
        )

    def channel_cube(self, spectra, quantity):
        """Pack (lenslet,band) integrals into a sky Image (band,y,x).

        Unoccupied hex axial pixels are zero; lenslet_indices identifies the
        occupied pixels. Square channel order remains x-major.
        """
        ir = self.ir
        data = jnp.zeros(
            (ir.n_wav, ir.lenslet_grid.shape[0] * ir.lenslet_grid.shape[1]),
            dtype=spectra.dtype,
        )
        data = data.at[:, ir.lenslet_indices].set(spectra.T)
        return Image(
            data.reshape(ir.n_wav, *ir.lenslet_grid.shape),
            ir.lenslet_grid,
            ("wavelength",),
            "per_pixel",
            quantity,
            "nearest",
        )

    def channel_spectra(self, cube: Image):
        """Unpack a lenslet cube into (lenslet,band) integrals."""
        if cube.grid != self.ir.lenslet_grid or cube.measure != "per_pixel":
            raise ValueError("expected a per_pixel Image on the lenslet grid")
        if cube.leading_axes != ("wavelength",) or cube.data.shape[0] != self.ir.n_wav:
            raise ValueError("cube wavelength axis must match bands")
        return cube.data.reshape(self.ir.n_wav, -1)[:, self.ir.lenslet_indices].T

    def _detector_data(self, detector):
        if (
            detector.grid != self.ir.detector_grid
            or detector.measure != "per_pixel"
            or detector.leading_axes
        ):
            raise ValueError("expected a per_pixel detector Image without leading axes")
        return detector.data

    @eqx.filter_jit
    def adjoint(self, detector: Image):
        """Return H_mono^T y as a lenslet-grid Image (band,y,x)."""
        z = self.H_mono.T @ self._detector_data(detector).ravel()
        return self.channel_cube(
            z.reshape(self.ir.n_channels, self.ir.n_wav), detector.quantity
        )

    def _output_quantity(self, cube, output_quantity):
        if cube.measure == "per_unit_area":
            if output_quantity is None:
                raise ValueError(
                    "area integration changes density units; supply output_quantity"
                )
            return output_quantity
        if output_quantity is not None and output_quantity != cube.quantity:
            raise ValueError("per_pixel rendering preserves the input quantity")
        return cube.quantity

    @eqx.filter_jit
    def forward_spmv(self, cube: Image, *, output_quantity=None):
        """Integrate cells then disperse to a per-pixel detector Image.

        For a density input, output_quantity must name the area-integrated
        unit (e.g. Jy for Jy/arcsec2). Per-pixel input units are preserved.
        """
        z = spatial_sample(cube, self.ir)
        return self.detector_image(
            self.H_mono @ z.ravel(), self._output_quantity(cube, output_quantity)
        )

    @eqx.filter_jit
    def forward_streaming(self, cube: Image, *, output_quantity=None):
        """Integrate cells then disperse with a per-band scatter-add scan.

        output_quantity follows forward_spmv: required for density inputs.
        """
        ir = self.ir
        z = spatial_sample(cube, ir)

        def body(detector, w):
            contrib = ir.det_vals[:, w, :] * z[:, w, None]
            return detector.at[ir.det_rows[:, w, :].ravel()].add(contrib.ravel()), None

        det0 = jnp.zeros(ir.det_shape[0] * ir.det_shape[1], dtype=z.dtype)
        detector, _ = jax.lax.scan(body, det0, jnp.arange(ir.n_wav))
        return self.detector_image(
            detector, self._output_quantity(cube, output_quantity)
        )
