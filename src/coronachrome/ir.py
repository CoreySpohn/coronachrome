"""Grid geometry and sparse dispersion footprints of the IFS operator."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
from hwoutils.leaves import as_leaf
from jax import Array
from optixstuff.grids import CellOverlap, PixelGrid
from optixstuff.spectral import Band

from coronachrome.grids import LensletLayout


def _leaf_overlap(overlap):
    """Store the overlap weights as strong leaves; indices stay integers."""

    def leaves(group):
        return None if group is None else tuple(as_leaf(a) for a in group)

    return dataclasses.replace(
        overlap, axes=leaves(overlap.axes), sparse=leaves(overlap.sparse)
    )


class SpatialChannelIR(eqx.Module):
    """Band-integrated dispersion operator with static sampling geometry."""

    det_rows: Array = eqx.field(converter=jnp.asarray)
    det_vals: Array = eqx.field(converter=jnp.asarray)
    lenslet_indices: Array = eqx.field(converter=jnp.asarray)
    bands: tuple[Band, ...] = eqx.field(static=True)
    fp_grid: PixelGrid = eqx.field(static=True)
    lenslet_grid: PixelGrid = eqx.field(static=True)
    detector_grid: PixelGrid = eqx.field(static=True)
    layout: LensletLayout = eqx.field(static=True)
    collection_grids: tuple[PixelGrid, ...] = eqx.field(static=True)

    collection_overlap: CellOverlap = eqx.field(converter=_leaf_overlap)

    @property
    def n_channels(self):
        """Number of spatial lenslet channels (distinct from spectral bands)."""
        return self.layout.n_lenslets**2

    @property
    def n_wav(self):
        """Number of spectral bands."""
        return len(self.bands)

    @property
    def fp_shape(self):
        """Focal-plane array shape (ny,nx)."""
        return self.fp_grid.shape

    @property
    def det_shape(self):
        """Detector array shape (ny,nx)."""
        return self.detector_grid.shape

    @property
    def fp_pixel_scale_arcsec(self):
        """Isotropic axis-aligned focal-plane spacing [arcsec/pixel]."""
        return self.fp_grid.step

    @property
    def fp_px_per_lenslet(self):
        """Lenslet x-pitch divided by isotropic focal-plane spacing."""
        (a, _), (c, _) = self.lenslet_grid.basis
        return (a * a + c * c) ** 0.5 / self.fp_pixel_scale_arcsec

    def __check_init__(self):
        """Reject inconsistent footprints and image geometry."""
        if not self.bands or not all(isinstance(b, Band) for b in self.bands):
            raise ValueError("bands must be a nonempty sequence of Band")
        if self.det_rows.shape != self.det_vals.shape or self.det_rows.ndim != 3:
            raise ValueError(
                "det_rows and det_vals must share shape (n_channels,n_wav,n_psf)"
            )
        if self.det_rows.shape[:2] != (self.n_channels, self.n_wav):
            raise ValueError("footprints must match n_channels and n_wav")
        if self.lenslet_indices.shape != (self.n_channels,):
            raise ValueError("lenslet_indices must match n_channels")
        if self.fp_grid.unit != "arcsec" or self.lenslet_grid.unit != "arcsec":
            raise ValueError("focal-plane and lenslet grids must use arcsec")
        expected = 1 if self.layout.grid_kind == "square" else self.layout.n_lenslets
        if len(self.collection_grids) != expected:
            raise ValueError("collection_grids must cover the layout rows")
        if self.collection_overlap.shape != (
            self.n_channels,
            self.fp_shape[0] * self.fp_shape[1],
        ):
            raise ValueError("collection_overlap must map focal pixels to channels")
        if self.detector_grid.unit != "meter":
            raise ValueError("detector_grid must use meter")
