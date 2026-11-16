"""Lenslet layouts, with explicit index-origin conventions."""

import math
from dataclasses import dataclass
from operator import index

import jax
import jax.numpy as jnp
from hwoutils.grids import PixelGrid


def _origin(n, lenslet_origin):
    if lenslet_origin == "geometric":
        return (n - 1) / 2
    if lenslet_origin == "central_index":
        return n // 2
    raise ValueError("lenslet_origin must be geometric or central_index")


def square_grid(n_lenslets, *, lenslet_origin="geometric"):
    """Return (x,y) centres, x-major order, in lenslet pitches.

    geometric subtracts (n-1)/2; central_index subtracts n//2 (the historical
    convention). For even n geometric moves both coordinates by +0.5 pitch.
    """
    idx = jnp.arange(n_lenslets) - _origin(n_lenslets, lenslet_origin)
    xx, yy = jnp.meshgrid(idx, idx, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)


def hex_grid(n_lenslets, *, lenslet_origin="geometric"):
    """Return staggered-row (x,y) centres, row-major, in lenslet pitches.

    central_index subtracts n//2 from columns and rows, retaining the odd-row
    +0.5 x offset. geometric additionally subtracts the mean stagger, so the
    centroid of all centres is zero, for both even and odd n.
    """
    origin = _origin(n_lenslets, lenslet_origin)
    row, col = jnp.indices((n_lenslets, n_lenslets))
    stagger_mean = (n_lenslets // 2) / (2 * n_lenslets)
    x = col - origin + 0.5 * (row % 2)
    if lenslet_origin == "geometric":
        x = x - stagger_mean
    y = (row - origin) * math.sqrt(3) / 2
    return jnp.stack([x.ravel(), y.ravel()], axis=1)


@jax.tree_util.register_static
@dataclass(frozen=True)
class LensletLayout:
    """Layout geometry independent of the disperser hardware.

    lenslet_origin is geometric (centroid at zero) or central_index (the
    historical n//2 index, retaining the hex odd-row offset).
    """

    n_lenslets: int
    grid_kind: str = "square"
    lenslet_origin: str = "geometric"

    def __post_init__(self):
        """Reject invalid layout choices before building geometry."""
        if index(self.n_lenslets) < 1:
            raise ValueError("n_lenslets must be positive")
        if self.grid_kind not in ("square", "hex"):
            raise ValueError("grid_kind must be square or hex")
        _origin(self.n_lenslets, self.lenslet_origin)

    def positions(self):
        """Return centres in the documented square/hex channel order."""
        fn = square_grid if self.grid_kind == "square" else hex_grid
        return fn(self.n_lenslets, lenslet_origin=self.lenslet_origin)

    def image_geometry(self, pitch_arcsec, angle_rad, orientation):
        """Return an affine sky grid and channel-to-flat-pixel indices.

        Hex offset rows are packed into an axial parallelogram grid; unused
        pixels are zero. This retains exact sky coordinates without claiming
        that staggered rows are a rectangular PixelGrid.
        """
        n = self.n_lenslets
        origin = _origin(n, self.lenslet_origin)
        if self.grid_kind == "square":
            shape = (n, n)
            center = (origin, origin)
            basis = ((1.0, 0.0), (0.0, 1.0))
            indices = jnp.arange(n * n).reshape(n, n).T.ravel()
        else:
            pad = (n - 1) // 2
            shape = (n, n + pad)
            stagger = (n // 2) / (2 * n) if self.lenslet_origin == "geometric" else 0
            center = (pad + origin / 2 + stagger, origin)
            basis = ((1.0, 0.5), (0.0, math.sqrt(3) / 2))
            row, col = jnp.indices((n, n))
            indices = (row * shape[1] + col - row // 2 + pad).ravel()
        ca, sa = math.cos(angle_rad), math.sin(angle_rad)
        (a, b), (c, d) = basis
        rotated = (
            (pitch_arcsec * (ca * a - sa * c), pitch_arcsec * (ca * b - sa * d)),
            (pitch_arcsec * (sa * a + ca * c), pitch_arcsec * (sa * b + ca * d)),
        )
        return PixelGrid(
            shape, *center, rotated, "arcsec", orientation=orientation
        ), indices

    def collection_grids(self, pitch_arcsec, angle_rad, orientation):
        """Square collection-cell grids; hex centres retain square apertures.

        Square mode uses one grid. Hex mode describes each staggered row on its
        own rectangular grid, preserving the historical square collection
        cells (not hexagonal Voronoi apertures). The axial output grid only
        represents the sample centres and does not change the aperture area.
        """
        if self.grid_kind == "square":
            return (self.image_geometry(pitch_arcsec, angle_rad, orientation)[0],)
        n = self.n_lenslets
        origin = _origin(n, self.lenslet_origin)
        stagger = (n // 2) / (2 * n) if self.lenslet_origin == "geometric" else 0
        ca, sa = math.cos(angle_rad), math.sin(angle_rad)
        basis = (
            (pitch_arcsec * ca, -pitch_arcsec * sa),
            (pitch_arcsec * sa, pitch_arcsec * ca),
        )
        return tuple(
            PixelGrid(
                (1, n),
                origin - 0.5 * (row % 2) + stagger,
                -(row - origin) * math.sqrt(3) / 2,
                basis,
                "arcsec",
                orientation=orientation,
            )
            for row in range(n)
        )
