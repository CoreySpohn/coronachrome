"""Lenslet layouts centred on their geometric centroid."""

import math
from dataclasses import dataclass
from operator import index

import jax
import jax.numpy as jnp
from hwoutils.radial import get_center
from optixstuff.grids import PixelGrid


def square_grid(n_lenslets):
    """Return (x,y) centres, x-major order, in lenslet pitches.

    Subtract the geometric pixel centre (n-1)/2 from each index.
    """
    idx = jnp.arange(n_lenslets) - get_center((n_lenslets, n_lenslets))[0]
    xx, yy = jnp.meshgrid(idx, idx, indexing="ij")
    return jnp.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)


def hex_grid(n_lenslets):
    """Return staggered-row (x,y) centres, row-major, in lenslet pitches.

    Subtract the geometric row/column centre and the mean odd-row
    x stagger, so the centroid is zero for both even and odd n.
    """
    origin = get_center((n_lenslets, n_lenslets))[0]
    row, col = jnp.indices((n_lenslets, n_lenslets))
    stagger_mean = (n_lenslets // 2) / (2 * n_lenslets)
    x = col - origin + 0.5 * (row % 2)
    x = x - stagger_mean
    y = (row - origin) * math.sqrt(3) / 2
    return jnp.stack([x.ravel(), y.ravel()], axis=1)


@jax.tree_util.register_static
@dataclass(frozen=True)
class LensletLayout:
    """Layout geometry independent of the disperser hardware.

    The centroid of the square or staggered hex array is zero.
    """

    n_lenslets: int
    grid_kind: str = "square"

    def __post_init__(self):
        """Reject invalid layout choices before building geometry."""
        if index(self.n_lenslets) < 1:
            raise ValueError("n_lenslets must be positive")
        if self.grid_kind not in ("square", "hex"):
            raise ValueError("grid_kind must be square or hex")

    def positions(self):
        """Return centres in the documented square/hex channel order."""
        fn = square_grid if self.grid_kind == "square" else hex_grid
        return fn(self.n_lenslets)

    def image_geometry(self, pitch_arcsec, angle_rad, orientation):
        """Return an affine sky grid and channel-to-flat-pixel indices.

        Hex offset rows are packed into an axial parallelogram grid; unused
        pixels are zero. This retains exact sky coordinates without claiming
        that staggered rows are a rectangular PixelGrid.
        """
        n = self.n_lenslets
        origin = get_center((n, n))[0]
        if self.grid_kind == "square":
            shape = (n, n)
            center = (origin, origin)
            basis = ((1.0, 0.0), (0.0, 1.0))
            indices = jnp.arange(n * n).reshape(n, n).T.ravel()
        else:
            pad = (n - 1) // 2
            shape = (n, n + pad)
            stagger = (n // 2) / (2 * n)
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
        origin = get_center((n, n))[0]
        stagger = (n // 2) / (2 * n)
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
