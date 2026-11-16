"""Host-built sparse integrals of piecewise-constant focal-plane density."""

import math

import equinox as eqx
import numpy as np


class CellOverlap(eqx.Module):
    """Immutable host COO matrix: lenslet rows, focal-pixel columns, arcsec^2.

    Tuples keep geometry hashable and static under JAX transformations. Only
    intersecting pixels are stored; no dense cell-by-image array is formed.
    """

    indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    areas_arcsec2: tuple[float, ...] = eqx.field(static=True)
    shape: tuple[int, int] = eqx.field(static=True)


def _clip(polygon, axis, bound, sign):
    """Clip a convex polygon to sign * (coordinate - bound) >= 0."""
    if not polygon:
        return []
    clipped = []
    previous = polygon[-1]
    prev_distance = sign * (previous[axis] - bound)
    for current in polygon:
        distance = sign * (current[axis] - bound)
        if (distance >= 0) != (prev_distance >= 0):
            fraction = prev_distance / (prev_distance - distance)
            crossing = [
                previous[k] + fraction * (current[k] - previous[k]) for k in (0, 1)
            ]
            crossing[axis] = bound
            clipped.append(tuple(crossing))
        if distance >= 0:
            clipped.append(current)
        previous, prev_distance = current, distance
    return clipped


def cell_overlap(fp_grid, collection_grids, channel_order):
    """Clip each cell against candidate pixels, once, using static host grids.

    Sutherland-Hodgman clipping is exact up to floating-point roundoff.
    Work in focal-pixel coordinates, then multiply areas by pixel_area.
    The affine change of coordinates also supports nonsquare focal pixels.
    No edge renormalization: missing focal-plane area contributes zero.
    """
    ny, nx = fp_grid.shape
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    polygons = []
    for grid in collection_grids:
        transform = np.linalg.solve(np.asarray(fp_grid.basis), grid.basis)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                polygon = corners + np.array([col - grid.center_x, row - grid.center_y])
                polygons.append(
                    polygon @ transform.T + [fp_grid.center_x, fp_grid.center_y]
                )
    indices, areas = [], []
    for cell, flat_index in enumerate(channel_order):
        polygon = polygons[flat_index]
        lo_x, lo_y = np.floor(polygon.min(axis=0) + 0.5).astype(int)
        hi_x, hi_y = np.ceil(polygon.max(axis=0) + 0.5).astype(int)
        for row in range(max(0, lo_y), min(ny, hi_y)):
            for col in range(max(0, lo_x), min(nx, hi_x)):
                # Local coordinates avoid cancellation in the shoelace area.
                clipped = [tuple(p) for p in polygon - [col, row]]
                for axis in (0, 1):
                    clipped = _clip(clipped, axis, -0.5, 1)
                    clipped = _clip(clipped, axis, 0.5, -1)
                if len(clipped) < 3:
                    continue
                area = (
                    abs(
                        math.fsum(
                            p[0] * q[1] - p[1] * q[0]
                            for p, q in zip(
                                clipped, clipped[1:] + clipped[:1], strict=True
                            )
                        )
                    )
                    / 2
                )
                if area > 0:
                    indices.append((cell, row * nx + col))
                    areas.append(area * fp_grid.pixel_area)
    return CellOverlap(tuple(indices), tuple(areas), (len(channel_order), ny * nx))
