"""Shared pytest fixtures and Hypothesis configuration for coronachrome tests."""

import importlib.util
import os

from hypothesis import settings
from optixstuff.grids import Measure

# Released hwoutils versions may not include the optional budget plugin yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")
if importlib.util.find_spec("hwoutils.precision_guard") is not None:
    pytest_plugins.append("hwoutils.precision_guard")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


def bands_at(wavelengths_nm, width_nm=1e-3):
    """Tiny explicit Bands with the requested geometric reference wavelengths.

    These monochromatic-limit fixtures separate geometry/profile tests from
    band-integration tests, which supply their physical intervals directly.
    """
    import math

    from optixstuff.spectral import Band

    return tuple(
        Band(
            math.sqrt(float(w) ** 2 + width_nm**2 / 4) - width_nm / 2,
            math.sqrt(float(w) ** 2 + width_nm**2 / 4) + width_nm / 2,
        )
        for w in wavelengths_nm
    )


def sky_grid(shape, step=0.1, **kwargs):
    """Geometrically centred focal-plane PixelGrid in arcsec."""
    from optixstuff.grids import PixelGrid

    return PixelGrid.regular(shape, step, "arcsec", **kwargs)


def focal_image(data, grid, *, reconstruction="nearest", measure=None):
    """Explicit band-integrated test Image, including stacks of mode cubes."""
    from optixstuff.grids import Image

    axes = ("mode", "band") if data.ndim == 4 else ("band",)
    return Image(
        data,
        grid,
        axes,
        Measure.integral() if measure is None else measure,
        "e/s",
        reconstruction,
    )


def square_rectangle_area(center, angle_rad, side, bounds):
    """Independent polygon oracle by enumerating boundary-line intersections.

    Intersect eight half-planes, keep all feasible line intersections, then
    angularly order the vertices. Unlike the implementation, no polygon is
    successively clipped. All inputs are physical coordinates.
    """
    from itertools import combinations

    import numpy as np

    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    normals = np.array(
        [[ca, sa], [-ca, -sa], [-sa, ca], [sa, -ca], [1, 0], [-1, 0], [0, 1], [0, -1]]
    )
    xmin, xmax, ymin, ymax = bounds
    limits = np.r_[normals[:4] @ center + side / 2, xmax, -xmin, ymax, -ymin]
    vertices = []
    for i, j in combinations(range(8), 2):
        pair = normals[[i, j]]
        if abs(np.linalg.det(pair)) < 1e-14:
            continue
        point = np.linalg.solve(pair, limits[[i, j]])
        if np.all(normals @ point <= limits + 2e-15):
            vertices.append(point)
    if len(vertices) < 3:
        return 0.0
    vertices = np.array(vertices)
    vertices -= vertices.mean(axis=0)
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
    return (
        abs(
            np.sum(
                vertices[:, 0] * np.roll(vertices[:, 1], -1)
                - vertices[:, 1] * np.roll(vertices[:, 0], -1)
            )
        )
        / 2
    )
