"""Plotting for coronachrome geometry and extraction products, on eyepiece.

Requires the ``viz`` extra (``pip install 'coronachrome[viz]'``), which brings
eyepiece and, through it, matplotlib and hwostyle. The base install stays free
of all three: names are re-exported lazily (PEP 562), so importing this
package imports no plotting stack, and the eyepiece requirement is checked
only when a plot function is first touched.

Every position these views draw comes from coronachrome's own geometry
functions (:func:`coronachrome.build.lenslet_cell_centers`,
:func:`coronachrome.build.detector_centroids`,
:func:`coronachrome.build.detector_trace_origin`) or from the footprints in a
built :class:`~coronachrome.SpatialChannelIR`, so a figure shows the operator
the forward model applies rather than a second rendering of it.
"""

import importlib

_LAZY = {
    "plot_channel_covariance": "coronachrome.viz.covariance",
    "plot_lenslet_cells": "coronachrome.viz.lenslets",
    "plot_traces": "coronachrome.viz.lenslets",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve a lazy re-export, checking the eyepiece requirement first.

    Args:
        name: Attribute being looked up on ``coronachrome.viz``.

    Returns:
        The requested plot function.

    Raises:
        AttributeError: If ``name`` is not one of the lazy re-exports.
    """
    if name in _LAZY:
        from coronachrome.viz import _require

        _require.eyepiece()
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'coronachrome.viz' has no attribute {name!r}")


def __dir__():
    """List the lazy re-exports alongside the module's real attributes.

    Returns:
        Sorted attribute names, including the lazily provided functions.
    """
    return sorted(set(globals()) | set(__all__))
