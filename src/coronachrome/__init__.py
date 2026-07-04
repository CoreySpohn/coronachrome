"""coronachrome: a JAX lenslet-IFS forward model for HWO."""

from coronachrome.build import build_ir
from coronachrome.extract import (
    lstsq,
    matched_filter,
    spectrum_covariance,
    spectrum_errorbars,
)
from coronachrome.ir import SpatialChannelIR
from coronachrome.render import IFSRenderer, spatial_sample
from coronachrome.spectral import (
    channel_centers,
    channel_edges,
    n_nyquist_channels,
    rebin_channels,
    spectral_grid,
)
from coronachrome.templates import (
    PsfletPack,
    analytic_psflet_pack,
    load_psflet_pack,
    save_psflet_pack,
    template_weights,
)

__all__ = [
    "IFSRenderer",
    "PsfletPack",
    "SpatialChannelIR",
    "analytic_psflet_pack",
    "build_ir",
    "channel_centers",
    "channel_edges",
    "load_psflet_pack",
    "lstsq",
    "matched_filter",
    "n_nyquist_channels",
    "rebin_channels",
    "save_psflet_pack",
    "spatial_sample",
    "spectral_grid",
    "spectrum_covariance",
    "spectrum_errorbars",
    "template_weights",
]
