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

__all__ = [
    "IFSRenderer",
    "SpatialChannelIR",
    "build_ir",
    "channel_centers",
    "channel_edges",
    "lstsq",
    "matched_filter",
    "n_nyquist_channels",
    "rebin_channels",
    "spatial_sample",
    "spectral_grid",
    "spectrum_covariance",
    "spectrum_errorbars",
]
