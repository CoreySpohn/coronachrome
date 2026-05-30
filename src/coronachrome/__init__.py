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

__all__ = [
    "IFSRenderer",
    "SpatialChannelIR",
    "build_ir",
    "lstsq",
    "matched_filter",
    "spatial_sample",
    "spectrum_covariance",
    "spectrum_errorbars",
]
