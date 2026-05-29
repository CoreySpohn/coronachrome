"""Spatial Channel IR: the geometry-agnostic forward-operator description.

Two factored steps, both stored as sparse footprint arrays so the operator
stays affordable at HWO scale:

- spatial sampling: per-channel focal-plane footprint (wavelength-independent).
- dispersion: per channel-wavelength PSFlet footprint on the detector.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class SpatialChannelIR(eqx.Module):
    """Standardized arrays specifying the IFS forward operator H."""

    spatial_src: Array = eqx.field(converter=jnp.asarray)
    spatial_w: Array = eqx.field(converter=jnp.asarray)
    det_rows: Array = eqx.field(converter=jnp.asarray)
    det_vals: Array = eqx.field(converter=jnp.asarray)
    n_channels: int = eqx.field(static=True)
    n_wav: int = eqx.field(static=True)
    fp_shape: tuple[int, int] = eqx.field(static=True)
    det_shape: tuple[int, int] = eqx.field(static=True)

    def __check_init__(self):
        """Validate that footprint arrays are consistent."""
        if self.spatial_src.shape != self.spatial_w.shape:
            raise ValueError("spatial_src and spatial_w must have the same shape")
        if self.det_rows.shape != self.det_vals.shape:
            raise ValueError("det_rows and det_vals must have the same shape")
        if self.det_rows.shape[0] != self.n_channels:
            raise ValueError("det_rows leading dim must equal n_channels")
        if self.det_rows.shape[1] != self.n_wav:
            raise ValueError("det_rows second dim must equal n_wav")
        if self.spatial_src.shape[0] != self.n_channels:
            raise ValueError("spatial_src leading dim must equal n_channels")
