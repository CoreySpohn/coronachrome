"""Shared pytest fixtures and JAX configuration for coronachrome tests."""

import jax

jax.config.update("jax_enable_x64", True)
