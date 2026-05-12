"""Tanh squashing helpers for MPO compatibility.

MPO assumes an unbounded Gaussian policy, but environments require actions in
[-1, 1]. We sample raw Gaussians, apply tanh on the way to the env, and store
the *raw* (pre-tanh) action in the replay buffer so the loss sees unbounded
Gaussian samples.
"""
import jax.numpy as jnp


_BOUNDARY_EPS = 1e-6


def bind(raw_action: jnp.ndarray) -> jnp.ndarray:
    """Map unbounded action to the open interval (-1, 1) via tanh.

    For large-magnitude inputs, ``jnp.tanh`` saturates to ``±1`` in float32,
    so we clip the output to keep it strictly inside the open interval. This
    keeps ``unbind(bind(x))`` finite without relying on ``unbind``'s clip.
    """
    return jnp.clip(
        jnp.tanh(raw_action), -1.0 + _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS
    )


def unbind(bound_action: jnp.ndarray) -> jnp.ndarray:
    """Inverse of bind. Clips inputs away from ±1 to avoid atanh -> ±inf."""
    clipped = jnp.clip(bound_action, -1.0 + _BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS)
    return jnp.arctanh(clipped)
