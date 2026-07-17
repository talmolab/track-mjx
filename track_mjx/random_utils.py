"""Utilities for working with JAX typed PRNG keys."""

import jax


def is_batched_prng_key(key: jax.Array) -> bool:
    """Return whether a typed PRNG key has a leading batch axis."""
    return key.ndim > 0


def split_prng_key(key: jax.Array, count: int = 2) -> tuple[jax.Array, ...]:
    """Split one key or each key in a one-dimensional key batch."""
    if not is_batched_prng_key(key):
        return tuple(jax.random.split(key, count))
    if key.ndim != 1:
        raise ValueError(f"Expected a scalar or 1D PRNG key batch, got {key.shape}")
    split_keys = jax.vmap(lambda item: jax.random.split(item, count))(key)
    return tuple(split_keys[:, index] for index in range(count))


def sample_normal(key: jax.Array, reference: jax.Array) -> jax.Array:
    """Sample normal noise matching a tensor and its optional key batch."""
    if not is_batched_prng_key(key):
        return jax.random.normal(key, reference.shape, dtype=reference.dtype)
    if key.ndim != 1:
        raise ValueError(f"Expected a scalar or 1D PRNG key batch, got {key.shape}")
    if reference.ndim == 0 or key.shape[0] != reference.shape[0]:
        raise ValueError(
            "A batched PRNG key must match the tensor's leading dimension: "
            f"key={key.shape}, tensor={reference.shape}"
        )
    return jax.vmap(
        lambda item_key, item: jax.random.normal(item_key, item.shape, dtype=item.dtype)
    )(key, reference)
