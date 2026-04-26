"""Per-feature mean/std normalizer; stats frozen after pre-training."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class FeatureNormalizer:
    mean: jnp.ndarray  # (feat_dim,)
    std: jnp.ndarray   # (feat_dim,) clipped to >= 1e-3

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.mean) / self.std

    def invert(self, z: jnp.ndarray) -> jnp.ndarray:
        return z * self.std + self.mean


def fit_normalizer(x: np.ndarray) -> FeatureNormalizer:
    mean = x.mean(axis=tuple(range(x.ndim - 1)))
    std = x.std(axis=tuple(range(x.ndim - 1)))
    std = np.maximum(std, 1e-3)
    return FeatureNormalizer(
        mean=jnp.asarray(mean, dtype=jnp.float32),
        std=jnp.asarray(std, dtype=jnp.float32),
    )
