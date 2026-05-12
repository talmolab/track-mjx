"""Shared MLP block: Dense -> ELU -> LayerNorm."""
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class Mlp(nn.Module):
    """Stack of Dense -> ELU -> LayerNorm layers.

    `activate_final` controls whether the final layer also applies ELU+LayerNorm.
    Set to False when the final output should be a linear projection
    (e.g. an encoder head or a logit).
    """

    layer_sizes: Sequence[int]
    activate_final: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, h in enumerate(self.layer_sizes):
            x = nn.Dense(h, kernel_init=self.kernel_init, name=f"dense_{i}")(x)
            is_final = i == len(self.layer_sizes) - 1
            if not is_final or self.activate_final:
                x = nn.elu(x)
                x = nn.LayerNorm(name=f"ln_{i}")(x)
        return x
