"""Motion encoder: window of m_t -> (mean, logvar) of latent z."""
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionEncoder(nn.Module):
    """Encoder q(z | S^motion_t) returning (mean, logvar) of a diagonal Gaussian."""

    layer_sizes: Sequence[int] = (256, 128)
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim)
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(x)
        mean = nn.Dense(self.latent_dim, name="mean_head")(h)
        logvar = nn.Dense(self.latent_dim, name="logvar_head")(h)
        return mean, logvar
