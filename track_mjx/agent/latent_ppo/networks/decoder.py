"""Motion decoder used only for autoencoder pre-training; discarded for RL."""
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionDecoder(nn.Module):
    """p(S^motion_t | z) reconstructing the full input window."""

    layer_sizes: Sequence[int] = (128, 256)
    window_len: int = 10
    feat_dim: int = 77

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(z)
        flat_out = nn.Dense(self.window_len * self.feat_dim, name="recon_head")(h)
        return flat_out.reshape(batch, self.window_len, self.feat_dim)
