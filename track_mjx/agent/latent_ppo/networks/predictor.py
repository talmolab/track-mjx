"""Motion predictor: z_t -> next N motion frames."""
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionPredictor(nn.Module):
    """f(z_t) = S_hat^motion_{t+1} (a window of length `horizon`)."""

    layer_sizes: Sequence[int] = (256, 128)
    horizon: int = 5
    feat_dim: int = 77

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(z)
        flat_out = nn.Dense(self.horizon * self.feat_dim, name="pred_head")(h)
        return flat_out.reshape(batch, self.horizon, self.feat_dim)
