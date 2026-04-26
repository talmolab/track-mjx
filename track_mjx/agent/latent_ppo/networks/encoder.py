"""Motion encoder: window of m_t -> (mean, logvar) of latent z."""
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


class MotionEncoder(nn.Module):
    """Flat-MLP encoder q(z | S^motion_t) returning (mean, logvar)."""

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


class MotionEncoderConv1D(nn.Module):
    """Temporal-conv encoder q(z | S^motion_t) returning (mean, logvar).

    Treats the input window as a (time, feat) sequence and applies stacked
    1D convolutions with translation invariance along the time axis. Far
    more parameter-efficient than flattening the window into a flat MLP, and
    gives the encoder an inductive bias for motion patterns (it can learn to
    extract velocity from positional deltas across adjacent frames).
    """

    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim) — flax Conv interprets last axis as channels.
        h = x
        for i, ch in enumerate(self.conv_channels):
            h = nn.Conv(
                features=ch,
                kernel_size=(self.kernel_size,),
                padding="VALID",
                name=f"conv_{i}",
            )(h)
            h = nn.elu(h)
        h = h.reshape(h.shape[0], -1)
        if self.head_layer_sizes:
            h = Mlp(
                layer_sizes=tuple(self.head_layer_sizes),
                activate_final=True,
                name="head",
            )(h)
        mean = nn.Dense(self.latent_dim, name="mean_head")(h)
        logvar = nn.Dense(self.latent_dim, name="logvar_head")(h)
        return mean, logvar
