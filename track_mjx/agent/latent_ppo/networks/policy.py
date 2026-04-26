"""Flat-MLP policy/value for LatentMimic (paper §IV).

Inputs: dict with keys 'proprioception', 'o_history', 'z_target'.
Policy outputs concat([mean, log_std]) of shape (..., 2*action_dim).
Value outputs scalar baseline.
"""
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.latent_ppo.networks.mlp import Mlp


def _flatten_obs(obs: dict) -> jnp.ndarray:
    parts = [obs["proprioception"], obs["o_history"], obs["z_target"]]
    return jnp.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=-1)


class LatentMimicPolicy(nn.Module):
    layer_sizes: Sequence[int] = (512, 256, 128)
    action_dim: int = 0
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs: dict) -> jnp.ndarray:
        x = _flatten_obs(obs)
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(x)
        mean = nn.Dense(self.action_dim, name="mean_head")(h)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(self.log_std_init),
            (self.action_dim,),
        )
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        log_std = jnp.broadcast_to(log_std, mean.shape)
        return jnp.concatenate([mean, log_std], axis=-1)


class LatentMimicValue(nn.Module):
    layer_sizes: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, obs: dict) -> jnp.ndarray:
        x = _flatten_obs(obs)
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(x)
        return nn.Dense(1, name="value_head")(h).squeeze(-1)
