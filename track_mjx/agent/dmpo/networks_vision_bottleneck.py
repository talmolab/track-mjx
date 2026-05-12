"""Bottleneck DMPO vision networks (apples-to-apples vs PPO binocular_shared_vision_task_obs).

Mirrors `make_dmpo_vision_networks` in shape but adds a fusion-MLP →
`vision_latent_size`-D bottleneck → decoder-MLP on the policy side, matching
PPO's binocular_shared_vision_task_obs policy head. The critic mirrors PPO's
value head: CNN + task_obs + action → value MLP → CategoricalCriticHead, with
no bottleneck.

Each torso has its OWN BinocularVisionEncoder (NOT shared between policy and
critic) — matches the existing DMPO convention in `networks_vision.py`.
Cross-head weight sharing can be added later if needed.
"""
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks import (
    GaussianPolicyHead,
    CategoricalCriticHead,
    DMPONetworks,
)
from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder

tfd = tfp.distributions


class _BottleneckVisionPolicyNet(nn.Module):
    """Binocular CNN + task_obs → fusion → vision_latent_size bottleneck → decoder → Gaussian."""

    action_size: int
    vision_shape: tuple
    vision_latent_size: int
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    fusion_hidden_layer_sizes: Sequence[int]
    decoder_hidden_layer_sizes: Sequence[int]
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, obs) -> tfd.Distribution:
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(obs["vision"])
        h = jnp.concatenate([vis, obs["imitation_target"]], axis=-1)
        # Fusion MLP
        for size in self.fusion_hidden_layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        # Bottleneck projection
        z = nn.Dense(self.vision_latent_size)(h)
        # Decoder MLP
        h = z
        for size in self.decoder_hidden_layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return GaussianPolicyHead(action_size=self.action_size)(h)


class _ValueVisionCriticNet(nn.Module):
    """Binocular CNN + task_obs + action → value MLP → Categorical critic.

    No bottleneck on the critic — symmetric to PPO's value head.
    """

    layer_sizes: Sequence[int]
    num_atoms: int
    vmin: float
    vmax: float
    vision_shape: tuple
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, obs, action: jnp.ndarray) -> tfd.Distribution:
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(obs["vision"])
        h = jnp.concatenate([vis, obs["imitation_target"], action], axis=-1)
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


def make_dmpo_bottleneck_vision_networks(
    *,
    task_obs_size: int,
    action_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    vision_latent_size: int = 16,
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    fusion_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
) -> DMPONetworks:
    """Build vision-aware DMPO networks with a fusion bottleneck on the policy.

    Mirrors PPO `binocular_shared_vision_task_obs` shape: policy goes
    CNN+task_obs → fusion MLP → vision_latent_size-D Dense → decoder MLP → Gaussian.
    Critic is CNN+task_obs+action → value MLP → Categorical (no bottleneck).
    """
    del task_obs_size  # unused directly
    return DMPONetworks(
        policy=_BottleneckVisionPolicyNet(
            action_size=action_size,
            vision_shape=vision_shape,
            vision_latent_size=vision_latent_size,
            cnn_feature_size=cnn_feature_size,
            cnn_channels=tuple(cnn_channels),
            mono_channels=mono_channels,
            shared_weights=shared_weights,
            fusion_hidden_layer_sizes=tuple(fusion_hidden_layer_sizes),
            decoder_hidden_layer_sizes=tuple(decoder_hidden_layer_sizes),
        ),
        critic=_ValueVisionCriticNet(
            layer_sizes=tuple(value_hidden_layer_sizes),
            num_atoms=cfg.num_atoms,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            vision_shape=vision_shape,
            cnn_feature_size=cnn_feature_size,
            cnn_channels=tuple(cnn_channels),
            mono_channels=mono_channels,
            shared_weights=shared_weights,
        ),
    )
