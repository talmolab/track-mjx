"""Vision-aware DMPO networks for the binocular-vision high-level transfer task.

Same Gaussian policy + categorical critic shapes as ``networks.py``, but each
torso is preceded by a ``BinocularVisionEncoder`` whose output is concatenated
with the (already-flat) ``imitation_target`` (= flattened task_obs).

NOTE: feedforward only. The GRU used by the existing PPO recurrent-binocular
network is intentionally omitted — see Path-3 design in
ClaudeCode_PromptHistory/2026-04-28-3-dmpo-jax-linen-port.md.

Each torso has its OWN vision encoder (not shared between policy and critic).
This matches the existing ``ff_ppo`` convention (separate per-head torsos) and
keeps the implementation simple. Weight-sharing across the policy/critic
torsos can be reintroduced later if needed.
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


class _VisionPolicyNet(nn.Module):
    """CNN-encoded binocular vision + flat task_obs -> Gaussian policy."""

    layer_sizes: Sequence[int]
    action_size: int
    vision_shape: tuple
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    activation: Callable = nn.silu

    @nn.compact
    def __call__(self, obs) -> tfd.Distribution:
        # obs is a dict: {"vision": [..., H, W, 2C], "imitation_target": [..., T]}.
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(obs["vision"])
        h = jnp.concatenate([vis, obs["imitation_target"]], axis=-1)
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return GaussianPolicyHead(action_size=self.action_size)(h)


class _VisionCriticNet(nn.Module):
    """CNN-encoded binocular vision + flat task_obs + action -> Categorical critic."""

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


def make_dmpo_vision_networks(
    *,
    task_obs_size: int,
    action_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
) -> DMPONetworks:
    """Build vision-aware (policy, critic) flax modules for DMPO.

    Args:
        task_obs_size: Flattened task-observation dimensionality (the
            ``imitation_target`` leaf of the dict observation).
        action_size: Action dimensionality.
        vision_shape: Shape of the vision tensor as ``(H, W, 2*C)`` for
            channel-stacked binocular images. Stored on the modules for
            downstream reference.
        cfg: A ``DMPOConfig`` (or anything exposing ``policy_layer_sizes``,
            ``critic_layer_sizes``, ``num_atoms``, ``vmin``, ``vmax``).
        cnn_feature_size: Per-eye CNN feature dimensionality. Total CNN
            output size is ``2 * cnn_feature_size`` (left + right).
        cnn_channels: Channel sizes for each conv layer in the per-eye CNN.
        mono_channels: Number of channels per eye (1 grayscale, 3 RGB).
        shared_weights: If True, use Siamese architecture (shared CNN
            weights across L/R). If False, two independent encoders.

    Returns:
        ``DMPONetworks(policy, critic)`` ready to be ``init``'d with a dict
        observation template (``{"vision": ..., "imitation_target": ...}``)
        and a dummy action.
    """
    del task_obs_size  # unused directly; the network reads it from obs at init.
    return DMPONetworks(
        policy=_VisionPolicyNet(
            layer_sizes=tuple(cfg.policy_layer_sizes),
            action_size=action_size,
            vision_shape=vision_shape,
            cnn_feature_size=cnn_feature_size,
            cnn_channels=tuple(cnn_channels),
            mono_channels=mono_channels,
            shared_weights=shared_weights,
        ),
        critic=_VisionCriticNet(
            layer_sizes=tuple(cfg.critic_layer_sizes),
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
