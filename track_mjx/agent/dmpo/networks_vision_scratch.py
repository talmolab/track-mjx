"""Vision-aware DMPO networks for the from-scratch (no-prior) training path.

Mirrors the shape of ``ff_ppo.binocular_shared_vision_network.BinocularSharedVisionPolicyValueModule``
so that DMPO and PPO baselines train comparable policies. Differences from the
PPO module:

  * Policy returns ``tfd.MultivariateNormalDiag`` (loc + scale_diag) — required
    by the MPO loss, which assumes a pre-tanh Gaussian.
  * Critic is a separate categorical critic head (DMPO is off-policy with C51).
  * Each head has its OWN CNN encoder (no shared CNN); matches the existing
    ``networks_vision.py`` convention. Sharing across heads is left for a
    follow-up (see future.md).
  * No intention bottleneck inside the policy by default — the user explicitly
    requested no 16-D bottleneck for the first from-scratch run. A bottleneck
    variant lives in the queued follow-up config.
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


class _VisionScratchPolicyNet(nn.Module):
    """CNN(vision) ⊕ task_obs ⊕ proprio → MLP → MultivariateNormalDiag.

    Three-modality input matching PPO baseline ``BinocularSharedVisionPolicyValueModule``,
    minus the 16-D fusion bottleneck (per user directive).
    """

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
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(obs["vision"])
        h = jnp.concatenate(
            [vis, obs["imitation_target"], obs["proprioception"]], axis=-1
        )
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return GaussianPolicyHead(action_size=self.action_size)(h)


class _VisionScratchCriticNet(nn.Module):
    """CNN(vision) ⊕ task_obs ⊕ proprio ⊕ action → MLP → CategoricalCriticHead.

    Vnl-ray-faithful flat-critic shape (`networks.LayerNormMLP +
    DiscreteValuedHead`). Critic input is the FULL observation plus the
    action — strictly more information than the policy receives. This is by
    design: DMPO's off-policy critic benefits from richer state.
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
        h = jnp.concatenate(
            [vis, obs["imitation_target"], obs["proprioception"], action],
            axis=-1,
        )
        for size in self.layer_sizes:
            h = nn.Dense(size)(h)
            h = nn.LayerNorm()(h)
            h = self.activation(h)
        return CategoricalCriticHead(
            num_atoms=self.num_atoms, vmin=self.vmin, vmax=self.vmax
        )(h)


def make_dmpo_vision_scratch_networks(
    *,
    task_obs_size: int,
    proprio_size: int,
    action_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
) -> DMPONetworks:
    """Build (policy, critic) for the from-scratch vision DMPO path.

    Args:
        task_obs_size: Flattened ``imitation_target`` (task_obs) dimensionality.
        proprio_size: Flattened ``proprioception`` dimensionality.
        action_size: Action dimensionality (38 for the rodent).
        vision_shape: Shape of the vision tensor as ``(H, W, 2*mono_channels)``
            for channel-stacked binocular images.
        cfg: A ``DMPOConfig`` providing ``policy_layer_sizes``,
            ``critic_layer_sizes``, ``num_atoms``, ``vmin``, ``vmax``.
        cnn_feature_size: Per-eye CNN output feature size.
        cnn_channels: Channel sizes for each conv layer in the per-eye CNN.
        mono_channels: 1 (grayscale) or 3 (RGB) per eye.
        shared_weights: If True, Siamese binocular CNN.

    Returns:
        ``DMPONetworks(policy, critic)`` ready to be ``init``'d with the dict
        observation template emitted by ``EndToEndWrapper``.
    """
    del task_obs_size, proprio_size  # not used directly; networks read from obs at init
    return DMPONetworks(
        policy=_VisionScratchPolicyNet(
            layer_sizes=tuple(cfg.policy_layer_sizes),
            action_size=action_size,
            vision_shape=vision_shape,
            cnn_feature_size=cnn_feature_size,
            cnn_channels=tuple(cnn_channels),
            mono_channels=mono_channels,
            shared_weights=shared_weights,
        ),
        critic=_VisionScratchCriticNet(
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
