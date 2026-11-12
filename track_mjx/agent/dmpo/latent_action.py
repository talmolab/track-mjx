"""Latent-action DMPO transfer (``transfer.mode: latent_action``).

In the default (motor-action, "Variant B") transfer the MPO policy is the frozen decoder's Gaussian over the 38-D motor
action at ``z = mu_prior + residual``. The residual is deterministic and the only exploration is the decoder's fixed
motor noise. SCAMPER's own transfer (PPO) instead samples the latent: the policy is a Gaussian over the latent residual
``delta`` with a learnable spread, and the frozen prior + decoder turn ``z = mu_prior + delta`` into motor commands.

This module provides that action space for the DMPO kl-anchor entry, without touching the learner:

* :class:`LatentActionPolicyNet` is a Gaussian over the RAW (pre-tanh) latent residual. Its mean comes from the same
  vision + task_obs + proprio trunk as the motor-action head (``_PolicyHeadModule``, zero-init last layer, so the mean
  residual is 0 at step 0). Its spread is a learnable, state-independent vector initialised to ``init_std``.
* :func:`latent_to_ctrl` and :class:`LatentActionWrapper` (env side) map the bound action ``delta = tanh(raw)`` in
  [-1, 1]^L to ``z = mu_prior(proprio) + scale * delta``, then to the decoder's deterministic motor command
  ``tanh(mu_decoder([z, proprio]))``. With ``scale = 1`` this is exactly SCAMPER's paper parameterization
  (delta in [-1, 1]^16, absolute latent units, Appendix E.4).

The rollout's ``bind`` (tanh + clip) produces ``delta``, and the critic sees the bound ``delta`` (aligned F3, "executed").
MPO samples and fits in the latent space, so its KL trust regions and its learned standard deviation act on the latent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_kl_anchor import _PolicyHeadModule
from track_mjx.agent.dmpo.networks_vision_bottleneck import _ValueVisionCriticNet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

tfd = tfp.distributions

__all__ = [
    "LatentActionPolicyNet",
    "LatentActionWrapper",
    "latent_to_ctrl",
    "make_dmpo_latent_action_networks",
]


def _inv_softplus(y: float) -> float:
    import math

    return math.log(math.expm1(y))


class LatentActionPolicyNet(nn.Module):
    """Gaussian over the raw latent residual; mean from the motor head's trunk, learnable state-independent spread."""

    latent_size: int
    vision_shape: tuple
    policy_head_layer_sizes: Sequence[int]
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    init_std: float = 0.7
    min_std: float = 1e-3

    @nn.compact
    def __call__(self, obs):
        mean = _PolicyHeadModule(
            layer_sizes=tuple(self.policy_head_layer_sizes),
            latent_size=self.latent_size,
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            name="policy_head",
        )(obs["vision"], obs["imitation_target"], obs["proprioception"])
        raw_std = self.param(
            "raw_std",
            nn.initializers.constant(_inv_softplus(self.init_std - self.min_std)),
            (self.latent_size,),
        )
        std = jax.nn.softplus(raw_std) + self.min_std
        return tfd.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.broadcast_to(std, mean.shape)
        )


def make_dmpo_latent_action_networks(
    *,
    latent_size: int,
    vision_shape: tuple,
    cfg: Any,
    policy_head_layer_sizes: Sequence[int] = (256, 256, 256),
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    critic_use_proprio: bool = False,
    critic_privileged_dim: int = 0,
    init_std: float = 0.7,
) -> DMPONetworks:
    """Latent-action policy + the same categorical vision critic the motor-action path uses."""
    policy = LatentActionPolicyNet(
        latent_size=int(latent_size),
        vision_shape=tuple(vision_shape),
        policy_head_layer_sizes=tuple(policy_head_layer_sizes),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        init_std=float(init_std),
    )
    critic = _ValueVisionCriticNet(
        layer_sizes=tuple(value_hidden_layer_sizes),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        vision_shape=tuple(vision_shape),
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        use_proprio=critic_use_proprio,
        privileged_dim=critic_privileged_dim,
    )
    return DMPONetworks(policy=policy, critic=critic)


def latent_to_ctrl(
    prior_fn: Callable,
    decoder_logits_fn: Callable,
    proprio: jax.Array,
    delta: jax.Array,
    scale: float,
    motor_size: int,
) -> jax.Array:
    """``tanh(mu_decoder([mu_prior(proprio) + scale * delta, proprio]))``: the decoder's deterministic motor command.

    ``prior_fn`` / ``decoder_logits_fn`` are the ``prior_utils`` makers' functions; they normalise the raw proprio with
    the prior checkpoint's own statistics, exactly as the motor-action path's anchor wrapper does.
    """
    mu, _ = prior_fn(proprio)
    z = mu + scale * delta
    logits, _ = decoder_logits_fn(jnp.concatenate([z, proprio], axis=-1))
    return jnp.tanh(logits[..., :motor_size])


def _end_to_end_wrapper_cls():
    from vnl_playground.tasks.wrappers import EndToEndWrapper

    return EndToEndWrapper


class LatentActionWrapper:
    """Factory shim: builds an ``EndToEndWrapper`` subclass whose action is the bound latent residual.

    The observation routing is EndToEndWrapper's (``{imitation_target, proprioception, vision}``), so the policy sees
    exactly what the motor-action path's policy sees. ``action_size`` becomes the latent size.
    """

    def __new__(
        cls,
        env,
        *,
        prior_fn: Callable,
        decoder_logits_fn: Callable,
        latent_size: int,
        scale: float = 1.0,
        policy_obs_key: str = "state",
        highlvl_obs_key: str = "task_obs",
        lowlvl_obs_key: str = "proprioception",
    ):
        base = _end_to_end_wrapper_cls()

        class _LatentActionWrapper(base):
            def __init__(self):
                super().__init__(
                    env,
                    policy_obs_key=policy_obs_key,
                    highlvl_obs_key=highlvl_obs_key,
                    lowlvl_obs_key=lowlvl_obs_key,
                )
                self._prior_fn = prior_fn
                self._decoder_logits_fn = decoder_logits_fn
                self._latent_size = int(latent_size)
                self._scale = float(scale)
                self._motor_size = int(env.action_size)

            def step(self, state, action):
                full = state.info["_full_obs"]
                proprio = jnp.nan_to_num(
                    jax.flatten_util.ravel_pytree(
                        full[self._policy_obs_key][self._lowlvl_obs_key]
                    )[0]
                )
                ctrl = latent_to_ctrl(
                    self._prior_fn,
                    self._decoder_logits_fn,
                    proprio,
                    action,
                    self._scale,
                    self._motor_size,
                )
                return self._process_state(self.env.step(state, ctrl))

            @property
            def action_size(self) -> int:
                return self._latent_size

        return _LatentActionWrapper()
