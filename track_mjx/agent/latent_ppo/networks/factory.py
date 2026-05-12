"""Build Brax-compatible PPONetworks for LatentMimic.

Returns a `PPOImitationNetworks` (the project-local equivalent of Brax's
`PPONetworks`) so that the existing `ff_ppo` trainer can consume it without
modification.

The policy/value `apply` functions match the existing intention-network
contract:
  - accept `(processor_params, network_params, obs, key, deterministic=...,
    get_activation=...)`  for the policy
  - accept `(processor_params, network_params, obs)` for the value
  - normalize obs via `normalize_dict_obs` (which also flattens the env's
    nested `{state: {task_obs, proprioception}}` schema to the flat
    `{imitation_target, proprioception}` schema the LatentMimic policy reads)
  - policy returns `(logits, latent_mean, latent_logvar)` — latent_*
    are zeros since LatentMimic has no internal VAE.
"""
from typing import Sequence

import jax.numpy as jnp
from brax.training import distribution
from brax.training.networks import FeedForwardNetwork

from track_mjx.agent.ff_ppo.ppo_networks import PPOImitationNetworks
from track_mjx.agent.latent_ppo.networks.policy import (
    LatentMimicPolicy, LatentMimicValue,
)
from track_mjx.agent.observation_utils import normalize_dict_obs


def make_latent_mimic_ppo_networks(
    observation_size,
    action_size: int,
    policy_layer_sizes: Sequence[int] = (512, 256, 128),
    value_layer_sizes: Sequence[int] = (512, 256, 128),
) -> PPOImitationNetworks:
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    pi = LatentMimicPolicy(
        layer_sizes=tuple(policy_layer_sizes),
        action_dim=parametric_action_distribution.param_size // 2,
    )
    v = LatentMimicValue(layer_sizes=tuple(value_layer_sizes))

    def policy_apply(processor_params, params, obs, key=None, *,
                     deterministic=False, get_activation=False):
        # normalize_dict_obs flattens env's nested {state:{task_obs, prop}} to
        # the flat {imitation_target, proprioception} the policy consumes,
        # AND applies running-mean/std normalization.
        obs = normalize_dict_obs(obs, processor_params)
        logits = pi.apply(params, obs)
        batch_shape = logits.shape[:-1]
        zero_latent = jnp.zeros((*batch_shape, 1), dtype=logits.dtype)
        if get_activation:
            return logits, zero_latent, zero_latent, {}
        return logits, zero_latent, zero_latent

    def value_apply(processor_params, params, obs):
        obs = normalize_dict_obs(obs, processor_params)
        return v.apply(params, obs)

    def _dummy_obs():
        # observation_size at this point is post-flatten {imitation_target, proprioception}.
        return {k: jnp.zeros((1, observation_size[k])) for k in observation_size}

    policy_network = FeedForwardNetwork(
        init=lambda key: pi.init(key, _dummy_obs()), apply=policy_apply
    )
    value_network = FeedForwardNetwork(
        init=lambda key: v.init(key, _dummy_obs()), apply=value_apply
    )
    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
