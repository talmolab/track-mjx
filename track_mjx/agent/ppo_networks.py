"""
Custom network definitions.
This is needed because we need to route the observations 
to proper places in the network in the case of the VAE (CoMic, Hasenclever 2020)
"""

import dataclasses
from typing import Any, Callable, Sequence, Tuple, Optional
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP

from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from track_mjx.agent import intention_network #TODO: still might need this for typing


@flax.struct.dataclass
class PPOImitationNetworks:
    policy_network: intention_network.IntentionNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPOImitationNetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False, get_activation: bool = True, use_lstm: bool = True,
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        # can modify this to provide stochastic action + noise
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
            hidden_state: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            activations = None
            
            # here determines if use hidden states
            if get_activation:
                if use_lstm:
                    print('In custom_ppo_network using LSTM + Activation')
                    logits, latent_mean, latent_logvar, new_hidden_state, activations = policy_network.apply(*params, observations, key_network, hidden_state, get_activation=get_activation, use_lstm=use_lstm)
                else:
                    logits, latent_mean, latent_logvar, activations = policy_network.apply(*params, observations, key_network, hidden_state, get_activation=get_activation, use_lstm=use_lstm)
                    # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)
            else:
                if use_lstm:
                    logits, latent_mean, latent_logvar, new_hidden_state = policy_network.apply(*params, observations, key_network, hidden_state, get_activation=get_activation, use_lstm=use_lstm)
                else:
                    logits, latent_mean, latent_logvar, = policy_network.apply(*params, observations, key_network, hidden_state, get_activation=get_activation, use_lstm=use_lstm)
            
            if deterministic:
                # returning hidden_state here
                if get_activation:
                    if use_lstm:
                        return ppo_networks.parametric_action_distribution.mode(logits), {"activations": activations}, new_hidden_state # swapped order from network return
                    else:
                        return ppo_networks.parametric_action_distribution.mode(logits), {"activations": activations}
                
                else:
                    if use_lstm:
                        return ppo_networks.parametric_action_distribution.mode(logits), {}, new_hidden_state
                    else:
                        return ppo_networks.parametric_action_distribution.mode(logits), {}

            # action sampling is happening here, according to distribution parameter logits
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )

            # probability of selection specific action, actions with higher reward should have higher probability
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)

            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            
            if use_lstm:
                return postprocessed_actions, {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                    "activations": activations,
                }, new_hidden_state
            else:
                return postprocessed_actions, {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                    "activations": activations,
                }
                

        return policy

    return make_policy


def make_logging_inference_fn(ppo_networks: PPOImitationNetworks):
    """Creates params and inference function for the PPO agent with LSTM support."""

    def make_logging_policy(
        deterministic: bool = False, get_activation: bool = False, use_lstm: bool = True
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
            hidden_state: jnp.ndarray,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            activations = None

            if use_lstm:
                logits, latent_mean, latent_logvar, new_hidden_state = policy_network.apply(
                    *params,
                    observations,
                    key_network,
                    hidden_state,
                    get_activation=get_activation,
                    use_lstm=use_lstm,
                )
            else:
                logits, latent_mean, latent_logvar, = policy_network.apply(
                    *params,
                    observations,
                    key_network,
                    hidden_state, # will not be used
                    get_activation=get_activation,
                    use_lstm=use_lstm,
                )

            if deterministic:
                if use_lstm:
                    return ppo_networks.parametric_action_distribution.mode(logits), {"latent_mean": latent_mean, "latent_logvar": latent_logvar}, new_hidden_state
                else:
                    return ppo_networks.parametric_action_distribution.mode(logits), {"latent_mean": latent_mean, "latent_logvar": latent_logvar}

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)

            if use_lstm:
                return postprocessed_actions, {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                    "hidden_state": new_hidden_state,
                }, new_hidden_state
            else:
                return postprocessed_actions, {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                }

        return logging_policy

    return make_logging_policy


# intention policy
def make_intention_ppo_networks(
    observation_size: int,
    reference_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    hidden_state_size: int = 128,
    hidden_layer_num: int = 2,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    get_activation: bool = True,
    use_lstm: bool = True,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = intention_network.make_intention_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        hidden_state_size=hidden_state_size,
        hidden_layer_num=hidden_layer_num,
        total_obs_size=observation_size,
        reference_obs_size=reference_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        get_activation=get_activation,
        use_lstm=use_lstm,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
