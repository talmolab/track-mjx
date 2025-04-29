"""
Custom network definitions.
This is needed because we need to route the observations
to proper places in the network in the case of the VAE (CoMic, Hasenclever 2020)
"""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
from pathlib import Path
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.types import PRNGKey

import jax
from jax import numpy as jnp
import flax

from track_mjx.agent import intention_network, masked_running_statistics, checkpointing

from omegaconf import DictConfig, OmegaConf


@flax.struct.dataclass
class PPOImitationNetworks:
    policy_network: intention_network.IntentionNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPOImitationNetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
        get_activation: bool = False,
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        # can modify this to provide stochastic action + noise
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            activations = None
            if get_activation:
                logits, latent_mean, latent_logvar, activations = policy_network.apply(
                    *params, observations, key_network, get_activation=True
                )
                # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)
            else:
                logits, latent_mean, latent_logvar = policy_network.apply(
                    *params, observations, key_network
                )
            if deterministic:
                if get_activation:
                    return jnp.array(
                        ppo_networks.parametric_action_distribution.mode(logits)
                    ), {
                        "activations": activations,
                        "latent_mean": latent_mean,
                        "latent_logvar": latent_logvar,
                    }
                return jnp.array(
                    ppo_networks.parametric_action_distribution.mode(logits)
                ), {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                }

            # action sampling is happening here, according to distribution parameter logits
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )

            # probability of selection specific action, actions with higher reward should have higher probability
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)

            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return jnp.array(postprocessed_actions), {
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
    """Creates params and inference function for the PPO agent.
    The policy takes the params as an input, so different sets of params can be used.
    """

    def make_logging_policy(deterministic: bool = False) -> types.Policy:
        policy_network = ppo_networks.policy_network
        # can modify this to provide stochastic action + noise
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            logits, latent_mean, latent_logvar = policy_network.apply(
                *params, observations, key_network
            )
            # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)

            if deterministic:
                return jnp.array(
                    ppo_networks.parametric_action_distribution.mode(logits)
                ), {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                }

            # action sampling is happening here, according to distribution parameter logits
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )

            # probability of selection specific action, actions with higher reward should have higher probability
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)

            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return jnp.array(postprocessed_actions), {
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
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = intention_network.make_intention_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        reference_obs_size=reference_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
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


def make_decoder_policy_fn(ckpt_path: str | Path, step: int = None):

    def make_decoder_policy(
        params, policy_network, parametric_action_distribution
    ) -> types.Policy:
        def policy(
            observations: types.Observation,
        ) -> Tuple[types.Action, types.Extra]:
            logits, extras = policy_network.apply(*params, observations)
            return parametric_action_distribution.mode(logits), extras

        return policy

    cfg = checkpointing.load_config_from_checkpoint(ckpt_path, step=step)
    observation_size = cfg["network_config"]["observation_size"]
    reference_obs_size = cfg["network_config"]["reference_obs_size"]
    action_size = cfg["network_config"]["action_size"]
    intention_latent_size = cfg["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = cfg["network_config"]["decoder_layer_sizes"]

    intention_policy_params = checkpointing.load_policy(ckpt_path, cfg, step=step)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = intention_network.make_decoder_policy(
        parametric_action_distribution.param_size,
        decoder_obs_size=(observation_size - reference_obs_size)
        + intention_latent_size,
        preprocess_observations_fn=masked_running_statistics.normalize,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    decoder_normalizer_params = masked_running_statistics.RunningStatisticsState(
        count=jnp.zeros(()),
        mean=intention_policy_params[0].mean[reference_obs_size:],
        summed_variance=intention_policy_params[0].summed_variance[reference_obs_size:],
        std=intention_policy_params[0].std[reference_obs_size:],
    )
    decoder_params = (
        decoder_normalizer_params,
        {"params": intention_policy_params[1]["params"]["decoder"]},
    )
    decoder_policy = make_decoder_policy(
        decoder_params, policy_network, parametric_action_distribution
    )
    return decoder_policy


def make_decoder_policy_fn(ckpt_path: str | Path, step: int = None):

    def make_decoder_policy(
        params, policy_network, parametric_action_distribution
    ) -> types.Policy:
        def policy(
            observations: types.Observation,
        ) -> Tuple[types.Action, types.Extra]:
            logits, extras = policy_network.apply(*params, observations)
            return parametric_action_distribution.mode(logits), extras

        return policy

    cfg = checkpointing.load_config_from_checkpoint(ckpt_path, step=step)
    observation_size = cfg["network_config"]["observation_size"]
    reference_obs_size = cfg["network_config"]["reference_obs_size"]
    action_size = cfg["network_config"]["action_size"]
    intention_latent_size = cfg["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = cfg["network_config"]["decoder_layer_sizes"]

    intention_policy_params = checkpointing.load_policy(ckpt_path, cfg, step=step)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = intention_network.make_decoder_policy(
        parametric_action_distribution.param_size,
        decoder_obs_size=(observation_size - reference_obs_size)
        + intention_latent_size,
        preprocess_observations_fn=masked_running_statistics.normalize,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    decoder_normalizer_params = masked_running_statistics.RunningStatisticsState(
        count=jnp.zeros(()),
        mean=intention_policy_params[0].mean[reference_obs_size:],
        summed_variance=intention_policy_params[0].summed_variance[reference_obs_size:],
        std=intention_policy_params[0].std[reference_obs_size:],
    )
    decoder_params = (
        decoder_normalizer_params,
        {"params": intention_policy_params[1]["params"]["decoder"]},
    )
    decoder_policy = make_decoder_policy(
        decoder_params, policy_network, parametric_action_distribution
    )
    return decoder_policy
