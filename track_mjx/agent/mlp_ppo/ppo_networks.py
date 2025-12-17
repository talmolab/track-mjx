"""PPO network definitions for intention-based imitation learning.

This module provides network architectures for PPO training with VAE-style
intention encoding.

The key components are:
- PPOImitationNetworks: Container for policy, value, and action distribution
- Inference functions that route observations through encoder/decoder
- Factory functions for creating intention-based PPO networks
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import flax
import jax
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from jax import numpy as jnp

from track_mjx.agent import checkpointing, masked_running_statistics
from track_mjx.agent.mlp_ppo import intention_network


@flax.struct.dataclass
class PPOImitationNetworks:
    """Container for PPO imitation learning network components.

    Attributes:
        policy_network: Intention-based encoder-decoder policy network.
        value_network: Feedforward value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
    """

    policy_network: intention_network.IntentionNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(
    ppo_networks: PPOImitationNetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory function for inference.

    Returns a function that creates policy functions with fixed parameters.
    The policy function maps observations to actions with optional extras.

    Args:
        ppo_networks: PPO network components.

    Returns:
        A make_policy function with signature:
            make_policy(params, deterministic, get_activation) -> policy_fn
    """

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
        get_activation: bool = False,
    ) -> types.Policy:
        """Create a policy function with fixed parameters.

        Args:
            params: Tuple of (normalizer_params, policy_params).
            deterministic: If True, return mode of action distribution.
            get_activation: If True, include network activations in extras.

        Returns:
            Policy function: (obs, key) -> (action, extras_dict).
        """
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            activations = None

            if get_activation:
                logits, latent_mean, latent_logvar, activations = policy_network.apply(
                    *params,
                    observations,
                    key_network,
                    deterministic=deterministic,
                    get_activation=True,
                )
            else:
                logits, latent_mean, latent_logvar = policy_network.apply(
                    *params, observations, key_network, deterministic=deterministic
                )

            if deterministic:
                action = jnp.array(parametric_action_distribution.mode(logits))
                extras = {"latent_mean": latent_mean, "latent_logvar": latent_logvar}
                if get_activation:
                    extras["activations"] = activations
                return action, extras

            # Sample action from distribution
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
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


def make_logging_inference_fn(
    ppo_networks: PPOImitationNetworks,
) -> Callable[[bool], Callable]:
    """Create a policy factory for logging/evaluation with explicit params.

    Unlike make_inference_fn, the returned policy takes params as an argument,
    allowing evaluation with different parameter sets without recreating the policy.

    Args:
        ppo_networks: PPO network components.

    Returns:
        A make_logging_policy function with signature:
            make_logging_policy(deterministic) -> logging_policy_fn

        Where logging_policy_fn has signature:
            (params, obs, key) -> (action, extras)
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        """Create a logging policy that takes params as input.

        Args:
            deterministic: If True, return mode of action distribution.

        Returns:
            Policy function: (params, obs, key) -> (action, extras_dict).
        """
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            logits, latent_mean, latent_logvar = policy_network.apply(
                *params, observations, key_network
            )

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                }

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
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
    """Create intention-based PPO networks for imitation learning.

    Creates an encoder-decoder policy network where the encoder processes
    reference trajectory observations and the decoder generates actions
    conditioned on proprioceptive state and latent intention.

    Args:
        observation_size: Total observation dimension.
        reference_obs_size: Dimension of reference trajectory observations
            (processed by encoder).
        action_size: Action dimension.
        preprocess_observations_fn: Observation preprocessing (e.g., normalize).
        intention_latent_size: Dimension of VAE latent space.
        encoder_hidden_layer_sizes: MLP layer sizes for encoder.
        decoder_hidden_layer_sizes: MLP layer sizes for decoder.
        value_hidden_layer_sizes: MLP layer sizes for value network.

    Returns:
        PPOImitationNetworks containing policy, value, and action distribution.
    """
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


def make_decoder_policy_fn(
    ckpt_path: str | Path,
    step: int | None = None,
) -> types.Policy:
    """Load a decoder-only policy from a trained intention network checkpoint.

    Extracts the decoder portion of a trained intention network for use as
    a standalone policy. Useful for downstream tasks that provide their own
    latent intentions.

    Args:
        ckpt_path: Path to checkpoint directory.
        step: Checkpoint step to load. If None, loads latest.

    Returns:
        Policy function: (obs) -> (action, extras).
        Note: This policy is deterministic (returns mode of action distribution).
    """

    def make_decoder_policy(
        params: tuple,
        policy_network: networks.FeedForwardNetwork,
        parametric_action_distribution: distribution.ParametricDistribution,
    ) -> types.Policy:
        def policy(
            observations: types.Observation,
        ) -> tuple[types.Action, types.Extra]:
            logits, extras = policy_network.apply(*params, observations)
            return parametric_action_distribution.mode(logits), extras

        return policy

    # Load config and policy from checkpoint
    cfg = checkpointing.load_config_from_checkpoint(ckpt_path, step=step)
    observation_size = cfg["network_config"]["observation_size"]
    reference_obs_size = cfg["network_config"]["reference_obs_size"]
    action_size = cfg["network_config"]["action_size"]
    intention_latent_size = cfg["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = cfg["network_config"]["decoder_layer_sizes"]

    intention_policy_params = checkpointing.load_policy(ckpt_path, cfg, step=step)

    # Create decoder-only network
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = intention_network.make_decoder_policy(
        parametric_action_distribution.param_size,
        decoder_obs_size=(observation_size - reference_obs_size) + intention_latent_size,
        preprocess_observations_fn=masked_running_statistics.normalize,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )

    # Extract decoder normalizer params (proprioceptive portion only)
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

    return make_decoder_policy(
        decoder_params, policy_network, parametric_action_distribution
    )
