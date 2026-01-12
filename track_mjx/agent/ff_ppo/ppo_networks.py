"""PPO network definitions for intention-based imitation learning.

This module provides network architectures for PPO training with VAE-style
intention encoding.

The key components are:
- PPOImitationNetworks: Container for policy, value, and action distribution
- Inference functions that route observations through encoder/decoder
- Factory functions for creating intention-based PPO networks

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import flax
import jax
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey
from jax import numpy as jnp

from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import intention_network
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)


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

            # Get batch size from observations for per-sample key generation
            obs_leaf = jax.tree_util.tree_leaves(observations)[0]
            batch_size = obs_leaf.shape[0]

            # Generate per-sample keys for deterministic replay
            per_sample_keys = jax.random.split(key_network, batch_size)

            if get_activation:
                logits, latent_mean, latent_logvar, activations = policy_network.apply(
                    *params,
                    observations,
                    per_sample_keys,
                    deterministic=deterministic,
                    get_activation=True,
                )
            else:
                logits, latent_mean, latent_logvar = policy_network.apply(
                    *params, observations, per_sample_keys, deterministic=deterministic
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
                "policy_rng": per_sample_keys,
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

            # Get batch size and generate per-sample keys
            obs_leaf = jax.tree_util.tree_leaves(observations)[0]
            batch_size = obs_leaf.shape[0]
            per_sample_keys = jax.random.split(key_network, batch_size)

            logits, latent_mean, latent_logvar = policy_network.apply(
                *params, observations, per_sample_keys
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


def make_dict_value_network(
    obs_sizes: Mapping[str, int],
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts dictionary observations.

    The value network flattens the dict observation internally and normalizes
    each component before concatenating.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        hidden_layer_sizes: MLP layer sizes for value network.

    Returns:
        FeedForwardNetwork that accepts dict observations.
    """
    total_obs_size = sum(obs_sizes.values())

    # Create underlying value network with flat observations
    base_value_network = networks.make_value_network(
        total_obs_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        value_params,
        obs: Mapping[str, jnp.ndarray],
    ):
        """Apply value network with dict observation normalization."""
        # Normalize each component and flatten
        normalized_obs = normalize_dict_obs(obs, processor_params)
        flat_obs = concat_flat_dict_obs(normalized_obs)
        return base_value_network.apply((), value_params, flat_obs)

    return networks.FeedForwardNetwork(
        init=lambda key: base_value_network.init(key),
        apply=apply,
    )


def make_intention_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
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
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 3716, "proprioception": 226}.
        action_size: Action dimension.
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
        obs_sizes=obs_sizes,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )

    value_network = make_dict_value_network(
        obs_sizes=obs_sizes,
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
        decoder_obs_size=(observation_size - reference_obs_size)
        + intention_latent_size,
        preprocess_observations_fn=running_statistics.normalize,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )

    # Extract decoder normalizer params (proprioceptive portion only)
    decoder_normalizer_params_dict = jax.tree.map(
        lambda x: (
            x[reference_obs_size:] if isinstance(x, jnp.ndarray) and x.ndim >= 1 else x
        ),
        intention_policy_params[0].__dict__,
    )

    decoder_normalizer_params = running_statistics.RunningStatisticsState(
        **decoder_normalizer_params_dict
    )

    decoder_params = (
        decoder_normalizer_params,
        {"params": intention_policy_params[1]["params"]["decoder"]},
    )

    return make_decoder_policy(
        decoder_params, policy_network, parametric_action_distribution
    )
