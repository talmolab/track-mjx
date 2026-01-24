"""PPO network definitions for VQ-VAE intention-based imitation learning.

This module provides network architectures for PPO training with VQ-VAE
style intention encoding using discrete codebooks.

The key components are:
- VQPPOImitationNetworks: Container for policy, value, and action distribution
- Inference functions that route observations through encoder/quantizer/decoder
- Factory functions for creating VQ-VAE intention-based PPO networks

Observations are expected as dictionaries with keys:
- "imitation_target": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Callable, Mapping, Sequence

import flax
import jax
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from jax import numpy as jnp

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    concat_flat_dict_obs,
    normalize_dict_obs,
)

# Import from local scratch directory
from vq_intention_network import make_vq_intention_policy


@flax.struct.dataclass
class VQPPOImitationNetworks:
    """Container for VQ-VAE PPO imitation learning network components.

    Attributes:
        policy_network: VQ-VAE encoder-quantizer-decoder policy network.
        value_network: Feedforward value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
        num_codes: Number of codebook entries.
        latent_dim: Dimension of latent/codebook embeddings.
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    num_codes: int = 512
    latent_dim: int = 60


def make_vq_inference_fn(
    ppo_networks: VQPPOImitationNetworks,
) -> Callable[..., types.Policy]:
    """Create a policy factory function for inference with VQ-VAE.

    Returns a function that creates policy functions with fixed parameters.
    The policy function maps observations to actions with optional extras.

    Args:
        ppo_networks: VQ PPO network components.

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
                Note: VQ quantization itself is always deterministic.
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
                logits, z_e, indices, activations = policy_network.apply(
                    *params,
                    observations,
                    key_network,
                    deterministic=deterministic,
                    get_activation=True,
                )
            else:
                logits, z_e, indices = policy_network.apply(
                    *params, observations, key_network, deterministic=deterministic
                )

            if deterministic:
                action = jnp.array(parametric_action_distribution.mode(logits))
                extras = {
                    "z_e": z_e,
                    "indices": indices,
                }
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
                "z_e": z_e,
                "indices": indices,
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
                "activations": activations,
            }

        return policy

    return make_policy


def make_vq_logging_inference_fn(
    ppo_networks: VQPPOImitationNetworks,
) -> Callable[[bool], Callable]:
    """Create a policy factory for logging/evaluation with explicit params.

    Unlike make_vq_inference_fn, the returned policy takes params as an argument,
    allowing evaluation with different parameter sets without recreating the policy.

    Args:
        ppo_networks: VQ PPO network components.

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
            logits, z_e, indices = policy_network.apply(
                *params, observations, key_network
            )

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), {
                    "z_e": z_e,
                    "indices": indices,
                }

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return jnp.array(postprocessed_actions), {
                "z_e": z_e,
                "indices": indices,
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
            }

        return logging_policy

    return make_logging_policy


def make_vq_dict_value_network(
    obs_sizes: Mapping[str, int],
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts dictionary observations.

    The value network normalizes each observation component, flattens them,
    and concatenates before passing to the MLP.

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


def make_vq_intention_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    latent_dim: int = 60,
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    codebook_init_scale: float = 1.0,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    encoder_type: str = "mlp",
    temporal_stride: int = 1,
    encoder_hidden_channels: Sequence[int] | None = None,
    encoder_kernel_size: int = 3,
) -> VQPPOImitationNetworks:
    """Create VQ-VAE intention-based PPO networks for imitation learning.

    Creates an encoder-quantizer-decoder policy network where the encoder
    processes reference trajectory observations, the quantizer maps to
    discrete codebook entries, and the decoder generates actions conditioned
    on proprioceptive state and quantized intention.

    Supports two encoder types:
    - "mlp": Standard MLP encoder (original behavior, no temporal downsampling)
    - "temporal_conv": Conv1D encoder with temporal downsampling, forcing
      codes to represent chunks of frames (behavioral primitives)

    Args:
        obs_sizes: Dict mapping observation keys to their sizes, e.g.
            {"imitation_target": 3716, "proprioception": 226}.
        action_size: Action dimension.
        latent_dim: Dimension of VQ-VAE latent/codebook embeddings.
        num_codes: Number of codebook entries (vocabulary size).
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        encoder_hidden_layer_sizes: MLP layer sizes for MLP encoder.
        decoder_hidden_layer_sizes: MLP layer sizes for decoder.
        value_hidden_layer_sizes: MLP layer sizes for value network.
        encoder_type: Type of encoder ("mlp" or "temporal_conv").
        temporal_stride: Temporal downsampling factor for temporal_conv encoder.
            E.g., 4 means each code represents ~4 frames.
        encoder_hidden_channels: Channel sizes for temporal conv encoder.
            Defaults to (256, 256) if not provided.
        encoder_kernel_size: Kernel size for temporal convolutions.

    Returns:
        VQPPOImitationNetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = make_vq_intention_policy(
        action_param_size=parametric_action_distribution.param_size,
        latent_dim=latent_dim,
        obs_sizes=obs_sizes,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        num_codes=num_codes,
        commitment_cost=commitment_cost,
        codebook_init_scale=codebook_init_scale,
        encoder_type=encoder_type,
        temporal_stride=temporal_stride,
        encoder_hidden_channels=encoder_hidden_channels,
        encoder_kernel_size=encoder_kernel_size,
    )

    value_network = make_vq_dict_value_network(
        obs_sizes=obs_sizes,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return VQPPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        num_codes=num_codes,
        latent_dim=latent_dim,
    )
