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
        num_codes: Number of codebook entries per level.
        latent_dim: Dimension of latent/codebook embeddings.
        stickiness_bias: Bias for temporal code persistence.
        rvq_depth: Number of RVQ depth levels.
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    num_codes: int = 512
    latent_dim: int = 60
    stickiness_bias: float = 0.0
    rvq_depth: int = 1


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
                logits, z_e, all_indices, logvar, activations = policy_network.apply(
                    *params,
                    observations,
                    key_network,
                    deterministic=deterministic,
                    get_activation=True,
                )
            else:
                logits, z_e, all_indices, logvar = policy_network.apply(
                    *params, observations, key_network, deterministic=deterministic
                )

            # all_indices is tuple of D arrays; primary level for compat
            indices = all_indices[0] if isinstance(all_indices, tuple) else all_indices

            if deterministic:
                action = jnp.array(parametric_action_distribution.mode(logits))
                extras = {
                    "z_e": z_e,
                    "indices": indices,
                    "all_indices": all_indices,
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
                "all_indices": all_indices,
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
            (params, obs, key, prev_indices) -> (action, extras)
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        """Create a logging policy that takes params as input.

        Args:
            deterministic: If True, return mode of action distribution.

        Returns:
            Policy function: (params, obs, key, prev_indices) -> (action, extras_dict).
        """
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            key_sample: PRNGKey,
            prev_indices: tuple[jnp.ndarray, ...] | jnp.ndarray | None = None,
        ) -> tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            logits, z_e, all_indices, logvar = policy_network.apply(
                *params,
                observations,
                key_network,
                deterministic=deterministic,
                prev_indices=prev_indices,
            )

            # Primary level indices for backward compat
            indices = all_indices[0] if isinstance(all_indices, tuple) else all_indices

            if deterministic:
                return jnp.array(parametric_action_distribution.mode(logits)), {
                    "z_e": z_e,
                    "indices": indices,
                    "all_indices": all_indices,
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
                "all_indices": all_indices,
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


def make_vq_augmented_value_network(
    obs_sizes: Mapping[str, int],
    num_codes: int,
    commitment_horizon: int,
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> networks.FeedForwardNetwork:
    """Create a value network augmented with D0 code identity and timer.

    The value function V(s, c^(0), tau) receives the standard observation
    plus a one-hot encoding of the D0 code and a normalized timer value.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        num_codes: Number of D0 codebook entries (for one-hot encoding).
        commitment_horizon: H, commitment horizon (for tau normalization).
        hidden_layer_sizes: MLP layer sizes for value network.

    Returns:
        FeedForwardNetwork that accepts dict observations plus code info.
    """
    total_obs_size = sum(obs_sizes.values())

    # Input size: flat obs + one-hot D0 code + tau/H scalar
    augmented_input_size = total_obs_size + num_codes + 1

    base_value_network = networks.make_value_network(
        augmented_input_size,
        preprocess_observations_fn=types.identity_observation_preprocessor,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        value_params,
        obs: Mapping[str, jnp.ndarray],
        d0_code_idx: jnp.ndarray | None = None,
        tau: jnp.ndarray | None = None,
    ):
        """Apply augmented value network.

        Args:
            processor_params: Running statistics for normalization.
            value_params: Value network parameters.
            obs: Dictionary observation.
            d0_code_idx: D0 code indices, shape [...]. None pads with zeros.
            tau: Timer values, shape [...]. None pads with zeros.
        """
        # Normalize obs first, then append code info (NOT normalized)
        normalized_obs = normalize_dict_obs(obs, processor_params)
        flat_obs = concat_flat_dict_obs(normalized_obs)

        if d0_code_idx is not None:
            code_one_hot = jax.nn.one_hot(d0_code_idx, num_codes)
            tau_normalized = tau.astype(jnp.float32) / commitment_horizon
            # Expand tau to match batch dims
            tau_expanded = jnp.expand_dims(tau_normalized, axis=-1)
            augmented = jnp.concatenate([flat_obs, code_one_hot, tau_expanded], axis=-1)
        else:
            # Pad with zeros for backward compat / edge cases
            zeros_code = jnp.zeros(
                flat_obs.shape[:-1] + (num_codes,), dtype=flat_obs.dtype
            )
            zeros_tau = jnp.zeros(flat_obs.shape[:-1] + (1,), dtype=flat_obs.dtype)
            augmented = jnp.concatenate([flat_obs, zeros_code, zeros_tau], axis=-1)

        return base_value_network.apply((), value_params, augmented)

    return networks.FeedForwardNetwork(
        init=lambda key: base_value_network.init(key),
        apply=apply,
    )


def make_vq_chunked_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    commitment_horizon: int,
    latent_dim: int = 60,
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    codebook_init_scale: float = 1.0,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    stickiness_bias: float | tuple[float, ...] = 0.0,
    rvq_depth: int = 2,
    use_rotation: bool = False,
    coupled_residual_grad: bool = False,
    proprio_noise_scale: float = 0.0,
    use_continuous_latent: bool = False,
    continuous_latent_dim: int = 4,
) -> VQPPOImitationNetworks:
    """Create VQ-VAE PPO networks with D0 temporal commitment (code chunking).

    Uses an augmented value function V(s, c^(0), tau) that conditions on the
    committed D0 code and timer.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension.
        commitment_horizon: H, number of steps to hold D0 code.
        latent_dim: Dimension of VQ-VAE latent/codebook embeddings.
        num_codes: Number of codebook entries per level.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        encoder_hidden_layer_sizes: MLP layer sizes for encoder.
        decoder_hidden_layer_sizes: MLP layer sizes for decoder.
        value_hidden_layer_sizes: MLP layer sizes for value network.
        stickiness_bias: Per-level stickiness bias. Float or tuple.
        rvq_depth: Number of RVQ depth levels (>= 1).
        use_rotation: If True, use Householder rotation-augmented STE.
        coupled_residual_grad: If True, couple depth gradients. Must be False
            when rvq_depth >= 2 with chunking (to avoid zero-gradient at
            intermediate steps).
        proprio_noise_scale: Gaussian noise std on normalized proprio.
        use_continuous_latent: If True, encoder outputs (mean, logvar).

    Returns:
        VQPPOImitationNetworks containing policy and augmented value network.
    """
    if rvq_depth >= 2:
        assert not coupled_residual_grad, (
            "coupled_residual_grad must be False when using code chunking. "
            "Coupled gradients through held D0 codes produce zero D1 gradients "
            "at worker steps ((H-1)/H of all steps)."
        )

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
        stickiness_bias=stickiness_bias,
        rvq_depth=rvq_depth,
        use_rotation=use_rotation,
        coupled_residual_grad=coupled_residual_grad,
        proprio_noise_scale=proprio_noise_scale,
        use_continuous_latent=use_continuous_latent,
        continuous_latent_dim=continuous_latent_dim,
    )

    value_network = make_vq_augmented_value_network(
        obs_sizes=obs_sizes,
        num_codes=num_codes,
        commitment_horizon=commitment_horizon,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return VQPPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        num_codes=num_codes,
        latent_dim=latent_dim,
        stickiness_bias=stickiness_bias,
        rvq_depth=rvq_depth,
    )


def make_vq_chunked_inference_fn(
    ppo_networks: VQPPOImitationNetworks,
    commitment_horizon: int,
) -> Callable[..., Callable]:
    """Create a stateful policy factory for chunked rollout.

    The returned policy carries chunk state (held_d0_idx, tau) through
    the rollout, matching the Semi-MDP temporal commitment pattern.

    Args:
        ppo_networks: VQ PPO network components.
        commitment_horizon: H, number of steps to hold D0 code.

    Returns:
        A make_policy function with signature:
            make_policy(params, deterministic) -> policy_fn

        Where policy_fn has signature:
            (obs, chunk_state, key) -> (action, extras, new_chunk_state)
    """

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> Callable:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            chunk_state: tuple[jnp.ndarray, jnp.ndarray],
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, tuple[jnp.ndarray, jnp.ndarray]]:
            held_d0_idx, tau = chunk_state
            key_sample, key_network = jax.random.split(key_sample)

            logits, z_e, all_indices, logvar, new_chunk = (
                policy_network.apply_step_chunked(
                    *params,
                    observations,
                    held_d0_idx=held_d0_idx,
                    tau=tau,
                    commitment_horizon=commitment_horizon,
                    key=key_network,
                    deterministic=deterministic,
                )
            )

            d0_idx = all_indices[0]

            if deterministic:
                action = jnp.array(parametric_action_distribution.mode(logits))
                extras = {
                    "z_e": z_e,
                    "indices": d0_idx,
                    "all_indices": all_indices,
                    "d0_indices": d0_idx,
                    "tau": tau,
                }
                return action, extras, new_chunk

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return (
                jnp.array(postprocessed_actions),
                {
                    "z_e": z_e,
                    "indices": d0_idx,
                    "all_indices": all_indices,
                    "d0_indices": d0_idx,
                    "tau": tau,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                },
                new_chunk,
            )

        return policy

    return make_policy


def make_vq_chunked_logging_inference_fn(
    ppo_networks: VQPPOImitationNetworks,
    commitment_horizon: int,
) -> Callable[[bool], Callable]:
    """Create a chunked policy factory for logging/evaluation with explicit params.

    Like make_vq_chunked_inference_fn but the returned policy takes params as
    an argument, matching the logging inference fn pattern.

    Args:
        ppo_networks: VQ PPO network components.
        commitment_horizon: H, number of steps to hold D0 code.

    Returns:
        A make_logging_policy function with signature:
            make_logging_policy(deterministic) -> logging_policy_fn

        Where logging_policy_fn has signature:
            (params, obs, chunk_state, key) -> (action, extras, new_chunk_state)
    """

    def make_logging_policy(deterministic: bool = False) -> Callable:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            chunk_state: tuple[jnp.ndarray, jnp.ndarray],
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, tuple[jnp.ndarray, jnp.ndarray]]:
            held_d0_idx, tau = chunk_state
            key_sample, key_network = jax.random.split(key_sample)

            logits, z_e, all_indices, logvar, new_chunk = (
                policy_network.apply_step_chunked(
                    *params,
                    observations,
                    held_d0_idx=held_d0_idx,
                    tau=tau,
                    commitment_horizon=commitment_horizon,
                    key=key_network,
                    deterministic=deterministic,
                )
            )

            d0_idx = all_indices[0]

            if deterministic:
                return (
                    jnp.array(parametric_action_distribution.mode(logits)),
                    {
                        "z_e": z_e,
                        "indices": d0_idx,
                        "all_indices": all_indices,
                    },
                    new_chunk,
                )

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            return (
                jnp.array(postprocessed_actions),
                {
                    "z_e": z_e,
                    "indices": d0_idx,
                    "all_indices": all_indices,
                    "log_prob": log_prob,
                    "raw_action": raw_actions,
                    "logits": logits,
                },
                new_chunk,
            )

        return logging_policy

    return make_logging_policy


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
    stickiness_bias: float | tuple[float, ...] = 0.0,
    rvq_depth: int = 1,
    use_rotation: bool = False,
    coupled_residual_grad: bool = False,
    proprio_noise_scale: float = 0.0,
    use_continuous_latent: bool = False,
    continuous_latent_dim: int = 4,
) -> VQPPOImitationNetworks:
    """Create VQ-VAE intention-based PPO networks for imitation learning.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension.
        latent_dim: Dimension of VQ-VAE latent/codebook embeddings.
        num_codes: Number of codebook entries per level.
        commitment_cost: Weight for commitment loss (beta).
        codebook_init_scale: Scale for codebook initialization.
        encoder_hidden_layer_sizes: MLP layer sizes for encoder.
        decoder_hidden_layer_sizes: MLP layer sizes for decoder.
        value_hidden_layer_sizes: MLP layer sizes for value network.
        stickiness_bias: Per-level stickiness bias. Float or tuple.
        rvq_depth: Number of RVQ depth levels. 1 = vanilla VQ.
        use_rotation: If True, use Householder rotation-augmented STE.
        coupled_residual_grad: If True and use_rotation, couple depth gradients.

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
        stickiness_bias=stickiness_bias,
        rvq_depth=rvq_depth,
        use_rotation=use_rotation,
        coupled_residual_grad=coupled_residual_grad,
        proprio_noise_scale=proprio_noise_scale,
        use_continuous_latent=use_continuous_latent,
        continuous_latent_dim=continuous_latent_dim,
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
        stickiness_bias=stickiness_bias,
        rvq_depth=rvq_depth,
    )
