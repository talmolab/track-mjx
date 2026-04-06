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
import flax.linen as nn
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

            # Check if observations are batched (ndim >= 2) or unbatched (ndim == 1)
            # Get first leaf array from nested observation structure
            obs_leaves = jax.tree_util.tree_leaves(observations)
            obs_leaf = obs_leaves[0]
            if obs_leaf.ndim >= 2:
                # Batched observations - generate per-sample keys for deterministic replay
                batch_size = obs_leaf.shape[0]
                per_sample_keys = jax.random.split(key_network, batch_size)
            else:
                # Unbatched observation - use single key
                per_sample_keys = key_network

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
                activations = None

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

            # Check if observations are batched (ndim >= 2) or unbatched (ndim == 1)
            # Get first leaf array from nested observation structure
            obs_leaves = jax.tree_util.tree_leaves(observations)
            obs_leaf = obs_leaves[0]
            if obs_leaf.ndim >= 2:
                # Batched observations - generate per-sample keys
                batch_size = obs_leaf.shape[0]
                per_sample_keys = jax.random.split(key_network, batch_size)
            else:
                # Unbatched observation - use single key
                per_sample_keys = key_network

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
    # Only count obs keys that concat_flat_dict_obs uses (excludes vision)
    total_obs_size = obs_sizes.get("imitation_target", 0) + obs_sizes.get(
        "proprioception", 0
    )

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
    encoder_noise_std: float = 0.0,
    proprioception_noise_std: float = 0.0,
    proprioception_noise_mode: str = "multiplicative",
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    activation: networks.ActivationFn = nn.silu,
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
        encoder_noise_std: Stddev for additive Gaussian noise on the
            encoder's imitation_target input during stochastic passes.
        proprioception_noise_std: Stddev for Gaussian noise on decoder
            proprioception input during stochastic training passes.
        proprioception_noise_mode: "multiplicative" or "additive".
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
        encoder_noise_std=encoder_noise_std,
        proprioception_noise_std=proprioception_noise_std,
        proprioception_noise_mode=proprioception_noise_mode,
        activation=activation,
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


def make_vision_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    vision_shape: tuple[int, int, int] = (64, 64, 3),
    vision_latent_size: int = 8,
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_channels: Sequence[int] = (2, 4, 8, 16),
    activation: networks.ActivationFn = nn.silu,
) -> PPOImitationNetworks:
    """Create vision-based PPO networks (CNN encoder + MLP decoder).

    The policy uses a CNN to encode raw pixels into a latent feature vector,
    which is concatenated with proprioception and decoded into actions.
    No imitation target is used. The value network uses only proprioception.

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension.
        vision_shape: Shape of the vision input (H, W, C).
        vision_latent_size: Dimension of the CNN latent output.
        decoder_hidden_layer_sizes: MLP layer sizes for policy decoder.
        value_hidden_layer_sizes: MLP layer sizes for value network.
        vision_channels: Channel sizes for CNN conv layers.

    Returns:
        PPOImitationNetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = intention_network.make_vision_only_policy(
        parametric_action_distribution.param_size,
        latent_size=vision_latent_size,
        obs_sizes=obs_sizes,
        vision_shape=vision_shape,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        vision_channels=vision_channels,
        activation=activation,
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


def make_vision_value_network(
    vision_shape: tuple[int, int, int],
    hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_latent_size: int = 8,
    vision_channels: Sequence[int] = (2, 4, 8, 16),
) -> networks.FeedForwardNetwork:
    """Create a value network that processes vision through a CNN.

    Used when proprioception/imitation_target are not available to the value
    network (e.g. vision-only high-level transfer). The CNN encodes pixels
    to a feature vector which feeds into a standard value MLP.

    Args:
        vision_shape: Shape of the vision input (H, W, C).
        hidden_layer_sizes: MLP layer sizes for value head.
        vision_latent_size: Output dimension of the CNN encoder.
        vision_channels: Channel sizes for each CNN conv layer.

    Returns:
        FeedForwardNetwork that accepts dict observations with a "vision" key.
    """
    from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

    class VisionValueNetwork(intention_network.nn.Module):
        """Value network with CNN vision encoder."""

        value_layers: Sequence[int]
        latent_size: int
        channels: Sequence[int]

        def setup(self):
            self.vision_encoder = VisionEncoder(
                feature_size=self.latent_size,
                channels=self.channels,
            )
            self.value_head = intention_network.Decoder(
                layer_sizes=list(self.value_layers) + [1],
            )

        def __call__(self, vision: jnp.ndarray) -> jnp.ndarray:
            z = self.vision_encoder(vision)
            value, _ = self.value_head(z)
            return jnp.squeeze(value, axis=-1)

    value_module = VisionValueNetwork(
        value_layers=list(hidden_layer_sizes),
        latent_size=vision_latent_size,
        channels=list(vision_channels),
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        value_params,
        obs: Mapping[str, jnp.ndarray],
    ):
        """Apply vision value network."""
        normalized_obs = normalize_dict_obs(obs, processor_params)
        vision = normalized_obs["vision"]
        return value_module.apply(value_params, vision)

    dummy_vision = jnp.zeros((1,) + vision_shape)

    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_vision),
        apply=apply,
    )


def make_vision_highlvl_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    vision_shape: tuple[int, int, int] = (32, 32, 1),
    vision_latent_size: int = 8,
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_channels: Sequence[int] = (2, 4, 8, 16),
    activation: networks.ActivationFn = nn.silu,
) -> PPOImitationNetworks:
    """Create vision-only PPO networks for high-level transfer training.

    Unlike ``make_vision_ppo_networks``, the value network also uses a CNN to
    process vision instead of relying on proprioception. This is needed when
    the high-level policy receives only vision (proprioception is 0-dim),
    as in vision-based transfer learning with a frozen decoder.

    Policy: CNN(pixels) -> z -> MLP -> latent intentions
    Value:  CNN(pixels) -> z -> MLP -> scalar value

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension (latent intention size).
        vision_shape: Shape of the vision input (H, W, C).
        vision_latent_size: Dimension of the CNN latent output.
        decoder_hidden_layer_sizes: MLP layer sizes for policy head.
        value_hidden_layer_sizes: MLP layer sizes for value head.
        vision_channels: Channel sizes for CNN conv layers.

    Returns:
        PPOImitationNetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = intention_network.make_vision_only_policy(
        parametric_action_distribution.param_size,
        latent_size=vision_latent_size,
        obs_sizes=obs_sizes,
        vision_shape=vision_shape,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        vision_channels=vision_channels,
        activation=activation,
    )

    value_network = make_vision_value_network(
        vision_shape=vision_shape,
        hidden_layer_sizes=value_hidden_layer_sizes,
        vision_latent_size=vision_latent_size,
        vision_channels=vision_channels,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_vision_task_obs_value_network(
    vision_shape: tuple[int, int, int],
    task_obs_size: int,
    hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_latent_size: int = 8,
    vision_channels: Sequence[int] = (2, 4, 8, 16),
) -> networks.FeedForwardNetwork:
    """Create a value network that processes both vision and task observations.

    Used for vision + task_obs high-level transfer where the value network
    needs access to both visual input and task-relevant body signals. The CNN
    encodes pixels to a feature vector which is concatenated with task_obs
    and fed into a standard value MLP.

    Args:
        vision_shape: Shape of the vision input (H, W, C).
        task_obs_size: Dimension of the task observation vector.
        hidden_layer_sizes: MLP layer sizes for value head.
        vision_latent_size: Output dimension of the CNN encoder.
        vision_channels: Channel sizes for each CNN conv layer.

    Returns:
        FeedForwardNetwork that accepts dict observations with "vision" and
        "imitation_target" keys.
    """
    from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

    class VisionTaskObsValueNetwork(intention_network.nn.Module):
        """Value network with CNN vision encoder and task observation input."""

        value_layers: Sequence[int]
        latent_size: int
        channels: Sequence[int]

        def setup(self):
            self.vision_encoder = VisionEncoder(
                feature_size=self.latent_size,
                channels=self.channels,
            )
            self.value_head = intention_network.Decoder(
                layer_sizes=list(self.value_layers) + [1],
            )

        def __call__(
            self, vision: jnp.ndarray, task_obs: jnp.ndarray
        ) -> jnp.ndarray:
            z = self.vision_encoder(vision)
            combined = jnp.concatenate([z, task_obs], axis=-1)
            value, _ = self.value_head(combined)
            return jnp.squeeze(value, axis=-1)

    value_module = VisionTaskObsValueNetwork(
        value_layers=list(hidden_layer_sizes),
        latent_size=vision_latent_size,
        channels=list(vision_channels),
    )

    def apply(
        processor_params: DictRunningStatisticsState,
        value_params,
        obs: Mapping[str, jnp.ndarray],
    ):
        """Apply vision + task_obs value network."""
        normalized_obs = normalize_dict_obs(obs, processor_params)
        vision = normalized_obs["vision"]
        task_obs = normalized_obs["imitation_target"]
        return value_module.apply(value_params, vision, task_obs)

    dummy_vision = jnp.zeros((1,) + vision_shape)
    dummy_task_obs = jnp.zeros((1, task_obs_size))

    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_vision, dummy_task_obs),
        apply=apply,
    )


def make_vision_task_obs_highlvl_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    vision_shape: tuple[int, int, int] = (32, 32, 1),
    vision_latent_size: int = 16,
    vision_feature_size: int = 8,
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_channels: Sequence[int] = (2, 4, 8, 16),
    fusion_hidden_layer_sizes: Sequence[int] = (256,),
    activation: networks.ActivationFn = nn.silu,
) -> PPOImitationNetworks:
    """Create vision + task_obs PPO networks for high-level transfer training.

    Unlike ``make_vision_highlvl_ppo_networks`` which uses vision only, this
    factory provides the value network with both visual input and task-relevant
    body signals (imitation_target). The policy uses a VisionTaskObsNetwork
    that fuses CNN vision features with task observations through a fusion MLP.

    Policy: CNN(pixels) + task_obs -> fusion MLP -> z -> MLP -> latent intentions
    Value:  CNN(pixels) + task_obs -> MLP -> scalar value

    Args:
        obs_sizes: Dict mapping observation keys to their sizes.
        action_size: Action dimension (latent intention size).
        vision_shape: Shape of the vision input (H, W, C).
        vision_latent_size: Dimension of the fusion MLP output / latent vector.
        vision_feature_size: Output dimension of the vision encoder CNN.
        decoder_hidden_layer_sizes: MLP layer sizes for policy decoder.
        value_hidden_layer_sizes: MLP layer sizes for value head.
        vision_channels: Channel sizes for CNN conv layers.
        fusion_hidden_layer_sizes: Hidden layer sizes for the fusion MLP.

    Returns:
        PPOImitationNetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = intention_network.make_vision_task_obs_policy(
        parametric_action_distribution.param_size,
        latent_size=vision_latent_size,
        obs_sizes=obs_sizes,
        vision_shape=vision_shape,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        vision_feature_size=vision_feature_size,
        vision_channels=vision_channels,
        fusion_hidden_layer_sizes=fusion_hidden_layer_sizes,
        activation=activation,
    )

    task_obs_size = obs_sizes.get("imitation_target", 0)

    value_network = make_vision_task_obs_value_network(
        vision_shape=vision_shape,
        task_obs_size=task_obs_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
        vision_latent_size=vision_latent_size,
        vision_channels=vision_channels,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_shared_vision_task_obs_highlvl_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    vision_shape: tuple[int, int, int] = (32, 32, 1),
    vision_latent_size: int = 16,
    vision_feature_size: int = 32,
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_channels: Sequence[int] = (4, 8, 16, 32),
    fusion_hidden_layer_sizes: Sequence[int] = (256,),
    activation: networks.ActivationFn = nn.silu,
) -> tuple["PPOImitationNetworks", "SharedVisionPolicyValueModule"]:
    """Create shared-CNN vision + task_obs PPO networks.

    Unlike ``make_vision_task_obs_highlvl_ppo_networks`` which creates
    separate CNNs for policy and value, this shares a single CNN.  Both
    ``policy_loss`` and ``v_loss`` gradients flow through the shared CNN,
    providing a stronger learning signal for vision features.

    All parameters (CNN + policy head + value head) are stored in
    ``params.policy``.  ``params.value`` is empty.

    Returns:
        Tuple of (ppo_networks, shared_module).  The shared_module is
        needed by ``compute_shared_vision_ppo_loss``.
    """
    from track_mjx.agent.ff_ppo.shared_vision_network import (
        make_shared_vision_policy,
        make_shared_vision_value_stub,
    )

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network, shared_module = make_shared_vision_policy(
        parametric_action_distribution.param_size,
        latent_size=vision_latent_size,
        obs_sizes=obs_sizes,
        vision_shape=vision_shape,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        vision_feature_size=vision_feature_size,
        vision_channels=vision_channels,
        fusion_hidden_layer_sizes=fusion_hidden_layer_sizes,
        activation=activation,
    )

    value_network = make_shared_vision_value_stub()

    ppo_networks = PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

    return ppo_networks, shared_module


def make_binocular_shared_vision_task_obs_highlvl_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    vision_shape: tuple[int, int, int] = (32, 32, 2),
    vision_latent_size: int = 16,
    vision_feature_size: int = 32,
    decoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    vision_channels: Sequence[int] = (4, 8, 16, 32),
    fusion_hidden_layer_sizes: Sequence[int] = (256,),
    mono_channels: int = 1,
    shared_weights: bool = True,
    activation: networks.ActivationFn = nn.silu,
) -> tuple["PPOImitationNetworks", "BinocularSharedVisionPolicyValueModule"]:
    """Create shared binocular-CNN vision + task_obs PPO networks.

    Mirrors ``make_shared_vision_task_obs_highlvl_ppo_networks`` but uses a
    ``BinocularVisionEncoder`` instead of a monocular ``VisionEncoder``.
    Both ``policy_loss`` and ``v_loss`` gradients flow through the shared
    binocular CNN.

    The binocular encoder supports two modes:
    - ``shared_weights=True`` (Siamese): A single CNN processes both eyes.
    - ``shared_weights=False`` (Independent): Two separate CNNs per eye.

    All parameters (binocular CNN + policy head + value head) are stored in
    ``params.policy``.  ``params.value`` is empty.

    Returns:
        Tuple of (ppo_networks, shared_module).  The shared_module is
        needed by ``compute_shared_vision_ppo_loss``.
    """
    from track_mjx.agent.ff_ppo.binocular_shared_vision_network import (
        make_binocular_shared_vision_policy,
        make_binocular_shared_vision_value_stub,
    )

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network, shared_module = make_binocular_shared_vision_policy(
        parametric_action_distribution.param_size,
        latent_size=vision_latent_size,
        obs_sizes=obs_sizes,
        vision_shape=vision_shape,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        vision_feature_size=vision_feature_size,
        vision_channels=vision_channels,
        fusion_hidden_layer_sizes=fusion_hidden_layer_sizes,
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        activation=activation,
    )

    value_network = make_binocular_shared_vision_value_stub()

    ppo_networks = PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

    return ppo_networks, shared_module


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
    network_config = cfg["network_config"]

    # Handle both new (dict-based) and legacy (flat) config formats
    if "obs_sizes" in network_config:
        # New dict-based format
        obs_sizes = network_config["obs_sizes"]
        reference_obs_size = obs_sizes["imitation_target"]
        observation_size = obs_sizes["imitation_target"] + obs_sizes["proprioception"]
    else:
        # Legacy flat format
        observation_size = network_config["observation_size"]
        reference_obs_size = network_config["reference_obs_size"]

    action_size = network_config["action_size"]
    intention_latent_size = network_config["intention_size"]
    decoder_hidden_layer_sizes = network_config["decoder_layer_sizes"]

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
    normalizer_state = intention_policy_params[0]

    if "obs_sizes" in network_config:
        # New dict-based format: normalizer has separate imitation_target/proprioception
        decoder_normalizer_params = normalizer_state.proprioception
    else:
        # Legacy flat format: slice the arrays to get proprioception portion
        decoder_normalizer_params_dict = jax.tree.map(
            lambda x: (
                x[reference_obs_size:]
                if isinstance(x, jnp.ndarray) and x.ndim >= 1
                else x
            ),
            normalizer_state.__dict__,
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
