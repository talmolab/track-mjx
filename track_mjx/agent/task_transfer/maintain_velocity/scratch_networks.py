"""Network architectures for scratch mode training.

Scratch mode trains both a policy and decoder from random initialization.
The policy takes observations and outputs latent intentions, which are then
combined with proprioception and passed through the decoder to produce actions.

Unlike decoder_only mode where the decoder is frozen, here both networks
are trainable end-to-end.

Handles dict observations directly (no wrapper needed).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks
from brax.training.acme import running_statistics
from flax import linen as nn

from track_mjx.agent.ff_ppo.intention_network import Decoder
from track_mjx.agent.task_transfer.maintain_velocity.observation_utils import (
    flatten_obs_dict,
    concat_flat_dict_obs,
)


class ScratchPolicy(nn.Module):
    """MLP policy that outputs latent intentions from observations.

    Attributes:
        layer_sizes: Hidden layer sizes for the MLP.
        latent_size: Dimension of the latent intention output.
        activation: Activation function.
    """

    layer_sizes: Sequence[int]
    latent_size: int
    activation: networks.ActivationFn = nn.silu

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the policy.

        Args:
            obs: Flattened observation array.

        Returns:
            Latent intention vector (bounded by tanh).
        """
        x = obs
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(hidden_size, name=f"hidden_{i}")(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)

        # Output latent with tanh to bound it
        latent = nn.Dense(self.latent_size, name="latent")(x)
        latent = nn.tanh(latent)

        return latent


class ScratchPolicyDecoder(nn.Module):
    """Combined policy + decoder network for scratch mode training.

    The policy processes full observations to produce latent intentions,
    which are then combined with proprioception and passed through the
    decoder to produce action distribution parameters.

    Attributes:
        policy_layers: Hidden layer sizes for the policy MLP.
        decoder_layers: Layer sizes for the decoder (including output).
        latent_size: Dimension of the latent intention space.
        proprio_size: Size of proprioceptive observations (at end of flat obs).
    """

    policy_layers: Sequence[int]
    decoder_layers: Sequence[int]
    latent_size: int
    proprio_size: int

    def setup(self):
        """Initialize policy and decoder submodules."""
        self.policy = ScratchPolicy(
            layer_sizes=self.policy_layers,
            latent_size=self.latent_size,
        )
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(self, flat_obs: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Forward pass through policy and decoder.

        Args:
            flat_obs: Flattened normalized observation array where
                proprioception is the last proprio_size elements.

        Returns:
            Tuple of (action_params, extras_dict) where action_params are
            the parameters for the action distribution and extras contains
            the latent intention.
        """
        # Extract proprioception from end of flat observation
        proprio = flat_obs[..., -self.proprio_size:]

        # Policy: full observation -> latent
        latent = self.policy(flat_obs)

        # Decoder: [latent, proprio] -> action params
        decoder_input = jnp.concatenate([latent, proprio], axis=-1)
        action_params, _ = self.decoder(decoder_input)

        return action_params, {"latent": latent}


def make_scratch_policy(
    action_param_size: int,
    task_obs_size: int,
    proprio_size: int,
    latent_size: int,
    policy_hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """Create a scratch policy network with combined policy and decoder.

    Args:
        action_param_size: Output dimension (typically 2x action_size for
            Gaussian mean and variance).
        task_obs_size: Size of task-specific observations.
        proprio_size: Size of proprioceptive observations.
        latent_size: Dimension of the latent intention space.
        policy_hidden_layer_sizes: Hidden layer sizes for policy MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.

    Returns:
        FeedForwardNetwork with init and apply methods.
    """
    total_obs_size = task_obs_size + proprio_size

    policy_module = ScratchPolicyDecoder(
        policy_layers=list(policy_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [action_param_size],
        latent_size=latent_size,
        proprio_size=proprio_size,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, Any],
    ) -> tuple[jnp.ndarray, dict]:
        """Apply policy with observation normalization.

        Args:
            processor_params: Brax's running statistics for dict observation.
            policy_params: Network parameters.
            obs: Dict observation with 'proprioception' and task keys.

        Returns:
            Tuple of (action_params, extras_dict).
        """
        # Normalize dict observation first (Brax normalizer handles dict structure)
        normalized_obs = running_statistics.normalize(obs, processor_params)

        # Then flatten the normalized dict to array (task_obs, proprio)
        flat_obs = flatten_obs_dict(normalized_obs)
        flat_obs_array = concat_flat_dict_obs(flat_obs)

        # Apply the policy module
        return policy_module.apply(policy_params, flat_obs_array)

    dummy_obs = jnp.zeros((1, total_obs_size))

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )


def make_scratch_value_network(
    task_obs_size: int,
    proprio_size: int,
    hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts dict observations.

    Args:
        task_obs_size: Size of task-specific observations.
        proprio_size: Size of proprioceptive observations.
        hidden_layer_sizes: MLP layer sizes for value network.

    Returns:
        FeedForwardNetwork that accepts dict observations.
    """
    total_obs_size = task_obs_size + proprio_size

    base_value_network = networks.make_value_network(
        total_obs_size,
        hidden_layer_sizes=hidden_layer_sizes,
    )

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        value_params,
        obs: Mapping[str, Any],
    ):
        """Apply value network with dict observation normalization."""
        # Normalize dict observation first (Brax normalizer handles dict structure)
        normalized_obs = running_statistics.normalize(obs, processor_params)

        # Then flatten the normalized dict to array
        flat_obs = flatten_obs_dict(normalized_obs)
        flat_obs_array = concat_flat_dict_obs(flat_obs)

        return base_value_network.apply((), value_params, flat_obs_array)

    return networks.FeedForwardNetwork(
        init=lambda key: base_value_network.init(key),
        apply=apply,
    )


@flax.struct.dataclass
class ScratchPPONetworks:
    """Container for scratch mode PPO network components.

    Attributes:
        policy_network: Combined policy + decoder network.
        value_network: Feedforward value function network.
        parametric_action_distribution: Action distribution (NormalTanh).
    """

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_scratch_ppo_networks(
    task_obs_size: int,
    proprio_size: int,
    action_size: int,
    latent_size: int,
    policy_hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    value_hidden_layer_sizes: Sequence[int] = (1024, 512, 256),
) -> ScratchPPONetworks:
    """Create PPO networks for scratch mode training.

    Creates a combined policy+decoder network where both components are
    trainable from random initialization.

    Args:
        task_obs_size: Size of task-specific observations.
        proprio_size: Size of proprioceptive observations.
        action_size: Action dimension.
        latent_size: Dimension of the latent intention space.
        policy_hidden_layer_sizes: Hidden layer sizes for policy MLP.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder MLP.
        value_hidden_layer_sizes: Hidden layer sizes for value network.

    Returns:
        ScratchPPONetworks containing policy, value, and action distribution.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_network = make_scratch_policy(
        action_param_size=parametric_action_distribution.param_size,
        task_obs_size=task_obs_size,
        proprio_size=proprio_size,
        latent_size=latent_size,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )

    value_network = make_scratch_value_network(
        task_obs_size=task_obs_size,
        proprio_size=proprio_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return ScratchPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_scratch_inference_fn(
    ppo_networks: ScratchPPONetworks,
) -> Callable:
    """Create a policy factory function for inference.

    Args:
        ppo_networks: Scratch PPO network components.

    Returns:
        A make_policy function with signature:
            make_policy(params, deterministic) -> policy_fn
    """

    def make_policy(
        params: tuple,
        deterministic: bool = False,
    ) -> Callable:
        """Create a policy function with fixed parameters.

        Args:
            params: Tuple of (normalizer_params, policy_params).
            deterministic: If True, return mode of action distribution.

        Returns:
            Policy function: (obs, key) -> (action, extras_dict).
        """
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(observations: Mapping[str, Any], key_sample: jax.Array):
            logits, extras = policy_network.apply(*params, observations)

            if deterministic:
                action = parametric_action_distribution.mode(logits)
                return jnp.array(action), extras

            # Sample action from distribution
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            extras["log_prob"] = log_prob
            extras["raw_action"] = raw_actions
            extras["logits"] = logits

            return jnp.array(postprocessed_actions), extras

        return policy

    return make_policy
