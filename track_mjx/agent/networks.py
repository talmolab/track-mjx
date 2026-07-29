"""Shared network factories for dictionary-based observations."""

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from brax.training import networks
from brax.training.acme import running_statistics
from flax import linen as nn


def make_dict_value_network(
    obs_sizes: Mapping[str, int],
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_obs_key: str = "state",
    activation: networks.ActivationFn = nn.relu,
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts nested dictionary observations.

    Wraps Brax's ``make_value_network`` with a preprocessor that normalizes
    the inner dict and concatenates its leaves into a flat vector.

    Args:
        obs_sizes: Dict with 'task_obs' and 'proprioception' sizes.
        hidden_layer_sizes: MLP layer sizes for value network.
        value_obs_key: Top-level observation key for value network.
        activation: Activation function for hidden layers. Defaults to ReLU
            to match Brax's previous implicit default.

    Returns:
        FeedForwardNetwork that accepts nested dict observations.
    """
    total_obs_size = obs_sizes["task_obs"] + obs_sizes["proprioception"]

    def preprocess(observation, preprocessor_params):
        normalized = running_statistics.normalize(observation, preprocessor_params)
        return jnp.concatenate(
            [normalized["task_obs"], normalized["proprioception"]], axis=-1
        )

    return networks.make_value_network(
        obs_size=total_obs_size,
        preprocess_observations_fn=preprocess,
        hidden_layer_sizes=hidden_layer_sizes,
        obs_key=value_obs_key,
        activation=activation,
    )
