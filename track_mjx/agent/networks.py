"""Shared network factories for dictionary-based observations."""

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from brax.training import networks
from brax.training.acme import running_statistics


def make_dict_value_network(
    obs_sizes: Mapping[str, int],
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts nested dictionary observations.

    Wraps Brax's ``make_value_network`` with a preprocessor that normalizes
    the inner dict and concatenates its leaves into a flat vector.

    Accepts obs dicts produced by either the old ('task_obs') or the
    current ('imitation_target') naming convention so it stays compatible
    with both observation_utils.flatten_obs and any legacy wrappers.

    Args:
        obs_sizes: Dict with 'imitation_target' (or legacy 'task_obs')
            and 'proprioception' sizes.
        hidden_layer_sizes: MLP layer sizes for value network.
        value_obs_key: Top-level observation key for value network.

    Returns:
        FeedForwardNetwork that accepts nested dict observations.
    """
    if "imitation_target" in obs_sizes:
        imit_key = "imitation_target"
    elif "task_obs" in obs_sizes:
        imit_key = "task_obs"
    else:
        raise KeyError(
            "obs_sizes must contain 'imitation_target' or 'task_obs'; "
            f"got keys={list(obs_sizes)}"
        )
    total_obs_size = obs_sizes[imit_key] + obs_sizes["proprioception"]

    def preprocess(observation, preprocessor_params):
        normalized = running_statistics.normalize(observation, preprocessor_params)
        return jnp.concatenate(
            [normalized[imit_key], normalized["proprioception"]], axis=-1
        )

    return networks.make_value_network(
        obs_size=total_obs_size,
        preprocess_observations_fn=preprocess,
        hidden_layer_sizes=hidden_layer_sizes,
        obs_key=value_obs_key,
    )
