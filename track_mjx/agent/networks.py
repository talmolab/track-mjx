"""Shared network factories for dictionary-based observations."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from brax.training import networks
from brax.training.acme import running_statistics
from flax import linen

from track_mjx.agent.observation_utils import normalize_dict_obs


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

    # We hand-roll the value network rather than wrapping
    # brax.training.networks.make_value_network: when obs is a Mapping (it
    # always is for the imitation env), brax's apply() routes through
    # running_statistics.normalizer_select(processor_params, obs_key) which
    # accesses .count on the normalizer state. Our DictRunningStatisticsState
    # is a dataclass that holds two RunningStatisticsState sub-states
    # (imitation_target, proprioception) — it has no .count attribute, so
    # the brax path raises. The hand-rolled version sidesteps that by always
    # taking the dict-aware preprocess() path.

    value_module = networks.MLP(
        layer_sizes=list(hidden_layer_sizes) + [1],
        activation=linen.relu,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def preprocess(observation, preprocessor_params):
        normalized = normalize_dict_obs(observation, preprocessor_params)
        # normalize_dict_obs always emits 'imitation_target' key regardless
        # of the input naming, so look there even if obs_sizes used the
        # legacy 'task_obs' name.
        return jnp.concatenate(
            [normalized["imitation_target"], normalized["proprioception"]], axis=-1
        )

    def apply(processor_params, value_params, obs):
        flat_obs = preprocess(obs, processor_params)
        return jnp.squeeze(value_module.apply(value_params, flat_obs), axis=-1)

    dummy_obs = jnp.zeros((1, total_obs_size))
    return networks.FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs),
        apply=apply,
    )
