"""Shared network factories for dictionary-based observations."""

import dataclasses
from collections.abc import Mapping, Sequence

import jax.numpy as jnp
from brax.training import networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks


def make_dict_value_network(
    obs_sizes: Mapping[str, int],
    hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    """Create a value network that accepts nested dictionary observations.

    Wraps Brax's ``make_value_network`` with a preprocessor that normalizes
    the inner dict and concatenates its leaves into a flat vector.

    Args:
        obs_sizes: Dict with 'task_obs' and 'proprioception' sizes.
        hidden_layer_sizes: MLP layer sizes for value network.
        value_obs_key: Top-level observation key for value network.

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
    )


def make_ppo_networks(
    *args,
    min_std: float | None = None,
    max_std: float | None = None,
    **kwargs,
) -> ppo_networks.PPONetworks:
    """Brax ``make_ppo_networks`` with an optional clamp on the policy std.

    For ``distribution_type="normal"`` the policy head returns ``(mean, std)`` and
    nothing bounds ``std``: with a state-independent log-std the entropy bonus has
    a constant-sign gradient once the PPO surrogate is clipped out, so std can run
    away (seen as a geometric blow-up of ``training/policy_dist_mean_std``). This
    clips std to ``[min_std, max_std]`` at the one seam every consumer goes through
    (rollout sampling, the loss and its KL metric, deterministic eval), without
    touching the parameter pytree, so existing checkpoints still restore.

    Args:
        *args, **kwargs: Forwarded to ``brax...ppo.networks.make_ppo_networks``.
        min_std: Lower bound on the policy std, or None for no floor.
        max_std: Upper bound on the policy std, or None for no ceiling.

    Returns:
        PPONetworks, with ``policy_network.apply`` wrapped when a bound is set.

    Raises:
        ValueError: if a bound is given for a non-``normal`` distribution.
    """
    nets = ppo_networks.make_ppo_networks(*args, **kwargs)
    if min_std is None and max_std is None:
        return nets
    if kwargs.get("distribution_type", "tanh_normal") != "normal":
        raise ValueError(
            "min_std/max_std only apply to distribution_type='normal'; the "
            "tanh_normal head uses its own softplus std."
        )
    if min_std is not None and max_std is not None and min_std > max_std:
        raise ValueError(f"min_std ({min_std}) > max_std ({max_std}).")
    inner_apply = nets.policy_network.apply

    def apply(processor_params, policy_params, obs):
        mean, std = inner_apply(processor_params, policy_params, obs)
        return mean, jnp.clip(std, min_std, max_std)  # clip ignores None bounds

    policy_network = dataclasses.replace(nets.policy_network, apply=apply)
    return nets.replace(policy_network=policy_network)
