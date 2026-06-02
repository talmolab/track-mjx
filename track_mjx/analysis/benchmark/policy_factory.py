"""Build a correctly-shaped, randomly-initialized intention policy + inference fn.

No checkpoint required: timing depends on architecture (layer sizes), not weights.
Reuses the exact training network factory so shapes match production.
"""

import functools
from typing import Any, Callable

import jax
from brax.training.acme import running_statistics

from track_mjx.agent.ff_ppo import ppo_networks as ff_networks
from track_mjx.agent.observation_utils import get_obs_sizes, get_obs_shape


def build_inference_fn(
    cfg: Any,
    env: Any,
    state: Any,
    *,
    seed: int = 0,
    deterministic: bool = True,
) -> Callable:
    """Return ``inference_fn(obs, rng) -> (action, extras)`` for the config's policy.

    Args:
        cfg: merged config (from ``prepare_config``); reads ``cfg.network_config``.
        env: UNWRAPPED task env (exposes ``.action_size``).
        state: a batched env State (used only for obs sizes/shapes).
        seed: PRNG seed for random weight init.
        deterministic: if True, policy returns the distribution mode.
    """
    nc = cfg.network_config
    network_factory = functools.partial(
        ff_networks.make_intention_ppo_networks,
        intention_latent_size=int(nc.intention_size),
        encoder_hidden_layer_sizes=tuple(nc.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(nc.decoder_layer_sizes),
        proprioception_noise_std=float(nc.get("proprioception_noise_std", 0.0)),
        value_hidden_layer_sizes=tuple(nc.critic_layer_sizes),
    )
    obs_sizes = get_obs_sizes(state.obs)
    ppo_network = network_factory(obs_sizes, env.action_size)

    make_policy = ff_networks.make_inference_fn(ppo_network)
    policy_params = ppo_network.policy_network.init(jax.random.PRNGKey(seed))
    normalizer_params = running_statistics.init_state(get_obs_shape(state.obs))
    return make_policy((normalizer_params, policy_params), deterministic=deterministic)
