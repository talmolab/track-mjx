"""Tests for DMPO vision-aware networks (Path 3 of the binocular integration)."""
import jax
import jax.numpy as jnp

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_vision import make_dmpo_vision_networks


VISION_SHAPE = (32, 32, 2)  # binocular grayscale
TASK_OBS_SIZE = 100
ACTION_SIZE = 16


def test_vision_networks_init_and_forward(rng):
    cfg = DMPOConfig()
    nets = make_dmpo_vision_networks(
        task_obs_size=TASK_OBS_SIZE,
        action_size=ACTION_SIZE,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
    )

    obs = {
        "vision": jnp.zeros(VISION_SHAPE),
        "imitation_target": jnp.zeros((TASK_OBS_SIZE,)),
    }
    act = jnp.zeros((ACTION_SIZE,))

    pol_params = nets.policy.init(rng, obs)
    crit_params = nets.critic.init(rng, obs, act)

    dist = nets.policy.apply(pol_params, obs)
    q_dist = nets.critic.apply(crit_params, obs, act)
    assert dist.loc.shape == (ACTION_SIZE,)
    assert q_dist.logits_parameter().shape == (cfg.num_atoms,)


def test_vision_networks_batched(rng):
    cfg = DMPOConfig()
    nets = make_dmpo_vision_networks(
        task_obs_size=TASK_OBS_SIZE,
        action_size=ACTION_SIZE,
        vision_shape=VISION_SHAPE,
        cfg=cfg,
    )
    obs_unbatched = {
        "vision": jnp.zeros(VISION_SHAPE),
        "imitation_target": jnp.zeros((TASK_OBS_SIZE,)),
    }
    obs_batched = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (4,) + x.shape), obs_unbatched
    )
    act = jnp.zeros((4, ACTION_SIZE))

    pol_params = nets.policy.init(rng, obs_unbatched)
    crit_params = nets.critic.init(rng, obs_unbatched, act[0])

    dist = nets.policy.apply(pol_params, obs_batched)
    q_dist = nets.critic.apply(crit_params, obs_batched, act)
    assert dist.loc.shape == (4, ACTION_SIZE)
    assert q_dist.logits_parameter().shape == (4, cfg.num_atoms)
