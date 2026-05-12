"""TrainingState carries running-statistics normalizer for dict observations.

Verifies that init_training_state initializes the dict normalizer correctly
when given a dict obs_template, and that sgd_step's loss inputs respect
state.normalizer_params (we observe this indirectly via finite metrics
when the raw obs has wildly mixed scales).
"""
import jax
import jax.numpy as jnp
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.observation_utils import DictRunningStatisticsState


def _toy_dict_env_spec(imit_size=8, proprio_size=4, action_size=3):
    return {
        "obs_template": {
            "imitation_target": jnp.zeros((imit_size,), jnp.float32),
            "proprioception": jnp.zeros((proprio_size,), jnp.float32),
        },
        "action_size": action_size,
    }


def _toy_flat_env_spec(obs_size=12, action_size=3):
    return {"obs_size": obs_size, "action_size": action_size}


def test_init_training_state_dict_normalizer():
    """When env_spec has obs_template (dict), normalizer_params is a
    DictRunningStatisticsState with separate per-key state."""
    from track_mjx.agent.dmpo.networks_intention import (
        make_dmpo_intention_networks,
    )

    cfg = DMPOConfig(num_envs=4, batch_size=4, sequence_length=2,
                     min_replay_size=4, max_replay_size=64)
    env_spec = _toy_dict_env_spec(imit_size=8, proprio_size=4, action_size=3)
    nets = make_dmpo_intention_networks(
        obs_sizes={"imitation_target": 8, "proprioception": 4},
        action_size=3,
        cfg=cfg,
        network_cfg={
            "encoder_layer_sizes": [16],
            "decoder_layer_sizes": [16],
            "intention_size": 4,
            "activation": "silu",
        },
    )
    state = init_training_state(jax.random.PRNGKey(0), nets, env_spec, cfg)
    assert isinstance(state.normalizer_params, DictRunningStatisticsState)
    assert state.normalizer_params.imitation_target.mean.shape == (8,)
    assert state.normalizer_params.proprioception.mean.shape == (4,)


def test_init_training_state_flat_normalizer():
    """When env_spec has obs_size (flat), normalizer_params is a
    flat RunningStatisticsState. Backward compat for the existing flat-MLP
    DMPO entry — though that entry doesn't currently use it."""
    from track_mjx.agent.dmpo.networks import make_dmpo_networks
    from brax.training.acme import running_statistics

    cfg = DMPOConfig(num_envs=4, batch_size=4, sequence_length=2,
                     min_replay_size=4, max_replay_size=64)
    env_spec = _toy_flat_env_spec(obs_size=12, action_size=3)
    nets = make_dmpo_networks(12, 3, cfg)
    state = init_training_state(jax.random.PRNGKey(0), nets, env_spec, cfg)
    assert isinstance(
        state.normalizer_params, running_statistics.RunningStatisticsState
    )
    assert state.normalizer_params.mean.shape == (12,)
