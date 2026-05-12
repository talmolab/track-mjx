import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    init_dict_normalizer,
)
from track_mjx.agent.dmpo.normalizer_seeding import seed_proprio_from_imit


def _make_dmpo_normalizer(proprio_size, task_obs_size):
    obs = {
        "proprioception": jnp.zeros((1, proprio_size)),
        "imitation_target": jnp.zeros((1, task_obs_size)),
    }
    return init_dict_normalizer(obs)


def _make_imit_normalizer(proprio_size, imit_task_obs_size):
    """Build an imit-style DictRunningStatisticsState with non-trivial stats."""
    proprio_state = running_statistics.RunningStatisticsState(
        mean=jnp.arange(proprio_size, dtype=jnp.float32) * 0.1,
        std=1.0 + jnp.arange(proprio_size, dtype=jnp.float32) * 0.01,
        count=jnp.array(1_000_000, dtype=jnp.float32),
        summed_variance=jnp.ones((proprio_size,), dtype=jnp.float32),
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )
    target_state = running_statistics.RunningStatisticsState(
        mean=jnp.zeros((imit_task_obs_size,), dtype=jnp.float32),
        std=jnp.ones((imit_task_obs_size,), dtype=jnp.float32),
        count=jnp.array(1_000_000, dtype=jnp.float32),
        summed_variance=jnp.ones((imit_task_obs_size,), dtype=jnp.float32),
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )
    return DictRunningStatisticsState(
        imitation_target=target_state, proprioception=proprio_state,
    )


def test_seed_copies_proprio_fields_bit_identically():
    proprio_size = 226
    dmpo_norm = _make_dmpo_normalizer(proprio_size=proprio_size, task_obs_size=64)
    imit_norm = _make_imit_normalizer(proprio_size=proprio_size, imit_task_obs_size=512)

    seeded = seed_proprio_from_imit(dmpo_norm, imit_norm)

    np.testing.assert_array_equal(
        np.asarray(seeded.proprioception.mean), np.asarray(imit_norm.proprioception.mean)
    )
    np.testing.assert_array_equal(
        np.asarray(seeded.proprioception.std), np.asarray(imit_norm.proprioception.std)
    )
    assert int(seeded.proprioception.count) == int(imit_norm.proprioception.count)
    np.testing.assert_array_equal(
        np.asarray(seeded.proprioception.summed_variance),
        np.asarray(imit_norm.proprioception.summed_variance),
    )


def test_seed_leaves_imitation_target_untouched():
    proprio_size = 226
    task_obs_size = 64
    dmpo_norm = _make_dmpo_normalizer(proprio_size=proprio_size, task_obs_size=task_obs_size)
    # imit normalizer has imitation_target_size=512 (different from gap-jump's 64)
    imit_norm = _make_imit_normalizer(proprio_size=proprio_size, imit_task_obs_size=512)

    seeded = seed_proprio_from_imit(dmpo_norm, imit_norm)

    # imitation_target must be DMPO's (size 64), NOT imit's (size 512)
    assert seeded.imitation_target.mean.shape == (task_obs_size,)
    np.testing.assert_array_equal(
        np.asarray(seeded.imitation_target.mean), np.asarray(dmpo_norm.imitation_target.mean)
    )
    np.testing.assert_array_equal(
        np.asarray(seeded.imitation_target.std), np.asarray(dmpo_norm.imitation_target.std)
    )


def test_seed_raises_on_proprio_size_mismatch():
    """If the imit proprio dim doesn't match the DMPO env's proprio dim, fail loudly."""
    dmpo_norm = _make_dmpo_normalizer(proprio_size=226, task_obs_size=64)
    imit_norm = _make_imit_normalizer(proprio_size=999, imit_task_obs_size=512)

    with pytest.raises(ValueError, match="proprio.*shape"):
        seed_proprio_from_imit(dmpo_norm, imit_norm)
