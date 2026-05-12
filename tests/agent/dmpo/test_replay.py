"""Tests for the flashbax-backed trajectory replay wrapper.

Confirmed against flashbax (installed 0.1.3): the trajectory buffer's `add`
expects shape ``[add_batch_size, T, ...]`` per leaf, and `sample.experience`
comes back as ``[sample_batch_size, sample_sequence_length, ...]``.
"""
import jax
import jax.numpy as jnp
import pytest

from track_mjx.agent.dmpo.replay import make_replay


@pytest.fixture
def transition():
    """Acme-style single-transition template (unbatched, no time axis)."""
    return {
        "observation": jnp.zeros((53,), dtype=jnp.float32),
        "action": jnp.zeros((37,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": jnp.zeros((53,), dtype=jnp.float32),
    }


def test_replay_min_size_blocks_sampling(transition):
    """Buffer must refuse to sample until min_length_time_axis is reached."""
    rb = make_replay(
        max_size=128,
        min_size=64,
        sequence_length=5,
        sample_batch_size=4,
        add_batch_size=2,
        period=1,
    )
    state = rb.init(transition)
    assert not rb.can_sample(state)


def test_replay_add_and_sample(transition):
    """Pushing past min_size enables sampling and returns [B, T, ...] tensors."""
    rb = make_replay(
        max_size=128,
        min_size=8,
        sequence_length=5,
        sample_batch_size=4,
        add_batch_size=2,
        period=1,
    )
    state = rb.init(transition)

    # flashbax expects per-leaf shape [add_batch_size, T, ...].
    T = 10
    seq = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (2, T) + x.shape), transition
    )
    state = rb.add(state, seq)

    assert rb.can_sample(state)

    rng = jax.random.PRNGKey(0)
    sample = rb.sample(state, rng)

    # Shapes: [sample_batch_size, sequence_length, ...].
    assert sample.experience["observation"].shape == (4, 5, 53)
    assert sample.experience["action"].shape == (4, 5, 37)
    assert sample.experience["reward"].shape == (4, 5)
    assert sample.experience["discount"].shape == (4, 5)
    assert sample.experience["next_observation"].shape == (4, 5, 53)

    # Dtypes preserved end-to-end.
    assert sample.experience["observation"].dtype == jnp.float32
    assert sample.experience["reward"].dtype == jnp.float32
