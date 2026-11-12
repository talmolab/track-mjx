"""Tests for the flashbax-backed trajectory replay wrapper.

Confirmed against flashbax (installed 0.1.3): the trajectory buffer's `add`
expects shape ``[add_batch_size, T, ...]`` per leaf, and `sample.experience`
comes back as ``[sample_batch_size, sample_sequence_length, ...]``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from flashbax.buffers.trajectory_buffer import add as fbx_add

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.replay import make_replay, replay_add_sequence_length


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
    seq = jax.tree.map(lambda x: jnp.broadcast_to(x, (2, T, *x.shape)), transition)
    state = rb.add(state, seq)

    assert rb.can_sample(state)

    rng = jax.random.key(0)
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


# ---------------------------------------------------------------------------
# In-place add (make_replay(add_sequence_length=T)) and its config switch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_size,seq_len", [(12, 4), (24, 3)])
def test_inplace_add_matches_flashbax_add(transition, max_size, seq_len):
    """add_sequence_length: the in-place add gives the same buffer state as
    flashbax's scatter add over several wrap-arounds."""
    kw = dict(
        max_size=max_size,
        min_size=seq_len + 1,
        sequence_length=seq_len,
        sample_batch_size=2,
        add_batch_size=3,
        period=1,
    )
    rb_ref = make_replay(**kw)
    rb_new = make_replay(**kw, add_sequence_length=seq_len)
    assert rb_ref.add.func is fbx_add
    assert getattr(rb_new.add, "func", None) is not fbx_add
    s_ref, s_new = rb_ref.init(transition), rb_new.init(transition)
    add_ref, add_new = jax.jit(rb_ref.add), jax.jit(rb_new.add)
    key = jax.random.key(0)
    for i in range(3 * max_size // seq_len + 1):
        key, k = jax.random.split(key)
        batch = jax.tree.map(
            lambda x: jax.random.normal(k, (3, seq_len, *x.shape), x.dtype),  # noqa: B023 (applied eagerly, this iteration)
            transition,
        )
        s_ref, s_new = add_ref(s_ref, batch), add_new(s_new, batch)
        assert jax.tree.structure(s_ref) == jax.tree.structure(s_new)
        for a, b in zip(jax.tree.leaves(s_ref), jax.tree.leaves(s_new)):
            assert a.dtype == b.dtype
            assert bool(jnp.array_equal(a, b)), f"diverged at add {i}"
    # Several wraps happened, and sampling reads the same trajectories.
    assert bool(s_new.is_full)
    k = jax.random.key(1)
    for a, b in zip(
        jax.tree.leaves(rb_ref.sample(s_ref, k)),
        jax.tree.leaves(rb_new.sample(s_new, k)),
    ):
        assert bool(jnp.array_equal(a, b))


def test_inplace_add_falls_back_when_not_a_multiple(transition):
    """max_size % T != 0: a slice write could wrap, so make_replay keeps
    flashbax's add (same state as tb.add over several wrapping adds)."""
    import flashbax.buffers.trajectory_buffer as tb

    kw = dict(
        max_size=10,
        min_size=4,
        sequence_length=3,
        sample_batch_size=2,
        add_batch_size=2,
        period=1,
    )
    rb = make_replay(**kw, add_sequence_length=4)
    assert rb.add.func is fbx_add
    ref, out = rb.init(transition), rb.init(transition)
    for i in range(4):  # writes at 0, 4, 8 (wraps to 2), 2
        batch = jax.tree.map(
            lambda x: jnp.full((2, 4, *x.shape), float(i + 1), x.dtype),  # noqa: B023 (applied eagerly, this iteration)
            transition,
        )
        ref, out = tb.add(ref, batch), rb.add(out, batch)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(out)):
        assert bool(jnp.array_equal(a, b))


def test_inplace_add_rejects_other_lengths(transition):
    rb = make_replay(
        max_size=12,
        min_size=5,
        sequence_length=4,
        sample_batch_size=2,
        add_batch_size=2,
        period=1,
        add_sequence_length=4,
    )
    s = rb.init(transition)
    batch = jax.tree.map(lambda x: jnp.ones((2, 3, *x.shape), x.dtype), transition)
    with pytest.raises(ValueError, match="add_sequence_length"):
        rb.add(s, batch)
    # A leaf whose dtype or trailing shape does not fit is refused too.
    bad = jax.tree.map(lambda x: jnp.ones((2, 4, *x.shape), x.dtype), transition)
    bad["reward"] = bad["reward"].astype(jnp.float16)
    with pytest.raises(ValueError, match="does not fit"):
        rb.add(s, bad)
    with pytest.raises(ValueError, match=">= 1"):
        make_replay(
            max_size=12,
            min_size=5,
            sequence_length=4,
            sample_batch_size=2,
            add_batch_size=2,
            add_sequence_length=0,
        )


def test_replay_inplace_add_defaults_off():
    cfg = DMPOConfig()
    assert cfg.replay_inplace_add is False
    assert replay_add_sequence_length(cfg, max_size=7) is None


def test_replay_add_sequence_length_resolves_the_unroll():
    cfg = DMPOConfig(unroll_length=4, replay_inplace_add=True)
    assert replay_add_sequence_length(cfg, max_size=12) == 4
    with pytest.raises(ValueError, match="multiple of unroll_length 4"):
        replay_add_sequence_length(cfg, max_size=10)
    with pytest.raises(ValueError, match="replay_inplace_add"):
        replay_add_sequence_length(
            dataclasses.replace(cfg, replay_inplace_add="true"), max_size=12
        )


def test_replay_add_sequence_length_selects_the_inplace_add(transition):
    """Built through the config helper, the buffer uses the in-place add
    exactly when the switch is on."""
    kw = dict(
        max_size=12,
        min_size=5,
        sequence_length=4,
        sample_batch_size=2,
        add_batch_size=2,
        period=1,
    )
    off = DMPOConfig(unroll_length=4)
    on = dataclasses.replace(off, replay_inplace_add=True)
    rb_off = make_replay(**kw, add_sequence_length=replay_add_sequence_length(off, 12))
    rb_on = make_replay(**kw, add_sequence_length=replay_add_sequence_length(on, 12))
    assert rb_off.add.func is fbx_add
    assert getattr(rb_on.add, "func", None) is not fbx_add
