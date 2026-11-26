"""Typed-PRNG-key migration.

Main's track-mjx is fully on typed keys (``jax.random.key``); DMPO's tests
were migrated off raw ``jax.random.PRNGKey`` in lockstep with this file. Two
things must hold:

  1. A save -> restore -> resume cycle driven entirely by typed keys must be
     bit-identical to never having checkpointed at all.
  2. ``checkpoint.restore`` must accept a LEGACY checkpoint -- one whose
     ``rng`` leaf is a raw uint32 array (every DMPO checkpoint saved before
     this migration, including the real dense-reference archive
     exercised in ``test_warm_start_machinery.py``) -- into a typed-key
     template, and hand back a genuinely typed key rather than silently
     passing the raw array through.

Both states here are the SAME tiny synthetic ``TrainingState`` used across
the DMPO test suite (``tests.agent.dmpo.test_train_dmpo_fused._setup``),
saved under ``tmp_path`` -- never the real checkpoint archive.
"""

import jax
import jax.numpy as jnp
import numpy as np

from tests.agent.dmpo.test_train_dmpo_fused import _setup
from track_mjx.agent.dmpo.checkpoint import make_checkpointer, restore, save
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step


def _env_spec(env):
    return {"obs_size": env.obs_size, "action_size": env.action_size}


def _clone(pytree):
    """Independent copy of a pytree's leaves.

    ``make_fused_train_step``'s jitted step donates ``env_state``/``rb_state``
    (``donate_argnums=(1, 2)``) -- each buffer may only be passed into one
    call. Comparing an uncheckpointed continuation against a resumed one from
    the SAME post-step-1 env/replay state needs two independent buffers.
    """
    return jax.tree.map(jnp.array, pytree)


def _assert_states_equal(a, b):
    """Field-by-field equality, ``rng`` handled via ``key_data``.

    A blanket ``jax.tree_util.tree_map`` over the whole ``TrainingState``
    would hand the typed-key leaf to ``np.testing.assert_allclose``, which
    calls ``np.asarray`` internally and raises ``TypeError`` on a typed key
    (shape (), dtype ``key<fry>``). Drop ``rng`` from the tree-mapped
    comparison (``None`` is a zero-leaf pytree node, so ``_replace(rng=None)``
    removes it from traversal) and compare it separately via ``key_data``.
    """
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)),
        a._replace(rng=None),
        b._replace(rng=None),
    )
    assert jax.dtypes.issubdtype(a.rng.dtype, jax.dtypes.prng_key)
    assert jax.dtypes.issubdtype(b.rng.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(
        jax.random.key_data(a.rng), jax.random.key_data(b.rng)
    )


def test_typed_key_save_restore_resume_matches_uncheckpointed_continuation(tmp_path):
    """Save mid-training with a typed key, restore into a fresh typed
    template, and continue training -- the resumed step must exactly match
    a continuation that never touched a checkpoint at all."""
    s = _setup()
    cfg = s["cfg"]
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=1
    )

    rng0 = jax.random.key(0)
    state1, env_state1, rb_state1, _ = fused(s["state"], None, s["rb_state"], rng0)
    assert jax.dtypes.issubdtype(state1.rng.dtype, jax.dtypes.prng_key)

    # Ground truth: continue training with no checkpoint round-trip at all.
    # Clone env_state1/rb_state1 -- the jitted step donates them, and the
    # "resumed" continuation below needs its own independent copy of the
    # same post-step-1 values.
    rng1 = jax.random.key(1)
    expected_state2, _, _, _ = fused(
        state1, _clone(env_state1), _clone(rb_state1), rng1
    )

    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=1, state=state1)
    mgr.wait_until_finished()

    # A template built from a DIFFERENT typed seed, so a no-op restore would
    # be caught by the equality assertions below (values must come from disk).
    fresh_template = init_training_state(
        jax.random.key(999), s["nets"], _env_spec(s["env"]), cfg
    )
    restored_state1 = restore(mgr, fresh_template)

    _assert_states_equal(state1, restored_state1)

    # Resume: same external rng1 as the ground-truth continuation above.
    resumed_state2, _, _, _ = fused(restored_state1, env_state1, rb_state1, rng1)
    _assert_states_equal(expected_state2, resumed_state2)

    # The typed key must still be usable after the round-trip.
    k1, k2 = jax.random.split(resumed_state2.rng)
    assert jax.dtypes.issubdtype(k1.dtype, jax.dtypes.prng_key)
    assert jax.dtypes.issubdtype(k2.dtype, jax.dtypes.prng_key)


def test_restore_legacy_raw_key_checkpoint_into_typed_template(tmp_path):
    """A checkpoint saved before the typed-key migration stores ``rng`` as a
    raw uint32 array. ``checkpoint.restore`` must accept it into a
    typed-key template and hand back a genuinely typed key (not the raw
    array passed through), with equal ``key_data`` and a working ``split``.
    """
    s = _setup()
    cfg = s["cfg"]

    # Synthetic legacy raw key data: shape (2,) uint32, exactly what
    # jax.random.PRNGKey used to produce and what every DMPO checkpoint saved
    # before this migration has on disk. Built as a plain array (not via
    # jax.random.PRNGKey) so this file has zero PRNGKey call sites while
    # still exercising the raw-checkpoint restore path.
    raw_rng = jnp.array([0, 123], dtype=jnp.uint32)
    saved_state = s["state"]._replace(rng=raw_rng)

    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=0, state=saved_state)
    mgr.wait_until_finished()

    typed_template = init_training_state(
        jax.random.key(7), s["nets"], _env_spec(s["env"]), cfg
    )
    restored = restore(mgr, typed_template)

    assert jax.dtypes.issubdtype(restored.rng.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(jax.random.key_data(restored.rng), raw_rng)

    # A working split() afterwards -- the whole point of coercing the type.
    k1, k2 = jax.random.split(restored.rng)
    assert jax.dtypes.issubdtype(k1.dtype, jax.dtypes.prng_key)
    assert jax.dtypes.issubdtype(k2.dtype, jax.dtypes.prng_key)


def test_restore_raw_key_checkpoint_into_raw_template_is_unchanged(tmp_path):
    """Raw-into-raw must keep working exactly as before this migration."""
    s = _setup()

    raw_rng = jnp.array([7, 8], dtype=jnp.uint32)  # synthetic legacy key data
    saved_state = s["state"]._replace(rng=raw_rng)

    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=0, state=saved_state)
    mgr.wait_until_finished()

    raw_template = s["state"]._replace(rng=jnp.zeros((2,), dtype=jnp.uint32))
    restored = restore(mgr, raw_template)

    assert not jax.dtypes.issubdtype(restored.rng.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(restored.rng, raw_rng)
