"""Round-trip tests for ``track_mjx.agent.dmpo.checkpoint``."""

import tempfile

import jax
import jax.numpy as jnp

from track_mjx.agent.dmpo.checkpoint import make_checkpointer, restore, save
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks import make_dmpo_networks


def test_checkpoint_round_trip(rng, env_spec):
    """Save then restore -> identical leaves on a couple of pytrees."""
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)

    with tempfile.TemporaryDirectory() as tmp:
        mgr = make_checkpointer(tmp)
        save(mgr, step=0, state=state)
        mgr.wait_until_finished()

        restored = restore(mgr, state_template=state)
        assert restored is not None

        # Compare a few representative pytrees leaf-by-leaf.
        eq_dual = jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            state.dual_params,
            restored.dual_params,
        )
        assert all(jax.tree_util.tree_leaves(eq_dual))

        eq_pol = jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            state.policy_params,
            restored.policy_params,
        )
        assert all(jax.tree_util.tree_leaves(eq_pol))

        # Scalar bookkeeping survives too.
        assert int(restored.steps) == int(state.steps)


def test_restore_returns_none_when_empty(rng, env_spec):
    """An empty checkpoint dir restores to ``None`` (signals "fresh run")."""
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        mgr = make_checkpointer(tmp)
        result = restore(mgr, state_template=state)
        assert result is None
