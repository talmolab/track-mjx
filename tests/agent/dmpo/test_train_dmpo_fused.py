"""Verify the fused (rollout + add + scan-K) training step matches the
unfused path (jit_collect_rollout -> rb.add -> scan_k_sgd).

The fused step internally splits ``rng`` once into (rollout_key, sgd_key),
so to make the unfused path produce the SAME numbers we mirror that split
in the test. Same RNG -> same trajectory -> same rb_state -> same SGD
samples -> identical metrics and final state.
"""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state, make_optimizers
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.rollout import collect_rollout
from track_mjx.agent.dmpo.train_dmpo_sgd import make_scan_k_sgd
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step


# Tiny deterministic env mirroring tests/agent/dmpo/test_rollout.py.
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _MockEnvState:
    obs: jnp.ndarray
    done: jnp.ndarray

    def tree_flatten(self):
        return (self.obs, self.done), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obs, done = children
        return cls(obs=obs, done=done)


class _MockEnv:
    obs_size = 6
    action_size = 3
    pre_batched = False

    def reset(self, rng):
        return _MockEnvState(
            obs=jnp.zeros(self.obs_size, dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.float32),
        )

    def step(self, state, action):
        # Smooth, finite dynamics so the rollout doesn't NaN.
        new_obs = state.obs + jnp.pad(
            action, (0, max(0, self.obs_size - action.size))
        )[: self.obs_size]
        new_state = _MockEnvState(obs=new_obs, done=state.done)
        reward = jnp.linalg.norm(action).astype(jnp.float32)
        return new_state, reward


def _setup():
    """Build a real DMPO state + networks + replay against the toy env.

    Returns dict so test cases can pull out only what they need.
    """
    cfg = DMPOConfig(
        num_envs=4,
        batch_size=8,
        sequence_length=4,
        # unroll_length=8 (not 4) so one rollout fills past flashbax's
        # min_size=max(seq+1=5, min_replay/num_envs=2)=5 per env, making
        # rb.can_sample True from iteration 1. This matches production
        # configs where num_envs*unroll >> min_replay_size and lets the
        # fused-step gate (introduced for v4-parity) be a no-op here so
        # the numerical-equivalence test still compares apples to apples.
        unroll_length=8,
        max_replay_size=128,
        min_replay_size=8,
    )
    env = _MockEnv()
    nets = make_dmpo_networks(env.obs_size, env.action_size, cfg)
    optimizers = make_optimizers(cfg)
    rng = jax.random.PRNGKey(0)
    env_spec = {"obs_size": env.obs_size, "action_size": env.action_size}
    state = init_training_state(rng, nets, env_spec, cfg)

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    transition_template = {
        "observation": jnp.zeros((env.obs_size,), dtype=jnp.float32),
        "action": jnp.zeros((env.action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": jnp.zeros((env.obs_size,), dtype=jnp.float32),
    }
    rb_state = rb.init(transition_template)

    return dict(
        cfg=cfg, env=env, nets=nets, optimizers=optimizers,
        state=state, rb=rb, rb_state=rb_state,
    )


def test_fused_step_matches_unfused():
    """Path A (separate rollout + add + scan_k_sgd) and Path B (fused step)
    should produce numerically identical metrics for the same input RNG."""
    s = _setup()
    cfg = s["cfg"]
    K = 2
    rng = jax.random.PRNGKey(0)

    # Path A: replicate the fused step's internal rng.split layout, then
    # call the unfused primitives in the same order.
    # IMPORTANT: this split MUST mirror ``train_dmpo_step._step``'s
    # ``jax.random.split(rng, 3) -> (rng, k_roll, k_sgd)``. If you change
    # the production split, change this test in lockstep — otherwise the
    # numerical-equivalence assertion below silently regresses.
    a_state = s["state"]
    a_rng, a_k_roll, a_k_sgd = jax.random.split(rng, 3)
    traj, _, a_new_norm = collect_rollout(
        s["env"], s["nets"].policy.apply, a_state.policy_params,
        a_state.normalizer_params, a_k_roll,
        num_envs=cfg.num_envs, num_steps=cfg.unroll_length, init_state=None,
    )
    # Mirror the fused step: thread the updated normalizer into state BEFORE SGD.
    a_state = a_state._replace(normalizer_params=a_new_norm)
    a_rb_state = s["rb"].add(s["rb_state"], traj)
    scan_k_sgd = make_scan_k_sgd(s["rb"], s["nets"], s["optimizers"], cfg, K=K)
    a_state, a_metrics = scan_k_sgd(a_state, a_rb_state, a_k_sgd)

    # Path B: fused step. Reset state + rb_state to the same starting point.
    s2 = _setup()
    fused = make_fused_train_step(
        s2["env"], s2["nets"], s2["optimizers"], s2["rb"], cfg, K=K
    )
    b_state, _, b_rb_state, b_metrics = fused(
        s2["state"], None, s2["rb_state"], rng,
    )

    # Metrics dicts must have the same keys.
    assert set(a_metrics.keys()) == set(b_metrics.keys()), (
        f"metric key mismatch: {a_metrics.keys()} vs {b_metrics.keys()}"
    )
    # Each metric is a scalar; must agree to fp32 noise.
    for k in a_metrics:
        np.testing.assert_allclose(
            np.asarray(a_metrics[k]),
            np.asarray(b_metrics[k]),
            rtol=1e-4, atol=1e-5,
            err_msg=f"metric '{k}' disagrees: {a_metrics[k]} vs {b_metrics[k]}",
        )

    # The training-state policy params should also agree.
    a_leaves = jax.tree.leaves(a_state.policy_params)
    b_leaves = jax.tree.leaves(b_state.policy_params)
    for la, lb in zip(a_leaves, b_leaves):
        np.testing.assert_allclose(np.asarray(la), np.asarray(lb), rtol=1e-4, atol=1e-5)


def test_fused_step_advances_steps_by_K():
    """Sanity: after one fused step the training state has advanced by K SGD steps."""
    s = _setup()
    K = 3
    fused = make_fused_train_step(
        s["env"], s["nets"], s["optimizers"], s["rb"], s["cfg"], K=K,
    )
    new_state, env_state, new_rb_state, metrics = fused(
        s["state"], None, s["rb_state"], jax.random.PRNGKey(0),
    )
    assert int(new_state.steps) == int(s["state"].steps) + K
    # env_state must be a real pytree now (not None) so subsequent calls resume.
    assert env_state is not None
