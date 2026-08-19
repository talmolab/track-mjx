"""Warm-start transition machinery: schedules, reward remix, behavior mixing,
critic-only warmup, and the cross-run checkpoint loader.

Context. Dense->sparse reward handover for the gap task (2026-08-19): the
policy is warm-started from a dense-reward run's checkpoint, a FRESH critic
fits the warm policy's replay before the policy may move
(``critic_warmup_sgd_steps``), the dense reward component anneals to zero
(``reward_anneal_*``), and a decaying fraction of envs keeps acting with the
frozen warm-start policy (``behavior_mix_*``). Everything defaults to OFF;
the regression tests here pin that the off-path is bit-identical.

Run CPU-only (as the whole suite): JAX_PLATFORMS=cpu.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.rollout import collect_rollout
from track_mjx.agent.dmpo.schedules import (
    behavior_mix_frac,
    env_steps_estimate,
    reward_anneal_lambda,
)

from tests.agent.dmpo.test_rollout import MockEnv, _flat_normalizer
from tests.agent.dmpo.test_train_dmpo_fused import _setup


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def _t(x):
    return jnp.asarray(x, jnp.float32)


def test_env_steps_estimate_matches_the_counter_arithmetic():
    cfg = DMPOConfig(num_envs=2048, unroll_length=50)
    # K=50: one rollout = 102400 env steps = 50 SGD updates -> 2048 env
    # steps per update.
    est = env_steps_estimate(jnp.asarray(50, jnp.int32), cfg, K=50)
    assert float(est) == 2048 * 50
    est = env_steps_estimate(jnp.asarray(4900, jnp.int32), cfg, K=50)
    assert float(est) == pytest.approx(10_035_200)  # ~10M at warmup end


def test_reward_anneal_lambda_is_linear_then_zero():
    cfg = DMPOConfig(reward_anneal_env_steps=100)
    assert float(reward_anneal_lambda(_t(0), cfg)) == 1.0
    assert float(reward_anneal_lambda(_t(25), cfg)) == pytest.approx(0.75)
    assert float(reward_anneal_lambda(_t(100), cfg)) == 0.0
    assert float(reward_anneal_lambda(_t(1e9), cfg)) == 0.0


def test_reward_anneal_lambda_degenerate_zero_steps_means_sparse_only():
    cfg = DMPOConfig(reward_anneal_env_steps=0)
    assert float(reward_anneal_lambda(_t(0), cfg)) == 0.0


def test_behavior_mix_frac_hold_then_linear_then_zero():
    cfg = DMPOConfig(
        behavior_mix_init=1.0,
        behavior_mix_hold_env_steps=10,
        behavior_mix_end_env_steps=110,
    )
    assert float(behavior_mix_frac(_t(0), cfg)) == 1.0
    assert float(behavior_mix_frac(_t(10), cfg)) == 1.0
    assert float(behavior_mix_frac(_t(60), cfg)) == pytest.approx(0.5)
    assert float(behavior_mix_frac(_t(110), cfg)) == 0.0
    assert float(behavior_mix_frac(_t(1e9), cfg)) == 0.0


def test_behavior_mix_frac_no_decay_window_is_a_hard_step():
    cfg = DMPOConfig(
        behavior_mix_init=0.5,
        behavior_mix_hold_env_steps=10,
        behavior_mix_end_env_steps=0,
    )
    assert float(behavior_mix_frac(_t(9), cfg)) == 0.5
    assert float(behavior_mix_frac(_t(10), cfg)) == 0.0


# ---------------------------------------------------------------------------
# Reward remix at rollout time
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _MetricsEnvState:
    obs: jnp.ndarray
    done: jnp.ndarray
    metrics: dict

    def tree_flatten(self):
        return (self.obs, self.done, self.metrics), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obs, done, metrics = children
        return cls(obs=obs, done=done, metrics=metrics)


class _MetricsEnv(MockEnv):
    """MockEnv whose state carries per-term reward metrics like run_gap.

    Total reward is 2.0 per step, of which metrics["rewards/sparse"] = 0.5.
    """

    def reset(self, rng):
        return _MetricsEnvState(
            obs=jnp.zeros(self.obs_size),
            done=jnp.zeros(()),
            metrics={"rewards/sparse": jnp.zeros(())},
        )

    def step(self, state, action):
        new_state = _MetricsEnvState(
            obs=state.obs,
            done=state.done,
            metrics={"rewards/sparse": jnp.asarray(0.5)},
        )
        return new_state, jnp.asarray(2.0)


def _const_policy_apply(value):
    class _Dist:
        def __init__(self, v):
            self._v = v

        def sample(self, seed):
            return self._v

    def policy_apply(params, obs):
        # params is a dict {"v": scalar}; the action is params-dependent so
        # behavior mixing between two param sets is observable.
        v = params["v"] if params is not None else value
        return _Dist(jnp.full((MockEnv.action_size,), v))

    return policy_apply


@pytest.mark.parametrize("lam,expected", [(1.0, 2.0), (0.0, 0.5), (0.25, 0.875)])
def test_reward_remix_stores_sparse_plus_lambda_dense(lam, expected):
    env = _MetricsEnv()
    traj, _, _ = collect_rollout(
        env, _const_policy_apply(0.1), {"v": jnp.asarray(0.1)},
        _flat_normalizer(env.obs_size), jax.random.PRNGKey(0),
        num_envs=3, num_steps=4,
        reward_remix_key="rewards/sparse",
        reward_remix_lambda=jnp.asarray(lam, jnp.float32),
    )
    # reward = sparse + lam * (total - sparse) = 0.5 + lam * 1.5
    np.testing.assert_allclose(np.asarray(traj["reward"]), expected, rtol=1e-6)


def test_reward_without_remix_is_unchanged():
    env = _MetricsEnv()
    traj, _, _ = collect_rollout(
        env, _const_policy_apply(0.1), {"v": jnp.asarray(0.1)},
        _flat_normalizer(env.obs_size), jax.random.PRNGKey(0),
        num_envs=3, num_steps=4,
    )
    np.testing.assert_allclose(np.asarray(traj["reward"]), 2.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Behavior mixing
# ---------------------------------------------------------------------------

def _mix_actions(frac, num_envs=4):
    env = MockEnv()
    traj, _, _ = collect_rollout(
        env, _const_policy_apply(None),
        {"v": jnp.asarray(0.2)},                    # learner acts +0.2
        _flat_normalizer(env.obs_size), jax.random.PRNGKey(0),
        num_envs=num_envs, num_steps=2,
        frozen_policy_params={"v": jnp.asarray(-0.3)},   # frozen acts -0.3
        behavior_mix_frac=jnp.asarray(frac, jnp.float32),
    )
    return np.asarray(traj["action"])  # [num_envs, T, action]


def test_behavior_mix_splits_the_env_batch_front_first():
    acts = _mix_actions(0.5)
    np.testing.assert_allclose(acts[:2], -0.3, rtol=1e-6)  # frozen envs first
    np.testing.assert_allclose(acts[2:], 0.2, rtol=1e-6)


def test_behavior_mix_frac_zero_and_one():
    np.testing.assert_allclose(_mix_actions(0.0), 0.2, rtol=1e-6)
    np.testing.assert_allclose(_mix_actions(1.0), -0.3, rtol=1e-6)


def test_behavior_mix_frac_rounds_up():
    # ceil(0.26 * 4) = 2 envs frozen
    acts = _mix_actions(0.26)
    np.testing.assert_allclose(acts[:2], -0.3, rtol=1e-6)
    np.testing.assert_allclose(acts[2:], 0.2, rtol=1e-6)


def test_no_frozen_params_is_bit_identical_to_before():
    """The single-policy path must not change when the feature is unused."""
    env = MockEnv()
    kwargs = dict(
        num_envs=3, num_steps=4,
    )
    a, _, _ = collect_rollout(
        env, _const_policy_apply(None), {"v": jnp.asarray(0.2)},
        _flat_normalizer(env.obs_size), jax.random.PRNGKey(7), **kwargs,
    )
    b, _, _ = collect_rollout(
        env, _const_policy_apply(None), {"v": jnp.asarray(0.2)},
        _flat_normalizer(env.obs_size), jax.random.PRNGKey(7),
        frozen_policy_params=None, behavior_mix_frac=None,
        reward_remix_key=None, reward_remix_lambda=None, **kwargs,
    )
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


# ---------------------------------------------------------------------------
# Critic-only warmup gate (real learner via the fused step)
# ---------------------------------------------------------------------------

def _leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def test_critic_warmup_freezes_policy_and_duals_but_not_critic():
    from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

    s = _setup()
    cfg = dataclasses.replace(s["cfg"], critic_warmup_sgd_steps=4)
    K = 2  # 2 SGD updates per fused step -> warmup spans exactly 2 fused steps
    fused = make_fused_train_step(s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=K)

    state, env_state, rb_state = s["state"], None, s["rb_state"]
    rng = jax.random.PRNGKey(0)
    pol0, dual0, crit0 = state.policy_params, state.dual_params, state.critic_params

    for i in range(2):  # steps 1..4: inside the warmup window
        rng, k = jax.random.split(rng)
        state, env_state, rb_state, _ = fused(state, env_state, rb_state, k)
    assert int(state.steps) == 4
    assert _leaves_equal(state.policy_params, pol0), "policy moved during warmup"
    assert _leaves_equal(state.dual_params, dual0), "duals moved during warmup"
    assert not _leaves_equal(state.critic_params, crit0), "critic did NOT train during warmup"
    # target policy must still equal the (frozen) online policy
    assert _leaves_equal(state.target_policy_params, pol0), (
        "target policy picked up gated-off params -- the gate must run before "
        "the hard target copy"
    )

    for i in range(2):  # steps 5..8: gate open
        rng, k = jax.random.split(rng)
        state, env_state, rb_state, _ = fused(state, env_state, rb_state, k)
    assert int(state.steps) == 8
    assert not _leaves_equal(state.policy_params, pol0), "policy never unfroze"
    assert not _leaves_equal(state.dual_params, dual0), "duals never unfroze"


def test_warmup_zero_is_bit_identical_to_no_gate():
    """Default 0 must not perturb existing arms: same rng -> same state."""
    from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

    outs = []
    for cfg_mod in (dict(), dict(critic_warmup_sgd_steps=0)):
        s = _setup()
        cfg = dataclasses.replace(s["cfg"], **cfg_mod)
        fused = make_fused_train_step(
            s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=1
        )
        state, env_state, rb_state = s["state"], None, s["rb_state"]
        rng = jax.random.PRNGKey(3)
        for _ in range(3):
            rng, k = jax.random.split(rng)
            state, env_state, rb_state, _ = fused(state, env_state, rb_state, k)
        outs.append(state)
    assert _leaves_equal(outs[0].policy_params, outs[1].policy_params)
    assert _leaves_equal(outs[0].critic_params, outs[1].critic_params)
    assert _leaves_equal(outs[0].dual_params, outs[1].dual_params)


def test_behavior_mix_without_frozen_params_raises_at_build_time():
    from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

    s = _setup()
    cfg = dataclasses.replace(s["cfg"], behavior_mix_init=0.5)
    with pytest.raises(ValueError, match="frozen_behavior_params"):
        make_fused_train_step(s["env"], s["nets"], s["optimizers"], s["rb"], cfg, K=1)


# ---------------------------------------------------------------------------
# Cross-run checkpoint loader (uses the real dense-reference checkpoint)
# ---------------------------------------------------------------------------

_I1_CKPT = (
    "/home/talmolab/Desktop/SalkResearch/_implementation_log/DMPO/checkpoints/"
    "arm_i1_nstep100_proprio/DMPONetwork_297676800"
)


@pytest.mark.skipif(
    not __import__("pathlib").Path(_I1_CKPT).is_dir(),
    reason="dense reference checkpoint not on this machine",
)
def test_load_train_state_items_numpy_from_the_real_dense_checkpoint():
    from track_mjx.agent.dmpo.checkpoint import load_train_state_items_numpy

    ws = load_train_state_items_numpy(_I1_CKPT)
    assert set(ws) == {"policy_params", "target_policy_params", "normalizer_params"}
    # policy tree: params/{prior, decoder, policy_head}, 3.73M params, numpy
    pol = ws["policy_params"]
    assert set(pol["params"]) == {"prior", "decoder", "policy_head"}
    leaves = jax.tree.leaves(pol)
    assert sum(l.size for l in leaves) == 3_726_556
    assert all(isinstance(l, np.ndarray) for l in leaves)
    # normalizer: the two obs branches the policy was trained under
    assert set(ws["normalizer_params"]) >= {"imitation_target", "proprioception"}
    # online and target are distinct pytrees with identical structure
    assert jax.tree.structure(pol) == jax.tree.structure(ws["target_policy_params"])


@pytest.mark.skipif(
    not __import__("pathlib").Path(_I1_CKPT).is_dir(),
    reason="dense reference checkpoint not on this machine",
)
def test_missing_item_key_fails_loudly():
    from track_mjx.agent.dmpo.checkpoint import load_train_state_items_numpy

    with pytest.raises(KeyError, match="not_a_key"):
        load_train_state_items_numpy(_I1_CKPT, items=("not_a_key",))


def test_from_state_dict_rebuilds_the_flax_normalizer_struct():
    """The entry point grafts the numpy dict back into DictRunningStatisticsState
    via flax.serialization.from_state_dict — pin that round-trip."""
    import flax.serialization as ser
    from track_mjx.agent.observation_utils import init_dict_normalizer

    template = init_dict_normalizer(
        {"proprioception": jnp.zeros((5,)), "imitation_target": jnp.zeros((3,))}
    )
    state_dict = ser.to_state_dict(template)
    rebuilt = ser.from_state_dict(template, jax.tree.map(np.asarray, state_dict))
    assert type(rebuilt) is type(template)
    assert _leaves_equal(rebuilt, template)
