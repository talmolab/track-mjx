"""Eval reward metrics must report the reward the REPLAY BUFFER stored.

The bug this locks out: `train_dmpo_eval` read `state.reward` / `allenv["reward"]`
-- the raw env total -- while `rollout.py:200-207` stores
`sparse + lambda(t) * (total - sparse)` in replay. On arm_w2's FINAL eval
(297.7M env steps, lambda == 0 since 20M, i.e. the buffer held PURE gap bonus)
that made `eval/batch/mean_episode_reward` read 223.22 when the learner's own
episode return was 10.64 -- a ~21x overstatement, ~96% of it a `forward_velocity`
term whose stored weight had been exactly zero for 278M steps. It also broke the
invariant BASELINE_w2 asserts, that with `gap_crossing_bonus: weight 1.0` the
undiscounted episode return IS the gap count (223.22 vs 11.33 crossings).
"""

import numpy as np
import pytest

from track_mjx.agent.dmpo.train_dmpo_eval import (
    compute_batch_rollout_metrics,
    compute_rollout_metrics,
    remix_eval_reward,
)

T, N = 20, 8
KEY = "rewards/gap_crossing_bonus"


def _rollout(sparse_per_step=0.0, dense_per_step=1.0, with_key=True):
    """A [T, N] rollout where every env terminates once, at t=9.

    env total = sparse + dense, mirroring base.py:314-322 summing weighted terms.
    """
    sparse = np.full((T, N), sparse_per_step, dtype=np.float32)
    total = sparse + dense_per_step
    done = np.zeros((T, N), dtype=np.float32)
    done[9, :] = 1.0
    out = {"reward": total, "done": done}
    if with_key:
        out[KEY] = sparse
    return out


def test_no_remix_key_is_a_passthrough():
    """remix_key=None means rollout.py stored the env reward unchanged."""
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    r_train, measured = remix_eval_reward(allenv, None, None)
    np.testing.assert_allclose(r_train, np.asarray(allenv["reward"]))
    assert measured == 1.0


def test_lambda_one_reproduces_the_env_total():
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    r_train, measured = remix_eval_reward(allenv, KEY, 1.0)
    np.testing.assert_allclose(r_train, np.asarray(allenv["reward"]))
    assert measured == 1.0


def test_lambda_zero_leaves_only_the_sparse_term():
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    r_train, _ = remix_eval_reward(allenv, KEY, 0.0)
    np.testing.assert_allclose(r_train, np.full((T, N), 2.0))


def test_lambda_half_scales_only_the_dense_remainder():
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    r_train, _ = remix_eval_reward(allenv, KEY, 0.5)
    np.testing.assert_allclose(r_train, np.full((T, N), 2.5))


def test_nan_in_the_sparse_term_is_sanitised_like_rollout_py():
    """rollout.py nan_to_num's ONLY the sparse term; the env total is left alone."""
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    sparse = np.array(allenv[KEY])
    sparse[0, 0] = np.nan
    allenv[KEY] = sparse
    r_train, _ = remix_eval_reward(allenv, KEY, 0.0)
    assert r_train[0, 0] == 0.0
    assert np.isfinite(r_train).all()


def test_absent_sparse_key_is_flagged_not_silently_zero():
    """The configured key missing must not read as 'the reward was zero'."""
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0, with_key=False)
    r_train, measured = remix_eval_reward(allenv, KEY, 0.0)
    assert measured == 0.0
    np.testing.assert_allclose(r_train, np.asarray(allenv["reward"]))


def test_batch_metrics_default_call_is_unchanged():
    """One-arg calls must stay byte-identical -- test_gap_crossing_metric relies on it."""
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    assert compute_batch_rollout_metrics(allenv) == compute_batch_rollout_metrics(
        allenv, None, None
    )


def test_batch_primary_keys_report_the_buffer_reward():
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    out = compute_batch_rollout_metrics(allenv, KEY, 0.0)
    # 10 steps per complete episode, sparse-only => 2.0 * 10
    assert out["batch/mean_episode_reward"] == pytest.approx(20.0)
    assert out["batch/reward_per_step"] == pytest.approx(2.0)
    assert out["episode_reward"] == pytest.approx(20.0)
    assert out["batch/reward_anneal_lambda"] == pytest.approx(0.0)
    assert out["batch/reward_train_measured"] == 1.0


def test_batch_env_twins_keep_the_old_env_total_meaning():
    allenv = _rollout(sparse_per_step=2.0, dense_per_step=1.0)
    out = compute_batch_rollout_metrics(allenv, KEY, 0.0)
    legacy = compute_batch_rollout_metrics(allenv)
    assert out["batch/mean_episode_reward_env"] == pytest.approx(
        legacy["batch/mean_episode_reward"]
    )
    assert out["batch/reward_per_step_env"] == pytest.approx(
        legacy["batch/reward_per_step"]
    )
    assert out["episode_reward_env"] == pytest.approx(legacy["episode_reward"])
    assert out["episode_reward_std_env"] == pytest.approx(legacy["episode_reward_std"])


def test_baseline_w2_invariant_holds_once_the_reward_is_annealed():
    """With gap bonus weight 1.0 and lambda 0, episode return IS the gap count.

    This is the invariant BASELINE_w2 asserts and the old eval violated
    (223.22 vs 11.33 on the real run's final eval).
    """
    sparse = np.zeros((T, N), dtype=np.float32)
    sparse[2, :] = 1.0
    sparse[5, :] = 1.0          # 2 crossings inside each 10-step episode
    done = np.zeros((T, N), dtype=np.float32)
    done[9, :] = 1.0
    allenv = {
        "reward": sparse + 0.25,          # dense forward_velocity riding along
        "done": done,
        KEY: sparse,
        "info/just_crossed_gap": sparse,
    }
    out = compute_batch_rollout_metrics(allenv, KEY, 0.0)
    assert out["batch/mean_episode_reward"] == pytest.approx(
        out["batch/gap_crossings_per_episode"]
    )
    # ...and the old readout did NOT satisfy it
    assert out["batch/mean_episode_reward_env"] != pytest.approx(
        out["batch/gap_crossings_per_episode"]
    )


# --------------------------------------------------------------------------
# env-0 path (compute_rollout_metrics)
# --------------------------------------------------------------------------


class _FakeState:
    """Minimal stand-in for the env-0 State pytree compute_rollout_metrics walks."""

    def __init__(self, reward, sparse, done):
        self.reward = np.float32(reward)
        self.metrics = {KEY: np.float32(sparse)}
        self.info = {"just_crossed_gap": np.float32(sparse > 0)}
        self.done = np.float32(done)


def _envzero_rollout():
    """11 states: index 0 is the reset state and is skipped by the metric fn."""
    states = [_FakeState(0.0, 0.0, 0.0)]
    for t in range(10):
        sparse = 1.0 if t in (2, 5) else 0.0
        states.append(_FakeState(sparse + 0.25, sparse, 1.0 if t == 9 else 0.0))
    return states


def test_envzero_default_call_is_unchanged():
    r = _envzero_rollout()
    assert compute_rollout_metrics(r) == compute_rollout_metrics(r, None, None)


def test_envzero_primary_keys_report_the_buffer_reward():
    out = compute_rollout_metrics(_envzero_rollout(), KEY, 0.0)
    # lambda=0 => only the two gap bonuses count
    assert out["cumulative_reward"] == pytest.approx(2.0)
    assert out["mean_episode_reward"] == pytest.approx(2.0)
    assert out["mean_reward_per_step"] == pytest.approx(0.2)
    # env twin keeps the old meaning: 2 bonuses + 10 * 0.25 dense
    assert out["cumulative_reward_env"] == pytest.approx(4.5)
    assert out["mean_episode_reward_env"] == pytest.approx(4.5)
    assert out["mean_reward_per_step_env"] == pytest.approx(0.45)


# --------------------------------------------------------------------------
# lambda source agreement with the fused training step
# --------------------------------------------------------------------------


def test_eval_lambda_recipe_matches_the_fused_training_step():
    """Eval must derive lambda from state.steps, exactly as train_dmpo_step does.

    train_dmpo_step.py:98-99 computes
        t_env = env_steps_estimate(state.steps, cfg, K)
        remix_lambda = reward_anneal_lambda(t_env, cfg)
    The host's `env_steps` int is NOT the same quantity -- it ignores the replay
    warm-up offset -- so reading it here would make the eval drift from the
    buffer it is supposed to describe.
    """
    import jax.numpy as jnp

    from track_mjx.agent.dmpo.config import DMPOConfig
    from track_mjx.agent.dmpo.schedules import env_steps_estimate, reward_anneal_lambda

    cfg = DMPOConfig(
        num_envs=2048,
        unroll_length=10,
        reward_anneal_sparse_key=KEY,
        reward_anneal_env_steps=20_000_000,
    )
    K = 50
    for steps, expected in [(0, 1.0), (12_207, 0.75), (48_828, 0.0), (10**6, 0.0)]:
        t_env = env_steps_estimate(jnp.asarray(steps), cfg, K)
        lam = float(reward_anneal_lambda(t_env, cfg))
        assert lam == pytest.approx(expected, abs=2e-3), f"steps={steps}"


def test_lambda_is_zero_when_a_sparse_key_has_no_schedule():
    """Degenerate config: sparse key set, reward_anneal_env_steps 0 => sparse-only."""
    import jax.numpy as jnp

    from track_mjx.agent.dmpo.config import DMPOConfig
    from track_mjx.agent.dmpo.schedules import reward_anneal_lambda

    cfg = DMPOConfig(reward_anneal_sparse_key=KEY, reward_anneal_env_steps=0)
    assert float(reward_anneal_lambda(jnp.float32(0.0), cfg)) == 0.0
