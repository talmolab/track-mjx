"""A zero gap-crossing rate must mean "it never crossed", not "nobody measured".

The bug this locks out: `batch/gap_crossings_per_env` was computed from
`rewards/gap_crossing_bonus`, a REWARD metric that only exists when
`gap_crossing_bonus` is listed in `env_config.reward_terms`. Every frozen-prior
arm deliberately omits that term (the reward is kept velocity-only to match the
PPO reference), so the key was always absent and the old
`allenv.get(key, zeros_like(rew))` default reported 0.000 crossings on every eval
of every arm -- unfalsifiable, and indistinguishable from a real zero. It was
reported for weeks as though it were a behavioural finding.

The task maintains `info["just_crossed_gap"]` on every step regardless of reward
configuration (run_gap.py:542-546), so that is now the primary source, and
`batch/gap_measured` says which source was used.
"""

import numpy as np

from track_mjx.agent.dmpo.train_dmpo_eval import compute_batch_rollout_metrics

T, N = 20, 8


def _base(**extra):
    """A [T, N] rollout where every env terminates once, at t=9."""
    done = np.zeros((T, N), dtype=np.float32)
    done[9, :] = 1.0
    return {"reward": np.ones((T, N), dtype=np.float32), "done": done, **extra}


def test_absent_source_is_flagged_not_reported_as_zero():
    """Neither source present -> rate is 0 but gap_measured says don't trust it."""
    out = compute_batch_rollout_metrics(_base())
    assert out["batch/gap_crossings_per_env"] == 0.0
    assert out["batch/gap_measured"] == 0.0, (
        "with no crossing signal available the metric must advertise itself as "
        "unmeasured; otherwise a structural zero reads as a real zero"
    )


def test_info_flag_is_used_when_present():
    """The task's own flag is read even though the reward term is disabled."""
    jc = np.zeros((T, N), dtype=np.float32)
    jc[3, 0] = 1.0      # env 0 crosses once
    jc[5, 1] = 1.0      # env 1 crosses twice
    jc[7, 1] = 1.0
    out = compute_batch_rollout_metrics(_base(**{"info/just_crossed_gap": jc}))
    assert out["batch/gap_measured"] == 1.0
    assert out["batch/gap_crossings_per_env"] == 3.0 / N
    assert out["batch/frac_envs_crossing_gap"] == 2.0 / N


def test_info_flag_wins_over_the_reward_bonus():
    """If both exist they must agree; the info flag is authoritative.

    The bonus is weighted (it is a reward, not a count), so it can be any
    positive magnitude. Only its sign is meaningful, and only the info flag is
    guaranteed present.
    """
    jc = np.zeros((T, N), dtype=np.float32)
    jc[3, 0] = 1.0
    bonus = np.zeros((T, N), dtype=np.float32)
    bonus[3, 0] = 10.0                     # same event, weight 10
    bonus[4, 5] = 10.0                     # a disagreement, to prove precedence
    out = compute_batch_rollout_metrics(
        _base(**{"info/just_crossed_gap": jc, "rewards/gap_crossing_bonus": bonus})
    )
    assert out["batch/gap_crossings_per_env"] == 1.0 / N


def test_reward_bonus_still_works_as_a_fallback():
    """Configs that DO enable the bonus keep working, and weight is ignored."""
    bonus = np.zeros((T, N), dtype=np.float32)
    bonus[2, 3] = 10.0
    bonus[6, 3] = 10.0
    out = compute_batch_rollout_metrics(_base(**{"rewards/gap_crossing_bonus": bonus}))
    assert out["batch/gap_measured"] == 1.0
    assert out["batch/gap_crossings_per_env"] == 2.0 / N
    assert out["batch/frac_envs_crossing_gap"] == 1.0 / N


def test_a_measured_zero_is_distinguishable_from_an_unmeasured_one():
    """The whole point: crossing rate 0 with the signal present is a real result."""
    jc = np.zeros((T, N), dtype=np.float32)   # genuinely never crossed
    measured = compute_batch_rollout_metrics(_base(**{"info/just_crossed_gap": jc}))
    blind = compute_batch_rollout_metrics(_base())
    assert measured["batch/gap_crossings_per_env"] == blind["batch/gap_crossings_per_env"] == 0.0
    assert measured["batch/gap_measured"] == 1.0
    assert blind["batch/gap_measured"] == 0.0


def test_per_episode_and_per_env_denominators_differ():
    """`per_env` spans the whole window; `per_episode` is per attempt.

    An env runs several episodes inside the T-step eval window, so quoting the
    per-env figure as if it were per-episode overstates the crossing rate by
    roughly T/mean_episode_length. Here: every env terminates at t=9, so with
    T=20 each env completes 2 episodes.
    """
    done = np.zeros((T, N), dtype=np.float32)
    done[9, :] = 1.0
    done[19, :] = 1.0                      # 2 complete episodes per env
    jc = np.zeros((T, N), dtype=np.float32)
    jc[3, :] = 1.0                         # one crossing per env, in episode 1
    out = compute_batch_rollout_metrics(
        {"reward": np.ones((T, N), np.float32), "done": done, "info/just_crossed_gap": jc}
    )
    assert out["batch/n_complete_episodes"] == 2 * N
    assert out["batch/gap_crossings_per_env"] == 1.0          # N crossings / N envs
    assert out["batch/gap_crossings_per_episode"] == 0.5      # N crossings / 2N episodes


def test_anchor_metrics_are_aggregated_over_all_envs():
    """`anchor/*` must be a 2048-env mean, not one rat's episode.

    Reported from the env-0 path alone, `anchor/r_task` swung 0.847 -> 0.517 ->
    0.803 across three consecutive h1 evals while the all-env `reward_per_step`
    in the same log lines moved a few percent. `anchor/r_anchor` is the only
    continuous measure of drift from the frozen prior, and it is the primary
    readout for the decoder-thaw arm, so it has to be an estimator.
    """
    # env 0 is an outlier; the all-env mean must not follow it.
    a = np.full((T, N), 0.8, dtype=np.float32)
    a[:, 0] = 0.1
    out = compute_batch_rollout_metrics(_base(**{"anchor/r_task": a}))
    expected = (0.8 * (N - 1) + 0.1) / N
    assert out["batch/anchor/r_task"] == np.float32(expected).astype(float).item() or \
        abs(out["batch/anchor/r_task"] - expected) < 1e-6
    assert abs(out["batch/anchor/r_task"] - 0.1) > 0.5, "must not equal the env-0 value"
    assert out["batch/anchor/r_task_sem"] > 0.0


def test_anchor_sem_is_zero_when_every_env_agrees():
    a = np.full((T, N), 0.42, dtype=np.float32)
    out = compute_batch_rollout_metrics(_base(**{"anchor/r_anchor": a}))
    assert abs(out["batch/anchor/r_anchor"] - 0.42) < 1e-6
    assert out["batch/anchor/r_anchor_sem"] == 0.0
