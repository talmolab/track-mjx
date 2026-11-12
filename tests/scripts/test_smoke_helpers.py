"""The parts of the GPU smoke machinery that run without a GPU.

``tests/scripts/_smoke.py`` sizes each smoke to one training chunk (or a few)
and checks the log it leaves. These tests check that the overrides make that
one chunk end in an eval, that a longer run spans the steps asked for, and that
the log checks refuse a run whose eval did not run or failed, or whose training
episodes were cut to a step or two.
"""

import pytest

from tests.scripts import _smoke

TRAIN_CFG = {"unroll_length": 50, "sequence_length": 101}

_GOOD_LOG = """\
replay warm-up complete at 9600 env steps
chunk env_steps=16000 steps_per_sec=900 policy_loss=0.1 critic_loss=2.0
eval env_steps=16000 batch/n_complete_episodes=12 return=3.5
timing env_steps=16000 eval_wall_s=40.0 ckpt_wall_s=1.0
"""


def _values(overrides):
    return dict(o.removeprefix("++").split("=", 1) for o in overrides)


def test_forced_eval_fires_after_the_one_chunk():
    num_timesteps, _ = _smoke.one_chunk_sizes(TRAIN_CFG)
    values = _values(_smoke.forced_eval_overrides("train_config", TRAIN_CFG))
    eval_every = int(values["train_config.eval_every_steps"])
    chunk = _smoke.NUM_ENVS * TRAIN_CFG["unroll_length"] * _smoke.ITERS_PER_CHUNK
    # The loop evaluates after a chunk once eval_every_steps have passed since
    # step 0; the smoke runs exactly one chunk and stops at num_timesteps.
    assert eval_every == chunk <= num_timesteps
    assert values["eval_render_config.episode_length"] == str(
        _smoke.EVAL_EPISODE_LENGTH
    )


def test_forced_eval_overrides_follow_the_entry_layout():
    values = _values(
        _smoke.forced_eval_overrides(
            "train_setup.train_config", TRAIN_CFG, eval_render_prefix="eval"
        )
    )
    assert set(values) == {
        "train_setup.train_config.eval_every_steps",
        "eval.episode_length",
    }


def test_assert_eval_ran_accepts_a_run_that_evaluated():
    _smoke.assert_eval_ran(_GOOD_LOG)


@pytest.mark.parametrize(
    "line",
    [
        "WARNING Eval render failed: EGL not available",
        "ERROR Eval failed at env_steps=16000 (eval/failed=1); training continues",
        "WARNING SPOUT_EVAL failed: KeyError",
    ],
)
def test_assert_eval_ran_refuses_a_failed_eval(line):
    with pytest.raises(AssertionError, match="in the log"):
        _smoke.assert_eval_ran(_GOOD_LOG + line + "\n")


def test_assert_eval_ran_refuses_a_run_without_an_eval():
    log = _GOOD_LOG.replace("eval env_steps=", "chunk env_steps=")
    with pytest.raises(AssertionError, match="no 'eval env_steps=' line"):
        _smoke.assert_eval_ran(log)


@pytest.mark.parametrize("steps_per_env", [1, 150, 151, 1000])
def test_chunks_spanning_is_the_fewest_that_cover_the_steps(steps_per_env):
    chunks = _smoke.chunks_spanning(TRAIN_CFG, steps_per_env)
    per_env = _smoke.run_sizes(TRAIN_CFG, chunks)[0] // _smoke.NUM_ENVS
    assert chunks >= 1
    assert per_env >= steps_per_env
    if chunks > 1:
        fewer = _smoke.run_sizes(TRAIN_CFG, chunks - 1)[0] // _smoke.NUM_ENVS
        assert fewer < steps_per_env


def test_one_chunk_overrides_size_the_longer_run():
    values = _values(_smoke.one_chunk_overrides("train_config", TRAIN_CFG, chunks=4))
    num_timesteps, max_replay_size = _smoke.run_sizes(TRAIN_CFG, 4)
    assert int(values["train_config.num_timesteps"]) == num_timesteps
    assert int(values["train_config.max_replay_size"]) == max_replay_size
    assert num_timesteps > _smoke.one_chunk_sizes(TRAIN_CFG)[0]


def _episodes_line(per_done, mismatch=0):
    return (
        "training episodes: envs=64 env_steps=64000 dones=1600 "
        f"env_steps_per_done={per_done} reference_clock_mismatch={mismatch}\n"
    )


@pytest.mark.parametrize("per_done", ["40.0", "5.0", "inf"])
def test_episode_check_accepts_healthy_episodes(per_done):
    _smoke.assert_training_episodes_not_degenerate(
        _GOOD_LOG + _episodes_line(per_done), min_env_steps_per_done=5.0
    )


def test_episode_check_refuses_one_step_episodes():
    with pytest.raises(AssertionError, match="steps per done"):
        _smoke.assert_training_episodes_not_degenerate(
            _GOOD_LOG + _episodes_line("1.3"), min_env_steps_per_done=5.0
        )


def test_episode_check_refuses_a_clock_that_runs_on():
    with pytest.raises(AssertionError, match="reference clock off on 64 envs"):
        _smoke.assert_training_episodes_not_degenerate(
            _GOOD_LOG + _episodes_line("40.0", mismatch=64),
            min_env_steps_per_done=5.0,
        )


def test_episode_check_refuses_a_log_without_the_line():
    with pytest.raises(AssertionError, match="no 'training episodes:' line"):
        _smoke.assert_training_episodes_not_degenerate(
            _GOOD_LOG, min_env_steps_per_done=5.0
        )
