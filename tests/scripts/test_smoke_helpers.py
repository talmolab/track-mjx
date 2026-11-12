"""The parts of the GPU smoke machinery that run without a GPU.

``tests/scripts/_smoke.py`` sizes each smoke to one training chunk and checks
the log it leaves. These tests check that the overrides make that one chunk end
in an eval, and that the log check refuses a run whose eval did not run or
failed.
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
