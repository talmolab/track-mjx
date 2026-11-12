"""One-chunk GPU smokes of ``scripts/train_dmpo_highlvl_kl_anchor.py``.

One arm per head the entry supports, each run through a replay warm-up and one
training chunk on 64 envs. The chunk ends in an eval, which must log its
metrics and write its video without a failure:

* the feed-forward head on the maze task;
* the GRU head on the run-gap task, whose startup probe must report the frozen
  prior and decoder unchanged (``r_anchor`` near 1);
* the slow-core head with a privileged critic on the spout-maze task, grafted
  from the feed-forward maze checkpoint, whose startup probe must show the new
  inputs do not move the outputs.

Each needs ``DMPO_PRIOR_ROOT`` (the spout arm also ``DMPO_CKPT_ROOT``) and a GPU.
"""

import re
from pathlib import Path

import pytest

from tests.scripts import _smoke

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.artifact]

SCRIPT = "train_dmpo_highlvl_kl_anchor.py"
MIN_R_ANCHOR = 0.85

PRIOR, CKPT = _smoke.PRIOR_ROOT_VAR, _smoke.CKPT_ROOT_VAR

# smoke id -> (config name, roots it needs, strings its log must contain,
#              lowest acceptable startup r_anchor or None)
ARMS = {
    "maze_gen1_prior_mlp": (
        "maze/maze_gen1_prior_mlp",
        (PRIOR,),
        ("replay warm-up complete", "chunk env_steps="),
        None,
    ),
    "arm_r2_rnn_k50": (
        "run_gap/arm_r2_rnn_k50",
        (PRIOR,),
        ("RECURRENT policy head ACTIVE",),
        MIN_R_ANCHOR,
    ),
    "mm_l0a_k10_c3": (
        "spout_maze/mm_l0a_k10_c3",
        (PRIOR, CKPT),
        (
            "PRIVILEGED critic input ACTIVE: critic_privileged_dim=8",
            "SLOW CORE",
            # The format prints an exact zero as "0"; a trailing space keeps
            # "0.0001" from matching.
            "GRAFT invariance: policy_max_abs_diff=0 critic_max_abs_diff=0 ",
        ),
        None,
    ),
}

# Extra overrides per smoke. The spout arm's graft source was saved before the
# alignment fields existed, so the entry refuses it under the aligned defaults
# unless the mismatch is allowed explicitly.
EXTRA_OVERRIDES = {
    "mm_l0a_k10_c3": ("++transfer.graft_allow_algorithm_mismatch=true",),
}


@pytest.mark.parametrize("arm", list(ARMS))
def test_kl_anchor_smoke(arm, tmp_path):
    config_name, roots, markers, min_r_anchor = ARMS[arm]
    _smoke.require_gpu_run(*roots)
    train_cfg = _smoke.composed_section(REPO, config_name, "train_config")
    ckpt_dir = tmp_path / "ckpt"

    run = _smoke.run_entry(
        REPO,
        SCRIPT,
        config_name,
        [
            *_smoke.one_chunk_overrides("train_config", train_cfg),
            *_smoke.forced_eval_overrides("train_config", train_cfg),
            f"++checkpoint_dir={ckpt_dir}",
            *EXTRA_OVERRIDES.get(arm, ()),
        ],
        tmp_path,
    )

    for marker in markers:
        assert marker in run.log, f"{marker!r} not in the log"
    _smoke.assert_finite_chunk_losses(run.log)
    _smoke.assert_checkpoint_written(ckpt_dir, run.log)
    _smoke.assert_eval_ran(run.log)
    assert list(ckpt_dir.glob("eval_*.mp4")), f"no eval video in {ckpt_dir}"
    if min_r_anchor is not None:
        r_anchor = [float(v) for v in re.findall(r"r_anchor=([0-9.]+)", run.log)]
        assert r_anchor, "no r_anchor value in the log"
        assert min(r_anchor) > min_r_anchor, r_anchor
