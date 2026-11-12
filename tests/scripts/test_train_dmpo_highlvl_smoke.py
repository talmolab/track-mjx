"""One-chunk GPU smoke of ``scripts/train_dmpo_highlvl.py``.

Runs the plain high-level entry (frozen prior and decoder, bottleneck vision
head) on the run-gap task through a replay warm-up and one chunk on 64 envs.
Needs ``DMPO_PRIOR_ROOT`` and a GPU.
"""

from pathlib import Path

import pytest

from tests.scripts import _smoke

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.artifact]

SCRIPT = "train_dmpo_highlvl.py"
CONFIG = "run_gap/velocity_only"


def test_highlvl_smoke(tmp_path):
    _smoke.require_gpu_run(_smoke.PRIOR_ROOT_VAR)
    train_cfg = _smoke.composed_section(REPO, CONFIG, "train_config")
    ckpt_dir = tmp_path / "ckpt"

    run = _smoke.run_entry(
        REPO,
        SCRIPT,
        CONFIG,
        [
            *_smoke.one_chunk_overrides("train_config", train_cfg),
            f"++checkpoint_dir={ckpt_dir}",
        ],
        tmp_path,
    )

    _smoke.assert_finite_chunk_losses(run.log)
    _smoke.assert_checkpoint_written(ckpt_dir, run.log)
