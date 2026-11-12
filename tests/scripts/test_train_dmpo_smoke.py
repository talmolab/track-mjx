"""GPU smoke of ``scripts/train_dmpo.py``.

Runs the rodent imitation entry (intention encoder-decoder, reference clips)
through a replay warm-up and enough chunks on 64 envs for every env to take two
clips' worth of steps. Needs ``DMPO_PRIOR_ROOT`` to hold
``rodent_reference_clips.h5`` and a GPU.

The run length lets the smoke see degenerate episodes. A reference clock that
is not reset on done passes the clip end within one clip's worth of steps, and
from then on every step ends an episode.
"""

from pathlib import Path

import pytest

from tests.scripts import _smoke

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.artifact]

SCRIPT = "train_dmpo.py"
CONFIG = "rodent-imitation-intention"
REFERENCE_CLIPS = "rodent_reference_clips.h5"
# Clips' worth of steps each env takes.
CLIPS_PER_ENV = 2
# Fewest env steps per episode end that counts as non-degenerate. A clock that
# runs on across episodes gives fewer than 2 over a two-clip run.
MIN_ENV_STEPS_PER_DONE = 5.0


def test_imitation_smoke(tmp_path):
    roots = _smoke.require_gpu_run(_smoke.PRIOR_ROOT_VAR)
    clips = Path(roots[_smoke.PRIOR_ROOT_VAR]) / REFERENCE_CLIPS
    if not clips.is_file():
        pytest.skip(f"{REFERENCE_CLIPS} is not staged under DMPO_PRIOR_ROOT")
    train_cfg = _smoke.composed_section(REPO, CONFIG, "train_setup.train_config")
    env_cfg = _smoke.composed_section(REPO, CONFIG, "env_config")
    clip_steps = round(
        env_cfg["clip_length"] / env_cfg["mocap_hz"] / env_cfg["ctrl_dt"]
    )
    chunks = _smoke.chunks_spanning(train_cfg, CLIPS_PER_ENV * clip_steps)
    ckpt_dir = tmp_path / "ckpt"

    run = _smoke.run_entry(
        REPO,
        SCRIPT,
        CONFIG,
        [
            *_smoke.one_chunk_overrides(
                "train_setup.train_config", train_cfg, chunks=chunks
            ),
            # This entry writes to <model_path>/<run id>; checkpoint_dir is set
            # as well so nothing can land under the default root.
            f"++logging_config.model_path={ckpt_dir}",
            f"++checkpoint_dir={ckpt_dir}",
            f"++data_path={clips}",
        ],
        tmp_path,
    )

    _smoke.assert_finite_chunk_losses(run.log)
    _smoke.assert_checkpoint_written(ckpt_dir, run.log)
    _smoke.assert_training_episodes_not_degenerate(run.log, MIN_ENV_STEPS_PER_DONE)
