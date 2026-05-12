"""End-to-end smoke for train_dmpo_imitation.

Runs the full Hydra entry at tiny scale (num_envs=64, num_timesteps=200K)
and verifies:
  * The script runs to completion without raising.
  * At least one orbax checkpoint was written.
  * No 'nan' substring appears in the captured log lines for policy_loss/
    critic_loss (the failure mode we're guarding against).

Skipped on CPU CI by default — set RUN_SMOKE=1 to enable. Costs ~3-5 min
even at tiny scale because of MJX compile time on first call.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_SMOKE", "0") != "1",
    reason="Set RUN_SMOKE=1 to run the end-to-end DMPO imitation smoke.",
)
def test_train_dmpo_imitation_smoke(tmp_path):
    """Tiny full-stack run; checks finite losses and checkpoint creation."""
    ckpt_dir = tmp_path / "ckpt"
    cmd = [
        sys.executable, "-m", "track_mjx.train_dmpo_imitation",
        "--config-name=rodent-dmpo-imitation-intention",
        "train_setup.train_config.num_envs=64",
        "train_setup.train_config.num_timesteps=200_000",
        "train_setup.train_config.max_replay_size=20_000",
        "train_setup.train_config.min_replay_size=5_000",
        "train_setup.train_config.batch_size=32",
        "train_setup.train_config.iters_per_chunk=2",
        # Disable mid-run eval so the smoke doesn't try to render video
        # under a possibly-headless test runner.
        "train_setup.train_config.eval_every_steps=200_000",
        "train_setup.train_config.log_every_steps=10_000",
        f"checkpoint_dir={ckpt_dir}",
        f"logging_config.model_path={ckpt_dir}",
        "logging_config.exp_name=smoke_test",
    ]
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/home/talmolab/Desktop/SalkResearch/track-mjx",
        env=env,
        timeout=600,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"smoke run failed: {result.stderr[-500:]}"
    # The orbax checkpoint dir lives under a per-run subdirectory injected
    # by track_mjx.agent.checkpointing.load_from_run_state (mirrors PPO's
    # SLURM-resume convention). Glob recursively for the DMPONetwork_* dir.
    saved_steps = list(Path(ckpt_dir).glob("**/DMPONetwork_*"))
    assert len(saved_steps) >= 1, f"no checkpoint saved in {ckpt_dir}"
    for line in result.stdout.splitlines():
        if "policy_loss=" in line or "critic_loss=" in line:
            assert "nan" not in line.lower(), f"NaN detected: {line}"
