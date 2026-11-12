"""Machinery shared by the one-chunk smokes of the three DMPO entry scripts.

A smoke launches an entry in a subprocess, from the repository root, with
``PYTHONPATH`` pointing at that root so the checkout under test is the one
imported (not an editable install of another). The run is sized to finish a
replay warm-up and exactly one training chunk, writes its checkpoints under a
pytest temp directory and runs with Weights & Biases disabled.
``forced_eval_overrides`` makes that chunk end in an eval, which
``assert_eval_ran`` checks in the log.

The smokes need a CUDA device, so they refuse to run unless the caller opts in
by exporting ``JAX_PLATFORMS`` with a GPU platform. ``tests/conftest.py``
defaults it to ``cpu``, which keeps an unfiltered ``pytest`` run (or one that
only selects the ``artifact`` marker) from starting a training job on a card
another job is using.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path

PRIOR_ROOT_VAR = "DMPO_PRIOR_ROOT"
CKPT_ROOT_VAR = "DMPO_CKPT_ROOT"

# Smoke sizing. Both are overrides of the arm's own values.
NUM_ENVS = 64
ITERS_PER_CHUNK = 2
# Episode length of a forced eval: short, so its rollout and video are cheap.
EVAL_EPISODE_LENGTH = 50

# Wall-clock ceiling for one entry: two XLA compiles of the fused step on top of
# an env build with a warp backend dominate, minutes rather than seconds.
TIMEOUT_S = 45 * 60
_POLL_S = 2.0
_TAIL_LINES = 80


def require_gpu_run(*root_vars: str) -> dict[str, str]:
    """Skips the calling test unless it can run for real.

    Args:
        *root_vars: Names of environment variables that must point at staged
            artifacts (``DMPO_PRIOR_ROOT``, ``DMPO_CKPT_ROOT``).

    Returns:
        ``{name: value}`` for each requested variable.
    """
    missing = [v for v in root_vars if not os.environ.get(v)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set (staged artifacts absent)")
    platforms = {
        p.strip() for p in os.environ.get("JAX_PLATFORMS", "cpu").split(",") if p
    }
    if not platforms & {"cuda", "gpu"}:
        pytest.skip(
            "GPU smoke: JAX_PLATFORMS is "
            f"{os.environ.get('JAX_PLATFORMS', 'cpu')!r}; export "
            "JAX_PLATFORMS=cuda,cpu to run it"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        pytest.skip("GPU smoke: CUDA_VISIBLE_DEVICES hides every device")
    return {v: os.environ[v] for v in root_vars}


def composed_section(repo: Path, config_name: str, section: str) -> dict:
    """One section of a composed DMPO config, interpolations left unresolved.

    Args:
        repo: Repository root.
        config_name: Hydra config name, e.g. ``maze/maze_gen1_prior_mlp``.
        section: Dotted key of the section, e.g. ``train_config``.

    Returns:
        The section as a plain dict.
    """
    config_dir = repo / "track_mjx" / "config" / "dmpo"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    return OmegaConf.to_container(OmegaConf.select(cfg, section), resolve=False)


def one_chunk_sizes(train_cfg: dict) -> tuple[int, int]:
    """Sizes a run to a replay warm-up plus exactly one chunk.

    Warm-up ends once the replay holds ``sequence_length + 1`` steps per env
    (``min_replay_size`` is overridden to 0, so that is the floor), and every
    fused step adds ``unroll_length`` steps per env. One chunk is
    ``ITERS_PER_CHUNK`` more fused steps. The replay is sized to hold all of it
    so the smoke never depends on the ring buffer wrapping.

    Args:
        train_cfg: The arm's ``train_config`` (needs ``unroll_length`` and
            ``sequence_length``).

    Returns:
        ``(num_timesteps, max_replay_size)``.
    """
    unroll = int(train_cfg["unroll_length"])
    sequence_length = int(train_cfg["sequence_length"])
    warmup_iters = -(-(sequence_length + 1) // unroll)
    total_iters = warmup_iters + ITERS_PER_CHUNK
    return total_iters * NUM_ENVS * unroll, total_iters * unroll * NUM_ENVS


def one_chunk_overrides(prefix: str, train_cfg: dict) -> list[str]:
    """Hydra overrides that shrink a run to warm-up plus one chunk.

    Args:
        prefix: Dotted path of the train config in the composed config
            (``train_config`` or ``train_setup.train_config``).
        train_cfg: The arm's own ``train_config``.

    Returns:
        ``++`` overrides for envs, batch, replay and budget.
    """
    num_timesteps, max_replay_size = one_chunk_sizes(train_cfg)
    values = {
        "num_envs": NUM_ENVS,
        "batch_size": NUM_ENVS,
        "iters_per_chunk": ITERS_PER_CHUNK,
        "min_replay_size": 0,
        "max_replay_size": max_replay_size,
        "num_timesteps": num_timesteps,
    }
    return [f"++{prefix}.{key}={value}" for key, value in values.items()]


def forced_eval_overrides(
    prefix: str, train_cfg: dict, eval_render_prefix: str = "eval_render_config"
) -> list[str]:
    """Hydra overrides that make the smoke's one chunk end in an eval.

    The training loop evaluates after a chunk once ``eval_every_steps`` env
    steps have passed since the last eval (step 0 for a fresh run). One chunk's
    worth of steps is reached after the first chunk, the only one a smoke runs.
    The eval episode is shortened to ``EVAL_EPISODE_LENGTH``.

    Args:
        prefix: Dotted path of the train config in the composed config.
        train_cfg: The arm's own ``train_config`` (needs ``unroll_length``).
        eval_render_prefix: Dotted path of the entry's eval settings.

    Returns:
        ``++`` overrides for the eval cadence and the eval episode length.
    """
    chunk_steps = NUM_ENVS * int(train_cfg["unroll_length"]) * ITERS_PER_CHUNK
    return [
        f"++{prefix}.eval_every_steps={chunk_steps}",
        f"++{eval_render_prefix}.episode_length={EVAL_EPISODE_LENGTH}",
    ]


def saved_steps(ckpt_root: Path) -> list[int]:
    """Env-step numbers of the finished checkpoints under a directory."""
    steps = []
    for path in ckpt_root.rglob("DMPONetwork_*"):
        suffix = path.name.rsplit("_", 1)[-1]
        # Orbax stages a save in a ``DMPONetwork_<step>.orbax-checkpoint-tmp-*``
        # directory first; only the finished one ends in the bare step number.
        if path.is_dir() and suffix.isdigit():
            steps.append(int(suffix))
    return sorted(steps)


@dataclass
class SmokeRun:
    """Outcome of one entry launch."""

    log: str
    wall_s: float
    peak_gpu_mib: int | None


def _gpu_used_mib() -> int | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return max(int(v) for v in out.split())
    except (subprocess.SubprocessError, ValueError):
        return None


def _tail(log: str) -> str:
    return "\n".join(log.splitlines()[-_TAIL_LINES:])


def run_entry(
    repo: Path,
    script: str,
    config_name: str,
    overrides: list[str],
    work_dir: Path,
    timeout_s: float = TIMEOUT_S,
) -> SmokeRun:
    """Runs an entry script to completion and returns its log.

    Fails the calling test, with the log tail, on a non-zero exit or a timeout.

    Args:
        repo: Repository root; the working directory and the import root.
        script: File name under ``scripts/``.
        config_name: Hydra ``--config-name``.
        overrides: Extra hydra overrides.
        work_dir: Directory for the log and the hydra run directory.
        timeout_s: Wall-clock limit before the entry is killed.

    Returns:
        The captured log, the wall time and the peak GPU memory sampled while
        it ran (``None`` without ``nvidia-smi``).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "smoke.log"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(repo), os.environ.get("PYTHONPATH", "")) if p
        ),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "MUJOCO_GL": "egl",
        "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",
    }
    cmd = [
        sys.executable,
        str(repo / "scripts" / script),
        f"--config-name={config_name}",
        *overrides,
        f"hydra.run.dir={work_dir / 'hydra'}",
    ]
    start = time.monotonic()
    peak = None
    timed_out = False
    with log_path.open("w") as sink:
        proc = subprocess.Popen(
            cmd, cwd=repo, env=env, stdout=sink, stderr=subprocess.STDOUT
        )
        try:
            while True:
                try:
                    returncode = proc.wait(timeout=_POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    used = _gpu_used_mib()
                    if used is not None:
                        peak = max(peak or 0, used)
                    if time.monotonic() - start > timeout_s:
                        timed_out = True
                        returncode = None
                        break
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    wall_s = time.monotonic() - start
    log = log_path.read_text(errors="replace")
    print(
        f"[smoke] {config_name}: wall {wall_s:.0f}s, "
        f"peak GPU memory {peak} MiB, log {log_path}"
    )
    if timed_out:
        pytest.fail(
            f"{script} {config_name} timed out after {timeout_s:.0f}s:\n{_tail(log)}"
        )
    if returncode != 0:
        pytest.fail(f"{script} {config_name} exited {returncode}:\n{_tail(log)}")
    return SmokeRun(log=log, wall_s=wall_s, peak_gpu_mib=peak)


def assert_checkpoint_written(ckpt_root: Path, log: str) -> None:
    """The run left at least one finished checkpoint behind."""
    steps = saved_steps(ckpt_root)
    assert steps, f"no DMPONetwork_<step> directory under {ckpt_root}\n{_tail(log)}"


_CHUNK_LINE = re.compile(
    r"chunk env_steps=(\d+) steps_per_sec=\S+ policy_loss=(\S+) critic_loss=(\S+)"
)


def assert_finite_chunk_losses(log: str) -> None:
    """Every logged chunk reports a finite policy and critic loss."""
    chunks = _CHUNK_LINE.findall(log)
    assert chunks, f"no 'chunk env_steps=' line in the log\n{_tail(log)}"
    for env_steps, policy_loss, critic_loss in chunks:
        assert math.isfinite(float(policy_loss)) and math.isfinite(
            float(critic_loss)
        ), f"non-finite loss at env_steps={env_steps}: {policy_loss} {critic_loss}"


_EVAL_LINE = re.compile(r"eval env_steps=\d+ ")

# Log text of an eval that failed, or whose video, spout or maze statistics did.
EVAL_FAILURE_MARKERS = (
    "Eval render failed",
    "eval/failed",
    "SPOUT_EVAL failed",
    "MAZE_EVAL failed",
)


def assert_eval_ran(log: str) -> None:
    """The run logged an eval, and no part of an eval failed."""
    assert _EVAL_LINE.search(log), f"no 'eval env_steps=' line in the log\n{_tail(log)}"
    for marker in EVAL_FAILURE_MARKERS:
        assert marker not in log, f"{marker!r} in the log\n{_tail(log)}"
