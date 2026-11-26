"""Logging utilities specific to ``train_dmpo`` (separate from
``track_mjx.agent.wandb_logging`` so its lifecycle stays decoupled from
PPO).

Two responsibilities:
1. ``make_run_id(config_name, seed, git_sha)`` — construct a deterministic
   wandb run id of the form ``<config>_seed<N>_g<sha7>``. Same config + seed
   + commit always yields the same id, so wandb resume reattaches to the
   same run on restart.
2. ``save_wandb_state`` / ``load_wandb_state`` — persist the run id next to
   the checkpoint dir so a run that was started before the git commit
   advanced (or started by a different process) can still be resumed.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def make_run_id(config_name: str, seed: int, git_sha: str | None) -> str:
    """Construct a deterministic wandb run id.

    Args:
        config_name: hydra config name (e.g. ``rodent-dmpo-vision-scratch-position``).
        seed: integer seed.
        git_sha: full or truncated git sha; ``None`` / empty falls back to ``nogit``.

    Returns:
        ``"<config_name>_seed<seed>_g<sha7>"``.
    """
    sha7 = (git_sha or "nogit")[:7] or "nogit"
    return f"{config_name}_seed{int(seed)}_g{sha7}"


def detect_git_sha(repo_path: str | Path) -> str | None:
    """Return the HEAD git sha for ``repo_path`` or ``None`` on failure."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_wandb_state(checkpoint_path: str | Path, wandb_run_id: str) -> None:
    """Persist ``wandb_run_id`` next to the checkpoint dir."""
    state_file = Path(checkpoint_path) / "wandb_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"wandb_run_id": wandb_run_id}, indent=2))
    log.info("Saved wandb state to %s", state_file)


def load_wandb_state(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Read previously saved wandb state. Returns ``None`` if missing or invalid."""
    state_file = Path(checkpoint_path) / "wandb_state.json"
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text())
        if "wandb_run_id" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to read wandb state: %s", e)
        return None
