"""Orbax helpers for DMPO ``TrainingState``.

DMPO's ``TrainingState`` is a NamedTuple of arbitrary pytrees (params, optimizer
states, dual variables, RNG, step counter). Orbax's ``StandardSave`` /
``StandardRestore`` handlers serialize pytrees natively, so we adapt the
NamedTuple via ``_asdict()``/``**`` at the boundary and let orbax do the rest.

This is intentionally narrower than ``track_mjx.agent.checkpointing`` (which
embeds PPO-specific concerns like normalizer state and config snapshots).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import orbax.checkpoint as ocp

from track_mjx.agent.dmpo.learner import TrainingState


def make_checkpointer(
    directory: str,
    max_to_keep: int = 3,
) -> ocp.CheckpointManager:
    """Create (or open) an orbax ``CheckpointManager`` rooted at ``directory``.

    Args:
        directory: Filesystem path for checkpoint storage. Created if missing.
        max_to_keep: Number of historical step directories to retain.

    Returns:
        An ``ocp.CheckpointManager`` configured for DMPO ``TrainingState`` saves.
    """
    path = pathlib.Path(directory).absolute()
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(path, options=options)


def save(
    mgr: ocp.CheckpointManager,
    step: int,
    state: TrainingState,
) -> None:
    """Save ``state`` at ``step`` using ``StandardSave``.

    The NamedTuple is converted to a dict so orbax can serialize the pytree
    leaves directly. The save is asynchronous; call ``mgr.wait_until_finished``
    if you need the on-disk write to complete before continuing.
    """
    mgr.save(int(step), args=ocp.args.StandardSave(state._asdict()))


def restore(
    mgr: ocp.CheckpointManager,
    state_template: TrainingState,
) -> Optional[TrainingState]:
    """Restore the latest checkpoint into the shape of ``state_template``.

    Returns ``None`` if no checkpoint exists. Otherwise returns a freshly
    constructed ``TrainingState`` with leaves loaded from disk.
    """
    latest = mgr.latest_step()
    if latest is None:
        return None
    restored = mgr.restore(
        latest,
        args=ocp.args.StandardRestore(state_template._asdict()),
    )
    return TrainingState(**restored)
