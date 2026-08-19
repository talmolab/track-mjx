"""Orbax helpers for DMPO ``TrainingState``.

Mirrors the ``track_mjx.agent.checkpointing`` (ff_ppo) layout so DMPO and PPO
runs use the same on-disk convention:

- ``step_prefix="DMPONetwork"`` → directories are ``DMPONetwork_<step>/``.
- Composite save with three named items::

      policy       — StandardSave(state.policy_params)
      train_state  — StandardSave(state._asdict())
      config       — JsonSave(config_dict)

  Splitting ``policy`` from ``train_state`` lets analysis scripts load just
  the inference-side pytree without instantiating dual variables / optimizer
  state, matching ``track_mjx.agent.checkpointing.load_policy``.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional

import orbax.checkpoint as ocp

from track_mjx.agent.dmpo.learner import TrainingState

STEP_PREFIX = "DMPONetwork"

log = logging.getLogger(__name__)


def make_checkpointer(
    directory: str,
    max_to_keep: int = 3,
    create: bool = True,
    step_prefix: str | None = STEP_PREFIX,
) -> ocp.CheckpointManager:
    """Create (or open) an orbax ``CheckpointManager`` rooted at ``directory``.

    Uses ``step_prefix="DMPONetwork"`` by default so saved directories sort as
    ``DMPONetwork_<step>/`` — the same naming convention as ff_ppo's
    ``PPONetwork_<step>/``.

    Args:
        directory: Filesystem path for checkpoint storage.
        max_to_keep: Number of historical step directories to retain.
        create: If True, create the directory when missing (training mode).
            Pass ``False`` from analysis/restore scripts to fail fast on a
            wrong path.
        step_prefix: Override the directory prefix. Pass ``None`` (the legacy
            DMPO layout, bare integer dirs) to load v1/v2/v3 checkpoints saved
            before the prefix was introduced.
    """
    path = pathlib.Path(directory).absolute()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        step_prefix=step_prefix,
        create=create,
    )
    return ocp.CheckpointManager(path, options=options)


def save(
    mgr: ocp.CheckpointManager,
    step: int,
    state: TrainingState,
    config: dict[str, Any] | None = None,
) -> None:
    """Save ``state`` at ``step`` as a Composite (policy / train_state / config).

    The save is asynchronous; call ``mgr.wait_until_finished`` if you need
    the on-disk write to complete before continuing.

    Args:
        mgr: Orbax checkpoint manager from ``make_checkpointer``.
        step: SGD step (training-side counter, not env steps).
        state: Full DMPO ``TrainingState`` (NamedTuple of pytrees).
        config: Resolved config dict (``OmegaConf.to_container(cfg, resolve=True)``).
            Saved as JSON so analysis scripts can rebuild the network without
            the live hydra config. Optional only for legacy callers.
    """
    items: dict[str, ocp.args.CheckpointArgs] = {
        "policy": ocp.args.StandardSave(state.policy_params),
        "train_state": ocp.args.StandardSave(state._asdict()),
    }
    if config is not None:
        items["config"] = ocp.args.JsonSave(config)
    mgr.save(int(step), args=ocp.args.Composite(**items))


def restore(
    mgr: ocp.CheckpointManager,
    state_template: TrainingState,
    step: int | None = None,
) -> Optional[TrainingState]:
    """Restore the latest (or specified) checkpoint into the shape of ``state_template``.

    Reads only the ``train_state`` item — the ``policy`` and ``config`` items
    are redundant for resuming training. Use ``load_policy`` / ``load_config``
    when only the inference-side pytree is needed.

    Falls back to the legacy ``StandardSave(state._asdict())`` layout (v1/v2/v3
    checkpoints, before the Composite/step_prefix migration) when the
    Composite path raises ``KeyError`` / orbax handler mismatch errors.

    Returns ``None`` if no checkpoint exists.
    """
    target_step = step if step is not None else mgr.latest_step()
    if target_step is None:
        return None
    try:
        restored = mgr.restore(
            target_step,
            args=ocp.args.Composite(
                train_state=ocp.args.StandardRestore(state_template._asdict()),
            ),
        )
        return TrainingState(**restored["train_state"])
    except Exception as e:
        log.warning(
            "Composite restore failed (%s); falling back to legacy StandardSave layout",
            e,
        )
        legacy = mgr.restore(
            target_step,
            args=ocp.args.StandardRestore(state_template._asdict()),
        )
        return TrainingState(**legacy)


def load_policy(
    mgr: ocp.CheckpointManager,
    policy_template: Any,
    step: int | None = None,
) -> Any:
    """Load just the policy pytree from a checkpoint.

    Mirrors ``track_mjx.agent.checkpointing.load_policy`` for the DMPO layout.
    Useful for analysis scripts that don't need optimizer state or dual vars.
    """
    target_step = step if step is not None else mgr.latest_step()
    if target_step is None:
        raise FileNotFoundError("No DMPO checkpoint found in manager")
    restored = mgr.restore(
        target_step,
        args=ocp.args.Composite(policy=ocp.args.StandardRestore(policy_template)),
    )
    return restored["policy"]


def load_train_state_items_numpy(
    ckpt_step_dir: str | pathlib.Path,
    items: tuple[str, ...] = ("policy_params", "target_policy_params", "normalizer_params"),
) -> dict:
    """Load selected ``train_state`` subtrees from ANOTHER run's checkpoint.

    For warm-starting a FRESH run from a different run's ``DMPONetwork_<step>``
    directory. ``restore``/``load_policy`` above need a template whose shapes
    match the SAVED state — impossible here when e.g. the new run's critic has
    a different atom count than the checkpoint's. This loader instead reads
    the checkpoint's own metadata tree and restores each leaf as host numpy
    (``RestoreArgs(restore_type=np.ndarray)``), which sidesteps the sharding
    round-trip entirely (a GPU-saved checkpoint restores fine in a CPU test
    process and vice versa; JAX re-devices the arrays when they are grafted
    into a live TrainingState).

    Args:
      ckpt_step_dir: path to one step directory, e.g.
        ``.../checkpoints/arm_i1_nstep100_proprio/DMPONetwork_297676800``.
      items: top-level ``train_state`` keys to return. Defaults to the
        warm-start graft set: online policy, its target, and the observation
        normalizer (the policy is useless without the running stats it was
        trained under).

    Returns:
      dict mapping each requested key to its numpy pytree.
    """
    import jax
    import numpy as np

    path = pathlib.Path(ckpt_step_dir) / "train_state"
    if not path.is_dir():
        raise FileNotFoundError(f"no train_state item at {path}")
    ptc = ocp.PyTreeCheckpointer()
    meta_tree = ptc.metadata(str(path)).item_metadata.tree
    missing = [k for k in items if k not in meta_tree]
    if missing:
        raise KeyError(
            f"train_state at {path} has no {missing}; available: {sorted(meta_tree)}"
        )
    restore_args = jax.tree.map(
        lambda m: ocp.RestoreArgs(restore_type=np.ndarray), meta_tree
    )
    restored = ptc.restore(str(path), restore_args=restore_args)
    return {k: restored[k] for k in items}


def load_config(
    mgr: ocp.CheckpointManager,
    step: int | None = None,
) -> dict[str, Any]:
    """Load the saved config dict from a checkpoint.

    Returns the same dict that was passed to ``save(..., config=...)``.
    """
    target_step = step if step is not None else mgr.latest_step()
    if target_step is None:
        raise FileNotFoundError("No DMPO checkpoint found in manager")
    restored = mgr.restore(
        target_step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )
    return restored["config"]
