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
from collections.abc import Mapping
from typing import Any

import orbax.checkpoint as ocp

from track_mjx.agent.dmpo.config import ALIGNMENT_FIELDS, LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import TrainingState

STEP_PREFIX = "DMPONetwork"

log = logging.getLogger(__name__)


def make_checkpointer(
    directory: str,
    max_to_keep: int | None = None,
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
            ``None`` (the DEFAULT) means KEEP EVERY CHECKPOINT -- orbax prunes
            nothing. Do not set this to a small integer.

            WHY: a run that goes NaN keeps running unless something stops it.
            With checkpoints every ~10M steps and a retention of THREE, the
            last healthy weights are only ~2 saves from being pruned by NaN
            checkpoints. A diverged run silently eats its own history:
            the longer it runs after diverging, the less of it survives.
            Checkpoints here are ~74 MB, so retention is cheap and the
            replicability it buys is not.
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


def _coerce_legacy_rng(
    restored_train_state: dict[str, Any], state_template: TrainingState
) -> dict[str, Any]:
    """Re-wrap a legacy raw-key ``rng`` leaf to match ``state_template``'s type.

    Every checkpoint saved before the typed-PRNG-key migration stores ``rng``
    as a raw uint32 array (trailing dim 2, from ``jax.random.PRNGKey``).
    ``StandardRestore`` does not coerce a restored leaf's dtype to the
    template's — it silently hands back whatever dtype is on disk, even when
    ``state_template.rng`` is a typed key (``jax.random.key``). Detect exactly
    that mismatch and wrap the raw data with ``jax.random.wrap_key_data`` so
    callers always get back the key type they asked for. Typed-into-typed and
    raw-into-raw restores are untouched (this function is a no-op for both).
    The reverse mismatch — a typed key on disk restored into a raw-key
    template — is intentionally left untouched: no code path builds raw
    templates anymore, so there is nothing to coerce it for.
    """
    import jax
    import jax.numpy as jnp

    template_rng = state_template.rng
    restored_rng = restored_train_state.get("rng")
    if restored_rng is None:
        return restored_train_state
    template_is_typed = jax.dtypes.issubdtype(template_rng.dtype, jax.dtypes.prng_key)
    restored_is_typed = jax.dtypes.issubdtype(restored_rng.dtype, jax.dtypes.prng_key)
    if (
        template_is_typed
        and not restored_is_typed
        and restored_rng.dtype == jnp.uint32
        and restored_rng.shape[-1:] == (2,)
    ):
        restored_train_state = dict(restored_train_state)
        restored_train_state["rng"] = jax.random.wrap_key_data(
            restored_rng, impl=jax.random.key_impl(template_rng)
        )
    return restored_train_state


def restore(
    mgr: ocp.CheckpointManager,
    state_template: TrainingState,
    step: int | None = None,
) -> TrainingState | None:
    """Restore the latest (or specified) checkpoint into the shape of ``state_template``.

    Reads only the ``train_state`` item — the ``policy`` and ``config`` items
    are redundant for resuming training. Use ``load_policy`` / ``load_config``
    when only the inference-side pytree is needed.

    Falls back to the legacy ``StandardSave(state._asdict())`` layout (v1/v2/v3
    checkpoints, before the Composite/step_prefix migration) when the
    Composite path raises ``KeyError`` / orbax handler mismatch errors.

    Either layout may additionally predate the typed-PRNG-key migration: see
    ``_coerce_legacy_rng``, applied in both branches, for how a raw ``rng``
    checkpoint is restored into a typed-key ``state_template``.

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
        train_state = restored["train_state"]
    except Exception as e:
        log.warning(
            "Composite restore failed (%s); falling back to legacy StandardSave layout",
            e,
        )
        train_state = mgr.restore(
            target_step,
            args=ocp.args.StandardRestore(state_template._asdict()),
        )
    # Outside the try/except so a coercion bug raises as itself instead of
    # being swallowed into the "Composite restore failed" fallback path
    # above -- applies to whichever branch produced train_state.
    train_state = _coerce_legacy_rng(train_state, state_template)
    return TrainingState(**train_state)


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
    items: tuple[str, ...] = (
        "policy_params",
        "target_policy_params",
        "normalizer_params",
    ),
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
        ``.../checkpoints/<exp_name>/DMPONetwork_<env_steps>``.
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


# ---------------------------------------------------------------------------
# Algorithm settings recorded with a checkpoint
# ---------------------------------------------------------------------------


def _train_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The train config inside a saved entry config.

    The high-level entries keep it at ``train_config``; the imitation entry at
    ``train_setup.train_config``. A config with neither gives ``{}``.
    """
    if isinstance(config.get("train_config"), Mapping):
        return config["train_config"]
    setup = config.get("train_setup")
    if isinstance(setup, Mapping) and isinstance(setup.get("train_config"), Mapping):
        return setup["train_config"]
    return {}


def record_alignment(config: dict[str, Any], cfg: DMPOConfig) -> dict[str, Any]:
    """Write the alignment fields in effect into a config dict, in place.

    The YAML a run is launched from usually leaves the fields at their
    dataclass defaults, so the resolved hydra config alone would not say which
    algorithm trained the run. ``saved_alignment`` reads what this writes.

    Args:
      config: the resolved entry config that is saved with every checkpoint.
      cfg: the learner config the run trains with.

    Returns:
      ``config``, with each of ``ALIGNMENT_FIELDS`` set in its train section.
    """
    section = _train_section(config)
    if not section:
        section = config.setdefault("train_config", {})
    for name in ALIGNMENT_FIELDS:
        section[name] = getattr(cfg, name)
    return config


def saved_alignment(ckpt_step_dir: str | pathlib.Path) -> dict[str, Any]:
    """The six alignment fields from a checkpoint's saved config.

    A field the saved train config does not carry maps to its
    ``LEGACY_ALIGNMENT`` value: a checkpoint saved before the fields existed
    was trained by the pre-alignment learner. ``policy_loss_state`` is resolved
    (``"auto"`` becomes ``"next"`` or ``"current"`` by the saved
    ``rnn_bptt_length``), so the result compares directly with
    ``current_alignment``.

    Args:
      ckpt_step_dir: one ``DMPONetwork_<step>`` directory.

    Raises:
      ValueError: if the checkpoint's ``config`` item cannot be read.
    """
    path = pathlib.Path(ckpt_step_dir) / "config"
    try:
        config = ocp.Checkpointer(ocp.JsonCheckpointHandler()).restore(str(path))
    except Exception as exc:
        raise ValueError(
            f"could not read the saved config at {path} ({exc}); the "
            "checkpoint's algorithm settings cannot be checked."
        ) from exc
    train = _train_section(config or {})
    values = {
        name: train.get(name, LEGACY_ALIGNMENT[name]) for name in ALIGNMENT_FIELDS
    }
    recurrent = int(train.get("rnn_bptt_length", 0) or 0) > 0
    return _resolve(values, recurrent)


def current_alignment(cfg: DMPOConfig) -> dict[str, Any]:
    """The six alignment fields of ``cfg``, with ``policy_loss_state`` resolved."""
    values = {name: getattr(cfg, name) for name in ALIGNMENT_FIELDS}
    return _resolve(values, int(getattr(cfg, "rnn_bptt_length", 0) or 0) > 0)


def _resolve(values: dict[str, Any], recurrent: bool) -> dict[str, Any]:
    values = dict(values)
    values["bootstrap_on_truncation"] = bool(values["bootstrap_on_truncation"])
    if values["policy_loss_state"] == "auto":
        values["policy_loss_state"] = "current" if recurrent else "next"
    return values


def alignment_differences(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> list[str]:
    """One ``field: checkpoint=... this config=...`` line per differing field."""
    return [
        f"{name}: checkpoint={saved[name]!r} this config={current[name]!r}"
        for name in ALIGNMENT_FIELDS
        if saved[name] != current[name]
    ]


def check_alignment(
    saved: Mapping[str, Any],
    cfg: DMPOConfig,
    *,
    source: str,
    override_key: str,
    allow: bool,
) -> list[str]:
    """Refuse to continue from a checkpoint trained by different settings.

    Grafting or resuming across a change of any alignment field would switch
    the learning algorithm silently partway through a run. A mismatch raises
    unless the override is set, in which case it is logged as a WARNING.

    Args:
      saved: ``saved_alignment`` of the checkpoint.
      cfg: the learner config of this run.
      source: what the checkpoint is, for the message (e.g. ``"graft source
        <path>"``).
      override_key: the config key that allows the mismatch.
      allow: the value of that key.

    Returns:
      The differing fields, one line each (empty when they all match).

    Raises:
      ValueError: on a mismatch without the override, listing every field.
    """
    diffs = alignment_differences(saved, current_alignment(cfg))
    if not diffs:
        return diffs
    listing = "\n  ".join(f"train_config.{line}" for line in diffs)
    if not allow:
        raise ValueError(
            f"{source} was trained with different algorithm settings than this "
            f"config:\n  {listing}\nA field absent from a checkpoint's saved "
            "config holds its pre-alignment value (config/dmpo/"
            "legacy_pre_alignment). Match the settings (for example by adding "
            "/legacy_pre_alignment to the defaults), or set "
            f"{override_key}: true to continue under this config's settings."
        )
    log.warning(
        "%s was trained with different algorithm settings; continuing under "
        "this config's settings because %s is true:\n  %s",
        source,
        override_key,
        listing,
    )
    return diffs


def latest_step_dir(directory: str | pathlib.Path) -> pathlib.Path | None:
    """The newest finished ``DMPONetwork_<step>`` directory under ``directory``.

    Orbax stages a save in ``DMPONetwork_<step>.orbax-checkpoint-tmp-*``; only
    a directory named by the bare step number is finished. ``None`` when the
    directory holds no checkpoint (or does not exist).
    """
    root = pathlib.Path(directory)
    if not root.is_dir():
        return None
    steps = []
    for path in root.iterdir():
        prefix, _, suffix = path.name.rpartition("_")
        if path.is_dir() and prefix == STEP_PREFIX and suffix.isdigit():
            steps.append((int(suffix), path))
    return max(steps)[1] if steps else None


def check_resume_alignment(
    ckpt_dir: str | pathlib.Path,
    cfg: DMPOConfig,
    *,
    override_key: str,
    allow: bool,
) -> dict[str, Any] | None:
    """Check a run directory's newest checkpoint before resuming from it.

    Call this before anything is restored: the policy optimizer state of the
    kl-anchor entry has a different structure under each
    ``policy_grad_clip_scope``, so a mismatch would otherwise surface as an
    opaque restore error, or, where the structures agree, as a silent switch of
    algorithm mid-run.

    Returns:
      ``saved_alignment`` of the newest checkpoint, or ``None`` when
      ``ckpt_dir`` holds none (a fresh run).

    Raises:
      ValueError: on a mismatch without the override (see ``check_alignment``)
        or when the checkpoint's config cannot be read.
    """
    step_dir = latest_step_dir(ckpt_dir)
    if step_dir is None:
        return None
    saved = saved_alignment(step_dir)
    check_alignment(
        saved,
        cfg,
        source=f"the checkpoint to resume, {step_dir}",
        override_key=override_key,
        allow=allow,
    )
    return saved
