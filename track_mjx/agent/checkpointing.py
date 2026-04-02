"""Checkpoint management utilities for PPO training.

This module provides functions for saving and loading training checkpoints
using Orbax, including support for:
- Saving/loading policy parameters, training state, and config
- Automatic run state discovery for SLURM preemption recovery
- Cross-job checkpoint resumption with config validation
"""

import fcntl
import hashlib
import json
import logging
import os
import socket
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import jax
import orbax.checkpoint as ocp
from brax.training.acme import running_statistics, specs
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent.ff_ppo import losses as ff_ppo_losses
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from track_mjx.agent.recurrent_ppo import losses as recurrent_ppo_losses
from track_mjx.agent.recurrent_ppo import networks as recurrent_ppo_networks
from track_mjx.agent.observation_utils import (
    convert_flat_to_dict_normalizer,
    init_dict_normalizer,
)


def load_config_from_checkpoint(
    checkpoint_path: str,
    step_prefix: str = "PPONetwork",
    step: int | None = None,
) -> dict[str, Any]:
    """Load the configuration dictionary from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        step_prefix: Prefix used for checkpoint step directories.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Configuration dictionary stored in the checkpoint.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading config from {checkpoint_path} at step {step}")
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

        return OmegaConf.create(cfg)


def load_training_state(
    checkpoint_path: str,
    abstract_training_state: Any,
    step_prefix: str = "PPONetwork",
    step: int | None = None,
) -> Any:
    """Load training state from a checkpoint.

    Requires an abstract training state with the correct pytree structure
    to restore into.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        abstract_training_state: Template with the expected pytree structure.
        step_prefix: Prefix used for checkpoint step directories.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Restored training state matching the structure of abstract_training_state.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading training state from {checkpoint_path} at step {step}")

        return ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.StandardRestore(abstract_training_state),
            ),
        )["train_state"]


def load_policy(
    checkpoint_path: str,
    cfg: DictConfig | None = None,
    ckpt_mgr: ocp.CheckpointManager | None = None,
    step_prefix: str = "PPONetwork",
    step: int | None = None,
) -> tuple[Any, Any]:
    """Load policy parameters from a checkpoint.

    Creates an abstract policy from the config to define the pytree structure,
    then restores the actual parameters.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        cfg: Configuration dict. If None, loaded from checkpoint.
        ckpt_mgr: Existing checkpoint manager. If None, creates a new one.
        step_prefix: Prefix used for checkpoint step directories.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (normalizer_state, policy_params).
    """
    if cfg is None:
        cfg = load_config_from_checkpoint(checkpoint_path, step_prefix, step)

    abstract_policy = make_abstract_policy(cfg)

    if ckpt_mgr is None:
        mgr_options = ocp.CheckpointManagerOptions(
            create=False, step_prefix=step_prefix
        )
        ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    if step is None:
        step = ckpt_mgr.latest_step()

    return ckpt_mgr.restore(
        step,
        args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
    )["policy"]


def load_checkpoint_for_eval(
    checkpoint_path: str,
    step_prefix: str = "PPONetwork",
    step: int | None = None,
) -> dict[str, Any]:
    """Load a checkpoint for evaluation, returning config and policy.

    Convenience function that loads both the configuration and policy
    parameters needed for running inference.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        step_prefix: Prefix used for checkpoint step directories.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Dictionary with keys:
            - "cfg": OmegaConf configuration
            - "policy": Policy parameters (normalizer_state, policy_params)
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    if step is None:
        step = ckpt_mgr.latest_step()

    logging.info(f"Loading checkpoint from {checkpoint_path} at step {step}")

    cfg = OmegaConf.create(
        load_config_from_checkpoint(checkpoint_path, step_prefix, step)
    )
    policy = load_policy(checkpoint_path, cfg, ckpt_mgr, step_prefix, step)

    return {"cfg": cfg, "policy": policy}


def make_abstract_policy(
    cfg: DictConfig,
    seed: int = 1,
) -> tuple[Any, Any]:
    """Create an abstract policy structure for checkpoint restoration.

    Initializes a policy network with random parameters to establish the
    pytree structure needed for loading saved parameters.

    Args:
        cfg: Configuration with network_config section.
        seed: Random seed for parameter initialization.

    Returns:
        Tuple of (normalizer_state, policy_params) with correct structure.
    """
    ppo_network = make_ppo_network_from_cfg(cfg)
    key_policy, key_value = jax.random.split(jax.random.key(seed))

    arch_name = cfg.network_config.get("arch_name")
    if arch_name is None:
        logging.warning(
            "arch_name not found in network_config, defaulting to 'intention'. "
            "If this is unexpected, check your config file."
        )
        arch_name = "intention"
    if arch_name == "recurrent_intention":
        init_params = recurrent_ppo_losses.RecurrentPPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )
    else:
        init_params = ff_ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

    # Handle both new (dict-based) and legacy (flat) config formats
    network_config = cfg["network_config"]
    if "obs_sizes" in network_config:
        # New dict-based format
        obs_sizes = network_config["obs_sizes"]
        dummy_obs = {
            "task_obs": jnp.zeros((1, obs_sizes["task_obs"])),
            "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
        }
        normalizer_state = init_dict_normalizer(dummy_obs)
    else:
        # Legacy flat format
        normalizer_state = running_statistics.init_state(
            specs.Array(network_config["observation_size"], jnp.dtype("float32"))
        )

    return (normalizer_state, init_params.policy)


def load_inference_fn(
    cfg: DictConfig,
    policy_params: tuple[Any, Any],
    deterministic: bool = True,
    get_activation: bool = True,
) -> Callable:
    """Create a policy inference function from loaded parameters.

    Args:
        cfg: Configuration with network_config section.
        policy_params: Tuple of (normalizer_state, policy_params).
        deterministic: If True, use mean action (no sampling).
        get_activation: If True, return network activations in extras.
            Only used for feedforward networks.

    Returns:
        For feedforward: Inference function (obs, rng) -> (action, extras).
        For recurrent: Inference function (obs, hidden, rng) -> (action, extras, new_hidden).
    """
    ppo_network = make_ppo_network_from_cfg(cfg)
    arch_name = cfg.network_config.get("arch_name")
    if arch_name is None:
        logging.warning(
            "arch_name not found in network_config, defaulting to 'intention'. "
            "If this is unexpected, check your config file."
        )
        arch_name = "intention"

    if arch_name == "recurrent_intention":
        make_policy = recurrent_ppo_networks.make_inference_fn(ppo_network)
    else:
        make_policy = ff_ppo_networks.make_inference_fn(ppo_network)

    # Convert legacy flat normalizer to dict normalizer if needed
    normalizer_state, network_params = policy_params
    network_config = cfg.network_config

    # Check if this is a legacy flat normalizer by looking at config format
    is_legacy = not (
        hasattr(network_config, "obs_sizes") or "obs_sizes" in network_config
    )

    if is_legacy:
        # Convert flat normalizer to dict normalizer
        reference_obs_size = network_config.reference_obs_size
        normalizer_state = convert_flat_to_dict_normalizer(
            normalizer_state, reference_obs_size
        )
        policy_params = (normalizer_state, network_params)

    if arch_name == "recurrent_intention":
        return make_policy(policy_params, deterministic=deterministic)
    else:
        return make_policy(
            policy_params, deterministic=deterministic, get_activation=get_activation
        )


def make_ppo_network_from_cfg(cfg: DictConfig) -> Any:
    """Create a PPO network from configuration.

    Args:
        cfg: Configuration with network_config and train_setup sections.

    Returns:
        PPONetworks or RecurrentPPONetworks with policy_network and value_network.

    Raises:
        ValueError: If network architecture is not recognized.
    """
    arch_name = cfg.network_config.get("arch_name")
    if arch_name is None:
        logging.warning(
            "arch_name not found in network_config, defaulting to 'intention'. "
            "If this is unexpected, check your config file."
        )
        arch_name = "intention"
    network_config = cfg.network_config

    # Handle both new (dict-based) and legacy (flat) config formats
    if hasattr(network_config, "obs_sizes") or "obs_sizes" in network_config:
        # New dict-based format
        obs_sizes = dict(network_config.obs_sizes)
    else:
        # Legacy flat format - convert to obs_sizes dict
        obs_sizes = {
            "task_obs": network_config.reference_obs_size,
            "proprioception": network_config.observation_size
            - network_config.reference_obs_size,
        }

    if arch_name == "intention":
        return ff_ppo_networks.make_intention_ppo_networks(
            obs_sizes=obs_sizes,
            action_size=network_config.action_size,
            intention_latent_size=network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(network_config.critic_layer_sizes),
        )
    elif arch_name == "recurrent_intention":
        return recurrent_ppo_networks.make_recurrent_intention_ppo_networks(
            obs_sizes=obs_sizes,
            action_size=network_config.action_size,
            intention_latent_size=network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(network_config.encoder_layer_sizes),
            rnn_type=network_config.rnn_type,
            rnn_hidden_sizes=tuple(network_config.rnn_hidden_sizes),
            value_hidden_layer_sizes=tuple(network_config.critic_layer_sizes),
        )
    else:
        raise ValueError(f"Unknown network architecture: {arch_name}")


def save(
    ckpt_mgr: ocp.CheckpointManager,
    step: int,
    policy: tuple[Any, Any],
    training_state: Any,
    config: dict[str, Any],
    checkpoint_callback: Callable[[int], None] | None = None,
) -> None:
    """Save a training checkpoint.

    Saves policy parameters, training state, and config to the checkpoint
    manager. Optionally calls a callback after successful save.

    Args:
        ckpt_mgr: Orbax checkpoint manager.
        step: Training step number.
        policy: Policy parameters (normalizer_state, policy_params).
        training_state: Full training state for resumption.
        config: Configuration dictionary.
        checkpoint_callback: Optional callback called with step after save.
    """
    ckpt_mgr.save(
        step=step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardSave(policy),
            train_state=ocp.args.StandardSave(training_state),
            config=ocp.args.JsonSave(config),
        ),
    )

    if checkpoint_callback is not None:
        try:
            checkpoint_callback(step)
        except Exception as e:
            logging.warning(f"Checkpoint callback failed: {e}")


def _hash_config(cfg: DictConfig) -> str:
    """Create a short hash of the config for consistency checking.

    Args:
        cfg: Configuration to hash.

    Returns:
        12-character MD5 hash of the JSON-serialized config.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_str = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:12]


def _get_job_identifier() -> str:
    """Get a unique identifier for the current job.

    Checks for job scheduler environment variables in order:
    SLURM -> PBS -> SGE -> local (hostname + PID).

    Returns:
        String identifier like "slurm_12345_0" or "local_hostname_1234".
    """
    # SLURM array job
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id and task_id:
        return f"slurm_{job_id}_{task_id}"

    # SLURM single job
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"slurm_{job_id}"

    # PBS
    pbs_job_id = os.environ.get("PBS_JOBID")
    if pbs_job_id:
        return f"pbs_{pbs_job_id}"

    # SGE
    sge_job_id = os.environ.get("JOB_ID")
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_job_id:
        if sge_task_id:
            return f"sge_{sge_job_id}_{sge_task_id}"
        return f"sge_{sge_job_id}"

    # Local fallback
    return f"local_{socket.gethostname()}_{os.getpid()}"


def _get_run_state_file_path(cfg: DictConfig) -> Path:
    """Get the path to the run state file for this job/config combination.

    Args:
        cfg: Configuration with logging_config.model_path.

    Returns:
        Path to the run state JSON file.
    """
    base_path = Path(cfg.logging_config.model_path).resolve()
    job_id = _get_job_identifier()
    config_hash = _hash_config(cfg)
    return base_path / f"run_state_{job_id}_{config_hash}.json"


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON data to a file.

    Writes to a temporary file then renames, ensuring no partial writes.

    Args:
        file_path: Destination path.
        data: Data to serialize as JSON.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", dir=file_path.parent, delete=False, suffix=".tmp"
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_name = tmp_file.name

    Path(tmp_name).rename(file_path)


def _read_json_with_lock(file_path: Path) -> dict[str, Any] | None:
    """Read JSON file with shared file lock.

    Args:
        file_path: Path to JSON file.

    Returns:
        Parsed JSON data, or None if file doesn't exist or is invalid.
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Failed to read run state file {file_path}: {e}")
        return None


def discover_existing_run_state(cfg: DictConfig) -> dict[str, Any] | None:
    """Discover existing run state for preemption recovery.

    Looks for a run state file matching the current job/config combination,
    validates it, and checks that the checkpoint directory exists.

    Args:
        cfg: Current configuration.

    Returns:
        Run state dict with "latest_checkpoint_step" added if valid,
        None if no valid run state found.
    """
    state_file_path = _get_run_state_file_path(cfg)
    logging.info(f"Looking for existing run state at: {state_file_path}")

    run_state = _read_json_with_lock(state_file_path)
    if not run_state:
        logging.info("No existing run state found")
        return None

    # Validate required keys
    required_keys = ["run_id", "checkpoint_path", "wandb_run_id", "config_hash"]
    if not all(key in run_state for key in required_keys):
        logging.warning("Run state file is missing required keys, ignoring")
        return None

    # Validate config hash
    current_config_hash = _hash_config(cfg)
    if run_state["config_hash"] != current_config_hash:
        logging.warning(
            f"Config hash mismatch (saved: {run_state['config_hash']}, "
            f"current: {current_config_hash}), ignoring run state"
        )
        return None

    # Validate checkpoint directory
    checkpoint_path = Path(run_state["checkpoint_path"])
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint directory {checkpoint_path} not found, ignoring")
        return None

    # Validate checkpoint contents
    try:
        ckpt_mgr = ocp.CheckpointManager(
            checkpoint_path,
            options=ocp.CheckpointManagerOptions(
                create=False, step_prefix="PPONetwork"
            ),
        )
        latest_step = ckpt_mgr.latest_step()
        if latest_step is None:
            logging.warning("No valid checkpoints found in directory, ignoring")
            return None

        run_state["latest_checkpoint_step"] = latest_step
        logging.info(f"Found valid run state with checkpoint at step {latest_step}")
        return run_state

    except Exception as e:
        logging.warning(f"Failed to access checkpoint manager: {e}, ignoring run state")
        return None


def save_run_state(
    cfg: DictConfig,
    run_id: str,
    checkpoint_path: Path | str,
    wandb_run_id: str,
    latest_step: int | None = None,
) -> None:
    """Save run state for preemption recovery.

    Args:
        cfg: Current configuration.
        run_id: Unique run identifier.
        checkpoint_path: Path to checkpoint directory.
        wandb_run_id: Wandb run ID for resuming logging.
        latest_step: Most recent checkpoint step, if known.
    """
    state_file_path = _get_run_state_file_path(cfg)

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    run_state = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "wandb_run_id": wandb_run_id,
        "config_hash": _hash_config(cfg),
        "timestamp": time.time(),
    }

    if latest_step is not None:
        run_state["latest_checkpoint_step"] = latest_step

    try:
        _atomic_write_json(state_file_path, run_state)
        logging.info(f"Saved run state to {state_file_path}")
    except Exception as e:
        logging.error(f"Failed to save run state: {e}")


def cleanup_run_state(cfg: DictConfig) -> None:
    """Remove run state file after successful training completion.

    Args:
        cfg: Configuration used to locate the run state file.
    """
    state_file_path = _get_run_state_file_path(cfg)

    try:
        if state_file_path.exists():
            state_file_path.unlink()
            logging.info(f"Cleaned up run state file: {state_file_path}")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state file: {e}")


def create_checkpoint_callback(
    cfg: DictConfig,
    run_id: str,
    checkpoint_path: Path | str,
    wandb_run_id: str,
) -> Callable[[int], None]:
    """Create a callback that updates run state after each checkpoint save.

    Args:
        cfg: Current configuration.
        run_id: Unique run identifier.
        checkpoint_path: Path to checkpoint directory.
        wandb_run_id: Wandb run ID.

    Returns:
        Callback function that takes a step number and updates run state.
    """

    def checkpoint_callback(step: int) -> None:
        try:
            save_run_state(
                cfg=cfg,
                run_id=run_id,
                checkpoint_path=checkpoint_path,
                wandb_run_id=wandb_run_id,
                latest_step=step,
            )
            logging.debug(f"Updated run state after checkpoint save at step {step}")
        except Exception as e:
            logging.warning(f"Failed to update run state after checkpoint save: {e}")

    return checkpoint_callback


def load_from_run_state(
    cfg: DictConfig,
) -> tuple[str, str, dict[str, Any] | None]:
    """Load or create run state, handling preemption and manual restoration.

    This function handles three scenarios:
    1. Automatic preemption recovery: Discovers existing run state for this job
    2. Manual restoration: Uses restore_from_run_state config path
    3. Fresh start: Creates new run_id and checkpoint path

    If restoring, loads the checkpoint's config and optionally updates
    num_timesteps from the submitted config.

    Args:
        cfg: Current configuration. May be modified to set checkpoint_to_restore.

    Returns:
        Tuple of (run_id, checkpoint_path, existing_run_state).
        existing_run_state is None for fresh starts.
    """
    existing_run_state = discover_existing_run_state(cfg)

    # Case 1: Automatic preemption recovery
    if existing_run_state:
        run_id = existing_run_state["run_id"]
        checkpoint_path = _resolve_path(existing_run_state["checkpoint_path"])
        logging.info(f"Resuming from existing run: {run_id}")
        cfg.train_setup.checkpoint_to_restore = checkpoint_path

    # Case 2: Manual restoration from specified run state file
    elif cfg.train_setup.restore_from_run_state:
        base_path = Path(cfg.logging_config.model_path).resolve()
        full_path = base_path / cfg.train_setup.restore_from_run_state

        with open(full_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            existing_run_state = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        run_id = existing_run_state["run_id"]
        checkpoint_path = _resolve_path(existing_run_state["checkpoint_path"])
        logging.info(f"Restoring from specified run state: {run_id}")
        cfg.train_setup.checkpoint_to_restore = checkpoint_path

    # Case 3: Fresh start
    else:
        run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        model_path = Path(cfg.logging_config.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        checkpoint_path = str(model_path / run_id)

    # If restoring from checkpoint, load and merge configs
    if cfg.train_setup.checkpoint_to_restore is not None:
        checkpoint_to_restore = _resolve_path(cfg.train_setup.checkpoint_to_restore)
        submitted_timesteps = cfg.train_setup.train_config.num_timesteps

        # Load checkpoint config
        cfg = OmegaConf.create(load_config_from_checkpoint(checkpoint_to_restore))
        cfg.train_setup.checkpoint_to_restore = checkpoint_to_restore

        # Allow overriding num_timesteps to extend training
        restored_timesteps = cfg.train_setup.train_config.num_timesteps
        if submitted_timesteps != restored_timesteps:
            logging.info(
                f"Updating num_timesteps: {restored_timesteps} -> {submitted_timesteps}"
            )
            cfg.train_setup.train_config.num_timesteps = submitted_timesteps

        checkpoint_path = checkpoint_to_restore
        run_id = os.path.basename(checkpoint_path)

    logging.info(f"Run ID: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")

    return (run_id, checkpoint_path, existing_run_state)


def _resolve_path(path: str | Path) -> str:
    """Resolve a path to absolute, relative to cwd if needed.

    Args:
        path: Path string or Path object.

    Returns:
        Absolute path as string.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj
    return str(path_obj)
