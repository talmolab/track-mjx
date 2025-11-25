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
from typing import Any, Dict, Optional, Union, Tuple, Callable

from brax.training.acme import running_statistics, specs

from orbax import checkpoint as ocp
from omegaconf import OmegaConf, DictConfig
import orbax.checkpoint as ocp

from track_mjx.agent.mlp_ppo import (
    ppo_networks as mlp_ppo_networks,
    losses as mlp_losses,
)
from jax import numpy as jnp
import jax


def load_config_from_checkpoint(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int | None = None
):
    """Load the config from a checkpoint."""
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading config from {checkpoint_path} at step {step}")
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                config=ocp.args.JsonRestore(),
            ),
        )["config"]

        return cfg


def load_training_state(
    checkpoint_path: str,
    abstract_training_state,
    step_prefix: str = "PPONetwork",
    step: int | None = None,
):
    """Load the training state from checkpoint, given an arbitrary reference training state."""
    mgr_options = ocp.CheckpointManagerOptions(
        create=False,
        step_prefix=step_prefix,
    )
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
    checkpoint_path: str, cfg=None, ckpt_mgr=None, step_prefix="PPONetwork", step=None
):
    if cfg is None:
        cfg = load_config_from_checkpoint(checkpoint_path, step_prefix, step)

    # Make an abstract policy to get the pytree structure
    abstract_policy = make_abstract_policy(cfg)
    if ckpt_mgr is None:
        mgr_options = ocp.CheckpointManagerOptions(
            create=False,
            step_prefix=step_prefix,
        )
        ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    if step is None:
        step = ckpt_mgr.latest_step()

    # Then load the policy given the pytree structure
    return ckpt_mgr.restore(
        step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardRestore(abstract_policy),
        ),
    )["policy"]


def load_checkpoint_for_eval(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int | None = None
):
    """Load a checkpoint's config and policy. Creates an abstract state to define structure.

    Returns: {
        cfg: config,
        policy: policy params
        }
    """
    mgr_options = ocp.CheckpointManagerOptions(
        create=False,
        step_prefix=step_prefix,
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    if step is None:
        step = ckpt_mgr.latest_step()

    logging.info(f"Loading checkpoint from {checkpoint_path} at step {step}")
    print(f"Loading checkpoint from {checkpoint_path} at step {step}")

    # First load the config
    cfg = OmegaConf.create(
        load_config_from_checkpoint(checkpoint_path, step_prefix, step)
    )

    policy = load_policy(checkpoint_path, cfg, ckpt_mgr, step_prefix, step)

    return {"cfg": cfg, "policy": policy}


def make_abstract_policy(cfg: OmegaConf, seed: int = 1):
    """
    Create a random policy from a config.
    """

    losses = mlp_losses

    ppo_network = make_ppo_network_from_cfg(cfg)
    key_policy, key_value = jax.random.split(jax.random.key(seed))

    init_params = losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )

    return (
        running_statistics.init_state(
            specs.Array(cfg["network_config"]["observation_size"], jnp.dtype("float32"))
        ),
        init_params.policy,
    )


def load_inference_fn(
    cfg, policy_params, deterministic: bool = True, get_activation: bool = True
) -> Callable:
    """
    Create a ppo policy inference function from a checkpoint.
    """
    ppo_networks = mlp_ppo_networks

    ppo_network = make_ppo_network_from_cfg(cfg)
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return make_policy(
        policy_params, deterministic=deterministic, get_activation=get_activation
    )


def make_ppo_network_from_cfg(cfg):
    """
    Create a PPONetwork from a config.
    """
    ppo_networks = mlp_ppo_networks

    normalize = lambda x, y: x
    if cfg.train_setup.train_config.normalize_observations:
        normalize = running_statistics.normalize

    if cfg.network_config.arch_name == "intention":
        # Handle backward compatibility for checkpoints without prior_layer_sizes
        prior_layer_sizes = cfg.network_config.get("prior_layer_sizes", cfg.network_config.encoder_layer_sizes)
        
        ppo_network = ppo_networks.make_intention_ppo_networks(
            observation_size=cfg.network_config.observation_size,
            reference_obs_size=cfg.network_config.reference_obs_size,
            action_size=cfg.network_config.action_size,
            intention_latent_size=cfg.network_config.intention_size,
            preprocess_observations_fn=normalize,
            encoder_hidden_layer_sizes=tuple(
                cfg.network_config.encoder_layer_sizes
            ),
            decoder_hidden_layer_sizes=tuple(
                cfg.network_config.decoder_layer_sizes
            ),
            prior_hidden_layer_sizes=tuple(prior_layer_sizes),
            value_hidden_layer_sizes=tuple(
                cfg.network_config.critic_layer_sizes
            ),
        )
    else:
        raise ValueError(
            f"Unknown network architecture: {cfg.network_config.arch_name}"
        )
    return ppo_network


def save(ckpt_mgr, step, policy, training_state, config, checkpoint_callback=None):
    """Save a checkpoint during training.
    Consists of policy, training state and config.

    Args:
    ckpt_mgr: Orbax checkpoint manager
    step: Training step number
    policy: Policy parameters
    training_state: Training state
    config: Config dictionary
    checkpoint_callback: Optional callback function to call after successful save
    """
    ckpt_mgr.save(
        step=step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardSave(policy),
            train_state=ocp.args.StandardSave(training_state),
            config=ocp.args.JsonSave(config),
        ),
    )

    # Call the callback after successful checkpoint save
    if checkpoint_callback is not None:
        try:
            checkpoint_callback(step)
        except Exception as e:
            logging.warning(f"Checkpoint callback failed: {e}")

def _hash_config(cfg: DictConfig) -> str:
    """Create a hash of the config for consistency checking."""
    # Convert to dict and create a stable hash
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_str = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:12]


def _get_job_identifier() -> str:
    """Get a unique identifier for the current job (SLURM-aware)."""
    # Try SLURM first
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if job_id and task_id:
        return f"slurm_{job_id}_{task_id}"

    # Fallback to single job ID
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"slurm_{job_id}"

    # For non-SLURM environments, create a unique identifier
    # Use process ID + hostname for uniqueness

    hostname = socket.gethostname()
    pid = os.getpid()

    # Also check for other job schedulers
    pbs_job_id = os.environ.get("PBS_JOBID")
    if pbs_job_id:
        return f"pbs_{pbs_job_id}"

    sge_job_id = os.environ.get("JOB_ID")
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_job_id:
        if sge_task_id:
            return f"sge_{sge_job_id}_{sge_task_id}"
        return f"sge_{sge_job_id}"

    # For truly local runs, use hostname + pid for uniqueness
    return f"local_{hostname}_{pid}"


def _get_run_state_file_path(cfg: DictConfig) -> Path:
    """Get the path to the run state file."""
    base_path = Path(cfg.logging_config.model_path).resolve()
    job_id = _get_job_identifier()
    config_hash = _hash_config(cfg)

    # Create a unique filename based on job and config
    filename = f"run_state_{job_id}_{config_hash}.json"
    return base_path / filename


def _atomic_write_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Atomically write JSON data to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first, then rename (atomic operation)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=file_path.parent, delete=False, suffix=".tmp"
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_name = tmp_file.name

    # Atomic rename
    Path(tmp_name).rename(file_path)


def _read_json_with_lock(file_path: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file with file locking to prevent race conditions."""
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r") as f:
            # Use file locking to prevent concurrent access
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Failed to read run state file {file_path}: {e}")
        return None


def discover_existing_run_state(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """
    Discover existing run state for the current job/config combination.

    Returns:
        Dict with run state if found and valid, None otherwise.
    """
    state_file_path = _get_run_state_file_path(cfg)

    logging.info(f"Looking for existing run state at: {state_file_path}")

    # Try to read existing state
    run_state = _read_json_with_lock(state_file_path)
    if not run_state:
        logging.info("No existing run state found")
        return None

    # Validate the run state
    required_keys = ["run_id", "checkpoint_path", "wandb_run_id", "config_hash"]
    if not all(key in run_state for key in required_keys):
        logging.warning("Run state file is missing required keys, ignoring")
        return None

    # Check config consistency
    current_config_hash = _hash_config(cfg)
    if run_state["config_hash"] != current_config_hash:
        logging.warning(
            f"Config hash mismatch (saved: {run_state['config_hash']}, "
            f"current: {current_config_hash}), ignoring run state"
        )
        return None

    # Check if checkpoint directory exists
    checkpoint_path = Path(run_state["checkpoint_path"])
    if not checkpoint_path.exists():
        logging.warning(
            f"Checkpoint directory {checkpoint_path} not found, ignoring run state"
        )
        return None

    # Try to find the latest checkpoint
    try:
        # Use the same step_prefix as used during training
        ckpt_mgr = ocp.CheckpointManager(
            checkpoint_path,
            options=ocp.CheckpointManagerOptions(
                create=False, step_prefix="PPONetwork"
            ),
        )
        latest_step = ckpt_mgr.latest_step()
        if latest_step is None:
            logging.warning(
                "No valid checkpoints found in directory, ignoring run state"
            )
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
    checkpoint_path: Union[Path, str],
    wandb_run_id: str,
    latest_step: Optional[int] = None,
) -> None:
    """
    Save the current run state to disk.

    Args:
        cfg: Hydra config
        run_id: Current run ID
        checkpoint_path: Path to checkpoint directory (Path or str)
        wandb_run_id: Wandb run ID for resuming
        latest_step: Latest checkpoint step (optional)
    """
    state_file_path = _get_run_state_file_path(cfg)

    # Ensure checkpoint_path is a Path object
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
    """Clean up the run state file after successful completion."""
    state_file_path = _get_run_state_file_path(cfg)

    try:
        if state_file_path.exists():
            state_file_path.unlink()
            logging.info(f"Cleaned up run state file: {state_file_path}")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state file: {e}")


def create_checkpoint_callback(
    cfg: DictConfig, run_id: str, checkpoint_path: Union[Path, str], wandb_run_id: str
):
    """
    Create a callback function that updates run state whenever a checkpoint is saved.

    Returns:
        A callback function that can be called after checkpoint saves.
    """

    def checkpoint_callback(step: int):
        """Callback to update run state after checkpoint save."""
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

def load_from_run_state(cfg: DictConfig) -> Tuple[str, str, Optional[Dict[str, Any]]]:

    existing_run_state = discover_existing_run_state(cfg)

    # If existing run state found, adjust config for resuming (this handles preemption)
    if existing_run_state:
        # Resume existing run
        run_id = existing_run_state["run_id"]
        checkpoint_path = existing_run_state["checkpoint_path"]
        
        # Ensure checkpoint_path is absolute
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_absolute():
            checkpoint_path_obj = Path.cwd() / checkpoint_path_obj
        checkpoint_path = str(checkpoint_path_obj)

        logging.info(f"Resuming from existing run: {run_id}")

        # Add checkpoint path to config to use orbax for resuming
        cfg.train_setup.checkpoint_to_restore = checkpoint_path

    # If manually passing an existing run_state
    elif cfg.train_setup.restore_from_run_state:
        # Access file path
        base_path = Path(cfg.logging_config.model_path).resolve()
        full_path = base_path / cfg.train_setup.restore_from_run_state

        # Read json with file locking to prevent concurrent access
        with open(full_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            existing_run_state = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        run_id = existing_run_state["run_id"]
        checkpoint_path = existing_run_state["checkpoint_path"]

        # Ensure checkpoint_path is absolute
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_absolute():
            checkpoint_path_obj = Path.cwd() / checkpoint_path_obj
        checkpoint_path = str(checkpoint_path_obj)

        logging.info(f"Restoring from specified run state: {run_id}")

        # Add checkpoint path to config to use orbax for resuming
        cfg.train_setup.checkpoint_to_restore = checkpoint_path

    # If no existing run state is found, generate a new run_if and checkpoint path
    else:
        run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        # Use a base path given by the config, ensure it's absolute
        model_path = Path(cfg.logging_config.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        checkpoint_path = str(model_path / run_id)

    # Load checkpoint's config
    if cfg.train_setup.checkpoint_to_restore is not None:
        checkpoint_to_restore = cfg.train_setup.checkpoint_to_restore

        # Ensure checkpoint_to_restore is absolute
        checkpoint_to_restore_obj = Path(checkpoint_to_restore)
        if not checkpoint_to_restore_obj.is_absolute():
            checkpoint_to_restore_obj = Path.cwd() / checkpoint_to_restore_obj
        checkpoint_to_restore = str(checkpoint_to_restore_obj)

        # Get submitted config's num_timesteps
        sub_timesteps = cfg.train_setup.train_config.num_timesteps

        # Load the checkpoint's config and update the run_id and checkpoint_path
        cfg = OmegaConf.create(
            load_config_from_checkpoint(checkpoint_to_restore)
        )
        cfg.train_setup.checkpoint_to_restore = checkpoint_to_restore

        # Get restored config's num_timesteps
        restored_timesteps = cfg.train_setup.train_config.num_timesteps

        # This allows user to resume a run with an different num_timesteps
        if sub_timesteps != restored_timesteps:
            logging.info(
                f"Original config num_timesteps: {restored_timesteps}, "
                f"Submitted config num_timesteps: {sub_timesteps}, "
                f"Updating restored config to use submitted num_timesteps."
            )
            cfg.train_setup.train_config.num_timesteps = sub_timesteps
            

        checkpoint_path = checkpoint_to_restore
        run_id = os.path.basename(checkpoint_path)

    logging.info(f"Run ID: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")

    return (run_id, checkpoint_path, existing_run_state)