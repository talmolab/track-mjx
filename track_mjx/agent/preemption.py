"""
Preemption handling utilities for robust training continuation.
Handles run state persistence and recovery for array jobs and preemptible environments.
"""

import json
import hashlib
import os
import tempfile
import fcntl
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
from omegaconf import DictConfig, OmegaConf
import orbax.checkpoint as ocp

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
    import socket
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
        mode='w', 
        dir=file_path.parent, 
        delete=False,
        suffix='.tmp'
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
        with open(file_path, 'r') as f:
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
    required_keys = ['run_id', 'checkpoint_path', 'wandb_run_id', 'config_hash']
    if not all(key in run_state for key in required_keys):
        logging.warning("Run state file is missing required keys, ignoring")
        return None
    
    # Check config consistency
    current_config_hash = _hash_config(cfg)
    if run_state['config_hash'] != current_config_hash:
        logging.warning(
            f"Config hash mismatch (saved: {run_state['config_hash']}, "
            f"current: {current_config_hash}), ignoring run state"
        )
        return None
    
    # Check if checkpoint directory exists
    checkpoint_path = Path(run_state['checkpoint_path'])
    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint directory {checkpoint_path} not found, ignoring run state")
        return None
    
    # Try to find the latest checkpoint
    try:
        # Use the same step_prefix as used during training
        ckpt_mgr = ocp.CheckpointManager(
            checkpoint_path,
            options=ocp.CheckpointManagerOptions(
                create=False,
                step_prefix="PPONetwork"
            )
        )
        latest_step = ckpt_mgr.latest_step()
        if latest_step is None:
            logging.warning("No valid checkpoints found in directory, ignoring run state")
            return None
        
        run_state['latest_checkpoint_step'] = latest_step
        logging.info(f"Found valid run state with checkpoint at step {latest_step}")
        return run_state
        
    except Exception as e:
        logging.warning(f"Failed to access checkpoint manager: {e}, ignoring run state")
        return None


def save_run_state(cfg: DictConfig, run_id: str, checkpoint_path: Union[Path, str], 
                   wandb_run_id: str, latest_step: Optional[int] = None) -> None:
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
        'run_id': run_id,
        'checkpoint_path': str(checkpoint_path.resolve()),
        'wandb_run_id': wandb_run_id,
        'config_hash': _hash_config(cfg),
        'timestamp': time.time(),
    }
    
    if latest_step is not None:
        run_state['latest_checkpoint_step'] = latest_step
    
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


def create_checkpoint_callback(cfg: DictConfig, run_id: str, checkpoint_path: Union[Path, str], wandb_run_id: str):
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
                latest_step=step
            )
            logging.debug(f"Updated run state after checkpoint save at step {step}")
        except Exception as e:
            logging.warning(f"Failed to update run state after checkpoint save: {e}")
    
    return checkpoint_callback
