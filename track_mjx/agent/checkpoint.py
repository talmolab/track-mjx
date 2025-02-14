from typing import Any, Dict, Tuple, Union

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from etils import epath
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax import checkpoint as ocp


def load_config(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
):
    """Load the config from a checkpoint."""
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix=step_prefix,
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    if step is None:
        step = ckpt_mgr.latest_step()
    return ckpt_mgr.restore(
        step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )["config"]


def add_to_network_config(
    cfg: dict,
    observation_size: int,
    action_size: int,
    normalize_observations: bool,
    **kwargs,
):
    """Add to the network config."""
    cfg["network_config"]["normalize_observations"] = normalize_observations
    cfg["network_config"]["action_size"] = action_size
    cfg["network_config"]["observation_size"] = observation_size
    cfg["network_config"].update(kwargs)
    return cfg
