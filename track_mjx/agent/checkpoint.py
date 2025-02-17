from brax.training.acme import running_statistics, specs

from orbax import checkpoint as ocp
from track_mjx.agent import ppo_networks, ppo
from typing import Callable

from track_mjx.agent import ppo_networks, losses
from jax import numpy as jnp
import jax
from omegaconf import OmegaConf


def load_config_from_checkpoint(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
):
    """Load the config from a checkpoint."""
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        return ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                config=ocp.args.JsonRestore(),
            ),
        )["config"]


def load_training_state(
    checkpoint_path: str,
    abstract_training_state,
    step_prefix: str = "PPONetwork",
    step: int = None,
) -> ppo.TrainingState:
    """Load the training state from checkpoint, given an arbitrary reference training state."""
    mgr_options = ocp.CheckpointManagerOptions(
        create=False,
        step_prefix=step_prefix,
    )
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        return ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.StandardRestore(abstract_training_state),
            ),
        )["train_state"]


def load_checkpoint_for_eval(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
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

    # First load the config
    cfg = OmegaConf.create(
        load_config_from_checkpoint(checkpoint_path, step_prefix, step)
    )

    # Then make an abstract policy to get the pytree structure
    abstract_policy = make_abstract_policy(cfg)

    # Then load the policy given the pytree structure
    policy = ckpt_mgr.restore(
        step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardRestore(abstract_policy),
        ),
    )["policy"]

    return {"cfg": cfg, "policy": policy}


def make_abstract_policy(cfg):
    """Create a random policy from a config."""
    ppo_network = make_ppo_network_from_cfg(cfg)
    key_policy, key_value = jax.random.split(jax.random.key(1))

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
    ppo_network = make_ppo_network_from_cfg(cfg)
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return make_policy(
        policy_params, deterministic=deterministic, get_activation=get_activation
    )


def make_ppo_network_from_cfg(cfg):
    """Create a PPONetwork from a config."""
    normalize = lambda x, y: x
    if cfg["network_config"]["normalize_observations"]:
        normalize = running_statistics.normalize

    if cfg["network_config"]["arch_name"] == "intention":
        ppo_network = ppo_networks.make_intention_ppo_networks(
            observation_size=cfg["network_config"]["observation_size"],
            reference_obs_size=cfg["network_config"]["reference_obs_size"],
            action_size=cfg["network_config"]["action_size"],
            intention_latent_size=cfg["network_config"]["intention_size"],
            preprocess_observations_fn=normalize,
            encoder_hidden_layer_sizes=tuple(
                cfg["network_config"]["encoder_layer_sizes"]
            ),
            decoder_hidden_layer_sizes=tuple(
                cfg["network_config"]["decoder_layer_sizes"]
            ),
            value_hidden_layer_sizes=tuple(cfg["network_config"]["critic_layer_sizes"]),
        )
    else:
        raise ValueError(
            f"Unknown network architecture: {cfg['network_config']['arch_name']}"
        )
    return ppo_network


def save(ckpt_mgr, step, policy, training_state, config):
    """Save a checkpoint during training.
    Consists of policy, training state and config.
    """
    ckpt_mgr.save(
        step=step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardSave(policy),
            train_state=ocp.args.StandardSave(training_state),
            config=ocp.args.JsonSave(config),
        ),
    )
