from brax.training.acme import running_statistics, specs

from orbax import checkpoint as ocp
from track_mjx.agent import ppo_networks
from typing import Callable

from track_mjx.agent import ppo_networks, losses
from jax import numpy as jnp
from flax import linen as nn
import jax
from omegaconf import OmegaConf

import logging


def load_config_from_checkpoint(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
):
    """Load the config from a checkpoint."""
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)
    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        logging.info(f"Loading config from {checkpoint_path} at step {step}")
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


def make_dummy_lstm_hidden(cfg: OmegaConf) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Read hidden_layer_num and lstm_features from `cfg`, then
    build a dummy (h, c) tuple each of shape [num_layers, hidden_dim].
    """
    num_layers = cfg['network_config']['hidden_layer_num']
    hidden_dim  = cfg['network_config']['hidden_state_size']
    batch_size = cfg['train_setup']['train_config']['batch_size']
    lstm_cell = nn.LSTMCell(features=hidden_dim)
    env_rngs = jax.random.split(jax.random.PRNGKey(0), batch_size)

    def init_layer(rng):
        return lstm_cell.initialize_carry(rng, ())

    def init_per_env(rng):
        layer_rngs = jax.random.split(rng, num_layers)
        carries   = [init_layer(lr) for lr in layer_rngs]  # list of (c, h)
        c_list, h_list = zip(*carries)
        return jnp.stack(h_list), jnp.stack(c_list) # ([num_layers, hidden_dim], …)

    h_batch, c_batch = jax.vmap(init_per_env)(env_rngs)
    # h0 = jnp.squeeze(h_batch, axis=0)
    # c0 = jnp.squeeze(c_batch, axis=0)

    return (h_batch, c_batch)


def make_abstract_policy(cfg: OmegaConf, seed: int = 1):
    """Create a random policy from a config."""
    ppo_network = make_ppo_network_from_cfg(cfg)
    key_policy, key_value = jax.random.split(jax.random.key(seed))

    dummy_hidden_state = make_dummy_lstm_hidden(cfg)
    
    init_params = losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy,  hidden_state=dummy_hidden_state),
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
        policy_params, deterministic=deterministic, get_activation=get_activation, use_lstm=False
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
