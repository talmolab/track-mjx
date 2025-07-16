from brax.training.acme import running_statistics, specs

from orbax import checkpoint as ocp
from typing import Callable

from track_mjx.agent.lstm_ppo import (
    ppo_networks as lstm_ppo_networks,
    losses as lstm_losses,
)
from track_mjx.agent.mlp_ppo import (
    ppo_networks as mlp_ppo_networks,
    losses as mlp_losses,
)
from jax import numpy as jnp
import jax
from omegaconf import OmegaConf

import logging
from flax import linen as nn


def load_config_from_checkpoint(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
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
         # TODO: Fill in missing cfg keys--remove once cfg structure is stable
        if "use_lstm" not in cfg["train_setup"]["train_config"]:
            cfg["train_setup"]["train_config"]["use_lstm"] = False
        if "get_activation" not in cfg["train_setup"]["train_config"]:
            cfg["train_setup"]["train_config"]["get_activation"] = False
        if "deterministic_eval" not in cfg["train_setup"]["train_config"]:
            cfg["train_setup"]["train_config"]["deterministic_eval"] = False
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

    # TODO: Fill in missing cfg keys--remove once cfg structure is stable
    if "use_lstm" not in cfg["train_setup"]["train_config"]:
        cfg["train_setup"]["train_config"]["use_lstm"] = False
    if "get_activation" not in cfg["train_setup"]["train_config"]:
        cfg["train_setup"]["train_config"]["get_activation"] = False
    if "deterministic_eval" not in cfg["train_setup"]["train_config"]:
        cfg["train_setup"]["train_config"]["deterministic_eval"] = False

    policy = load_policy(checkpoint_path, cfg, ckpt_mgr, step_prefix, step)

    return {"cfg": cfg, "policy": policy}


def make_dummy_lstm_hidden(cfg: OmegaConf) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Read hidden_layer_num and lstm_features from `cfg`, then
    build a dummy (h, c) tuple each of shape [num_layers, hidden_dim].
    """
    num_layers = cfg["network_config"]["hidden_layer_num"]
    hidden_dim = cfg["network_config"]["hidden_state_size"]
    batch_size = cfg["train_setup"]["train_config"]["batch_size"]
    lstm_cell = nn.LSTMCell(features=hidden_dim)
    env_rngs = jax.random.split(jax.random.PRNGKey(0), batch_size)

    def init_layer(rng):
        return lstm_cell.initialize_carry(rng, ())

    def init_per_env(rng):
        layer_rngs = jax.random.split(rng, num_layers)
        carries = [init_layer(lr) for lr in layer_rngs]  # list of (c, h)
        c_list, h_list = zip(*carries)
        return jnp.stack(h_list), jnp.stack(c_list)  # ([num_layers, hidden_dim], …)

    h_batch, c_batch = jax.vmap(init_per_env)(env_rngs)
    # h0 = jnp.squeeze(h_batch, axis=0)
    # c0 = jnp.squeeze(c_batch, axis=0)

    return (h_batch, c_batch)


def make_abstract_policy(cfg: OmegaConf, seed: int = 1):
    """
    Create a random policy from a config.
    """
    use_lstm = False #cfg["train_setup"]["train_config"]["use_lstm"]
    if use_lstm:
        losses = lstm_losses
    else:
        losses = mlp_losses

    ppo_network = make_ppo_network_from_cfg(cfg)
    key_policy, key_value = jax.random.split(jax.random.key(seed))

    if use_lstm:
        dummy_hidden_state = make_dummy_lstm_hidden(cfg)

        init_params = losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(
                key_policy, hidden_state=dummy_hidden_state
            ),
            value=ppo_network.value_network.init(key_value),
        )
    else:
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
    if cfg.train_setup.train_config.use_lstm:
        ppo_networks = lstm_ppo_networks
    else:
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
    lstm = False #cfg["train_setup"]["train_config"]["use_lstm"]
    if lstm:
        ppo_networks = lstm_ppo_networks
    else:
        ppo_networks = mlp_ppo_networks

    normalize = lambda x, y: x
    if cfg["network_config"]["normalize_observations"]:
        normalize = running_statistics.normalize

    if cfg["network_config"]["arch_name"] == "intention":

        if lstm:
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
                value_hidden_layer_sizes=tuple(
                    cfg["network_config"]["critic_layer_sizes"]
                ),
                hidden_state_size=cfg["network_config"]["hidden_state_size"],
                hidden_layer_num=cfg["network_config"]["hidden_layer_num"],
            )

        else:
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
                value_hidden_layer_sizes=tuple(
                    cfg["network_config"]["critic_layer_sizes"]
                ),
            )
    else:
        raise ValueError(
            f"Unknown network architecture: {cfg['network_config']['arch_name']}"
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
