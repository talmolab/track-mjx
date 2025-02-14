from brax.training.acme import running_statistics

from orbax import checkpoint as ocp
from track_mjx.agent import ppo_networks
from typing import Callable


def load_checkpoint(
    checkpoint_path: str, step_prefix: str = "PPONetwork", step: int = None
):
    """Load a checkpoint."""
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix=step_prefix,
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    if step is None:
        step = ckpt_mgr.latest_step()
    return ckpt_mgr.restore(
        ckpt_mgr.latest_step(),
        args=ocp.args.Composite(
            config=ocp.args.JsonRestore(),
            policy=ocp.args.PyTreeRestore(),
            train_state=ocp.args.PyTreeRestore(),
        ),
    )


def load_inference_fn(
    ckpt, deterministic: bool = True, get_activation: bool = True
) -> Callable:
    """
    Create a policy inference function from a checkpoint.
    """
    cfg = ckpt["config"]
    policy_params = ckpt["policy"]
    policy_params[0] = running_statistics.RunningStatisticsState(**policy_params[0])

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

    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return make_policy(
        policy_params, deterministic=deterministic, get_activation=get_activation
    )


def add_to_network_config(
    network_config: dict,
    observation_size: int,
    action_size: int,
    normalize_observations: bool,
    **kwargs,
):
    """Add network info to the network config."""
    network_config["normalize_observations"] = normalize_observations
    network_config["action_size"] = action_size
    network_config["observation_size"] = observation_size
    network_config.update(kwargs)
    return network_config
