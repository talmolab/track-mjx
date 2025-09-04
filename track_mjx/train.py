"""
Entry point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os
import sys

# Limit to a particular GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Either preallocate memory for JAX or disable it
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
#     "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"
# )
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "osmesa")

import jax

import hydra
from omegaconf import DictConfig, OmegaConf
import functools
import wandb
from brax import envs
import orbax.checkpoint as ocp
from track_mjx.agent.mlp_ppo import ppo as mlp_ppo, ppo_networks as mlp_ppo_networks
from track_mjx.agent.lstm_ppo import ppo as lstm_ppo, ppo_networks as lstm_ppo_networks
import warnings
from pathlib import Path
from datetime import datetime
import logging
import json

from track_mjx.io import load
from track_mjx.io.load import load_reaching_data, ReferenceClipReach
from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment.task.multi_clip_reaching import MultiClipReaching
from track_mjx.environment.task.single_clip_reaching import SingleClipReaching
from track_mjx.environment import wrappers
from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.analysis import render
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.reacher.mouse_arm import MouseArm
from track_mjx.environment.task.reward import RewardConfig, RewardConfigReach

warnings.filterwarnings("ignore", category=DeprecationWarning)

_WALKERS = {
    "rodent": Rodent,
    "fly": Fly,
}

_REACHERS = {
    "mouse_arm": MouseArm,
}


@hydra.main(version_base=None, config_path="config", config_name="mouse-arm")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    # Register environments
    envs.register_environment("mouse_arm_multi_clip", MultiClipReaching)
    envs.register_environment("mouse_arm_single_clip", SingleClipReaching)
    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    # Generate a new run_id and associated checkpoint path
    run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    # TODO: Use a base path given by the config
    checkpoint_path = hydra.utils.to_absolute_path(
        f"./{cfg.logging_config.model_path}/{run_id}"
    )

    # Load the checkpoint's config
    if cfg.train_setup["checkpoint_to_restore"] is not None:
        # TODO: We set the restored config's checkpoint_to_restore to itself
        # Because that restored config is used from now on. This is a hack.
        checkpoint_to_restore = cfg.train_setup["checkpoint_to_restore"]
        # Load the checkpoint's config and update the run_id and checkpoint path
        cfg = OmegaConf.create(
            checkpointing.load_config_from_checkpoint(
                cfg.train_setup["checkpoint_to_restore"]
            )
        )
        cfg.train_setup["checkpoint_to_restore"] = checkpoint_to_restore
        checkpoint_path = Path(checkpoint_to_restore)
        run_id = checkpoint_path.name

    # Convert config to dict TODO: do we need this?
    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=cfg.train_setup["checkpoint_max_to_keep"],
        keep_period=cfg.train_setup["checkpoint_keep_period"],
        step_prefix="PPONetwork",
    )

    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    logging.info(f"run_id: {run_id}")
    logging.info(f"Training checkpoint path: {checkpoint_path}")

    env_args = cfg.env_config["env_args"]
    env_rewards = cfg.env_config["reward_weights"]
    train_config = cfg.train_setup["train_config"]
    if cfg.env_config.get("task_type", "tracking") == "reaching":
        agent_config = cfg["reacher_config"]
    else:
        agent_config = cfg["walker_config"]
    traj_config = cfg["reference_config"]

    logging.info(f"Loading data: {cfg.data_path}")

    # Create agent and reward config based on task type
    task_type = cfg.env_config.get("task_type", "tracking")
    logging.info(f"Task type: {task_type}")
    
    if task_type == "reaching":
        # Use reacher for reaching tasks
        if "reacher_name" not in cfg.env_config:
            raise ValueError("reacher_name must be specified for reaching tasks")
        reacher_name = cfg.env_config.reacher_name
        if reacher_name not in _REACHERS:
            raise ValueError(f"Unknown reacher: {reacher_name}")
        agent = _REACHERS[reacher_name](**agent_config)
        if "joint_penalty_weight" not in env_rewards:
            env_rewards["joint_penalty_weight"] = 1.0
        if "joint_penalty_threshold" not in env_rewards:
            env_rewards["joint_penalty_threshold"] = 1.0
        
        reward_config = RewardConfigReach(**env_rewards)
        logging.info(f"Created reacher: {reacher_name}")
        
        # Load reaching data
        train_clips = load_reaching_data(cfg.data_path)
        logging.info(f"Loaded reaching data: {train_clips.joints.shape=}, {train_clips.body_positions.shape=}")
    else:
        # Use walker for tracking tasks
        if "walker_name" not in cfg.env_config:
            raise ValueError("walker_name must be specified for tracking tasks")
        walker_name = cfg.env_config.walker_name
        if walker_name not in _WALKERS:
            raise ValueError(f"Unknown walker: {walker_name}")
        agent = _WALKERS[walker_name](**agent_config)
        reward_config = RewardConfig(**env_rewards)
        logging.info(f"Created walker: {walker_name}")
        
        # Load tracking data
        train_clips = load.load_data(cfg.data_path)
        logging.info(f"Loaded tracking data: {train_clips.position.shape=}, {train_clips.quaternion.shape=}")

    # setup test set evaluator
    if cfg.train_setup["train_test_split_info"] is not None:
        if task_type == "reaching":
            all_clips = load_reaching_data(cfg.data_path)
        else:
            all_clips = load.load_data(cfg.data_path)
        # load the train/test split data from the disk
        with open(cfg.train_setup["train_test_split_info"], "r") as f:
            split_info = json.load(f)
        test_idx = split_info["test"]
        if cfg.train_setup["train_subset_ratio"] is None:
            train_idx = split_info["train"]
        else:
            train_idx = split_info["train_subset"][
                f"{cfg.train_setup['train_subset_ratio']:.2f}"
            ]
        test_clips = load.select_clips(all_clips, test_idx)
        train_clips = load.select_clips(all_clips, train_idx)
        if task_type == "reaching":
            logging.info(
                f"Train set length:{train_clips.joints.shape[0]}, Test set length: {test_clips.joints.shape[0]}"
            )
        else:
            logging.info(
                f"Train set length:{train_clips.position.shape[0]}, Test set length: {test_clips.quaternion.shape[0]}"
            )
        test_env = create_environment(cfg, agent, reward_config, test_clips, **env_args, **traj_config)
    elif cfg.train_setup["train_subset_ratio"] is not None:
        if task_type == "reaching":
            all_clips = load_reaching_data(cfg.data_path)
        else:
            all_clips = load.load_data(cfg.data_path)
        train_clips, test_clips = load.generate_train_test_split(
            all_clips, test_ratio=1 - cfg.train_setup["train_subset_ratio"]
        )
        if task_type == "reaching":
            logging.info(
                f"Train set length:{train_clips.joints.shape[0]}, Test set length: {test_clips.joints.shape[0]}"
            )
        else:
            logging.info(
                f"Train set length:{train_clips.position.shape[0]}, Test set length: {test_clips.quaternion.shape[0]}"
            )
        test_env = create_environment(cfg, agent, reward_config, test_clips, **env_args, **traj_config)
    else:
        # eval with the train set
        test_env = None

    env = create_environment(cfg, agent, reward_config, train_clips, **env_args, **traj_config)

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    episode_length = (
        traj_config.clip_length
        - traj_config.random_init_range
        - traj_config.traj_length
    ) * env._steps_for_cur_frame
    print(f"episode_length {episode_length}")
    logging.info(f"episode_length {episode_length}")

    if cfg.train_setup.train_config.use_lstm:
        print("Using LSTM Pipeline Now")
        ppo = lstm_ppo
        ppo_networks = lstm_ppo_networks
        if task_type == "reaching":
            render_wrapper = wrappers.RenderRolloutWrapperTrackingLSTM  # You may need to create a reaching-specific wrapper
        else:
            render_wrapper = wrappers.RenderRolloutWrapperTrackingLSTM
        network_factory = functools.partial(
            ppo_networks.make_intention_ppo_networks,
            intention_latent_size=cfg.network_config.intention_size,
            hidden_state_size=cfg.network_config.hidden_state_size,
            hidden_layer_num=cfg.network_config.hidden_layer_num,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )

    else:
        print("Using MLP Pipeline Now")
        ppo = mlp_ppo
        ppo_networks = mlp_ppo_networks
        if task_type == "reaching":
            render_wrapper = wrappers.RenderRolloutWrapperMulticlipTracking  # You may need to create a reaching-specific wrapper
        else:
            render_wrapper = wrappers.RenderRolloutWrapperMulticlipTracking
        network_factory = functools.partial(
            ppo_networks.make_intention_ppo_networks,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )

    train_fn = functools.partial(
        ppo.train,
        **train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        kl_weight=cfg.network_config.kl_weight,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=cfg.network_config.kl_schedule,
        eval_env_test_set=test_env,
        freeze_decoder=(
            False
            if "freeze_decoder" not in cfg.train_setup
            else cfg.train_setup.freeze_decoder
        ),
    )

    run_id = f"{cfg.env_config.env_name}_{cfg.env_config.task_name}_{cfg.logging_config.algo_name}_{run_id}"
    wandb.init(
        project=cfg.logging_config.project_name,
        config=OmegaConf.to_container(cfg, resolve=True, structured_config_mode=True),
        notes=f"",
        id=run_id,
        resume="allow",
        group=cfg.logging_config.group_name,
    )

    def wandb_progress(num_steps, metrics):
        metrics["num_steps_thousands"] = num_steps
        wandb.log(metrics, commit=False)

    if cfg.train_setup.train_config.use_lstm:
        rollout_env = render_wrapper(
            env=env,
            lstm_features=cfg.network_config.hidden_state_size,
            hidden_layer_num=cfg.network_config.hidden_layer_num,
        )
    else:
        rollout_env = render_wrapper(env=env)

    # define the jit reset/step functions
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    renderer, mj_model, mj_data, scene_option = render.make_rollout_renderer(cfg)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
        renderer,
        mj_model,
        mj_data,
        scene_option,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,  # fill in the rest in training
    )


def create_environment(cfg, agent, reward_config, clips, **kwargs):
    """Create environment based on task type."""
    task_type = cfg.env_config.get("task_type", "tracking")
    
    # Add the use_emg parameter
    use_emg = cfg.env_config.get("use_emg", False)
    
    if task_type == "reaching":
        # For reaching tasks, use reaching environments
        return envs.get_environment(
            env_name=cfg.env_config.env_name,
            reference_clip=clips,
            reacher=agent,
            reward_config=reward_config,
            use_emg=use_emg,  # Add this line
            **kwargs,
        )
    else:
        # For tracking tasks, use tracking environments
        return envs.get_environment(
            env_name=cfg.env_config.env_name,
            reference_clip=clips,
            walker=agent,
            reward_config=reward_config,
            **kwargs,
        )


if __name__ == "__main__":
    main()