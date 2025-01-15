"""
Entries point for track-mjx. Load the config file, create environments, initialize network, and start training.
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True "
)
os.environ["PYOPENGL_PLATFORM"] = "egl"

from absl import flags
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid

from pathlib import Path

import functools
import jax
import wandb
from brax import envs
import orbax.checkpoint as ocp
import track_mjx.agent.custom_ppo as ppo
from track_mjx.agent import custom_ppo
import numpy as np
import pickle
import warnings
from jax import numpy as jp

from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.io.preprocess.mjx_preprocess import process_clip_to_train
from track_mjx.io import preprocess as preprocessing  # the pickle file needs it
from track_mjx.environment import custom_wrappers
from track_mjx.agent import custom_ppo_networks
from track_mjx.agent.logging import setup_training_logging
from track_mjx.environment.walker.rodent import Rodent
import logging
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.task.reward import RewardConfig

FLAGS = flags.FLAGS
warnings.filterwarnings("ignore", category=DeprecationWarning)



@hydra.main(version_base=None, config_path="config", config_name="fly-mc-intention")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    flags.DEFINE_enum("solver", "cg", ["cg", "newton"], "constraint solver")
    flags.DEFINE_integer("iterations", 4, "number of solver iterations")
    flags.DEFINE_integer("ls_iterations", 4, "number of linesearch iterations")

    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    logging.info(f"Configs: {OmegaConf.to_container(cfg, resolve=True)}")
    env_cfg = OmegaConf.to_container(env_cfg, resolve=True)
    env_args = cfg.env_config["env_args"]
    env_rewards = cfg.env_config["reward_weights"]
    train_config = cfg.train_setup["train_config"]
    walker_config = cfg["walker_config"]
    traj_config = cfg["reference_config"]

    # TODO: Fix this dependency issue
    import sys

    sys.modules["preprocessing"] = preprocessing
    # TODO(Scott): move this to track_mjx.io module
    input_data_path = hydra.utils.to_absolute_path(cfg.data_path)
    logging.info(f"Loading data: {input_data_path}")
    with open(input_data_path, "rb") as file:
        reference_clip = pickle.load(file)

    walker_map = {
        "rodent": Rodent,
        "fly": Fly,
    }

    walker_class = walker_map[env_cfg["walker_type"]]
    walker = walker_class(**walker_config)

    # didn't use args** since penalty_pos_distance_scale need conversion
    reward_config = RewardConfig(
        too_far_dist=env_rewards.too_far_dist,
        bad_pose_dist=env_rewards.bad_pose_dist,
        bad_quat_dist=env_rewards.bad_quat_dist,
        ctrl_cost_weight=env_rewards.ctrl_cost_weight,
        ctrl_diff_cost_weight=env_rewards.ctrl_diff_cost_weight,
        pos_reward_weight=env_rewards.pos_reward_weight,
        quat_reward_weight=env_rewards.quat_reward_weight,
        joint_reward_weight=env_rewards.joint_reward_weight,
        angvel_reward_weight=env_rewards.angvel_reward_weight,
        bodypos_reward_weight=env_rewards.bodypos_reward_weight,
        endeff_reward_weight=env_rewards.endeff_reward_weight,
        healthy_z_range=env_rewards.healthy_z_range,
        pos_reward_exp_scale=env_rewards.pos_reward_exp_scale,
        quat_reward_exp_scale=env_rewards.quat_reward_exp_scale,
        joint_reward_exp_scale=env_rewards.joint_reward_exp_scale,
        angvel_reward_exp_scale=env_rewards.angvel_reward_exp_scale,
        bodypos_reward_exp_scale=env_rewards.bodypos_reward_exp_scale,
        endeff_reward_exp_scale=env_rewards.endeff_reward_exp_scale,
        penalty_pos_distance_scale=jp.array(env_rewards.penalty_pos_distance_scale),
    )

    # Automatically match dict keys and func needs
    env = envs.get_environment(
        env_name=cfg.env_config.env_name,
        reference_clip=reference_clip,
        walker=walker,
        reward_config=reward_config,
        **env_args,
        **traj_config,
    )

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    # Will work on not hardcoding these values later
    episode_length = (250 - 50 - 5) * env._steps_for_cur_frame
    logging.info(f"episode_length {episode_length}")

    # Generates a completely random UUID (version 4), take the first 8 characters
    run_id = uuid.uuid4().hex[:6]
    model_path = f"./{cfg.logging_config.model_path}/{cfg.env_config.walker_name}_{cfg.data_path}_{run_id}"
    model_path = hydra.utils.to_absolute_path(model_path)
    logging.info(f"Model Checkpoint Path: {model_path}")

    # initialize orbax checkpoint manager
    # TODO (Scott): add the checkpoint parameter to config file.
    mgr_options = ocp.CheckpointManagerOptions(create=True, max_to_keep=3, keep_period=2, step_prefix="PPONetwork")
    ckpt_mgr = ocp.CheckpointManager(model_path, options=mgr_options)


    train_fn = functools.partial(
        custom_ppo.train,
        **train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        episode_length=episode_length,
        kl_weight=cfg.network_config.kl_weight,
        network_factory=functools.partial(
            custom_ppo_networks.make_intention_ppo_networks,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        ),
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
    )

    wandb.init(
        project=cfg.logging_config.project_name,
        config=OmegaConf.to_container(cfg, resolve=True, structured_config_mode=True),
        notes=f"clip_id: {cfg.logging_config.clip_id}",
    )
    wandb.run.name = f"{cfg.env_config.env_name}_{cfg.env_config.task_name}_{cfg.logging_config.algo_name}_{run_id}"

    def wandb_progress(num_steps, metrics):
        metrics["num_steps"] = num_steps
        wandb.log(metrics, commit=False)

    def policy_params_fn(current_step, make_policy, params, policy_params_fn_key):
        """wrapper function that pass in some args that this file has"""
        return setup_training_logging(
            current_step,
            make_policy,
            params,
            policy_params_fn_key,
            cfg=cfg,
            env=env,
            wandb=wandb,
            model_path=model_path,
        )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,
    )


if __name__ == "__main__":
    main()
