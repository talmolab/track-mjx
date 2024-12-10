"""
Entries point for track-mjx. Load the config file, create environments, initialize network, and start training. Uisng nnx train the ppo algorihtm
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True "
)

from absl import flags
import hydra
from omegaconf import DictConfig
import uuid

import functools
import jax
from typing import Dict
import wandb
import imageio
from brax import envs
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale

import track_mjx.agent.custom_ppo as ppo
from track_mjx.agent import custom_ppo
from brax.io import model
import numpy as np
import pickle
import warnings
from jax import numpy as jp

from track_mjx.environment.task.multi_clip_tracking import RodentMultiClipTracking
from track_mjx.environment.task.single_clip_tracking import RodentTracking
from track_mjx.io.preprocess.mjx_preprocess import process_clip_to_train
from track_mjx.io import preprocess as preprocessing  # the pickle file needs it
from track_mjx.environment import custom_wrappers

# nnx related imports
from track_mjx.agent import nnx_ppo 
from track_mjx.agent import nnx_ppo_network
from track_mjx.agent.logging import policy_params_fn


from track_mjx.environment.walker.rodent import Rodent

FLAGS = flags.FLAGS
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_path="config", config_name="rodent-mc-intention")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""

    try:
        n_devices = jax.device_count(backend="gpu")
        print(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        print("Not using GPUs")

    flags.DEFINE_enum("solver", "cg", ["cg", "newton"], "constraint solver")
    flags.DEFINE_integer("iterations", 4, "number of solver iterations")
    flags.DEFINE_integer("ls_iterations", 4, "number of linesearch iterations")

    envs.register_environment("single clip", RodentTracking)
    envs.register_environment("multi clip", RodentMultiClipTracking)
    
    input_data_path = hydra.utils.to_absolute_path(cfg.data_path)
    print(f"Loading data: {input_data_path}")
    with open(input_data_path, "rb") as file:
        # Use pickle.load() to load the data from the file
        reference_clip = pickle.load(file)

    # TODO (Kevin): add this as a yaml config
    walker = Rodent
    
    # instantiate the environment
    env = envs.get_environment(
        cfg.env_config.env_name,
        reference_clip=reference_clip,
        walker=walker,
        torque_actuators=cfg.env_config.torque_actuators,
        solver=cfg.env_config.solver,
        iterations=cfg.env_config.iterations,
        ls_iterations=cfg.env_config.ls_iterations,
        too_far_dist=cfg.env_config.reward_weights.too_far_dist,
        bad_pose_dist=cfg.env_config.reward_weights.bad_pose_dist,
        bad_quat_dist=cfg.env_config.reward_weights.bad_quat_dist,
        ctrl_cost_weight=cfg.env_config.reward_weights.ctrl_cost_weight,
        ctrl_diff_cost_weight=cfg.env_config.reward_weights.ctrl_diff_cost_weight,
        pos_reward_weight=cfg.env_config.reward_weights.pos_reward_weight,
        quat_reward_weight=cfg.env_config.reward_weights.quat_reward_weight,
        joint_reward_weight=cfg.env_config.reward_weights.joint_reward_weight,
        angvel_reward_weight=cfg.env_config.reward_weights.angvel_reward_weight,
        bodypos_reward_weight=cfg.env_config.reward_weights.bodypos_reward_weight,
        endeff_reward_weight=cfg.env_config.reward_weights.endeff_reward_weight,
        healthy_z_range=tuple(cfg.env_config.reward_weights.healthy_z_range),
        physics_steps_per_control_step=cfg.env_config.physics_steps_per_control_step,
    )
    
    ppo_cfg = nnx_ppo_network.PPOTrainConfig()
    
    nnx_ppo.train(env, ppo_cfg)
    
    
if __name__ == "__main__":
    main()