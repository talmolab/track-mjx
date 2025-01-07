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
import jax
import wandb
from brax import envs
import pickle
import warnings
import functools

from track_mjx.environment.task.multi_clip_tracking import RodentMultiClipTracking
from track_mjx.environment.task.single_clip_tracking import RodentTracking
from track_mjx.io import preprocess as preprocessing  # the pickle file needs it
from track_mjx.agent import nnx_ppo 
from track_mjx.agent import nnx_ppo_network
from track_mjx.agent.logging import policy_params_fn
from track_mjx.environment.walker.rodent import Rodent

FLAGS = flags.FLAGS
warnings.filterwarnings("ignore", category=DeprecationWarning)

# jax.config.update("jax_debug_nans", True)


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

    # Generates a completely random UUID (version 4)
    run_id = uuid.uuid4()
    model_path = hydra.utils.to_absolute_path(f"./{cfg.logging_config.model_path}/{run_id}")

    wandb.init(
        project=cfg.logging_config.project_name,
        config=dict(cfg),
        notes=f"clip_id: {cfg.logging_config.clip_id}",
    )
    wandb.run.name = f"{cfg.env_config.env_name}_{cfg.env_config.task_name}_{cfg.logging_config.algo_name}_{run_id}"  # type: ignore

    def wandb_progress(num_steps, metrics):
        metrics["num_steps"] = num_steps
        wandb.log(metrics)

    train_cfg = cfg.train_config
    network_cfg = cfg.network_config

    network_factory = functools.partial(
        nnx_ppo_network.make_intention_ppo_networks,
        encoder_layers=network_cfg.encoder_layer_sizes,
        decoder_layers=network_cfg.decoder_layer_sizes,
        value_layers=network_cfg.critic_layer_sizes,
    )

    ppo_cfg = nnx_ppo_network.PPOTrainConfig(
        num_timesteps=1000000,
        episode_length=250,
        action_repeat=1,
        num_envs=train_cfg.num_envs,
        network_factory=network_factory,
        encoder_layers=network_cfg.encoder_layer_sizes,
        decoder_layers=network_cfg.decoder_layer_sizes,
        value_layer_sizes=network_cfg.critic_layer_sizes,
        num_eval_envs=128,
        learning_rate=train_cfg.learning_rate,
        entropy_cost=train_cfg.entropy_cost,
        kl_weight=network_cfg.kl_weight,
        discounting=train_cfg.discounting,
        seed=0,
        unroll_length=train_cfg.unroll_length,
        batch_size=train_cfg.batch_size,
        num_minibatches=train_cfg.num_minibatches,  # minibatch_size=batch_size/num_minibatches
        num_updates_per_batch=train_cfg.num_updates_per_batch,
        num_evals=1,
        num_resets_per_eval=0,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.3,
        gae_lambda=0.95,
        deterministic_eval=False,
        normalize_advantage=True,
        progress_fn=wandb_progress,
        eval_env=None,
        checkpoint_logdir=model_path,
    )

    nnx_ppo.train(env, ppo_cfg)


if __name__ == "__main__":
    main()
