"""
Only inference for track-mjx. Load the config file, create environments, initialize network, and start inference.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=True "
)

import hydra
from omegaconf import DictConfig, OmegaConf
from absl import flags
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import sys

import orbax.checkpoint as orbax_cp

from brax import envs
from brax.io import model
from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.task.reward import RewardConfig
from track_mjx.io import preprocess as preprocessing
from track_mjx.agent import custom_ppo_networks

FLAGS = flags.FLAGS
warnings.filterwarnings("ignore", category=DeprecationWarning)

@hydra.main(version_base=None, config_path="config", config_name="fly-mc-intention")
def main(cfg: DictConfig):
    """
    Pure inference script. Loads a trained model checkpoint via Orbax,
    creates an environment, and runs a rollout for evaluation or visualization.
    """
    
    envs.register_environment("rodent_single_clip", SingleClipTracking)
    envs.register_environment("rodent_multi_clip", MultiClipTracking)
    envs.register_environment("fly_multi_clip", MultiClipTracking)

    try:
        n_devices = jax.device_count(backend="gpu")
        print(f"Using {n_devices} GPUs for inference")
    except:
        print("Not using GPUs for inference")

    sys.modules["preprocessing"] = preprocessing
    with open(hydra.utils.to_absolute_path(cfg.data_path), "rb") as file:
        reference_clip = pickle.load(file)
        
    # config files
    env_cfg = hydra.compose(config_name="fly-mc-intention")
    env_cfg = OmegaConf.to_container(env_cfg, resolve=True)
    env_args = cfg.env_config["env_args"]
    env_rewards = cfg.env_config["reward_weights"]
    walker_config = cfg["walker_config"]
    traj_config = cfg["reference_config"]

    walker_map = {
        "rodent": Rodent,
        "fly": Fly,
    }
    walker_class = walker_map[env_cfg["walker_type"]]
    walker = walker_class(**walker_config)

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
        penalty_pos_distance_scale=jnp.array(env_rewards.penalty_pos_distance_scale),
    )

    env = envs.get_environment(
        env_name=cfg.env_config.env_name,
        reference_clip=reference_clip,
        walker=walker,
        reward_config=reward_config,
        **env_args,
        **traj_config,
    )
    print("Environment created.")

    # Orbax: Load checkpoint for inference
    # checkpoint_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    # ckpt_options = orbax_cp.CheckpointManagerOptions(max_to_keep=3, create=False)
    # orbax_checkpointer = orbax_cp.PyTreeCheckpointer()
    # ckpt_manager = orbax_cp.CheckpointManager(
    #     checkpoint_dir, orbax_checkpointer, ckpt_options
    # )

    # # If you want latest checkpoint or a specific step
    # if cfg.inference_step == "latest":
    #     step_to_restore = ckpt_manager.latest_step()
    # else:
    #     step_to_restore = int(cfg.inference_step)

    # restored_state = ckpt_manager.restore(step=step_to_restore)
    # inference_params = restored_state["params"]
    # print(f"Loaded parameters from step {step_to_restore} for inference.")
    
    # currently just loading it with brax.model first
    final_save_path = hydra.utils.to_absolute_path(cfg.inference_params_path)
    print(f"Loading trained parameters from: {final_save_path}")
    inference_params = model.load_params(final_save_path)
    print("Parameters successfully loaded!")

    # Run rollout in environment
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng=rng)
    total_reward = 0.0
    
    # Build inference function
    policy_networks = custom_ppo_networks.make_intention_ppo_networks(
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        action_size=env.action_size,
        observation_size=state.obs.shape[-1],
        reference_obs_size=int(state.info["reference_obs_size"]),
    )
    make_policy_fn = custom_ppo_networks.make_inference_fn(policy_networks)
    policy = make_policy_fn(inference_params, deterministic=cfg.eval_config.deterministic)
    print('Policy created successfuly!')

    def select_action_fn(policy, obs, key):
        """
        Select an action using the policy function.

        Args:
            policy: The policy function created by make_policy.
            obs: Current observation from the environment.
            key: JAX PRNGKey.

        Returns:
            action: The action to take in the environment.
            extra: Additional info from the policy (e.g., log_prob).
        """
        action, extra = policy(obs, key)
        return action, extra

    def run_inference_and_record(env, policy, max_steps=cfg.eval_config.num_eval_steps):
        """
        Run one rollout with the given env and policy, storing all data.

        Args:
          env: Brax environment already created and ready to go.
          policy: Policy function to select actions.
          max_steps: Maximum number of steps to run (or until 'done').

        Returns:
          A dictionary containing arrays of observations, actions, rewards, etc., for the entire rollout.
        """
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_extra = []

        rng = jax.random.PRNGKey(2)
        state = env.reset(rng=rng)

        for step_i in range(max_steps):
            obs = state.obs
            all_obs.append(np.array(obs))

            rng, key = jax.random.split(rng)
            action, extra = select_action_fn(policy, obs, key)
            all_actions.append(np.array(action))
            all_extra.append(extra)

            state = env.step(state, action)
            all_rewards.append(float(state.reward))
            all_dones.append(bool(state.done))

            if state.done:
                print(f"Episode ended at step {step_i + 1} with reward {float(state.reward)}")
                break

        rollout_data = {
            "observations": np.stack(all_obs, axis=0),      # Shape: [T, obs_dim]
            "actions":      np.stack(all_actions, axis=0),   # Shape: [T, act_dim]
            "rewards":      np.array(all_rewards),           # Shape: [T]
            "dones":        np.array(all_dones),             # Shape: [T]
            "extra":        all_extra,                        # List of dicts
        }
        return rollout_data

    print("Starting inference rollout...")
    trajectory = run_inference_and_record(env, policy, max_steps=cfg.eval_config.num_eval_steps)
    print("Inference rollout completed.")

    print("Collected observations shape:", trajectory["observations"].shape)
    print("Collected actions shape:", trajectory["actions"].shape)
    print("Collected rewards shape:", trajectory["rewards"].shape)
    print("Collected dones shape:", trajectory["dones"].shape)
    print(f"Finished inference rollout. Total reward: {total_reward}")

if __name__ == "__main__":
    main()