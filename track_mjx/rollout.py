"""
Only inference for track-mjx. Load the config file, create environments, initialize network, and start inference.

Saving out full state for now

Ensure jax==0.4.35 mujoco==3.2.4 jaxlib==0.4.35
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
import imageio
import mujoco
from pathlib import Path

import orbax.checkpoint as ocp

from brax import envs
from brax.io import model
from brax.envs.base import PipelineEnv
from brax.training.acme import running_statistics, specs

from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.task.single_clip_tracking import SingleClipTracking
from track_mjx.environment.walker.rodent import Rodent
from track_mjx.environment.walker.fly import Fly
from track_mjx.environment.task.reward import RewardConfig
from track_mjx.io import preprocess as preprocessing
from track_mjx.agent import custom_ppo_networks
from track_mjx.agent.custom_ppo import TrainingState
from track_mjx.agent import custom_losses as ppo_losses
from track_mjx.agent import custom_ppo, custom_ppo_networks
from track_mjx.environment import custom_wrappers
import optax

import matplotlib.pyplot as plt
from PIL import Image

FLAGS = flags.FLAGS
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(
    version_base=None, config_path="config", config_name="rodent-full-intention"
)
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
        print(f"Loading data: {cfg.data_path}")

    # config files
    env_cfg = OmegaConf.to_container(cfg, resolve=True)
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
    use_only_mjdata = cfg.eval_config.use_only_mjdata

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

    eval_env = envs.get_environment(
        env_name=cfg.env_config.env_name,
        reference_clip=reference_clip,
        walker=walker,
        reward_config=reward_config,
        **env_args,
        **traj_config,
    )

    seed = 42

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 3)

    v_randomization_fn = None

    if isinstance(eval_env, envs.Env):
        wrap_for_training = custom_wrappers.wrap
    else:
        wrap_for_training = custom_wrappers.wrap

    episode_length = (
        traj_config.clip_length
        - traj_config.random_init_range
        - traj_config.traj_length
    ) * eval_env._steps_for_cur_frame
    print(f"episode_length {episode_length}")

    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=v_randomization_fn,
    )

    print("Environment created.")

    # final_save_path = hydra.utils.to_absolute_path(cfg.inference_params_path)
    # print(f"Loading trained parameters from: {final_save_path}")
    # inference_params = model.load_params(final_save_path)
    # print("Parameters successfully loaded!")

    # Run rollout in environment
    reset_fn = eval_env.reset
    key_envs = jax.random.split(key_env, 1)
    state = reset_fn(key_envs)

    normalize = running_statistics.normalize
    # Build inference function
    policy_networks = custom_ppo_networks.make_intention_ppo_networks(
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        action_size=eval_env.action_size,
        observation_size=state.obs.shape[-1],
        reference_obs_size=int(state.info["reference_obs_size"]),
        preprocess_observations_fn=normalize,
    )
    # make_policy_fn = custom_ppo_networks.make_inference_fn(policy_networks)
    # policy = make_policy_fn(
    #     inference_params, deterministic=cfg.eval_config.deterministic
    # )
    optimizer = optax.adam(learning_rate=1e-4)

    init_params = ppo_losses.PPONetworkParams(
        policy=policy_networks.policy_network.init(key_policy),
        value=policy_networks.value_network.init(key_value),
    )

    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(
            specs.Array(state.obs.shape[-1:], jnp.dtype("float32"))
        ),
        env_steps=0,
    )

    abstract_policy = (training_state.normalizer_params, training_state.params.policy)

    # orbax: restore the whole model (both policy module, and training state)
    checkpoint_dir = hydra.utils.to_absolute_path(cfg.checkpoint_dir)
    options = ocp.CheckpointManagerOptions(step_prefix="PPONetwork")
    with ocp.CheckpointManager(
        checkpoint_dir,
        options=options,
    ) as mngr:
        print(f"latest checkpoint step: {mngr.latest_step()}")
        policy = mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    print("Policy created successfuly!")

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
        all_state = []
        all_summed_pos_distance = []
        all_quat_distance = []
        all_joint_distance = []

        rollout_data_array, state_data_array = [], []

        rng = jax.random.PRNGKey(2)
        state = env.reset(rng=rng)

        for step_i in range(max_steps):
            print(f"Currently in step {step_i}")
            obs = state.obs
            all_obs.append(np.array(obs))

            rng, key = jax.random.split(rng)
            action, _ = select_action_fn(policy, obs, key)
            all_actions.append(np.array(action))

            state = env.step(state, action)
            all_rewards.append(float(state.reward))
            all_dones.append(bool(state.done))
            all_state.append(state)

            info = state.info
            all_summed_pos_distance.append(info.get("summed_pos_distance", 0.0))
            all_quat_distance.append(info.get("quat_distance", 0.0))
            all_joint_distance.append(info.get("joint_distance", 0.0))

            rollout_data_array.append((obs, action))
            state_data_array.append(state)

            if state.done:
                print(
                    f"Episode ended at step {step_i + 1} with reward {float(state.reward)}"
                )
                break

        inference_data = {
            "observations": np.stack(all_obs, axis=0),  # Shape: [T, obs_dim]
            "actions": np.stack(all_actions, axis=0),  # Shape: [T, act_dim]
            "rewards": np.array(all_rewards),
            "dones": np.array(all_dones),
            "states": all_state,
            "summed_pos_distance": np.array(all_summed_pos_distance),
            "quat_distance": np.array(all_quat_distance),
            "joint_distance": np.array(all_joint_distance),
        }

        return inference_data, rollout_data_array, state_data_array

    def render_rollout(
        rollout: dict,
        cfg: DictConfig,
        env: PipelineEnv,
        video_save_path: str,
        num_steps: int,
    ):
        """
        Render the rollout using MuJoCo and log the video to wandb.

        Args:
            rollout: The rollout data containing observations, actions, etc.
            cfg: Configuration dictionary.
            env: The Brax environment.
            video_save_path: Path to save the video.
            num_steps: Current step number for naming the video.
        """
        env_config = cfg.env_config
        walker_config = cfg.walker_config
        walker_type = cfg.walker_type

        pair_render_xml_path = env.walker._pair_rendering_xml_path
        _XML_PATH = (
            Path(__file__).resolve().parent
            / "environment"
            / "walker"
            / pair_render_xml_path
        )

        spec = mujoco.MjSpec()
        spec = spec.from_file(str(_XML_PATH))

        # Scale geoms and positions
        for geom in spec.geoms:
            if geom.size is not None:
                geom.size *= walker_config.rescale_factor
            if geom.pos is not None:
                geom.pos *= walker_config.rescale_factor

        mj_model = spec.compile()

        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }["cg"]

        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        state_data = mujoco.MjData(mj_model)

        site_id = [
            mj_model.site(i).id
            for i in range(mj_model.nsite)
            if "-0" in mj_model.site(i).name
        ]
        site_id = [site_id[0]]

        for id in site_id:
            mj_model.site(id).rgba = [1, 0, 0, 1]

        if walker_type == "rodent":
            for i in range(mj_model.ngeom):
                geom_name = mj_model.geom(i).name
                if "-1" in geom_name:  # ghost
                    mj_model.geom(i).rgba = [
                        1,
                        1,
                        1,
                        0.5,
                    ]  # White color, 50% transparent
                elif "-0" in geom_name:  # agent
                    mj_model.geom(i).rgba = [
                        0.3,
                        0.6,
                        1.0,
                        1.0,
                    ]  # Light blue color, fully opaque

        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
        renderer = mujoco.Renderer(mj_model, height=512, width=512)
        video_path = f"{video_save_path}/{num_steps}_inference_rollout.mp4"

        with imageio.get_writer(video_path, fps=env_config.render_fps) as video:

            qposes_rollout = np.array(
                [state.pipeline_state.qpos for state in rollout["states"]]
            )
            ref_traj = env._get_reference_clip(rollout["states"][0].info)
            print(f"clip_id:{rollout['states'][0].info}")

            qposes_ref = np.repeat(
                np.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints]),
                env._steps_for_cur_frame,
                axis=0,
            )

            for step_idx, (qpos1, qpos2) in enumerate(zip(qposes_rollout, qposes_ref)):
                # Ensure qpos dimensions match
                total_qpos_size = len(qpos1) + len(qpos2)
                if total_qpos_size != mj_model.nq:
                    raise ValueError(
                        f"Combined qpos sizes do not match MuJoCo model's nq: {total_qpos_size} vs {mj_model.nq}"
                    )
                state_data.qpos = np.append(qpos1, qpos2)
                mujoco.mj_forward(mj_model, state_data)
                renderer.update_scene(
                    state_data,
                    camera=env_config.render_camera_name,
                    scene_option=scene_option,
                )
                pixels = renderer.render()

                if not use_only_mjdata:
                    rewards = rollout["rewards"]
                    summed_pos_distance = rollout["summed_pos_distance"]
                    quat_distance = rollout["quat_distance"]
                    joint_distance = rollout["joint_distance"]

                    # Create a plot of cumulative rewards up to the current step
                    fig, axs = plt.subplots(3, 1, figsize=(6, 9))

                    # Reward per Step
                    axs[0].plot(
                        rewards[: step_idx + 1], label="Reward per Step", color="blue"
                    )
                    axs[0].set_xlabel("Step")
                    axs[0].set_ylabel("Reward")
                    axs[0].set_title("Reward over Time")
                    axs[0].legend()
                    axs[0].grid(True)

                    # Summed Positional Distance
                    axs[1].plot(
                        summed_pos_distance[: step_idx + 1],
                        label="Summed Positional Distance",
                        color="green",
                    )
                    axs[1].set_xlabel("Step")
                    axs[1].set_ylabel("Summed Positional Distance")
                    axs[1].set_title("Summed Positional Distance over Time")
                    axs[1].legend()
                    axs[1].grid(True)

                    # Quaternion and Joint Distances
                    axs[2].plot(
                        quat_distance[: step_idx + 1],
                        label="Quaternion Distance",
                        color="red",
                    )
                    axs[2].plot(
                        joint_distance[: step_idx + 1],
                        label="Joint Distance",
                        color="purple",
                    )
                    axs[2].set_xlabel("Step")
                    axs[2].set_ylabel("Distance")
                    axs[2].set_title("Quaternion and Joint Distances over Time")
                    axs[2].legend()
                    axs[2].grid(True)

                    plt.tight_layout()
                    # Save the plot to a temporary in-memory buffer
                    plot_path = f"{video_save_path}/temp_plot_{num_steps}.png"
                    plt.savefig(plot_path)
                    plt.close()

                    plot_image = Image.open(plot_path)
                    plot_image = plot_image.resize((512, 512))

                    # Convert simulation frame to PIL Image
                    sim_image = Image.fromarray(pixels)
                    sim_image = sim_image.resize(
                        (512, 512)
                    )  # Ensure same height as plot

                    # Combine simulation image and plot image side by side
                    combined_width = sim_image.width + plot_image.width
                    combined_image = Image.new(
                        "RGB", (combined_width, sim_image.height)
                    )
                    combined_image.paste(sim_image, (0, 0))
                    combined_image.paste(plot_image, (sim_image.width, 0))

                    combined_frame = np.array(combined_image)
                    video.append_data(combined_frame)

                    os.remove(plot_path)

                else:
                    video.append_data(pixels)

    if not use_only_mjdata:
        print("Starting inference rollout...")
        trajectory, rollout_data_array, state_data_array = run_inference_and_record(
            eval_env, policy, max_steps=cfg.eval_config.num_eval_steps
        )
        print("Inference rollout completed.")
        print("Collected observations shape:", trajectory["observations"].shape)
        print("Collected actions shape:", trajectory["actions"].shape)
        print("Collected rewards shape:", trajectory["rewards"].shape)
        print("Collected dones shape:", trajectory["dones"].shape)

        save_path = Path(hydra.utils.to_absolute_path(cfg.eval_config.save_eval_path))
        save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / "state_data.npy", np.array(state_data_array))
        np.savez(
            save_path / "rollout_data.npz",
            observations=np.array([data[0] for data in rollout_data_array]),
            actions=np.array([data[1] for data in rollout_data_array]),
        )
        print(f"Data saved at {save_path}.")

        print("Rendering rollout...")
        video_save_path = hydra.utils.to_absolute_path(cfg.eval_config.save_eval_path)
        os.makedirs(video_save_path, exist_ok=True)
        render_rollout(
            trajectory,
            cfg,
            eval_env,
            video_save_path,
            num_steps=trajectory.get("step", 0),
        )
    else:
        print("Loading data for simulatuon")
        save_path = Path(hydra.utils.to_absolute_path(cfg.eval_config.save_eval_path))
        video_save_path = hydra.utils.to_absolute_path(cfg.eval_config.save_eval_path)
        os.makedirs(video_save_path, exist_ok=True)

        state_data_array = np.load(save_path / "state_data.npy", allow_pickle=True)
        # rollout_data = np.load(save_path / "rollout_data.npz")

        rollout = {"states": state_data_array}

        render_rollout(
            rollout, cfg, eval_env, video_save_path, num_steps=len(state_data_array)
        )


if __name__ == "__main__":
    main()
