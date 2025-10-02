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
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import jax
import hydra
from omegaconf import DictConfig, OmegaConf
import functools
import wandb
from brax import envs
import orbax.checkpoint as ocp
from track_mjx.agent.mlp_ppo import ppo as mlp_ppo, ppo_networks as mlp_ppo_networks
from pathlib import Path
from datetime import datetime
import logging
import json
import fcntl

from track_mjx.io import load
from track_mjx.environment import wrappers
from track_mjx.agent import checkpointing
from track_mjx.agent import wandb_logging
from track_mjx.agent import preemption
from track_mjx.analysis import render
from track_mjx.environment.task.reward import RewardConfig

from vnl_mjx.tasks.rodent import imitation
from vnl_mjx.tasks.rodent import wrappers as vnl_wrappers
from mujoco_playground import wrapper as playground_wrappers


def _track_to_vnl_cfg(cfg):
    """Replace the config values with the ones in our hydra cfg"""
    env_cfg = imitation.default_config()

    # Map environment parameters directly
    env_args = cfg.env_config.env_args
    env_cfg.solver = env_args.solver
    env_cfg.iterations = env_args.iterations
    env_cfg.ls_iterations = env_args.ls_iterations
    env_cfg.sim_dt = env_args.mj_model_timestep
    env_cfg.mocap_hz = env_args.mocap_hz
    env_cfg.ctrl_dt = (
        env_args.mj_model_timestep * env_args.physics_steps_per_control_step
    )
    # Map walker parameters directly
    walker_cfg = cfg.walker_config
    env_cfg.torque_actuators = walker_cfg.torque_actuators
    env_cfg.rescale_factor = walker_cfg.rescale_factor

    # Map reference parameters directly
    ref_cfg = cfg.reference_config
    env_cfg.clip_length = ref_cfg.clip_length
    env_cfg.reference_length = ref_cfg.traj_length
    env_cfg.start_frame_range = [0, ref_cfg.random_init_range]

    # Map reward terms directly
    reward_weights = cfg.env_config.reward_weights

    # Map imitation rewards
    env_cfg.reward_terms["root_pos"] = {
        "exp_scale": 1.0 / (2 * reward_weights.pos_reward_exp_scale) ** 0.5,
        "weight": reward_weights.pos_reward_weight,
    }

    env_cfg.reward_terms["root_quat"] = {
        "exp_scale": 1.0 / (2 * reward_weights.quat_reward_exp_scale) ** 0.5,
        "weight": reward_weights.quat_reward_weight,
    }

    env_cfg.reward_terms["joints"] = {
        "exp_scale": 1.0 / (2 * reward_weights.joint_reward_exp_scale) ** 0.5,
        "weight": reward_weights.joint_reward_weight,
    }

    env_cfg.reward_terms["joints_vel"] = {
        "exp_scale": 1.0 / (2 * reward_weights.angvel_reward_exp_scale) ** 0.5,
        "weight": reward_weights.angvel_reward_weight,
    }

    env_cfg.reward_terms["bodies_pos"] = {
        "exp_scale": 1.0 / (2 * reward_weights.bodypos_reward_exp_scale) ** 0.5,
        "weight": reward_weights.bodypos_reward_weight,
    }

    env_cfg.reward_terms["end_eff"] = {
        "exp_scale": 1.0 / (2 * reward_weights.endeff_reward_exp_scale) ** 0.5,
        "weight": reward_weights.endeff_reward_weight,
    }

    # Map cost terms (these exist in default config)
    env_cfg.reward_terms["control_cost"] = {"weight": reward_weights.ctrl_cost_weight}

    env_cfg.reward_terms["control_diff_cost"] = {
        "weight": reward_weights.ctrl_diff_cost_weight
    }

    # Handle energy_cost properly - it exists in default config but has different structure
    env_cfg.reward_terms["energy_cost"]["weight"] = reward_weights.energy_cost_weight

    # Map healthy z range (exists in default config)
    env_cfg.reward_terms["torso_z_range"] = {
        "healthy_z_range": tuple(reward_weights.healthy_z_range),
        "weight": 1.0,  # This doesnt exist in hydra cfg
    }

    # Map penalty parameters to termination criteria (these exist in default config)
    # env_cfg.termination_criteria["root_too_far"] = {
    #     "max_distance": reward_weights.too_far_dist
    # }

    # env_cfg.termination_criteria["pose_error"] = {
    #     "max_l2_error": reward_weights.bad_pose_dist
    # }

    # env_cfg.termination_criteria["root_too_rotated"] = {
    #     "max_degrees": reward_weights.bad_quat_dist
    # }

    # # SET TO WARP
    # env_cfg.mujoco_impl = "warp"
    # # SET NCONMAX FOR WARP
    # env_cfg.nconmax = env_cfg.nconmax * cfg.train_setup.train_config.num_envs
    return env_cfg


@hydra.main(version_base=None, config_path="config", config_name="rodent-full-clips")
def main(cfg: DictConfig):
    """Main function using Hydra configs"""
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except:
        n_devices = 1
        logging.info("Not using GPUs")

    # Check for existing run state (preemption handling)
    existing_run_state = preemption.discover_existing_run_state(cfg)

    # Auto-preemption resume logic
    if existing_run_state:
        # Resume from existing run
        run_id = existing_run_state["run_id"]
        checkpoint_path = existing_run_state["checkpoint_path"]
        # Ensure checkpoint_path is absolute
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_absolute():
            checkpoint_path_obj = Path.cwd() / checkpoint_path_obj
        checkpoint_path = str(checkpoint_path_obj)
        logging.info(f"Resuming from existing run: {run_id}")
        # Add checkpoint path to config to use orbax for resuming
        cfg.train_setup["checkpoint_to_restore"] = checkpoint_path
    # If manually passing json run_state
    elif cfg.train_setup["restore_from_run_state"] is not None:
        # Access file path
        base_path = Path(cfg.logging_config.model_path).resolve()
        full_path = base_path / cfg.train_setup["restore_from_run_state"]
        # Read json with file locking to prevent concurrent access
        with open(full_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            existing_run_state = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        run_id = existing_run_state["run_id"]
        checkpoint_path = existing_run_state["checkpoint_path"]
        # Ensure checkpoint_path is absolute
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.is_absolute():
            checkpoint_path_obj = Path.cwd() / checkpoint_path_obj
        checkpoint_path = str(checkpoint_path_obj)
        logging.info(f"Restoring from run state: {run_id}")
        # Add checkpoint path to config to use orbax for resuming
        cfg.train_setup["checkpoint_to_restore"] = checkpoint_path
    # If no existing run state, generate a new run_id and checkpoint path
    else:
        # Generate a new run_id and associated checkpoint path
        run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        # Use a base path given by the config, ensure it's absolute
        model_path = Path(cfg.logging_config.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        checkpoint_path = str(model_path / run_id)

    # Load the checkpoint's config
    if cfg.train_setup["checkpoint_to_restore"] is not None:
        # TODO: We set the restored config's checkpoint_to_restore to itself
        # Because that restored config is used from now on. This is a hack.
        checkpoint_to_restore = cfg.train_setup["checkpoint_to_restore"]
        # Ensure checkpoint_to_restore is an absolute path
        checkpoint_to_restore_path = Path(checkpoint_to_restore)
        if not checkpoint_to_restore_path.is_absolute():
            checkpoint_to_restore_path = Path.cwd() / checkpoint_to_restore_path
        checkpoint_to_restore = str(checkpoint_to_restore_path)

        # Load the checkpoint's config and update the run_id and checkpoint path
        cfg = OmegaConf.create(
            checkpointing.load_config_from_checkpoint(checkpoint_to_restore)
        )
        cfg.train_setup["checkpoint_to_restore"] = checkpoint_to_restore
        checkpoint_path = checkpoint_to_restore
        run_id = os.path.basename(checkpoint_path)

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

    train_config = cfg.train_setup["train_config"]
    traj_config = cfg["reference_config"]

    logging.info(f"Loading data: {cfg.data_path}")

    # use this custom fn to set values in the vnl config with our hydra cfg
    env_cfg = _track_to_vnl_cfg(cfg)

    # Eval with the train set
    # TODO: implement the train/test split for the vnl env (current init can only take data files)
    train_clips = load.load_data(cfg.data_path)
    logging.info(f"{train_clips.position.shape=}")
    test_env = None

    logging.info(f"Environment config: {env_cfg}")
    env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=env_cfg))

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    env_args = cfg.env_config.env_args
    steps_per_frame = (1 / env_args["mocap_hz"]) / (
        env_args["mj_model_timestep"] * env_args["physics_steps_per_control_step"]
    )
    episode_length = (
        traj_config.clip_length
        - traj_config.random_init_range
        - traj_config.traj_length
    ) * steps_per_frame
    print(f"episode_length {episode_length}")
    logging.info(f"episode_length {episode_length}")

    print("Using MLP Pipeline Now")
    ppo = mlp_ppo
    ppo_networks = mlp_ppo_networks
    network_factory = functools.partial(
        ppo_networks.make_intention_ppo_networks,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
    )

    run_id = f"{cfg.logging_config.exp_name}_{run_id}"

    # Determine wandb run ID for resuming
    if existing_run_state:
        wandb_run_id = existing_run_state["wandb_run_id"]
        wandb_resume = "must"  # Must resume the exact run
        logging.info(f"Resuming wandb run: {wandb_run_id}")
    else:
        wandb_run_id = run_id
        wandb_resume = "allow"  # Allow resuming if run exists
        logging.info(f"Starting new wandb run: {wandb_run_id}")
    cfg_for_wandb = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=True
    )
    cfg_for_wandb["mjx_env_config"] = env_cfg.to_dict()
    wandb.init(
        project=cfg.logging_config.project_name,
        config=cfg_for_wandb,
        notes=f"",
        id=wandb_run_id,
        resume=wandb_resume,
        group=cfg.logging_config.group_name,
    )

    # Save initial run state after wandb initialization
    if not existing_run_state:
        preemption.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    # Create the checkpoint callback with the correct wandb_run_id
    checkpoint_callback = preemption.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
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
        checkpoint_callback=checkpoint_callback,
        wrap_for_training=functools.partial(  # Testing full reset instead of setting to initial state
            playground_wrappers.wrap_for_brax_training, full_reset=True
        ),
    )

    def wandb_progress(num_steps, metrics):
        metrics["num_steps_thousands"] = num_steps
        wandb.log(metrics)

    # Set the render env start frame to always be 0
    render_cfg = env_cfg.copy_and_resolve_references()
    render_cfg.start_frame_range = [0, 0]
    rollout_env = vnl_wrappers.FlattenObsWrapper(imitation.Imitation(config=render_cfg))

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

    # Clean up run state after successful completion
    try:
        preemption.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
