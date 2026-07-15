"""Entry point for track-mjx training."""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import orbax.checkpoint as ocp
from brax.training.distribution import NormalTanhDistribution
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig
from vnl_playground import registry
from vnl_playground.tasks import wrappers as rodent_wrappers

import wandb
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.distribution import NormalSigmoidDistribution
from track_mjx.agent.domain_randomization import domain_randomization_maker
from track_mjx.agent.ff_ppo import ppo as ff_ppo
from track_mjx.agent.ff_ppo import ppo_networks as ff_networks
from track_mjx.agent.recurrent_ppo import (
    networks as recurrent_networks,
)
from track_mjx.agent.recurrent_ppo import (
    ppo as recurrent_ppo,
)
from track_mjx.config import utils


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(
    version_base=None,
    config_path="../track_mjx/config",
    config_name="rodent-full-clips",
)
def main(cfg: DictConfig) -> None:
    """Main training entry point using Hydra configs.

    Initializes JAX devices, loads reference clips, creates train/test
    environments, and runs PPO training with wandb logging.

    Args:
        cfg: Hydra configuration containing env_config, network_config,
            train_setup, and logging_config.
    """
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # Prepare config BEFORE load_from_run_state so the config hash is consistent
    # between discovery and saving (prepare_config modifies cfg by adding paths)
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Initialize wandb early so that clip loading and env creation are visible
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    env_name = cfg.env_config.env_name
    default_config = registry.get_default_config(env_name)

    reference_clips = registry.load_reference_clips(
        env_name,
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
        joint_names=cfg.env_config.get("joints", default_config.get("joints", None)),
        body_names=cfg.env_config.get("bodies", default_config.get("bodies", None)),
    )
    key_split, _ = jax.random.split(jax.random.key(cfg.train_setup.train_config.seed))
    train_clips, test_clips = reference_clips.split(
        train_ratio=cfg.train_setup.train_subset_ratio,
        seed=key_split,
    )
    logging.info(f"Training on {len(train_clips)} clips: {train_clips}")
    logging.info(f"Testing on {len(test_clips)} clips: {test_clips}")

    if len(train_clips) == 0:
        raise ValueError(
            f"train_subset_ratio={cfg.train_setup.train_subset_ratio} yields 0 "
            f"training clips. Increase the ratio or add more clips."
        )

    env = rodent_wrappers.TrackMjxObsWrapper(
        registry.load(env_name, config=env_cfg_ml, clips=train_clips, flatten_obs=False)
    )
    try:
        xml = env.save_spec("./env_spec.xml", return_str=True)
    except Exception as e:
        logging.warning(f"Failed to save environment spec: {e}")
        xml = None

    if len(test_clips) == 0:
        logging.warning(
            "No test clips after split — skipping held-out test-set evaluation."
        )
        test_env = None
    else:
        test_env = rodent_wrappers.TrackMjxObsWrapper(
            registry.load(
                env_name, config=env_cfg_ml, clips=test_clips, flatten_obs=False
            )
        )

    logging.info(f"Environment config: {cfg.env_config}")
    logging.info(f"Train environment: {env.unwrapped}")

    # Episode length is equal to (clip length - random init range - traj length) * steps per cur frame.
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (cfg.env_config.ctrl_dt)
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using PPO Pipeline")

    arch_name = cfg.network_config.get("arch_name", "intention")
    logging.info(f"Using architecture: {arch_name}")

    valid_arch_names = {"intention", "recurrent_intention"}
    if arch_name not in valid_arch_names:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Valid options are: {sorted(valid_arch_names)}"
        )
    action_dist = cfg.network_config.get("action_distribution", "tanh")
    if action_dist.lower() == "sigmoid":
        logging.info("Using sigmoid action distribution")
        action_distribution = NormalSigmoidDistribution
    else:
        logging.info("Using tanh action distribution")
        action_distribution = NormalTanhDistribution
    if arch_name == "recurrent_intention":
        # Validate required config keys for recurrent architecture
        required_keys = ["rnn_type", "rnn_hidden_sizes"]
        missing_keys = [k for k in required_keys if not hasattr(cfg.network_config, k)]
        if missing_keys:
            raise ValueError(
                f"recurrent_intention architecture requires these config keys: {missing_keys}. "
                f"Please add them to network_config in your YAML file."
            )

        # Recurrent intention network (MLP encoder + RNN decoder)
        network_factory = functools.partial(
            recurrent_networks.make_recurrent_intention_ppo_networks,
            action_distribution=action_distribution,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            rnn_type=cfg.network_config.rnn_type,
            rnn_hidden_sizes=tuple(cfg.network_config.rnn_hidden_sizes),
            proprioception_noise_std=cfg.network_config.get(
                "proprioception_noise_std", 0.0
            ),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )
        ppo_module = recurrent_ppo
    else:
        # Feedforward intention network (default)
        network_factory = functools.partial(
            ff_networks.make_intention_ppo_networks,
            action_distribution=action_distribution,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            proprioception_noise_std=cfg.network_config.get(
                "proprioception_noise_std", 0.0
            ),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )
        ppo_module = ff_ppo

    if xml is not None:
        wandb.log({"spec_file": wandb.Html(xml)}, commit=False)

    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    train_kwargs = dict(
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        latent_kl_weight=cfg.network_config.latent_kl_weight,
        latent_ar1_weight=cfg.network_config.latent_ar1_weight,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=cfg.network_config.kl_schedule,
        eval_env_test_set=test_env,
        checkpoint_callback=checkpoint_callback,
        wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
        randomization_fn=(
            domain_randomization_maker(
                floor_friction=cfg.env_config.domain_randomization.floor_friction,
                static_friction_scale=cfg.env_config.domain_randomization.static_friction_scale,
                armature_scale=cfg.env_config.domain_randomization.armature_scale,
                com_jitter=cfg.env_config.domain_randomization.com_jitter,
                link_mass_scale=cfg.env_config.domain_randomization.link_mass_scale,
                torso_mass_jitter=cfg.env_config.domain_randomization.torso_mass_jitter,
                qpos0_jitter=cfg.env_config.domain_randomization.qpos0_jitter,
            )
            if cfg.env_config.domain_randomization.use_domain_randomization
            else None
        ),
    )

    # Add get_activation only for feedforward PPO (not supported for recurrent)
    if arch_name != "recurrent_intention":
        train_kwargs["get_activation"] = cfg.train_setup.train_config.get(
            "get_activation", False
        )

    train_fn = functools.partial(ppo_module.train, **train_kwargs)

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = rodent_wrappers.TrackMjxObsWrapper(
        registry.load(env_name, config=rollout_cfg, clips=None, flatten_obs=False)
    )

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)
    policy_params_fn = functools.partial(
        wandb_logging.rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        checkpoint_path,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
