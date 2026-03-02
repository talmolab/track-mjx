"""Entry point for track-mjx training."""

import os
from pathlib import Path

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf
from vnl_playground import registry
from vnl_playground.tasks import wrappers as rodent_wrappers

from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.ff_ppo import ppo as ff_ppo, ppo_networks as ff_networks
from track_mjx.agent.recurrent_ppo import (
    ppo as recurrent_ppo,
    networks as recurrent_networks,
)
from track_mjx.agent.temporal_highlvl_ppo import (
    networks as temporal_highlvl_networks,
)
from track_mjx.agent.temporal_ppo import (
    ppo as temporal_ppo,
    networks as temporal_networks,
)
from track_mjx.agent.domain_randomization import domain_randomization_maker


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


_TEMPORAL_ARCHES = {"temporal_fixed_ppo", "temporal_learned_ppo"}
_TEMPORAL_HIGHLVL_ARCHES = {
    "temporal_fixed_highlvl_ppo",
    "temporal_learned_highlvl_ppo",
}
_HIGHLVL_WRAPPER_KEYS = {
    "policy_obs_key": "state",
    "value_obs_key": "state",
    "highlvl_obs_key": "task_obs",
    "lowlvl_obs_key": "proprioception",
}


def _resolve_decoder_checkpoint_path(checkpoint_path: str) -> Path:
    """Resolves a decoder checkpoint path from config."""
    path = Path(checkpoint_path)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "model_checkpoints" / path


def load_decoder_for_highlvl_wrapper(
    checkpoint_path: str,
    step: int | None = None,
) -> tuple[DictConfig, callable]:
    """Loads and validates a feedforward decoder for HighLevelWrapper."""
    resolved_path = _resolve_decoder_checkpoint_path(checkpoint_path)
    decoder_cfg = checkpointing.load_config_from_checkpoint(
        str(resolved_path), step=step
    )
    arch_name = decoder_cfg.network_config.get("arch_name", "intention")
    if arch_name != "intention":
        raise ValueError(
            f"High-level temporal PPO requires a feedforward decoder checkpoint. "
            f"Got arch_name='{arch_name}' at {resolved_path}."
        )

    required_keys = (
        ("network_config", "intention_size"),
        ("network_config", "decoder_layer_sizes"),
        ("network_config", "obs_sizes"),
    )
    missing = [
        ".".join(path)
        for path in required_keys
        if OmegaConf.select(decoder_cfg, ".".join(path), default=None) is None
    ]
    if missing:
        raise ValueError(
            f"Decoder checkpoint at {resolved_path} is missing required config keys: {missing}."
        )

    decoder_policy = ff_networks.make_decoder_policy_fn(str(resolved_path), step=step)
    return decoder_cfg, decoder_policy


def _wrap_with_highlvl_decoder(base_env, decoder_policy, latent_size: int):
    """Wraps a task env with the frozen feedforward decoder."""
    return rodent_wrappers.HighLevelWrapper(
        base_env,
        decoder_inference_fn=decoder_policy,
        latent_size=latent_size,
        **_HIGHLVL_WRAPPER_KEYS,
    )


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

    # Determine how to load from checkpoint
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Initialize checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="PPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    env_name = cfg.env_config.env_name
    arch_name = cfg.network_config.get("arch_name", "intention")
    logging.info(f"Using architecture: {arch_name}")

    valid_arch_names = {
        "intention",
        "recurrent_intention",
        "temporal_fixed_ppo",
        "temporal_learned_ppo",
        "temporal_fixed_highlvl_ppo",
        "temporal_learned_highlvl_ppo",
    }
    if arch_name not in valid_arch_names:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Valid options are: {sorted(valid_arch_names)}"
        )

    decoder_cfg = None
    decoder_policy = None
    decoder_latent_size = None
    if arch_name in _TEMPORAL_HIGHLVL_ARCHES:
        if env_name != "RodentImitation":
            raise ValueError(
                f"{arch_name} only supports env_name='RodentImitation'. Got '{env_name}'."
            )
        decoder_checkpoint_path = cfg.train_setup.get("decoder_checkpoint_path")
        if decoder_checkpoint_path is None:
            raise ValueError(
                f"{arch_name} requires train_setup.decoder_checkpoint_path."
            )
        decoder_cfg, decoder_policy = load_decoder_for_highlvl_wrapper(
            decoder_checkpoint_path,
            step=cfg.train_setup.get("decoder_checkpoint_step"),
        )
        if decoder_cfg.env_config.walker_name != cfg.env_config.walker_name:
            raise ValueError(
                "Decoder checkpoint walker does not match training config: "
                f"{decoder_cfg.env_config.walker_name} vs {cfg.env_config.walker_name}."
            )
        decoder_ctrl_dt = decoder_cfg.env_config.ctrl_dt
        cfg.env_config.ctrl_dt = decoder_ctrl_dt
        env_cfg_ml.ctrl_dt = decoder_ctrl_dt
        cfg_dict["env_config"]["ctrl_dt"] = decoder_ctrl_dt
        decoder_latent_size = int(decoder_cfg.network_config.intention_size)

    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    reference_clips = registry.load_reference_clips(
        env_name,
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.keep_clips_idx,
    )
    key_split, _ = jax.random.split(
        jax.random.PRNGKey(cfg.train_setup.train_config.seed)
    )
    train_clips, test_clips = reference_clips.split(
        train_ratio=cfg.train_setup.train_subset_ratio,
        seed=key_split,
    )

    if arch_name in _TEMPORAL_HIGHLVL_ARCHES:
        train_base_env = registry.load(
            env_name, config=env_cfg_ml, clips=train_clips, flatten_obs=False
        )
        test_base_env = registry.load(
            env_name, config=env_cfg_ml, clips=test_clips, flatten_obs=False
        )
        env = _wrap_with_highlvl_decoder(
            train_base_env, decoder_policy, decoder_latent_size
        )
        test_env = _wrap_with_highlvl_decoder(
            test_base_env, decoder_policy, decoder_latent_size
        )
        sample_state = env.reset(jax.random.PRNGKey(0))
        if "state" not in sample_state.obs:
            raise ValueError(
                f"Wrapped env observations must include 'state'. Got {sample_state.obs.keys()}."
            )
        if env.action_size != decoder_latent_size:
            raise ValueError(
                "Wrapped env action size does not match decoder latent size: "
                f"{env.action_size} vs {decoder_latent_size}."
            )
    else:
        env = rodent_wrappers.TrackMjxObsWrapper(
            registry.load(
                env_name, config=env_cfg_ml, clips=train_clips, flatten_obs=False
            )
        )
        test_env = rodent_wrappers.TrackMjxObsWrapper(
            registry.load(
                env_name, config=env_cfg_ml, clips=test_clips, flatten_obs=False
            )
        )

    logging.info(f"Environment config: {cfg.env_config}")

    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (cfg.env_config.ctrl_dt)
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")
    logging.info("Using PPO Pipeline")

    if arch_name == "recurrent_intention":
        required_keys = ["rnn_type", "rnn_hidden_sizes"]
        missing_keys = [k for k in required_keys if not hasattr(cfg.network_config, k)]
        if missing_keys:
            raise ValueError(
                f"recurrent_intention architecture requires these config keys: {missing_keys}. "
                f"Please add them to network_config in your YAML file."
            )

        network_factory = functools.partial(
            recurrent_networks.make_recurrent_intention_ppo_networks,
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
    elif arch_name in _TEMPORAL_ARCHES:
        required_keys = ["rnn_type", "rnn_hidden_sizes"]
        missing_keys = [k for k in required_keys if not hasattr(cfg.network_config, k)]
        if missing_keys:
            raise ValueError(
                f"{arch_name} architecture requires these config keys: {missing_keys}. "
                f"Please add them to network_config in your YAML file."
            )

        boundary_mode = "fixed" if arch_name == "temporal_fixed_ppo" else "learned"
        network_factory = functools.partial(
            temporal_networks.make_temporal_intention_ppo_networks,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            rnn_type=cfg.network_config.rnn_type,
            rnn_hidden_sizes=tuple(cfg.network_config.rnn_hidden_sizes),
            boundary_mode=boundary_mode,
            macro_horizon=cfg.network_config.get("macro_horizon", 16),
            min_macro_horizon=cfg.network_config.get("min_macro_horizon", 4),
            max_macro_horizon=cfg.network_config.get("max_macro_horizon", 64),
            eval_gate_threshold=cfg.network_config.get("eval_gate_threshold", 0.5),
            proprioception_noise_std=cfg.network_config.get(
                "proprioception_noise_std", 0.0
            ),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            condition_value_on_latent=cfg.network_config.get(
                "condition_value_on_latent", True
            ),
            horizon_ramp=cfg.network_config.get("horizon_ramp", False),
            horizon_ramp_steps=cfg.network_config.get("horizon_ramp_steps", 0),
        )
        ppo_module = temporal_ppo
    elif arch_name in _TEMPORAL_HIGHLVL_ARCHES:
        required_keys = ["rnn_type", "rnn_hidden_sizes"]
        missing_keys = [k for k in required_keys if not hasattr(cfg.network_config, k)]
        if missing_keys:
            raise ValueError(
                f"{arch_name} architecture requires these config keys: {missing_keys}. "
                f"Please add them to network_config in your YAML file."
            )

        boundary_mode = (
            "fixed" if arch_name == "temporal_fixed_highlvl_ppo" else "learned"
        )
        network_factory = functools.partial(
            temporal_highlvl_networks.make_temporal_highlvl_ppo_networks,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            rnn_type=cfg.network_config.rnn_type,
            rnn_hidden_sizes=tuple(cfg.network_config.rnn_hidden_sizes),
            boundary_mode=boundary_mode,
            macro_horizon=cfg.network_config.get("macro_horizon", 16),
            min_macro_horizon=cfg.network_config.get("min_macro_horizon", 4),
            max_macro_horizon=cfg.network_config.get("max_macro_horizon", 64),
            eval_gate_threshold=cfg.network_config.get("eval_gate_threshold", 0.5),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            condition_value_on_latent=cfg.network_config.get(
                "condition_value_on_latent", True
            ),
            horizon_ramp=cfg.network_config.get("horizon_ramp", False),
            horizon_ramp_steps=cfg.network_config.get("horizon_ramp_steps", 0),
        )
        ppo_module = temporal_ppo
    else:
        network_factory = functools.partial(
            ff_networks.make_intention_ppo_networks,
            intention_latent_size=cfg.network_config.intention_size,
            encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            proprioception_noise_std=cfg.network_config.get(
                "proprioception_noise_std", 0.0
            ),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        )
        ppo_module = ff_ppo

    # Determine wandb run ID for resuming
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    # Save initial run state after wandb initialization
    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    # Create the checkpoint callback with the correct wandb_run_id
    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Build common training arguments
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

    if arch_name in _TEMPORAL_ARCHES | _TEMPORAL_HIGHLVL_ARCHES:
        train_kwargs["gate_entropy_cost"] = cfg.network_config.get(
            "gate_entropy_cost", 1e-4
        )
        train_kwargs["latent_entropy_cost"] = cfg.network_config.get(
            "latent_entropy_cost", 0.0
        )
        train_kwargs["discounting_gate"] = cfg.network_config.get(
            "discounting_gate", None
        )
        train_kwargs["target_refresh_rate"] = cfg.network_config.get(
            "target_refresh_rate", None
        )
        train_kwargs["lambda_refresh_rate"] = cfg.network_config.get(
            "lambda_refresh_rate", 0.0
        )

    # Add get_activation only for feedforward PPO.
    if arch_name == "intention":
        train_kwargs["get_activation"] = cfg.train_setup.train_config.get(
            "get_activation", False
        )

    train_fn = functools.partial(ppo_module.train, **train_kwargs)

    # Set the render env start frame to always be 0
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    if arch_name in _TEMPORAL_HIGHLVL_ARCHES:
        rollout_base_env = registry.load(
            env_name, config=rollout_cfg, clips=None, flatten_obs=False
        )
        rollout_env = _wrap_with_highlvl_decoder(
            rollout_base_env, decoder_policy, decoder_latent_size
        )
    else:
        rollout_env = rodent_wrappers.TrackMjxObsWrapper(
            registry.load(env_name, config=rollout_cfg, clips=None, flatten_obs=False)
        )

    # define the jit reset/step functions
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

    # Clean up run state after successful completion
    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
