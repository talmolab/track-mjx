"""Phase 2 training entry: latent-mimic style policy on flat ground.

Mirrors track_mjx/train.py's structure (Hydra config, env build, wandb init,
periodic eval rollouts with video) but uses LatentMimicEnvWrapper around the
base imitation env and the latent_mimic network factory.
"""
import os

# Must set rendering backend before importing MuJoCo
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools
import logging
from pathlib import Path

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from flax import serialization
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf

from track_mjx.config import utils
from track_mjx.agent import checkpointing, network_registry
from track_mjx.agent import wandb_logging as base_wandb_logging
from track_mjx.agent.latent_ppo.env_wrapper import LatentMimicEnvWrapper
from track_mjx.agent.latent_ppo import wandb_logging as latent_wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker
from vnl_playground import registry


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def build_env(cfg: DictConfig, env_cfg_ml, clips):
    """Build a base imitation env then wrap it with LatentMimicEnvWrapper."""
    env_name = cfg.env_config.env_name
    base_env = registry.load(
        env_name, config=env_cfg_ml, clips=clips, flatten_obs=False
    )
    branch_kl_weights = cfg.latent_mimic.get("branch_kl_weights", None)
    if branch_kl_weights is not None:
        branch_kl_weights = tuple(float(x) for x in branch_kl_weights)
    return LatentMimicEnvWrapper(
        env=base_env,
        prior_dir=cfg.latent_mimic.prior_dir,
        n_joints=cfg.latent_mimic.n_joints,
        w_r=cfg.latent_mimic.w_r,
        history_len=cfg.latent_mimic.history_len,
        kl_mode=cfg.latent_mimic.get("kl_mode", "mean"),
        drop_dead_orientations=cfg.latent_mimic.get("drop_dead_orientations", False),
        proprio_var_threshold=cfg.latent_mimic.get("proprio_var_threshold", 1e-8),
        use_predictor=cfg.latent_mimic.get("use_predictor", True),
        sigma_clamp=float(cfg.latent_mimic.get("sigma_clamp", 0.0)),
        branch_kl_weights=branch_kl_weights,
    )


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="latent_mimic_phase2",
)
def main(cfg: DictConfig):
    """Phase 2 main entry: trains pi_style on flat ground."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("No GPUs detected")

    # Resolve walker paths and build cfg dict / ml_collections variant.
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)
    env_name = cfg.env_config.env_name

    # Run-state discovery / fresh-start setup.
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)
    mgr_options = ocp.CheckpointManagerOptions(create=True, step_prefix="LatentPhase2")
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Load reference clips and split.
    logging.info(f"Loading reference clips: {cfg.env_config.reference_data_path}")
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

    env = build_env(cfg, env_cfg_ml, train_clips)
    test_env = build_env(cfg, env_cfg_ml, test_clips)

    # Episode length matches track_mjx/train.py's calculation.
    clip_length = cfg.env_config.clip_length
    if clip_length is None:
        clip_length = reference_clips.qpos.shape[1]
        logging.info(f"Auto-detected clip_length: {clip_length}")
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = int(
        (clip_length - cfg.env_config.start_frame_range[-1] - cfg.env_config.reference_length)
        * steps_per_frame
    )
    logging.info(f"episode_length: {episode_length}")

    # Network factory + ppo trainer via the registry.
    arch_name = cfg.network_config.arch_name
    logging.info(f"Using architecture: {arch_name}")
    network_factory_fn = network_registry.get_network_factory(arch_name)
    ppo_module = network_registry.get_ppo_module(arch_name)
    network_factory = functools.partial(
        network_factory_fn,
        policy_layer_sizes=tuple(cfg.network_config.policy_layer_sizes),
        value_layer_sizes=tuple(cfg.network_config.value_layer_sizes),
    )

    # wandb init.
    base_wandb_logging.initialize_wandb_logging(
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

    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Eval rollout env: always start at frame 0 for consistent video.
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = build_env(cfg, rollout_cfg, None)
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    policy_params_fn = functools.partial(
        latent_wandb_logging.latent_mimic_rollout_logging_fn,
        rollout_env,
        jit_reset,
        jit_step,
        cfg,
        str(checkpoint_path),
    )

    # ff_ppo.train kwargs. The trainer also accepts intention-network specific
    # kwargs (latent_kl_weight, latent_ar1_weight, use_kl_schedule); we pass
    # zeros / False since the latent_mimic policy has no aux-VAE losses.
    train_kwargs = dict(
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        latent_kl_weight=0.0,
        latent_ar1_weight=0.0,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=False,
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
        freeze_decoder=False,
    )

    train_fn = functools.partial(ppo_module.train, **train_kwargs)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=base_wandb_logging.wandb_progress,
        policy_params_fn=policy_params_fn,
    )

    # Save final policy.
    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "policy.msgpack", "wb") as f:
        f.write(serialization.to_bytes(params))
    OmegaConf.save(cfg, out_dir / "config.yaml")
    logging.info(f"Phase 2 done; pi_style saved at {out_dir}")

    try:
        checkpointing.cleanup_run_state(cfg)
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")


if __name__ == "__main__":
    main()
