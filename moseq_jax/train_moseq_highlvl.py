"""Entry point for MoSeq high-level RNN intention training.

Trains an RNN that maps KPMS codes + proprioception to latent intentions,
routed through a frozen pretrained decoder.

Usage:
    cd moseq_jax
    python train_moseq_highlvl.py highlvl_config.mimic_checkpoint=260217_084318_560494

    # With random codes for smoke testing:
    python train_moseq_highlvl.py kpms_config.codes_path=null \
        highlvl_config.mimic_checkpoint=260217_084318_560494
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import functools
import json
import logging
from pathlib import Path

MOSEQ_DIR = Path(__file__).parent
REPO_ROOT = MOSEQ_DIR.parent
sys.path.insert(0, str(MOSEQ_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from vnl_playground.tasks import wrappers as rodent_wrappers

from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker
from track_mjx.agent.ff_ppo.ppo_networks import make_decoder_policy_fn

from moseq_highlvl_network import make_moseq_highlvl_rnn_networks
from moseq_highlvl_ppo import train as highlvl_train
from moseq_env_wrapper import MoSeqImitation
from moseq_highlvl_wrapper import MoSeqHighLevelWrapper

# Reuse eval functions from moseq decoder training
from train_moseq_decoder import _run_eval_rollouts, _log_eval_metrics


def _setup_environment() -> None:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _resolve_mimic_checkpoint(path_str: str) -> str:
    """Resolve mimic checkpoint path (absolute or relative to model_checkpoints)."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    for base in [
        REPO_ROOT,
        REPO_ROOT / "model_checkpoints",
        MOSEQ_DIR / "model_checkpoints",
    ]:
        candidate = base / path_str
        if candidate.exists():
            return str(candidate)
    return path_str


def moseq_highlvl_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg,
    model_path,
    current_step,
    jit_logging_inference_fn,
    params,
    policy_params_fn_key,
    render_video=True,
    ppo_network=None,
):
    """Rollout logging with code metrics and video rendering."""
    num_codes = cfg.network_config.num_codes
    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)
    n_rollouts = cfg.render_config.get("eval_rollouts_for_transition", 16)

    key = policy_params_fn_key

    all_indices, all_states, all_rewards, all_latents, key = _run_eval_rollouts(
        env,
        jit_reset,
        jit_step,
        jit_logging_inference_fn,
        params,
        key,
        episode_length,
        n_rollouts,
        use_rnn=True,
        ppo_network=ppo_network,
    )
    _log_eval_metrics(
        all_indices,
        all_states,
        all_rewards,
        env,
        cfg,
        model_path,
        current_step,
        num_codes,
        render_video,
        metric_prefix="highlvl",
        video_prefix="videos",
    )


@hydra.main(version_base=None, config_path="configs", config_name="moseq_highlvl")
def main(cfg: DictConfig) -> None:
    """Main high-level RNN intention training entry point."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # --- Frozen decoder ---
    mimic_ckpt = cfg.highlvl_config.mimic_checkpoint
    if mimic_ckpt is None:
        raise ValueError(
            "highlvl_config.mimic_checkpoint is required. "
            "Pass e.g. highlvl_config.mimic_checkpoint=260217_084318_560494"
        )
    mimic_ckpt_path = _resolve_mimic_checkpoint(str(mimic_ckpt))
    logging.info(f"Loading frozen decoder from: {mimic_ckpt_path}")

    frozen_decoder_fn = make_decoder_policy_fn(mimic_ckpt_path)
    mimic_cfg = checkpointing.load_config_from_checkpoint(mimic_ckpt_path)
    intention_size = mimic_cfg["network_config"]["intention_size"]
    logging.info(f"Intention size from mimic: {intention_size}")

    # --- Checkpoint setup ---
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(
        cfg
    )
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    mgr_options = ocp.CheckpointManagerOptions(
        create=True, step_prefix="MoSeqHighLevelNetwork"
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # --- Data ---
    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    balanced_split_path = cfg.env_config.get("balanced_split_path", None)
    if balanced_split_path:
        with open(balanced_split_path) as f:
            splits = json.load(f)
        train_indices = splits["balanced"]["train_indices"]
        test_indices = splits["balanced"]["test_indices"]
        train_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=np.array(train_indices),
        )
        test_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=np.array(test_indices),
        )
        logging.info(
            f"Loaded balanced splits: {len(train_indices)} train, "
            f"{len(test_indices)} test"
        )
    else:
        reference_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=cfg.env_config.keep_clips_idx,
        )
        key_split, _ = jax.random.split(
            jax.random.PRNGKey(cfg.train_setup.train_config.seed)
        )
        train_clips, test_clips = reference_clips.split(
            train_ratio=cfg.train_setup.train_subset_ratio, seed=key_split
        )

    # --- KPMS codes ---
    codes_path = cfg.kpms_config.get("codes_path", None)
    if codes_path and not Path(codes_path).is_absolute():
        resolved = REPO_ROOT / codes_path
        if resolved.exists():
            codes_path = str(resolved)

    num_codes = int(cfg.network_config.num_codes)
    if codes_path and Path(codes_path).exists():
        codes_data = np.load(codes_path)
        train_codes = codes_data["train_codes"]
        test_codes = codes_data["test_codes"]
        num_codes = int(np.max([train_codes.max(), test_codes.max()])) + 1
        logging.info(
            f"Loaded KPMS codes: train {train_codes.shape}, "
            f"test {test_codes.shape}, num_codes={num_codes}"
        )
    else:
        logging.warning(
            f"No KPMS codes found — generating random codes ({num_codes})"
        )
        n_train = train_clips.qpos.shape[0]
        n_test = test_clips.qpos.shape[0]
        n_frames = cfg.env_config.clip_length
        rng = np.random.RandomState(42)
        train_codes = rng.randint(0, num_codes, size=(n_train, n_frames))
        test_codes = rng.randint(0, num_codes, size=(n_test, n_frames))

    code_stack_size = int(cfg.network_config.get("code_stack_size", 1))
    rnn_hidden_sizes = tuple(cfg.network_config.get("rnn_hidden_sizes", [256]))

    # --- Environments ---
    def _make_env(clips, codes, config):
        return MoSeqHighLevelWrapper(
            rodent_wrappers.TrackMjxObsWrapper(
                MoSeqImitation(
                    config=config,
                    clips=clips,
                    kpms_codes=codes,
                    code_stack_size=code_stack_size,
                )
            ),
            decoder_inference_fn=frozen_decoder_fn,
            intention_size=intention_size,
        )

    env = _make_env(train_clips, train_codes, env_cfg_ml)
    test_env = _make_env(test_clips, test_codes, env_cfg_ml)

    logging.info(f"Environment action_size (intention_size): {env.action_size}")

    # Episode length
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    # --- Network factory ---
    network_factory = functools.partial(
        make_moseq_highlvl_rnn_networks,
        num_codes=num_codes,
        code_embed_dim=int(cfg.network_config.code_embed_dim),
        rnn_hidden_sizes=rnn_hidden_sizes,
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        code_stack_size=code_stack_size,
    )

    # --- WandB ---
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    wandb.config.update(
        {
            "arch": "moseq_highlvl_rnn",
            "num_codes": num_codes,
            "code_embed_dim": int(cfg.network_config.code_embed_dim),
            "code_stack_size": code_stack_size,
            "codes_path": str(codes_path) if codes_path else "random",
            "mimic_checkpoint": str(mimic_ckpt),
            "intention_size": intention_size,
            "rnn_hidden_sizes": list(rnn_hidden_sizes),
        }
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

    # --- Training ---
    train_fn = functools.partial(
        highlvl_train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
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
        rnn_hidden_sizes=rnn_hidden_sizes,
    )

    # Rollout env for logging
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = _make_env(test_clips, test_codes, rollout_cfg)

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    def _policy_params_fn_wrapper(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video=True,
        ppo_network=None,
    ):
        return moseq_highlvl_rollout_logging_fn(
            rollout_env,
            jit_reset,
            jit_step,
            cfg,
            checkpoint_path,
            current_step=current_step,
            jit_logging_inference_fn=jit_logging_inference_fn,
            params=params,
            policy_params_fn_key=policy_params_fn_key,
            render_video=render_video,
            ppo_network=ppo_network,
        )

    def _grouped_wandb_progress(num_steps: int, metrics: dict) -> None:
        grouped = {}
        for k, v in metrics.items():
            if k in ("total_loss", "policy_loss", "v_loss", "entropy_loss"):
                grouped[f"losses/{k}"] = v
            elif k in ("intention_norm", "intention_std", "hidden_state_norm"):
                grouped[f"rnn/{k}"] = v
            elif k in (
                "transition_rate",
                "perplexity",
                "codebook_utilization",
                "codes_used",
            ):
                grouped[f"codes/{k}"] = v
            grouped[k] = v
        wandb_logging.wandb_progress(num_steps, grouped)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=_grouped_wandb_progress,
        policy_params_fn=_policy_params_fn_wrapper,
    )

    try:
        wandb.run.summary.update({"training_completed": True})
    except Exception:
        pass

    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
