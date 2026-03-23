"""Entry point for MoSeq decoder-only RL training (Pipeline A).

Usage:
    cd moseq_jax
    python train_moseq_decoder.py

    # Override config values:
    python train_moseq_decoder.py network_config.num_codes=64

    # Use random codes for smoke testing (no KPMS sweep needed):
    python train_moseq_decoder.py kpms_config.codes_path=null
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
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker

from moseq_ppo_networks import make_moseq_decoder_ppo_networks
from moseq_ppo import train as moseq_train
from moseq_env_wrapper import MoSeqImitation


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def moseq_rollout_logging_fn(
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
    """Rollout logging with MoSeq code metrics and video rendering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_codes = cfg.network_config.num_codes

    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)

    n_rollouts = cfg.render_config.get("eval_rollouts_for_transition", 16)

    key = policy_params_fn_key
    all_rollout_indices: list[np.ndarray] = []
    all_rollout_states: list[list] = []
    all_rollout_rewards: list[list] = []

    for rollout_i in range(n_rollouts):
        key, subkey = jax.random.split(key)
        state = jit_reset(subkey)

        rollout_states = [state]
        rollout_indices = []
        rollout_rewards = []

        for _ in range(episode_length):
            key, subkey = jax.random.split(key)
            action, extras = jit_logging_inference_fn(params, state.obs, subkey, None)

            if "code_idx" in extras:
                rollout_indices.append(int(extras["code_idx"]))
            elif "indices" in extras:
                rollout_indices.append(int(extras["indices"]))

            state = jit_step(state, action)
            rollout_states.append(state)
            rollout_rewards.append(float(state.reward))

            if state.done:
                break

        all_rollout_indices.append(np.array(rollout_indices))
        all_rollout_states.append(rollout_states)
        all_rollout_rewards.append(rollout_rewards)

    # Use first rollout for metrics and video
    indices_array = all_rollout_indices[0] if all_rollout_indices else None

    if indices_array is not None and len(indices_array) > 0:
        # Code utilization metrics
        code_counts = np.bincount(indices_array, minlength=num_codes)
        probs = code_counts / (code_counts.sum() + 1e-8)
        perplexity = float(np.exp(-np.sum(probs * np.log(probs + 1e-8))))
        codes_used = int(np.sum(code_counts > 0))
        utilization = codes_used / num_codes

        code_transitions = int(np.sum(indices_array[1:] != indices_array[:-1]))
        transition_rate = code_transitions / max(len(indices_array) - 1, 1)

        wandb.log(
            {
                "moseq/perplexity": perplexity,
                "moseq/codebook_utilization": utilization,
                "moseq/codes_used": codes_used,
                "moseq/eval_transition_rate": transition_rate,
                "moseq/eval_transitions": code_transitions,
                "moseq/eval_steps": len(indices_array),
            },
            commit=False,
        )

        # Code sequence plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 4), height_ratios=[1, 2])

        axes[0].bar(range(num_codes), code_counts, edgecolor="none")
        axes[0].set_xlabel("Code Index")
        axes[0].set_ylabel("Count")
        axes[0].set_title(
            f"Code Usage (perplexity={perplexity:.2f}, used={codes_used}/{num_codes})"
        )
        axes[0].set_xlim(-0.5, num_codes - 0.5)

        timesteps = np.arange(len(indices_array))
        for i in range(len(indices_array) - 1):
            axes[1].axvspan(timesteps[i], timesteps[i + 1], alpha=0.8)
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Code")
        axes[1].set_title(
            f"Code Sequence (transitions={code_transitions}, "
            f"rate={transition_rate:.2%})"
        )
        axes[1].set_xlim(0, len(indices_array))
        axes[1].set_ylim(-0.5, num_codes - 0.5)

        plt.tight_layout()
        wandb.log({"moseq/code_sequence": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    # Render video from first rollout
    if render_video and all_rollout_states:
        import mujoco

        try:
            from vqvae_jax.analysis.rendering import render_rollout_to_video

            render_fps = cfg.render_config.render_fps
            num_videos = min(
                int(cfg.render_config.get("num_eval_rollout_videos", 1)),
                n_rollouts,
            )

            for vid_i in range(num_videos):
                vid_states = all_rollout_states[vid_i]
                vid_indices = (
                    all_rollout_indices[vid_i]
                    if vid_i < len(all_rollout_indices)
                    else None
                )
                video_path = f"{model_path}/{current_step}_vid{vid_i}.mp4"

                render_rollout_to_video(
                    env=env,
                    rollout_states=vid_states,
                    output_path=video_path,
                    camera=f"{cfg.render_config.render_camera_name}{env._suffix}",
                    width=640,
                    height=480,
                    fps=render_fps,
                    indices=vid_indices,
                    num_codes=num_codes,
                    code_bar_height=30,
                )

                wandb.log(
                    {f"videos/rollout_{vid_i}": wandb.Video(video_path, format="mp4")},
                    commit=False,
                )

        except Exception as e:
            logging.warning(f"Failed to render video: {e}")


@hydra.main(version_base=None, config_path="configs", config_name="moseq_decoder")
def main(cfg: DictConfig) -> None:
    """Main MoSeq decoder training entry point."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # Checkpoint path
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Prepare config
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    # Checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="MoSeqPPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Load balanced splits
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
            train_ratio=cfg.train_setup.train_subset_ratio,
            seed=key_split,
        )
        train_indices = list(range(train_clips.qpos.shape[0]))
        test_indices = list(range(test_clips.qpos.shape[0]))

    # Load or generate KPMS codes
    codes_path = cfg.kpms_config.get("codes_path", None)

    # Resolve relative paths against the repo root (Hydra may change CWD)
    if codes_path and not Path(codes_path).is_absolute():
        resolved = REPO_ROOT / codes_path
        if resolved.exists():
            codes_path = str(resolved)

    if codes_path and Path(codes_path).exists():
        codes_data = np.load(codes_path)
        train_codes = codes_data["train_codes"]
        test_codes = codes_data["test_codes"]
        # Derive num_codes from the actual KPMS data
        num_codes = int(np.max([train_codes.max(), test_codes.max()])) + 1
        logging.info(
            f"Loaded KPMS codes: train {train_codes.shape}, test {test_codes.shape}, "
            f"num_codes={num_codes} (from data)"
        )
    else:
        # Generate random codes for smoke testing
        num_codes = int(cfg.network_config.num_codes)
        logging.warning(
            f"No KPMS codes found — generating random codes ({num_codes}) for smoke testing"
        )
        n_train = train_clips.qpos.shape[0]
        n_test = test_clips.qpos.shape[0]
        n_frames = cfg.env_config.clip_length
        rng = np.random.RandomState(42)
        train_codes = rng.randint(0, num_codes, size=(n_train, n_frames))
        test_codes = rng.randint(0, num_codes, size=(n_test, n_frames))

    # Use MoSeqImitation (subclass of Imitation) which injects kpms_code
    # directly in _get_obs. This ensures the obs pytree structure is consistent
    # from the start — no wrapper needed, no pytree mismatches in jax.lax.scan.
    env = MoSeqImitation(config=env_cfg_ml, clips=train_clips, kpms_codes=train_codes)
    test_env = MoSeqImitation(
        config=env_cfg_ml, clips=test_clips, kpms_codes=test_codes
    )

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using MoSeq Decoder PPO Pipeline")

    # Continuous encoder config
    use_continuous_encoder = bool(
        cfg.network_config.get("use_continuous_encoder", False)
    )
    encoder_layer_sizes = tuple(
        cfg.network_config.get("encoder_layer_sizes", [256, 128])
    )
    continuous_latent_dim = int(cfg.network_config.get("continuous_latent_dim", 16))
    kl_weight = float(cfg.network_config.get("kl_weight", 0.0))

    # Network factory
    network_factory = functools.partial(
        make_moseq_decoder_ppo_networks,
        num_codes=num_codes,
        code_embed_dim=int(cfg.network_config.code_embed_dim),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
        use_continuous_encoder=use_continuous_encoder,
        encoder_layer_sizes=encoder_layer_sizes,
        continuous_latent_dim=continuous_latent_dim,
    )

    # WandB
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    wandb.config.update(
        {
            "arch": (
                "moseq_encoder_decoder" if use_continuous_encoder else "moseq_decoder"
            ),
            "num_codes": num_codes,
            "code_embed_dim": int(cfg.network_config.code_embed_dim),
            "codes_path": str(codes_path) if codes_path else "random",
            "use_continuous_encoder": use_continuous_encoder,
            "continuous_latent_dim": continuous_latent_dim,
            "kl_weight": kl_weight,
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

    # Training
    train_fn = functools.partial(
        moseq_train,
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
        freeze_decoder=cfg.train_setup.get("freeze_decoder", False),
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
        num_codes=num_codes,
        code_embed_dim=int(cfg.network_config.code_embed_dim),
        kl_weight=kl_weight,
    )

    # Rollout env for logging
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = MoSeqImitation(
        config=rollout_cfg, clips=test_clips, kpms_codes=test_codes
    )

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
        return moseq_rollout_logging_fn(
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

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=wandb_logging.wandb_progress,
        policy_params_fn=_policy_params_fn_wrapper,
    )

    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
