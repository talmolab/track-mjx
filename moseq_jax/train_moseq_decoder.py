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

import dataclasses
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

from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker

from moseq_ppo_networks import (
    make_moseq_decoder_ppo_networks,
    make_moseq_recurrent_decoder_ppo_networks,
)
from moseq_ppo import train as moseq_train
from moseq_env_wrapper import MoSeqImitation


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _run_eval_rollouts(
    env,
    jit_reset,
    jit_step,
    jit_logging_inference_fn,
    params,
    key,
    episode_length,
    n_rollouts,
    use_rnn,
    ppo_network,
):
    """Run eval rollouts and return raw data (no logging).

    Returns:
        ``(all_indices, all_states, all_rewards, final_key)``
    """
    all_rollout_indices: list[np.ndarray] = []
    all_rollout_states: list[list] = []
    all_rollout_rewards: list[list] = []

    for rollout_i in range(n_rollouts):
        key, subkey = jax.random.split(key)
        state = jit_reset(subkey)

        rollout_states = [state]
        rollout_indices = []
        rollout_rewards = []

        if use_rnn and ppo_network is not None:
            hidden = ppo_network.policy_network.init_hidden(1)
        else:
            hidden = None

        for _ in range(episode_length):
            key, subkey = jax.random.split(key)

            if hidden is not None:
                batched_obs = jax.tree_util.tree_map(lambda x: x[None], state.obs)
                action, extras, hidden = jit_logging_inference_fn(
                    params, batched_obs, hidden, subkey
                )
                action = jax.tree_util.tree_map(lambda x: x[0], action)
                extras = jax.tree_util.tree_map(
                    lambda x: x[0] if hasattr(x, "shape") else x, extras
                )
            else:
                action, extras = jit_logging_inference_fn(
                    params, state.obs, subkey, None
                )

            if "code_idx" in extras:
                rollout_indices.append(int(extras["code_idx"]))
            elif "indices" in extras:
                rollout_indices.append(int(extras["indices"]))

            state = jit_step(state, action)
            rollout_states.append(state)
            rollout_rewards.append(float(state.reward))

            if state.done:
                if hidden is not None and ppo_network is not None:
                    hidden = ppo_network.policy_network.init_hidden(1)
                break

        all_rollout_indices.append(np.array(rollout_indices))
        all_rollout_states.append(rollout_states)
        all_rollout_rewards.append(rollout_rewards)

    return all_rollout_indices, all_rollout_states, all_rollout_rewards, key


def _log_eval_metrics(
    all_rollout_indices,
    all_rollout_states,
    all_rollout_rewards,
    env,
    cfg,
    model_path,
    current_step,
    num_codes,
    render_video,
    metric_prefix="moseq",
    video_prefix="videos",
):
    """Log metrics, code plots, and videos from eval rollouts.

    Returns:
        ``episode_reward_mean`` (float) for reward gap computation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Episode reward stats
    episode_rewards = [sum(r) for r in all_rollout_rewards if len(r) > 0]
    episode_lengths = [len(r) for r in all_rollout_rewards if len(r) > 0]

    reward_mean = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    reward_std = float(np.std(episode_rewards)) if episode_rewards else 0.0
    length_mean = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    wandb.log(
        {
            f"{metric_prefix}/episode_reward_mean": reward_mean,
            f"{metric_prefix}/episode_reward_std": reward_std,
            f"{metric_prefix}/episode_length_mean": length_mean,
        },
        commit=False,
    )

    # Code metrics from first rollout
    indices_array = all_rollout_indices[0] if all_rollout_indices else None

    if indices_array is not None and len(indices_array) > 0:
        code_counts = np.bincount(indices_array, minlength=num_codes)
        probs = code_counts / (code_counts.sum() + 1e-8)
        perplexity = float(np.exp(-np.sum(probs * np.log(probs + 1e-8))))
        codes_used = int(np.sum(code_counts > 0))
        utilization = codes_used / num_codes

        code_transitions = int(np.sum(indices_array[1:] != indices_array[:-1]))
        transition_rate = code_transitions / max(len(indices_array) - 1, 1)

        wandb.log(
            {
                f"{metric_prefix}/perplexity": perplexity,
                f"{metric_prefix}/codebook_utilization": utilization,
                f"{metric_prefix}/codes_used": codes_used,
                f"{metric_prefix}/eval_transition_rate": transition_rate,
                f"{metric_prefix}/eval_transitions": code_transitions,
                f"{metric_prefix}/eval_steps": len(indices_array),
            },
            commit=False,
        )

        # Code sequence plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 4), height_ratios=[1, 2])

        axes[0].bar(range(num_codes), code_counts, edgecolor="none")
        axes[0].set_xlabel("Code Index")
        axes[0].set_ylabel("Count")
        axes[0].set_title(
            f"Code Usage (perplexity={perplexity:.2f}, "
            f"used={codes_used}/{num_codes})"
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
        wandb.log({f"{metric_prefix}/code_sequence": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    # Render video
    if render_video and all_rollout_states:
        try:
            from vqvae_jax.analysis.rendering import render_rollout_to_video

            render_fps = cfg.render_config.render_fps
            n_rollouts = len(all_rollout_states)
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
                video_path = (
                    f"{model_path}/{current_step}_{metric_prefix}_vid{vid_i}.mp4"
                )

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
                    {
                        f"{video_prefix}/{metric_prefix}_rollout_{vid_i}": wandb.Video(
                            video_path, format="mp4"
                        )
                    },
                    commit=False,
                )

        except Exception as e:
            logging.warning(f"Failed to render {metric_prefix} video: {e}")

    return reward_mean


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
    jit_decoder_only_inference_fn=None,
):
    """Rollout logging with MoSeq code metrics and video rendering.

    Runs full eval (z_e=1) and optionally decoder-only eval (z_e=0).
    """
    num_codes = cfg.network_config.num_codes

    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)
    n_rollouts = cfg.render_config.get("eval_rollouts_for_transition", 16)
    use_rnn = bool(cfg.network_config.get("use_rnn_decoder", False))

    key = policy_params_fn_key

    # --- Full eval (z_e=1.0) ---
    all_indices, all_states, all_rewards, key = _run_eval_rollouts(
        env,
        jit_reset,
        jit_step,
        jit_logging_inference_fn,
        params,
        key,
        episode_length,
        n_rollouts,
        use_rnn,
        ppo_network,
    )
    full_reward = _log_eval_metrics(
        all_indices,
        all_states,
        all_rewards,
        env,
        cfg,
        model_path,
        current_step,
        num_codes,
        render_video,
        metric_prefix="moseq",
        video_prefix="videos",
    )

    # --- Decoder-only eval (z_e=0.0) ---
    if jit_decoder_only_inference_fn is not None:
        do_indices, do_states, do_rewards, key = _run_eval_rollouts(
            env,
            jit_reset,
            jit_step,
            jit_decoder_only_inference_fn,
            params,
            key,
            episode_length,
            n_rollouts,
            use_rnn,
            ppo_network,
        )
        decoder_only_reward = _log_eval_metrics(
            do_indices,
            do_states,
            do_rewards,
            env,
            cfg,
            model_path,
            current_step,
            num_codes,
            render_video,
            metric_prefix="decoder_only",
            video_prefix="videos",
        )

        # Reward gap: how much performance is lost without z_e
        wandb.log(
            {"decoder_only/reward_gap": full_reward - decoder_only_reward},
            commit=False,
        )


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

    # Code stack size: how many consecutive codes to give the decoder
    code_stack_size = int(cfg.network_config.get("code_stack_size", 1))

    # MoSeqImitation overrides _get_obs to inject kpms_code, flatten nested
    # obs, and strip the "state" hierarchy — all inline. No wrapper chain
    # needed, since BraxDomainRandomizationVmapWrapper bypasses wrappers
    # via env.unwrapped.
    env = MoSeqImitation(
        config=env_cfg_ml, clips=train_clips, kpms_codes=train_codes,
        code_stack_size=code_stack_size,
    )
    test_env = MoSeqImitation(
        config=env_cfg_ml, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
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
    z_e_dropout_rate = float(cfg.network_config.get("z_e_dropout_rate", 0.0))
    z_e_at_action_head = bool(cfg.network_config.get("z_e_at_action_head", False))
    reinit_hidden_on_code = bool(cfg.network_config.get("reinit_hidden_on_code", False))
    learned_hidden_init = bool(cfg.network_config.get("learned_hidden_init", False))

    # RNN decoder config
    use_rnn_decoder = bool(cfg.network_config.get("use_rnn_decoder", False))
    rnn_hidden_sizes = tuple(cfg.network_config.get("rnn_hidden_sizes", [256]))
    rnn_cell_type = str(cfg.network_config.get("rnn_cell_type", "gru"))

    # Distillation head config
    use_distillation_head = bool(
        cfg.network_config.get("use_distillation_head", False)
    )
    distill_head_layer_sizes = tuple(
        cfg.network_config.get("distillation_head_layer_sizes", [256, 128])
    )
    distill_kl_weight = float(cfg.network_config.get("distill_kl_weight", 1.0))
    distillation_encoder_checkpoint = cfg.network_config.get(
        "distillation_encoder_checkpoint", None
    )
    distill_logvar_min = cfg.network_config.get("distill_logvar_min", None)
    distill_logvar_max = cfg.network_config.get("distill_logvar_max", None)
    if distill_logvar_min is not None:
        distill_logvar_min = float(distill_logvar_min)
    if distill_logvar_max is not None:
        distill_logvar_max = float(distill_logvar_max)
    use_pretrained_decoder = bool(
        cfg.network_config.get("use_pretrained_decoder", False)
    )
    decoder_layer_sizes_vae = tuple(
        cfg.network_config.get("decoder_layer_sizes_vae", [512, 256, 256, 256])
    )

    # Auto-disable / auto-override when distillation head is enabled
    if use_distillation_head:
        if not use_rnn_decoder:
            raise ValueError(
                "use_distillation_head=True requires use_rnn_decoder=True "
                "(distill head reads from RNN hidden state)"
            )
        if distillation_encoder_checkpoint is None:
            logging.warning(
                "use_distillation_head=True without encoder checkpoint — "
                "using randomly initialized encoder (for testing only)"
            )
        # Force continuous encoder ON (needed to produce distillation targets)
        use_continuous_encoder = True
        # z_e must NOT enter the action path
        z_e_at_action_head = False
        z_e_dropout_rate = 0.0
        # Disable z_e action-path KL schedule (z_e is detached from actions)
        kl_weight = 0.0

        logging.info(
            f"Distillation mode enabled: z_e detached from action path, "
            f"KL schedule disabled, distill_kl_weight={distill_kl_weight}, "
            f"encoder_checkpoint={distillation_encoder_checkpoint}"
        )

    # KL schedule config
    kl_schedule = str(cfg.network_config.get("kl_schedule", "none"))
    if use_distillation_head:
        kl_schedule = "none"  # Force disable when distilling
    kl_sched_cfg = cfg.network_config.get("kl_schedule_config", {})
    kl_sched_start = float(kl_sched_cfg.get("start_value", kl_weight))
    kl_sched_end = float(kl_sched_cfg.get("end_value", 0.5))
    kl_sched_start_frac = float(kl_sched_cfg.get("start_frac", 0.3))
    kl_sched_end_frac = float(kl_sched_cfg.get("end_frac", 0.7))
    kl_sched_num_cycles = int(kl_sched_cfg.get("num_cycles", 4))

    num_evals_total = int(
        cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
    )

    z_e_scale_fn = None  # No z_e_scale scheduling — use KL schedule instead
    kl_weight_fn = None

    if kl_schedule != "none" and use_continuous_encoder:
        _start = kl_sched_start_frac * num_evals_total
        _end = kl_sched_end_frac * num_evals_total
        _range = max(_end - _start, 1.0)
        _sv = kl_sched_start
        _ev = kl_sched_end

        if kl_schedule == "ramp":
            def kl_weight_fn(step, _s=_start, _r=_range, _e=_end, sv=_sv, ev=_ev):
                frac = jnp.clip((step - _s) / _r, 0.0, 1.0)
                w = sv + (ev - sv) * frac
                return jnp.where(step < _s, sv, jnp.where(step > _e, ev, w))

            logging.info(
                f"KL ramp: {_sv}→{_ev} over steps {_start:.0f}-{_end:.0f} "
                f"(of {num_evals_total} evals)"
            )

        elif kl_schedule == "cosine_anneal":
            _nc = kl_sched_num_cycles

            def kl_weight_fn(step, _s=_start, _n=num_evals_total, sv=_sv, ev=_ev, nc=_nc):
                progress = jnp.clip((step - _s) / max(_n - _s, 1.0), 0.0, 1.0)
                cos_val = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * nc * progress))
                # cos_val=1 at trough (low KL), cos_val=0 at peak (high KL)
                w = ev + (sv - ev) * cos_val
                return jnp.where(step < _s, sv, w)

            logging.info(
                f"KL cosine anneal: {_sv}↔{_ev}, {_nc} cycles, "
                f"starting at step {_start:.0f} (of {num_evals_total} evals)"
            )
        else:
            logging.warning(f"Unknown kl_schedule '{kl_schedule}', using constant kl_weight")

    # Network factory
    if use_rnn_decoder:
        network_factory = functools.partial(
            make_moseq_recurrent_decoder_ppo_networks,
            num_codes=num_codes,
            code_embed_dim=int(cfg.network_config.code_embed_dim),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            use_continuous_encoder=use_continuous_encoder,
            encoder_layer_sizes=encoder_layer_sizes,
            continuous_latent_dim=continuous_latent_dim,
            z_e_dropout_rate=z_e_dropout_rate,
            rnn_hidden_sizes=rnn_hidden_sizes,
            rnn_cell_type=rnn_cell_type,
            z_e_at_action_head=z_e_at_action_head,
            reinit_hidden_on_code=reinit_hidden_on_code,
            learned_hidden_init=learned_hidden_init,
            use_distillation_head=use_distillation_head,
            distill_head_layer_sizes=distill_head_layer_sizes,
            distill_logvar_min=distill_logvar_min,
            distill_logvar_max=distill_logvar_max,
            use_pretrained_decoder=use_pretrained_decoder,
            decoder_layer_sizes_vae=decoder_layer_sizes_vae,
        )
        logging.info(
            f"Using RNN decoder: cell={rnn_cell_type}, hidden={rnn_hidden_sizes}, "
            f"z_e_at_action_head={z_e_at_action_head}, reinit_hidden={reinit_hidden_on_code}, "
            f"learned_init={learned_hidden_init}, "
            f"distillation_head={use_distillation_head}"
        )
    else:
        network_factory = functools.partial(
            make_moseq_decoder_ppo_networks,
            num_codes=num_codes,
            code_embed_dim=int(cfg.network_config.code_embed_dim),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            use_continuous_encoder=use_continuous_encoder,
            encoder_layer_sizes=encoder_layer_sizes,
            continuous_latent_dim=continuous_latent_dim,
            z_e_dropout_rate=z_e_dropout_rate,
        )

    # Encoder loading: inject pre-trained encoder params from a VAE
    # checkpoint trained with train_mimic_encoder.py.  Both the VAE and
    # the distillation model use IntentionEncoder, so the encoder subtree
    # at params["encoder_module"] has identical structure — one-line drop-in.
    _post_init_params_fn = None
    if use_distillation_head and distillation_encoder_checkpoint is not None:
        _encoder_ckpt_path = str(Path(distillation_encoder_checkpoint))
        _encoder_step_prefix = str(
            cfg.network_config.get("distillation_encoder_step_prefix", "MimicEncoder")
        )

        def _post_init_params_fn(training_state):
            """Load encoder from VAE checkpoint (direct subtree assignment)."""
            mgr_options = ocp.CheckpointManagerOptions(
                create=False, step_prefix=_encoder_step_prefix,
            )
            with ocp.CheckpointManager(_encoder_ckpt_path, options=mgr_options) as mgr:
                restored = mgr.restore(
                    mgr.latest_step(),
                    args=ocp.args.Composite(policy=ocp.args.StandardRestore()),
                )
            _, loaded_policy_params = restored["policy"]

            # The VAE checkpoint stores encoder under params["encoder"].
            # Our model stores it under params["encoder_module"].
            # Both are the same IntentionEncoder Flax module.
            loaded_encoder = loaded_policy_params["params"]["encoder"]
            training_state.params.policy["params"]["encoder_module"] = loaded_encoder
            logging.info(
                f"Loaded encoder from {distillation_encoder_checkpoint} "
                f"(step_prefix={_encoder_step_prefix})"
            )

            # Also load decoder if use_pretrained_decoder is enabled
            if use_pretrained_decoder and "decoder" in loaded_policy_params["params"]:
                loaded_decoder = loaded_policy_params["params"]["decoder"]
                training_state.params.policy["params"]["decoder_module"] = loaded_decoder
                logging.info(
                    f"Loaded decoder from {distillation_encoder_checkpoint}"
                )

            return training_state

    # WandB
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    if use_distillation_head:
        arch_name = "moseq_rnn_distill"
    elif use_rnn_decoder:
        arch_name = (
            "moseq_rnn_encoder_decoder"
            if use_continuous_encoder
            else "moseq_rnn_decoder"
        )
    else:
        arch_name = (
            "moseq_encoder_decoder" if use_continuous_encoder else "moseq_decoder"
        )

    # Read KPMS model provenance (kappa, num_states).
    # Try .npz metadata first (self-contained), then fall back to sweep_results.json.
    kpms_kappa = None
    kpms_num_states = None
    kpms_model_type = None
    kpms_mean_duration = None
    if codes_path and Path(codes_path).exists():
        try:
            codes_meta = np.load(codes_path, allow_pickle=False)
            if "kappa" in codes_meta:
                kpms_kappa = float(codes_meta["kappa"])
                kpms_num_states = int(codes_meta["num_states"])
                kpms_model_type = str(codes_meta.get("model_type", "unknown"))
                kpms_mean_duration = float(codes_meta.get("mean_duration", 0.0))
                logging.info(
                    f"KPMS provenance (from .npz): kappa={kpms_kappa}, "
                    f"n_states={kpms_num_states}, model={kpms_model_type}, "
                    f"mean_duration={kpms_mean_duration:.1f}"
                )
        except Exception as e:
            logging.warning(f"Could not read .npz metadata: {e}")

        # Fallback: sweep_results.json alongside the codes file
        if kpms_kappa is None:
            sweep_results_path = Path(codes_path).parent / "sweep_results.json"
            if sweep_results_path.exists():
                try:
                    with open(sweep_results_path) as f:
                        sweep_results = json.load(f)
                    best = sweep_results.get("best_model", {})
                    kpms_kappa = best.get("kappa")
                    kpms_num_states = best.get("n_states")
                    kpms_model_type = best.get("model_type")
                    kpms_mean_duration = best.get("mean_duration")
                    logging.info(
                        f"KPMS provenance (from sweep_results.json): "
                        f"kappa={kpms_kappa}, n_states={kpms_num_states}, "
                        f"model={kpms_model_type}, mean_duration={kpms_mean_duration}"
                    )
                except Exception as e:
                    logging.warning(f"Could not read KPMS sweep results: {e}")

    wandb.config.update(
        {
            "arch": arch_name,
            "num_codes": num_codes,
            "code_embed_dim": int(cfg.network_config.code_embed_dim),
            "code_stack_size": code_stack_size,
            "codes_path": str(codes_path) if codes_path else "random",
            "kpms_kappa": kpms_kappa,
            "kpms_num_states": kpms_num_states,
            "kpms_model_type": kpms_model_type,
            "kpms_mean_duration": kpms_mean_duration,
            "use_continuous_encoder": use_continuous_encoder,
            "continuous_latent_dim": continuous_latent_dim,
            "kl_weight": kl_weight,
            "kl_schedule": kl_schedule,
            "use_rnn_decoder": use_rnn_decoder,
            "rnn_hidden_sizes": list(rnn_hidden_sizes) if use_rnn_decoder else None,
            "rnn_cell_type": rnn_cell_type if use_rnn_decoder else None,
            "z_e_dropout_rate": z_e_dropout_rate,
            "z_e_at_action_head": z_e_at_action_head,
            "reinit_hidden_on_code": reinit_hidden_on_code,
            "learned_hidden_init": learned_hidden_init,
            "use_distillation_head": use_distillation_head,
            "distill_kl_weight": distill_kl_weight if use_distillation_head else None,
            "distillation_encoder_checkpoint": (
                str(distillation_encoder_checkpoint) if distillation_encoder_checkpoint else None
            ),
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
        use_rnn_decoder=use_rnn_decoder,
        rnn_hidden_sizes=rnn_hidden_sizes,
        z_e_scale_fn=z_e_scale_fn,
        kl_weight_fn=kl_weight_fn,
        distill_kl_weight=distill_kl_weight if use_distillation_head else 0.0,
        post_init_params_fn=_post_init_params_fn,
    )

    # Rollout env for logging
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = MoSeqImitation(
        config=rollout_cfg, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    # Cache for decoder-only inference fn (created lazily once ppo_network available)
    _decoder_only_fn_cache = {}

    def _policy_params_fn_wrapper(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video=True,
        ppo_network=None,
    ):
        # Create decoder-only inference fn (z_e=0) from ppo_network
        # Skip in distillation mode (z_e never enters action path, so z_e=0 is
        # identical to z_e=1)
        jit_decoder_only_fn = None
        if use_continuous_encoder and not use_distillation_head and ppo_network is not None:
            if "fn" not in _decoder_only_fn_cache:
                from moseq_ppo_networks import (
                    make_moseq_recurrent_logging_inference_fn,
                    make_moseq_logging_inference_fn,
                )

                if use_rnn_decoder:
                    make_logging = make_moseq_recurrent_logging_inference_fn(
                        ppo_network
                    )
                else:
                    make_logging = make_moseq_logging_inference_fn(ppo_network)

                _decoder_only_fn_cache["fn"] = jax.jit(
                    make_logging(deterministic=True, z_e_scale=0.0)
                )
            jit_decoder_only_fn = _decoder_only_fn_cache["fn"]

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
            jit_decoder_only_inference_fn=jit_decoder_only_fn,
        )

    # Wrap wandb_progress to add `/`-prefixed metric groups for WandB panels
    def _grouped_wandb_progress(num_steps: int, metrics: dict) -> None:
        grouped = {}
        for k, v in metrics.items():
            if k in ("total_loss", "policy_loss", "v_loss", "entropy_loss"):
                grouped[f"losses/{k}"] = v
            elif k in ("kl_loss", "scaled_kl_loss", "z_e_norm", "z_e_std", "z_e_scale"):
                grouped[f"z_e/{k}"] = v
            elif k in ("transition_rate", "perplexity", "codebook_utilization", "codes_used"):
                grouped[f"codes/{k}"] = v
            elif k in ("hidden_state_norm",):
                grouped[f"rnn/{k}"] = v
            elif k in (
                "distill_kl_loss", "scaled_distill_kl_loss",
                "distill_mean_norm", "distill_logvar_mean",
            ):
                grouped[f"distillation/{k}"] = v
            # Keep originals for backward compat with shared code
            grouped[k] = v
        wandb_logging.wandb_progress(num_steps, grouped)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=_grouped_wandb_progress,
        policy_params_fn=_policy_params_fn_wrapper,
    )

    # Log final summary for sweep filtering
    try:
        wandb.run.summary.update({
            "training_completed": True,
            "kpms_kappa": kpms_kappa,
            "kpms_num_states": kpms_num_states,
        })
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
