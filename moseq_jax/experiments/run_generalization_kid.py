"""Experiment: Generalization KID Comparison (MIMIC vs C2A).

Samples long continuous segments from unseen data, runs KPMS to extract
syllable codes, then evaluates both MIMIC (oracle) and Code2Act on those
segments. Computes KID using the pre-trained VAE to quantify how close
each method's generalization distribution is to the training reference.

Usage:
    cd moseq_jax
    python -m experiments.run_generalization_kid
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from moseq_env_wrapper import MoSeqImitation

from experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    load_mimic_checkpoint,
    make_inference_fn,
    make_mimic_inference_fn,
    run_rollout,
)
from experiments.shared.plotting import set_nature_style

from experiments.run_generalization import (
    sample_segments,
    write_segmented_h5,
    extract_keypoints,
    run_kpms_inference,
)
from experiments.run_inception_distance import (
    preprocess_data,
    get_joint_start_index,
    compute_joint_normalization,
    normalize_joints,
    extract_features,
    compute_kid,
    compute_fid,
    filter_and_truncate,
    train_vae,
    collect_mimic_rollouts,
    _compute_vae_cache_key,
    _load_cached_vae,
    _save_vae_cache,
    _plot_vae_loss,
    compute_steps_per_frame,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def chunk_trajectories(
    raw_qpos_list: list[np.ndarray],
    chunk_control_steps: int,
    steps_per_frame: int,
) -> np.ndarray:
    """Chunk rollout trajectories and subsample to mocap rate.

    Each 2000-step trajectory is split into non-overlapping segments of
    ``chunk_control_steps`` (e.g., 450).  Each segment is subsampled to
    mocap rate, yielding ``chunk_control_steps // steps_per_frame`` mocap
    frames per chunk.

    Args:
        raw_qpos_list: List of ``(T_i, 74)`` arrays at control rate.
        chunk_control_steps: Control steps per chunk (450 for 225 mocap frames).
        steps_per_frame: Subsampling factor (2 for 100Hz ctrl → 50Hz mocap).

    Returns:
        ``(N_total_chunks, mocap_frames, 74)`` array.
    """
    mocap_frames = chunk_control_steps // steps_per_frame
    all_chunks = []
    for qpos in raw_qpos_list:
        # Drop the extra final qpos if present (run_rollout appends T+1)
        T = len(qpos)
        n_chunks = T // chunk_control_steps
        for c in range(n_chunks):
            start = c * chunk_control_steps
            chunk = qpos[start : start + chunk_control_steps]
            chunk_mocap = chunk[::steps_per_frame][:mocap_frames]
            all_chunks.append(chunk_mocap)
    return np.stack(all_chunks, axis=0) if all_chunks else np.zeros((0, mocap_frames, 74))


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------


def _train_or_load_vae(
    raw_list: list[np.ndarray],
    label: str,
    kid_cfg: DictConfig,
    output_dir: Path,
) -> tuple[list[tuple[dict, tuple, np.ndarray]], tuple | None]:
    """Filter/preprocess rollouts, train or load cached VAE, extract features.

    Generic helper used for both original-data and generalization-data VAEs.

    Args:
        raw_list: List of raw qpos arrays at control rate.
        label: Identifier for logging and loss plot filenames (e.g. "original", "generalization").
        kid_cfg: The kid_eval config section.
        output_dir: Experiment output directory.

    Returns:
        ``(seed_entries, joint_norm_params)``
    """
    import time

    vae_cfg = kid_cfg.vae
    pp = kid_cfg.preprocessing

    survival_threshold = int(kid_cfg.survival_threshold)
    steps_per_frame = int(kid_cfg.steps_per_frame)
    exclude_xy = bool(pp.exclude_xy)
    do_rotation = bool(pp.handle_rotation)
    do_normalize = bool(pp.normalize_joints)

    qpos_arr, n_kept, n_total = filter_and_truncate(
        raw_list, survival_threshold, steps_per_frame,
    )
    log.info(f"  [{label}] Reference clips: {n_kept}/{n_total} survived, shape {qpos_arr.shape}")

    real_data = preprocess_data(qpos_arr, exclude_xy, do_rotation)

    joint_norm_params = None
    if do_normalize:
        joint_start = get_joint_start_index(exclude_xy, do_rotation)
        joint_mean, joint_std = compute_joint_normalization(real_data, joint_start)
        joint_norm_params = (joint_start, joint_mean, joint_std)
        real_data = normalize_joints(real_data, joint_start, joint_mean, joint_std)

    input_size = int(np.prod(real_data.shape[1:]))
    log.info(f"  [{label}] Preprocessed: {real_data.shape}, input_size={input_size}")

    cache_dir = Path(str(kid_cfg.vae_cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include label in cache key so original vs generalization get distinct VAEs
    import hashlib as _hashlib
    base_key = _compute_vae_cache_key(
        real_data.shape,
        int(vae_cfg.latent_dim),
        list(vae_cfg.hidden_layers),
        int(vae_cfg.num_epochs),
        float(vae_cfg.beta),
        exclude_xy, do_rotation, do_normalize,
        float(vae_cfg.learning_rate),
        float(vae_cfg.weight_decay),
        float(vae_cfg.dropout_rate),
        bool(vae_cfg.use_layer_norm),
    )
    cache_key = _hashlib.sha256(f"{base_key}_{label}".encode()).hexdigest()[:16]
    log.info(f"  [{label}] VAE cache key: {cache_key}")

    seed_entries = []
    for seed in vae_cfg.seeds:
        seed = int(seed)

        cached_vae = _load_cached_vae(
            cache_dir, cache_key, seed,
            input_size=input_size,
            latent_dim=int(vae_cfg.latent_dim),
            hidden_layers=tuple(vae_cfg.hidden_layers),
            dropout_rate=float(vae_cfg.dropout_rate),
            use_layer_norm=bool(vae_cfg.use_layer_norm),
        )

        if cached_vae is not None:
            trained_params, network_fns = cached_vae
            log.info(f"  [{label}] Seed {seed}: loaded VAE from cache")
        else:
            warmup = vae_cfg.get("beta_warmup_epochs", None)
            if warmup is None:
                warmup = int(vae_cfg.num_epochs) // 2

            log.info(f"  [{label}] Seed {seed}: training VAE ({vae_cfg.num_epochs} epochs)...")
            t0 = time.time()
            trained_params, network_fns, train_metrics, epoch_losses = train_vae(
                data=real_data,
                input_size=input_size,
                latent_dim=int(vae_cfg.latent_dim),
                encoder_hidden_layer_sizes=tuple(vae_cfg.hidden_layers),
                decoder_hidden_layer_sizes=None,
                num_epochs=int(vae_cfg.num_epochs),
                batch_size=int(vae_cfg.batch_size),
                learning_rate=float(vae_cfg.learning_rate),
                weight_decay=float(vae_cfg.weight_decay),
                grad_clip_norm=float(vae_cfg.grad_clip_norm),
                dropout_rate=float(vae_cfg.dropout_rate),
                use_layer_norm=bool(vae_cfg.use_layer_norm),
                target_beta=float(vae_cfg.beta),
                beta_warmup_epochs=int(warmup),
                seed=seed,
            )
            _save_vae_cache(cache_dir, cache_key, seed, trained_params)
            _plot_vae_loss(
                epoch_losses,
                str(output_dir / f"vae_loss_{label}_seed{seed}.png"),
            )
            log.info(f"  [{label}] Seed {seed}: trained in {time.time() - t0:.1f}s")

        _, _, _, encode_fn = network_fns
        mu_ref = extract_features(trained_params, encode_fn, real_data)
        seed_entries.append((trained_params, network_fns, mu_ref))
        log.info(f"  [{label}] Seed {seed}: mu_ref shape {mu_ref.shape}")

    return seed_entries, joint_norm_params


def _collect_or_load_mimic_rollouts(
    rollout_cache: Path,
    label: str,
    clips: ReferenceClips,
    codes: np.ndarray,
    ckpt_cfg: DictConfig,
    mimic_params: tuple,
    mimic_ppo,
    n_rollouts: int,
    steps_per_frame: int,
    seed: int,
) -> list[np.ndarray]:
    """Collect mimic oracle rollouts or load from cache."""
    if rollout_cache.exists():
        log.info(f"Loading cached {label} mimic rollouts...")
        cached = np.load(rollout_cache, allow_pickle=True)
        n_clips = int(cached["n_clips"])
        raw_list = [cached[f"raw_{i}"] for i in range(n_clips)]
        log.info(f"  Loaded {n_clips} cached rollouts")
        return raw_list

    log.info(f"Collecting mimic oracle rollouts on {label} data ({n_rollouts} clips)...")
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    env_cfg.nconmax = 256
    env_cfg.njmax = 128
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

    env = MoSeqImitation(
        config=env_cfg,
        clips=clips,
        kpms_codes=codes,
        code_stack_size=code_stack_size,
    )
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    mimic_inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)
    clip_length = int(ckpt_cfg.env_config.clip_length)

    raw_list = collect_mimic_rollouts(
        env, mimic_inf_fn, mimic_params, mimic_ppo,
        n_clips=n_rollouts,
        clip_length=clip_length,
        steps_per_frame=steps_per_frame,
        seed=seed,
        jit_reset=jit_reset,
        jit_step=jit_step,
    )

    save_dict = {"n_clips": len(raw_list)}
    for i, q in enumerate(raw_list):
        save_dict[f"raw_{i}"] = q
    np.savez_compressed(rollout_cache, **save_dict)
    log.info(f"  Saved {len(raw_list)} rollouts to {rollout_cache}")
    return raw_list


def load_reference_and_vaes(
    cfg: DictConfig,
    ckpt_cfg: DictConfig,
    mimic_params: tuple,
    mimic_ppo,
    output_dir: Path,
    gen_seg_h5_path: Path | None = None,
    gen_codes: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Train/load two VAEs: one on original training data, one on generalization data.

    Args:
        gen_seg_h5_path: Path to the generalization segments H5 (for gen VAE).
        gen_codes: KPMS codes for generalization segments (for gen VAE env).

    Returns:
        ``(vae_results, joint_norm_params_dict)`` where:
        - vae_results: ``{"original": seed_entries, "generalization": seed_entries}``
        - joint_norm_params_dict: ``{"original": ..., "generalization": ...}``
    """
    kid_cfg = cfg.kid_eval
    steps_per_frame = int(kid_cfg.steps_per_frame)
    n_rollouts = int(kid_cfg.vae_training_clips)
    vae_seed = int(kid_cfg.vae_training_seed)

    data_dir = output_dir / "data"

    # --- Original data VAE ---
    log.info("\n--- Original data VAE ---")
    balanced_clips = ReferenceClips(
        data_path=str(kid_cfg.reference_data_path),
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
    )
    codes_data = np.load(str(kid_cfg.codes_path))
    orig_codes = codes_data["all_codes"]

    orig_raw = _collect_or_load_mimic_rollouts(
        rollout_cache=data_dir / "vae_training_rollouts_original.npz",
        label="original",
        clips=balanced_clips,
        codes=orig_codes,
        ckpt_cfg=ckpt_cfg,
        mimic_params=mimic_params,
        mimic_ppo=mimic_ppo,
        n_rollouts=n_rollouts,
        steps_per_frame=steps_per_frame,
        seed=vae_seed,
    )
    orig_entries, orig_norm = _train_or_load_vae(
        orig_raw, "original", kid_cfg, output_dir,
    )

    # --- Generalization data VAE ---
    log.info("\n--- Generalization data VAE ---")
    if gen_seg_h5_path is None or gen_codes is None:
        raise ValueError("gen_seg_h5_path and gen_codes required for generalization VAE")

    # Build clips from generalization segments (250-frame chunks)
    gen_clips = ReferenceClips(
        data_path=str(gen_seg_h5_path),
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
    )

    gen_raw = _collect_or_load_mimic_rollouts(
        rollout_cache=data_dir / "vae_training_rollouts_generalization.npz",
        label="generalization",
        clips=gen_clips,
        codes=gen_codes,
        ckpt_cfg=ckpt_cfg,
        mimic_params=mimic_params,
        mimic_ppo=mimic_ppo,
        n_rollouts=n_rollouts,
        steps_per_frame=steps_per_frame,
        seed=vae_seed + 1000,
    )
    gen_entries, gen_norm = _train_or_load_vae(
        gen_raw, "generalization", kid_cfg, output_dir,
    )

    vae_results = {"original": orig_entries, "generalization": gen_entries}
    norm_params = {"original": orig_norm, "generalization": gen_norm}
    return vae_results, norm_params


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Wong colorblind-safe palette
COLORS = {
    "mimic_mjx": "#0072B2",  # blue
    "code2act": "#D55E00",   # vermillion
}
LABELS = {
    "mimic_mjx": "MIMIC-MJX",
    "code2act": "Code2Act",
}


def plot_kid_comparison(
    results: dict,
    output_path: str,
) -> None:
    """Create KID comparison barplot: MIMIC vs C2A on generalization data."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    methods = ["mimic_mjx", "code2act"]
    x_pos = np.arange(len(methods))
    means = [results[m]["kid_mean"] for m in methods]
    stds = [results[m]["kid_std"] for m in methods]
    colors = [COLORS[m] for m in methods]
    labels = [LABELS[m] for m in methods]

    bars = ax.bar(
        x_pos, means, yerr=stds, width=0.5,
        color=colors, alpha=0.85, capsize=5, edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("KID (vs. training reference)")
    ax.set_title("Generalization Quality")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value annotations
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"KID barplot saved to: {output_path}")


def plot_reward_comparison(
    mimic_rewards: np.ndarray,
    c2a_rewards: np.ndarray,
    output_path: str,
) -> None:
    """Mean reward per timestep for MIMIC vs C2A."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(4, 2.5))

    T = min(mimic_rewards.shape[1], c2a_rewards.shape[1])
    ts = np.arange(T)
    ax.plot(ts, mimic_rewards[:, :T].mean(axis=0), color=COLORS["mimic_mjx"],
            linewidth=1, label=LABELS["mimic_mjx"])
    ax.plot(ts, c2a_rewards[:, :T].mean(axis=0), color=COLORS["code2act"],
            linewidth=1, label=LABELS["code2act"])

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.set_title("Mean Reward on Generalization Data")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Reward plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="generalization_kid_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Generalization KID Comparison: MIMIC vs C2A ===")

    output_dir = Path(cfg.output.base_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    K = int(cfg.K)
    frames_per_segment = int(cfg.frames_per_segment)
    seed = int(cfg.seed)
    n_render = int(cfg.n_render)

    # ==================================================================
    # Stage 1: Sample long segments from continuous unseen data
    # ==================================================================
    log.info(f"\n--- Stage 1: Sampling {K} segments of {frames_per_segment} frames ---")
    seg_data = sample_segments(
        cfg.new_data.path,
        n_segments=K,
        frames_per_segment=frames_per_segment,
        seed=seed,
    )

    seg_h5_path = data_dir / "segments.h5"
    write_segmented_h5(seg_data, str(seg_h5_path))

    # ==================================================================
    # Stage 2: FK → keypoints → KPMS → codes
    # ==================================================================
    log.info("\n--- Stage 2: Extracting keypoints + KPMS codes ---")

    codes_path = data_dir / "kpms_codes.npz"
    if codes_path.exists():
        log.info("  Loading cached KPMS codes...")
        codes = np.load(codes_path)["codes"]
    else:
        kps, kp_names = extract_keypoints(
            seg_data["qpos"], cfg.reference_h5, cfg.stac_xml,
        )
        kps = kps.astype(np.float64)

        # Batch KPMS to avoid OOM (10 segments per batch)
        kpms_batch_size = 10
        all_codes = []
        kps_3d = kps.reshape(K, frames_per_segment, -1, 3)

        for b_start in range(0, K, kpms_batch_size):
            b_end = min(b_start + kpms_batch_size, K)
            b_size = b_end - b_start
            log.info(f"  KPMS batch {b_start}-{b_end-1} ({b_size} segments)...")
            batch_kps = kps_3d[b_start:b_end].reshape(b_size * frames_per_segment, -1, 3)
            batch_codes = run_kpms_inference(
                batch_kps,
                n_segments=b_size,
                frames_per_segment=frames_per_segment,
                model_dir=cfg.kpms.model_dir,
                model_name=cfg.kpms.model_name,
                num_iters=int(cfg.kpms.num_iters),
            )
            all_codes.append(batch_codes)

        codes = np.concatenate(all_codes, axis=0)
        np.savez_compressed(codes_path, codes=codes)

    log.info(f"  Codes shape: {codes.shape}")

    # ==================================================================
    # Stage 3: Load checkpoints
    # ==================================================================
    log.info("\n--- Stage 3: Loading checkpoints ---")

    # C2A
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path,
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    c2a_params = (norm_state, policy_params)

    # MIMIC
    mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
        cfg.mimic_checkpoint.path,
    )
    mimic_params = (mimic_norm, mimic_policy)

    # ==================================================================
    # Stage 4: Run 2000-step rollouts in batches of 10
    # ==================================================================
    # Batching avoids slow JIT with large reference clip arrays.
    # Each batch loads 10 clips (matching hidden_trajectory's K=10).
    rollouts_path = data_dir / "rollouts.npz"
    if rollouts_path.exists():
        log.info("\n--- Stage 4: Loading cached rollouts ---")
        cached_rollouts = np.load(rollouts_path)
        c2a_qpos_list = list(cached_rollouts["c2a_qpos"])
        c2a_rewards_list = list(cached_rollouts["c2a_rewards"])
        mimic_qpos_list = list(cached_rollouts["mimic_qpos"])
        mimic_rewards_list = list(cached_rollouts["mimic_rewards"])
        c2a_mean_rew = np.mean([r.mean() for r in c2a_rewards_list])
        mimic_mean_rew = np.mean([r.mean() for r in mimic_rewards_list])
        log.info(f"  Loaded {len(c2a_qpos_list)} rollouts per method")
        log.info(f"  C2A mean reward: {c2a_mean_rew:.2f}")
        log.info(f"  MIMIC mean reward: {mimic_mean_rew:.2f}")
    else:
        log.info("\n--- Stage 4: Running rollouts (batched) ---")

    if not rollouts_path.exists():
        _, _, env_cfg = utils.prepare_config(ckpt_cfg)
        env_cfg.start_frame_range = [0, 0]
        env_cfg.domain_randomization.use_domain_randomization = False
        env_cfg.clip_length = frames_per_segment
        env_cfg.nconmax = 256
        env_cfg.njmax = 128
        code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

        c2a_inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)
        mimic_inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)

        batch_size = 10  # clips per batch (matches hidden_trajectory K=10)
        c2a_qpos_list = []
        c2a_rewards_list = []
        mimic_qpos_list = []
        mimic_rewards_list = []

        for b_start in range(0, K, batch_size):
            b_end = min(b_start + batch_size, K)
            b_size = b_end - b_start
            log.info(f"\n  Batch {b_start}-{b_end-1} ({b_size} clips)...")

            # Write batch H5
            batch_h5_path = data_dir / f"batch_{b_start}.h5"
            batch_seg = {
                k: seg_data[k].reshape(K, frames_per_segment, *seg_data[k].shape[1:])[b_start:b_end].reshape(-1, *seg_data[k].shape[1:])
                for k in ("qpos", "qvel", "xpos", "xquat")
            }
            batch_seg["names_qpos"] = seg_data["names_qpos"]
            batch_seg["names_xpos"] = seg_data["names_xpos"]
            if "config" in seg_data:
                batch_seg["config"] = seg_data["config"]
            write_segmented_h5(batch_seg, str(batch_h5_path))

            batch_clips = ReferenceClips(
                data_path=str(batch_h5_path),
                n_frames_per_clip=frames_per_segment,
            )
            batch_codes = codes[b_start:b_end]

            env = MoSeqImitation(
                config=env_cfg,
                clips=batch_clips,
                kpms_codes=batch_codes,
                code_stack_size=code_stack_size,
            )
            jit_reset = jax.jit(env.reset)
            jit_step = jax.jit(env.step)

            # C2A rollouts for this batch
            for i in range(b_size):
                ki = b_start + i
                key = jax.random.PRNGKey(seed + ki * 1000)
                result = run_rollout(
                    env, c2a_inf_fn, c2a_params, ppo_networks,
                    use_rnn=use_rnn, key=key,
                    max_steps=frames_per_segment,
                    code_override=batch_codes[i],
                    reset_clip_idx=i,
                    jit_reset=jit_reset, jit_step=jit_step,
                    model_type="code2act",
                    ignore_done=True,
                )
                c2a_qpos_list.append(result["qpos"][:frames_per_segment])
                c2a_rewards_list.append(result["rewards"])
                log.info(
                    f"    C2A {ki+1}/{K}: survival={result['survival']}, "
                    f"mean_reward={result['rewards'].mean():.1f}"
                )

            # MIMIC rollouts for this batch
            for i in range(b_size):
                ki = b_start + i
                key = jax.random.PRNGKey(seed + ki * 1000 + 500)
                result = run_rollout(
                    env, mimic_inf_fn, mimic_params, mimic_ppo,
                    use_rnn=False, key=key,
                    max_steps=frames_per_segment,
                    reset_clip_idx=i,
                    jit_reset=jit_reset, jit_step=jit_step,
                    model_type="mimic_mjx",
                    ignore_done=True,
                )
                mimic_qpos_list.append(result["qpos"][:frames_per_segment])
                mimic_rewards_list.append(result["rewards"])
                log.info(
                    f"    MIMIC {ki+1}/{K}: survival={result['survival']}, "
                    f"mean_reward={result['rewards'].mean():.1f}"
                )

            # Clean up batch H5
            batch_h5_path.unlink(missing_ok=True)

        # Save raw rollouts for rendering
        log.info("Saving rollout data...")
        np.savez_compressed(
            rollouts_path,
            c2a_qpos=np.stack(c2a_qpos_list),
            c2a_rewards=np.stack(c2a_rewards_list),
            mimic_qpos=np.stack(mimic_qpos_list),
            mimic_rewards=np.stack(mimic_rewards_list),
            codes=codes,
            n_render=n_render,
        )

        c2a_mean_rew = np.mean([r.mean() for r in c2a_rewards_list])
        mimic_mean_rew = np.mean([r.mean() for r in mimic_rewards_list])
        log.info(f"  C2A mean reward: {c2a_mean_rew:.2f}")
        log.info(f"  MIMIC mean reward: {mimic_mean_rew:.2f}")

        plot_reward_comparison(
            np.stack(mimic_rewards_list),
            np.stack(c2a_rewards_list),
            str(output_dir / "reward_comparison.png"),
        )

    # ==================================================================
    # Stage 4b: Training data rollouts (for in-distribution KID)
    # ==================================================================
    train_rollouts_path = data_dir / "train_rollouts.npz"

    if train_rollouts_path.exists():
        log.info("\n--- Stage 4b: Loading cached training data rollouts ---")
        cached_train = np.load(train_rollouts_path)
        train_c2a_qpos_list = list(cached_train["c2a_qpos"])
        train_mimic_qpos_list = list(cached_train["mimic_qpos"])
        log.info(f"  Loaded {len(train_c2a_qpos_list)} train rollouts per method")
    else:
        log.info(f"\n--- Stage 4b: Training data rollouts ({K} clips) ---")

        # Load training clips + codes
        balanced_clips = ReferenceClips(
            data_path=str(cfg.kid_eval.reference_data_path),
            n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
        )
        codes_data = np.load(str(cfg.kid_eval.codes_path))
        train_codes = codes_data["all_codes"]

        _, _, train_env_cfg = utils.prepare_config(ckpt_cfg)
        train_env_cfg.start_frame_range = [0, 0]
        train_env_cfg.domain_randomization.use_domain_randomization = False
        train_env_cfg.nconmax = 256
        train_env_cfg.njmax = 128
        code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

        c2a_inf_fn_tr = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)
        mimic_inf_fn_tr = make_mimic_inference_fn(mimic_ppo, deterministic=True)

        # Use first K clips from training data
        n_train_clips = min(K, len(train_codes))
        train_c2a_qpos_list = []
        train_mimic_qpos_list = []

        # Batched rollouts on training data
        batch_size_tr = 10
        for b_start in range(0, n_train_clips, batch_size_tr):
            b_end = min(b_start + batch_size_tr, n_train_clips)
            b_size = b_end - b_start
            log.info(f"  Train batch {b_start}-{b_end-1}...")

            batch_clips_tr = ReferenceClips(
                data_path=str(cfg.kid_eval.reference_data_path),
                n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
                keep_clips_idx=np.arange(b_start, b_end),
            )
            batch_codes_tr = train_codes[b_start:b_end]

            train_env = MoSeqImitation(
                config=train_env_cfg,
                clips=batch_clips_tr,
                kpms_codes=batch_codes_tr,
                code_stack_size=code_stack_size,
            )
            jit_reset_tr = jax.jit(train_env.reset)
            jit_step_tr = jax.jit(train_env.step)

            # clip_length in mocap frames -> control steps
            max_ctrl_steps = int(ckpt_cfg.env_config.clip_length) * int(cfg.kid_eval.steps_per_frame)

            for i in range(b_size):
                ki = b_start + i
                key = jax.random.PRNGKey(seed + ki * 1000 + 2000)
                result = run_rollout(
                    train_env, c2a_inf_fn_tr, c2a_params, ppo_networks,
                    use_rnn=use_rnn, key=key,
                    max_steps=max_ctrl_steps,
                    code_override=batch_codes_tr[i],
                    reset_clip_idx=i,
                    jit_reset=jit_reset_tr, jit_step=jit_step_tr,
                    model_type="code2act",
                    ignore_done=True,
                )
                train_c2a_qpos_list.append(result["qpos"][:max_ctrl_steps])

                key = jax.random.PRNGKey(seed + ki * 1000 + 2500)
                result = run_rollout(
                    train_env, mimic_inf_fn_tr, mimic_params, mimic_ppo,
                    use_rnn=False, key=key,
                    max_steps=max_ctrl_steps,
                    reset_clip_idx=i,
                    jit_reset=jit_reset_tr, jit_step=jit_step_tr,
                    model_type="mimic_mjx",
                    ignore_done=True,
                )
                train_mimic_qpos_list.append(result["qpos"][:max_ctrl_steps])
                log.info(f"    Train clip {ki+1}/{n_train_clips} done")

        np.savez_compressed(
            train_rollouts_path,
            c2a_qpos=np.stack(train_c2a_qpos_list),
            mimic_qpos=np.stack(train_mimic_qpos_list),
        )
        log.info(f"  Saved training rollouts: {train_rollouts_path}")

    # ==================================================================
    # Stage 5: Chunk all rollouts into 250-step segments for KID
    # ==================================================================
    log.info("\n--- Stage 5: Chunking and preprocessing ---")

    kid_cfg = cfg.kid_eval
    chunk_steps = int(kid_cfg.chunk_control_steps)  # 250
    steps_per_frame = int(kid_cfg.steps_per_frame)  # 2

    # Generalization rollouts
    c2a_chunks = chunk_trajectories(c2a_qpos_list, chunk_steps, steps_per_frame)
    mimic_chunks = chunk_trajectories(mimic_qpos_list, chunk_steps, steps_per_frame)

    n_chunks_per_body = frames_per_segment // chunk_steps
    log.info(
        f"  Gen C2A: {c2a_chunks.shape} "
        f"({K} bodies × {n_chunks_per_body} chunks)"
    )
    log.info(f"  Gen MIMIC: {mimic_chunks.shape}")

    # Training rollouts
    train_c2a_chunks = chunk_trajectories(train_c2a_qpos_list, chunk_steps, steps_per_frame)
    train_mimic_chunks = chunk_trajectories(train_mimic_qpos_list, chunk_steps, steps_per_frame)
    log.info(f"  Train C2A: {train_c2a_chunks.shape}")
    log.info(f"  Train MIMIC: {train_mimic_chunks.shape}")

    # ==================================================================
    # Stage 6: Train/load both VAEs, compute KID
    # ==================================================================
    log.info("\n--- Stage 6: VAE + KID computation ---")

    vae_results, norm_params = load_reference_and_vaes(
        cfg=cfg,
        ckpt_cfg=ckpt_cfg,
        mimic_params=mimic_params,
        mimic_ppo=mimic_ppo,
        output_dir=output_dir,
        gen_seg_h5_path=seg_h5_path,
        gen_codes=codes,
    )

    pp = kid_cfg.preprocessing
    exclude_xy = bool(pp.exclude_xy)
    do_rotation = bool(pp.handle_rotation)
    do_normalize = bool(pp.normalize_joints)
    vae_cfg = kid_cfg.vae
    kid_params = kid_cfg.kid

    # Compute KID for each (rollout_source, vae_type) combination
    # 1. gen rollouts + original VAE  ("tst_orig")
    # 2. gen rollouts + gen VAE       ("tst_gen")
    # 3. train rollouts + original VAE ("trn_orig")
    all_aggregated = {}

    kid_configs = [
        ("tst_orig",  c2a_chunks,       mimic_chunks,       "original"),
        ("tst_gen",   c2a_chunks,       mimic_chunks,       "generalization"),
        ("trn_orig",  train_c2a_chunks, train_mimic_chunks, "original"),
    ]

    for kid_label, c2a_ch, mimic_ch, vae_label in kid_configs:
        seed_entries = vae_results[vae_label]
        joint_norm_params = norm_params[vae_label]

        c2a_data = preprocess_data(c2a_ch, exclude_xy, do_rotation)
        mimic_data = preprocess_data(mimic_ch, exclude_xy, do_rotation)
        if joint_norm_params is not None:
            js, jm, jstd = joint_norm_params
            c2a_data = normalize_joints(c2a_data, js, jm, jstd)
            mimic_data = normalize_joints(mimic_data, js, jm, jstd)

        log.info(f"\n  [{kid_label}] C2A: {c2a_data.shape}, MIMIC: {mimic_data.shape}")

        all_seed_results = []
        for seed_idx, (vae_params, network_fns, mu_ref) in enumerate(seed_entries):
            vae_seed = list(vae_cfg.seeds)[seed_idx]
            _, _, _, encode_fn = network_fns

            mu_c2a = extract_features(vae_params, encode_fn, c2a_data)
            mu_mimic = extract_features(vae_params, encode_fn, mimic_data)

            split_seed = int(kid_params.split_seed)
            split_rng = np.random.default_rng(split_seed)
            mid = len(mu_ref) // 2
            idx = split_rng.permutation(len(mu_ref))
            mu_ref_half = mu_ref[idx[:mid]]

            c2a_kid_mean, c2a_kid_std = compute_kid(
                mu_ref_half, mu_c2a,
                degree=int(kid_params.degree),
                num_subsets=int(kid_params.num_subsets),
                subset_size=kid_params.get("subset_size", None),
                seed=split_seed,
            )
            mimic_kid_mean, mimic_kid_std = compute_kid(
                mu_ref_half, mu_mimic,
                degree=int(kid_params.degree),
                num_subsets=int(kid_params.num_subsets),
                subset_size=kid_params.get("subset_size", None),
                seed=split_seed,
            )

            c2a_fid = compute_fid(mu_ref_half, mu_c2a)
            mimic_fid = compute_fid(mu_ref_half, mu_mimic)

            all_seed_results.append({
                "seed": int(vae_seed),
                "code2act": {"kid_mean": c2a_kid_mean, "kid_std": c2a_kid_std, "fid": c2a_fid},
                "mimic_mjx": {"kid_mean": mimic_kid_mean, "kid_std": mimic_kid_std, "fid": mimic_fid},
            })
            log.info(
                f"  [{kid_label}] Seed {vae_seed}: C2A KID={c2a_kid_mean:.4f}, "
                f"MIMIC KID={mimic_kid_mean:.4f}"
            )

        aggregated = {}
        for method in ["mimic_mjx", "code2act"]:
            kid_means = [r[method]["kid_mean"] for r in all_seed_results]
            fids = [r[method]["fid"] for r in all_seed_results]
            aggregated[method] = {
                "kid_mean": float(np.mean(kid_means)),
                "kid_std": float(np.std(kid_means)),
                "fid_mean": float(np.mean(fids)),
                "fid_std": float(np.std(fids)),
            }
        all_aggregated[kid_label] = {
            "per_seed_results": all_seed_results,
            "aggregated": aggregated,
        }

    # ==================================================================
    # Stage 7: Save results + plot
    # ==================================================================
    log.info("\n--- Stage 7: Saving results ---")

    output_data = {
        "metadata": {
            "K": K,
            "frames_per_segment": frames_per_segment,
            "n_render": n_render,
            "chunk_control_steps": chunk_steps,
            "mocap_frames_per_chunk": chunk_steps // steps_per_frame,
            "chunks_per_body": c2a_chunks.shape[0] // K,
            "c2a_mean_reward": float(c2a_mean_rew),
            "mimic_mean_reward": float(mimic_mean_rew),
            "timestamp": datetime.now().isoformat(),
        },
        # All 3 KID conditions
        "tst_orig": all_aggregated["tst_orig"],
        "tst_gen": all_aggregated["tst_gen"],
        "trn_orig": all_aggregated["trn_orig"],
        # Backward compat
        "original_vae": all_aggregated["tst_orig"],
        "generalization_vae": all_aggregated["tst_gen"],
        "per_seed_results": all_aggregated["tst_orig"]["per_seed_results"],
        "aggregated": all_aggregated["tst_orig"]["aggregated"],
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to: {json_path}")

    # KID barplots
    for kid_label in all_aggregated:
        plot_kid_comparison(
            all_aggregated[kid_label]["aggregated"],
            str(output_dir / f"kid_comparison_{kid_label}.png"),
        )

    # Copy data to figures/data
    fig_data_dir = MOSEQ_DIR / "figures" / "data"
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / "rollouts.npz", fig_data_dir / "generalization_kid_rollouts.npz")
    shutil.copy2(json_path, fig_data_dir / "generalization_kid_results.json")
    log.info(f"Copied to: {fig_data_dir}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Bodies: {K}, Chunks/body: {c2a_chunks.shape[0] // K}")
    for kl in ["tst_orig", "tst_gen", "trn_orig"]:
        agg = all_aggregated[kl]["aggregated"]
        log.info(f"\n  [{kl}]")
        log.info(f"  C2A  : KID={agg['code2act']['kid_mean']:.4f} ± {agg['code2act']['kid_std']:.4f}")
        log.info(f"  MIMIC: KID={agg['mimic_mjx']['kid_mean']:.4f} ± {agg['mimic_mjx']['kid_std']:.4f}")


if __name__ == "__main__":
    main()
