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

import flax
import flax.serialization
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
from experiments.shared.vae_network import make_vae_network
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


def load_reference_and_vaes(
    original_rollout_path: str,
    survival_threshold: int,
    steps_per_frame: int,
    exclude_xy: bool,
    handle_rotation: bool,
    normalize_joints_flag: bool,
    vae_cache_dir: str,
    vae_cache_key: str,
    vae_latent_dim: int,
    vae_hidden_layers: tuple[int, ...],
    vae_dropout_rate: float,
    vae_use_layer_norm: bool,
    vae_seeds: list[int],
) -> tuple[
    list[tuple[dict, tuple, np.ndarray]],
    tuple[int, np.ndarray, np.ndarray] | None,
]:
    """Load original mimic rollouts, preprocess, load cached VAEs, extract features.

    Returns:
        ``(seed_entries, joint_norm_params)`` where:
        - seed_entries: List of ``(params, network_fns, mu_ref)`` per seed.
          ``mu_ref`` is the reference feature vector array for that seed's VAE.
        - joint_norm_params: ``(joint_start, mean, std)`` or None.
    """
    log.info("Loading original mimic rollouts for reference distribution...")
    cached = np.load(original_rollout_path, allow_pickle=True)
    n_clips = int(cached["n_clips"])
    raw_list = [cached[f"raw_{i}"] for i in range(n_clips)]

    # Filter and truncate (same as original experiment)
    qpos_arr, n_kept, n_total = filter_and_truncate(
        raw_list, survival_threshold, steps_per_frame,
    )
    log.info(f"  Reference clips: {n_kept}/{n_total} survived, shape {qpos_arr.shape}")

    # Preprocess
    real_data = preprocess_data(qpos_arr, exclude_xy, handle_rotation)

    # Normalize joints
    joint_norm_params = None
    if normalize_joints_flag:
        joint_start = get_joint_start_index(exclude_xy, handle_rotation)
        joint_mean, joint_std = compute_joint_normalization(real_data, joint_start)
        joint_norm_params = (joint_start, joint_mean, joint_std)
        real_data = normalize_joints(real_data, joint_start, joint_mean, joint_std)

    input_size = int(np.prod(real_data.shape[1:]))
    log.info(f"  Reference data shape: {real_data.shape}, input_size={input_size}")

    # Load cached VAEs and extract reference features
    cache_dir = Path(vae_cache_dir)
    seed_entries = []

    for seed in vae_seeds:
        cache_path = cache_dir / f"{vae_cache_key}_seed{seed}.msgpack"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"VAE cache not found: {cache_path}. "
                "Run inception_distance experiment first."
            )

        vae, init_fn, apply_fn, encode_fn = make_vae_network(
            input_size=input_size,
            latent_dim=vae_latent_dim,
            encoder_hidden_layer_sizes=vae_hidden_layers,
            dropout_rate=vae_dropout_rate,
            use_layer_norm=vae_use_layer_norm,
        )
        dummy_params = init_fn(jax.random.PRNGKey(0))
        with open(cache_path, "rb") as f:
            params = flax.serialization.from_bytes(dummy_params, f.read())

        mu_ref = extract_features(params, encode_fn, real_data)
        seed_entries.append((params, (vae, init_fn, apply_fn, encode_fn), mu_ref))
        log.info(f"  Seed {seed}: mu_ref shape {mu_ref.shape}")

    return seed_entries, joint_norm_params


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
    # Stage 5: Chunk 2000-step rollouts into 4 × 450 for KID
    # ==================================================================
    log.info("\n--- Stage 5: Chunking and preprocessing ---")

    kid_cfg = cfg.kid_eval
    chunk_steps = int(kid_cfg.chunk_control_steps)  # 450
    steps_per_frame = int(kid_cfg.steps_per_frame)  # 2

    # Each 2000-step rollout → 4 chunks of 450 steps → 225 mocap frames each
    c2a_chunks = chunk_trajectories(c2a_qpos_list, chunk_steps, steps_per_frame)
    mimic_chunks = chunk_trajectories(mimic_qpos_list, chunk_steps, steps_per_frame)

    n_chunks_per_body = frames_per_segment // chunk_steps
    log.info(
        f"  C2A: {c2a_chunks.shape} "
        f"({K} bodies × {n_chunks_per_body} chunks)"
    )
    log.info(f"  MIMIC: {mimic_chunks.shape}")

    # ==================================================================
    # Stage 6: Load reference + VAE, compute KID
    # ==================================================================
    log.info("\n--- Stage 6: KID computation ---")

    pp = kid_cfg.preprocessing
    exclude_xy = bool(pp.exclude_xy)
    do_rotation = bool(pp.handle_rotation)
    do_normalize = bool(pp.normalize_joints)

    vae_cfg = kid_cfg.vae
    seed_entries, joint_norm_params = load_reference_and_vaes(
        original_rollout_path=str(kid_cfg.original_mimic_rollouts),
        survival_threshold=int(kid_cfg.survival_threshold),
        steps_per_frame=steps_per_frame,
        exclude_xy=exclude_xy,
        handle_rotation=do_rotation,
        normalize_joints_flag=do_normalize,
        vae_cache_dir=str(kid_cfg.vae_cache_dir),
        vae_cache_key=str(kid_cfg.vae_cache_key),
        vae_latent_dim=int(vae_cfg.latent_dim),
        vae_hidden_layers=tuple(vae_cfg.hidden_layers),
        vae_dropout_rate=float(vae_cfg.dropout_rate),
        vae_use_layer_norm=bool(vae_cfg.use_layer_norm),
        vae_seeds=list(vae_cfg.seeds),
    )

    # Preprocess generalization data with SAME normalization as reference
    c2a_data = preprocess_data(c2a_chunks, exclude_xy, do_rotation)
    mimic_data = preprocess_data(mimic_chunks, exclude_xy, do_rotation)
    if joint_norm_params is not None:
        js, jm, jstd = joint_norm_params
        c2a_data = normalize_joints(c2a_data, js, jm, jstd)
        mimic_data = normalize_joints(mimic_data, js, jm, jstd)

    log.info(f"  C2A preprocessed: {c2a_data.shape}")
    log.info(f"  MIMIC preprocessed: {mimic_data.shape}")

    # Compute KID per seed, then aggregate
    kid_params = kid_cfg.kid
    all_seed_results = []

    for seed_idx, (vae_params, network_fns, mu_ref) in enumerate(seed_entries):
        vae_seed = list(vae_cfg.seeds)[seed_idx]
        _, _, _, encode_fn = network_fns

        mu_c2a = extract_features(vae_params, encode_fn, c2a_data)
        mu_mimic = extract_features(vae_params, encode_fn, mimic_data)

        # Use half of ref for KID (same protocol as original experiment)
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

        seed_result = {
            "seed": int(vae_seed),
            "code2act": {"kid_mean": c2a_kid_mean, "kid_std": c2a_kid_std, "fid": c2a_fid},
            "mimic_mjx": {"kid_mean": mimic_kid_mean, "kid_std": mimic_kid_std, "fid": mimic_fid},
        }
        all_seed_results.append(seed_result)
        log.info(
            f"  Seed {vae_seed}: C2A KID={c2a_kid_mean:.4f}, "
            f"MIMIC KID={mimic_kid_mean:.4f}"
        )

    # Aggregate across seeds
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
        "per_seed_results": all_seed_results,
        "aggregated": aggregated,
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to: {json_path}")

    # KID barplot
    plot_kid_comparison(aggregated, str(output_dir / "kid_comparison.png"))

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
    log.info(f"C2A  : KID={aggregated['code2act']['kid_mean']:.4f} ± {aggregated['code2act']['kid_std']:.4f}, "
             f"FID={aggregated['code2act']['fid_mean']:.2f}, reward={c2a_mean_rew:.2f}")
    log.info(f"MIMIC: KID={aggregated['mimic_mjx']['kid_mean']:.4f} ± {aggregated['mimic_mjx']['kid_std']:.4f}, "
             f"FID={aggregated['mimic_mjx']['fid_mean']:.2f}, reward={mimic_mean_rew:.2f}")


if __name__ == "__main__":
    main()
