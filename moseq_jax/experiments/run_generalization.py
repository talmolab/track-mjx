"""Experiment 3: Generalization to unseen continuous data.

Samples segments from a new unsegmented recording, runs a trained KPMS
model to extract syllable codes, then evaluates code2act vs mimic-mjx
on those segments.

Usage:
    cd moseq_jax
    python -m experiments.run_generalization
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
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
from experiments.shared.keypoint_fk import setup_stac_model, qpos_to_keypoints_fk
from experiments.shared.metrics import decompose_rewards
from experiments.shared.plotting import (
    set_nature_style,
    fig_to_image,
    get_trajectory_colors,
    get_code_colormap,
    MODE_COLORS,
    MODE_LABELS,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Sample & segment new data
# ---------------------------------------------------------------------------


def sample_segments(
    h5_path: str,
    n_segments: int,
    frames_per_segment: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sample non-overlapping segments from continuous recording.

    Args:
        h5_path: Path to continuous H5 (flat arrays, no snips_order).
        n_segments: Number of segments to sample.
        frames_per_segment: Frames per segment.
        seed: Random seed.

    Returns:
        Dict with ``qpos``, ``qvel``, ``xpos``, ``xquat`` each shaped
        ``[n_segments * frames_per_segment, ...]`` (concatenated flat),
        plus ``start_indices`` array.
    """
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, "r") as f:
        total_frames = f["qpos"].shape[0]
        max_start = total_frames - frames_per_segment

        # Sample non-overlapping start indices
        starts = []
        used = set()
        attempts = 0
        while len(starts) < n_segments and attempts < n_segments * 100:
            s = rng.randint(0, max_start)
            # Check no overlap with existing segments
            overlap = False
            for existing in starts:
                if abs(s - existing) < frames_per_segment:
                    overlap = True
                    break
            if not overlap:
                starts.append(s)
            attempts += 1

        starts = sorted(starts)
        log.info(f"  Sampled {len(starts)} segments from {total_frames} frames")

        # Extract data for each segment
        arrays = {k: [] for k in ("qpos", "qvel", "xpos", "xquat")}
        for s in starts:
            sl = slice(s, s + frames_per_segment)
            for k in arrays:
                arrays[k].append(f[k][sl])

        # Also grab metadata
        names_qpos = f["names_qpos"][:]
        names_xpos = f["names_xpos"][:]
        config_blob = f["config"][()] if "config" in f else None

    result = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    result["start_indices"] = np.array(starts)
    result["names_qpos"] = names_qpos
    result["names_xpos"] = names_xpos
    if config_blob is not None:
        result["config"] = config_blob
    return result


def write_segmented_h5(
    data: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Write sampled segments to an H5 file loadable by ReferenceClips."""
    with h5py.File(output_path, "w") as f:
        # Write float arrays as float32 (MuJoCo/Brax expects f32, not f64)
        for key in ("qpos", "qvel", "xpos", "xquat"):
            f.create_dataset(key, data=data[key].astype(np.float32))
        for key in ("names_qpos", "names_xpos"):
            f.create_dataset(key, data=data[key])
        if "config" in data:
            f.create_dataset("config", data=data["config"])
    log.info(f"  Written segmented H5: {output_path}")


# ---------------------------------------------------------------------------
# Step 2: Extract keypoints via FK
# ---------------------------------------------------------------------------


def extract_keypoints(
    qpos: np.ndarray,
    reference_h5: str,
    xml_path: str,
) -> tuple[np.ndarray, list[str]]:
    """Convert qpos to keypoints via MuJoCo FK.

    Args:
        qpos: Joint positions ``[N, 74]``.
        reference_h5: H5 with config/offsets/kp_names for FK setup.
        xml_path: Path to stac-mjx rodent XML.

    Returns:
        ``(keypoints, kp_names)`` where keypoints is ``[N, K, 3]``.
    """
    log.info("  Setting up FK model...")
    mj_model, mj_data, site_ids, kp_names = setup_stac_model(reference_h5, xml_path)
    log.info(f"  Running FK on {qpos.shape[0]} frames ({len(kp_names)} keypoints)...")
    kps = qpos_to_keypoints_fk(qpos, mj_model, mj_data, site_ids)
    return kps, kp_names


# ---------------------------------------------------------------------------
# Step 3: Run KPMS inference
# ---------------------------------------------------------------------------


def run_kpms_inference(
    keypoints: np.ndarray,
    n_segments: int,
    frames_per_segment: int,
    model_dir: str,
    model_name: str,
    num_iters: int = 200,
) -> np.ndarray:
    """Run trained KPMS model on new keypoint data.

    Args:
        keypoints: ``[n_segments * frames_per_segment, K, 3]``.
        n_segments: Number of segments.
        frames_per_segment: Frames per segment.
        model_dir: Path to KPMS model directory.
        model_name: Model checkpoint name.
        num_iters: Gibbs sampling iterations.

    Returns:
        Syllable codes ``[n_segments, frames_per_segment]``.
    """
    # keypoint_moseq.__init__ eagerly imports .analysis → panel → bokeh,
    # which crashes on newer numpy (np.bool8 removed). Patch before import.
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    import keypoint_moseq as kpms

    load_checkpoint = kpms.load_checkpoint
    load_config = kpms.load_config
    apply_model = kpms.apply_model
    format_data = kpms.format_data


    jax.config.update("jax_enable_x64", True)

    log.info(f"  Loading KPMS model from {model_dir}/{model_name}...")
    model, _, _, _ = load_checkpoint(
        project_dir=model_dir,
        model_name=model_name,
    )
    config = load_config(model_dir)

    # Reshape keypoints into per-segment dict
    kps_reshaped = keypoints.reshape(n_segments, frames_per_segment, -1, 3)
    coordinates = {f"seg_{i}": kps_reshaped[i] for i in range(n_segments)}

    # Use full config for format_data (matches training preprocessing)
    # but override noise to 0 at test time
    fmt_config = dict(config)
    fmt_config["added_noise_level"] = 0.0

    log.info(f"  Formatting data for KPMS ({n_segments} segments)...")
    data, metadata = format_data(coordinates, confidences=None, **fmt_config)

    log.info(f"  Running KPMS inference ({num_iters} iterations)...")
    # Pass config kwargs for init_model (anterior_idxs, error_estimator, etc.)
    # Do NOT pass noise_prior — it has training-data shape (484 segments).
    # error_estimator in config will be used to compute noise_prior for new data.
    init_kwargs = {
        k: v for k, v in config.items()
        if k in (
            "anterior_idxs", "posterior_idxs", "fix_heading", "whiten",
            "error_estimator", "conf_threshold", "PCA_fitting_num_frames",
            "trans_hypparams", "ar_hypparams", "obs_hypparams", "cen_hypparams",
        )
    }
    results = apply_model(
        model=model,
        data=data,
        metadata=metadata,
        num_iters=num_iters,
        ar_only=False,
        return_model=False,
        save_results=False,
        verbose=True,
        **init_kwargs,
    )

    # Extract syllable codes
    all_codes = []
    for i in range(n_segments):
        syllables = results[f"seg_{i}"]["syllable"]
        all_codes.append(syllables)

    codes = np.array(all_codes, dtype=np.int32)  # [n_segments, frames_per_segment]

    # Disable x64 — decoder/env need float32
    jax.config.update("jax_enable_x64", False)
    log.info(
        f"  KPMS codes: shape={codes.shape}, "
        f"active syllables={len(np.unique(codes))}/{model['params']['pi'].shape[0]}"
    )
    return codes


# ---------------------------------------------------------------------------
# Step 5: Plotting
# ---------------------------------------------------------------------------


COMPONENT_MARKERS = {
    "coarse": "--",
    "fine": ":",
}

COMPONENT_LABELS = {
    "coarse": "Coarse (root)",
    "fine": "Fine (joints+end-eff)",
}


def plot_reward_decomposition(
    results: dict[str, dict[str, np.ndarray]],
    max_steps: int,
) -> plt.Figure:
    """Reward decomposition: modes as colors, components as line styles."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for mode, color in MODE_COLORS.items():
        if mode not in results:
            continue
        mode_label = MODE_LABELS.get(mode, mode)
        for comp, ls in COMPONENT_MARKERS.items():
            curve = results[mode].get(comp)
            if curve is None:
                continue
            # Per-component normalization
            norm_factor = max(curve.mean(axis=0).max() if curve.ndim > 1 else curve.max(), 1e-8)
            mean = curve.mean(axis=0) / norm_factor if curve.ndim > 1 else curve / norm_factor
            label = f"{mode_label} — {COMPONENT_LABELS[comp]}"
            ax.plot(mean, color=color, linestyle=ls, label=label, linewidth=1.2)
            if curve.ndim > 1:
                sem = curve.std(axis=0) / (np.sqrt(curve.shape[0]) * norm_factor)
                ax.fill_between(range(len(mean)), mean - sem, mean + sem, alpha=0.15, color=color)

    ax.set_xlabel("Episode Timestep")
    ax.set_ylabel("Normalized reward")
    ax.set_title("Reward Decomposition (New Data)")
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="generalization")
def main(cfg: DictConfig) -> None:
    log.info("=== Generalization Experiment (new data → KPMS → Code2Act) ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = cfg.wandb.get("enabled", False)
    if wandb_enabled:
        run_name = f"moseq_gen_{datetime.now():%y%m%d_%H%M%S}"
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            config=dict(cfg),
        )

    n_segments = int(cfg.new_data.n_segments)
    frames_per_seg = int(cfg.new_data.frames_per_segment)
    seed = int(cfg.new_data.seed)

    # ===================================================================
    # Step 1: Sample & segment new data
    # ===================================================================
    log.info("\n--- Step 1: Sampling segments from new data ---")
    seg_data = sample_segments(
        cfg.new_data.path, n_segments, frames_per_seg, seed,
    )

    # Write temporary H5 for ReferenceClips
    seg_h5_path = str(output_dir / "segmented_new_data.h5")
    write_segmented_h5(seg_data, seg_h5_path)

    # ===================================================================
    # Step 2: Extract keypoints via FK
    # ===================================================================
    # Enable x64 for FK + KPMS (both need float64), disable before env/decoder
    jax.config.update("jax_enable_x64", True)

    log.info("\n--- Step 2: Extracting keypoints via FK ---")
    # Use the new data's own H5 for FK setup (has its own optimized offsets)
    keypoints, kp_names = extract_keypoints(
        seg_data["qpos"],
        cfg.new_data.path,
        cfg.stac_xml,
    )
    log.info(f"  Keypoints shape: {keypoints.shape}")

    # ===================================================================
    # Step 3: Run KPMS inference
    # ===================================================================
    log.info("\n--- Step 3: Running KPMS inference ---")
    codes = run_kpms_inference(
        keypoints, n_segments, frames_per_seg,
        cfg.kpms.model_dir,
        cfg.kpms.model_name,
        int(cfg.kpms.num_iters),
    )

    # Save codes
    np.savez_compressed(
        output_dir / "generalization_codes.npz",
        codes=codes,
        start_indices=seg_data["start_indices"],
        n_segments=n_segments,
        frames_per_segment=frames_per_seg,
    )

    # ===================================================================
    # Step 4: Load checkpoints & create env
    # ===================================================================
    log.info("\n--- Step 4: Loading checkpoints & creating env ---")

    # Code2Act decoder
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path, step=cfg.checkpoint.get("step"),
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    code2act_params = (norm_state, policy_params)

    # Mimic-MJX oracle
    mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
        cfg.mimic_checkpoint.path, step=cfg.mimic_checkpoint.get("step"),
    )
    mimic_params = (mimic_norm, mimic_policy)

    # Create env with new-data reference clips
    clips = ReferenceClips(
        data_path=seg_h5_path,
        n_frames_per_clip=frames_per_seg,
    )
    log.info(f"  ReferenceClips: {clips.qpos.shape}")

    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    env_cfg.clip_length = frames_per_seg  # Match segment length (may exceed training 250)

    # Disable termination criteria for open-loop evaluation
    env_cfg.termination_criteria.root_too_far.max_distance = 999.0
    env_cfg.termination_criteria.root_too_rotated.max_degrees = 999.0
    env_cfg.termination_criteria.pose_error.max_l2_error = 999.0

    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg, clips=clips, kpms_codes=codes,
        code_stack_size=code_stack_size,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Build inference functions
    inf_fn_code2act = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)
    inf_fn_mimic = make_mimic_inference_fn(mimic_ppo, deterministic=True)

    code_colors = get_code_colormap(num_codes)
    max_steps = int(cfg.inference.max_steps)

    # ===================================================================
    # Step 5: Run rollouts per segment per mode
    # ===================================================================
    log.info("\n--- Step 5: Running rollouts ---")

    mode_decomp: dict[str, dict[str, list]] = {}

    for mode in cfg.inference.modes:
        is_mimic = mode == "mimic_mjx"
        log.info(f"\n  Mode: {mode}")

        if is_mimic:
            inf_fn = inf_fn_mimic
            mode_params = mimic_params
            mode_ppo = mimic_ppo
            mode_rnn = False
            model_type = "mimic_mjx"
        else:
            inf_fn = inf_fn_code2act
            mode_params = code2act_params
            mode_ppo = ppo_networks
            mode_rnn = use_rnn
            model_type = "code2act"

        all_qpos = []
        all_rewards = []
        all_codes_out = []
        all_decomposed = []

        for si in range(n_segments):
            key = jax.random.PRNGKey(seed + si)
            result = run_rollout(
                env, inf_fn, mode_params, mode_ppo, mode_rnn, key,
                max_steps=max_steps,
                reset_clip_idx=si,
                jit_reset=jit_reset, jit_step=jit_step,
                model_type=model_type,
                ignore_done=True,
            )
            all_qpos.append(result["qpos"])
            all_rewards.append(result["rewards"])
            all_codes_out.append(result["code_indices"])
            all_decomposed.append(decompose_rewards(result["per_step_metrics"]))

            if (si + 1) % 5 == 0 or si == n_segments - 1:
                log.info(f"    Segment {si+1}/{n_segments}: reward={result['rewards'].mean():.3f}")

        # Aggregate decomposed curves
        min_len = min(len(r) for r in all_rewards)
        reward_matrix = np.array([r[:min_len] for r in all_rewards])
        decomp_arrays = {}
        for comp in ["total", "coarse", "fine", "penalty"]:
            curves = [d[comp][:min_len] for d in all_decomposed]
            decomp_arrays[comp] = np.array(curves)

        mode_decomp[mode] = decomp_arrays

        # Save per-mode data
        np.savez_compressed(
            output_dir / f"generalization_{mode}.npz",
            rewards=reward_matrix,
            qpos=np.array(all_qpos, dtype=object),
            code_indices=np.array(all_codes_out, dtype=object),
            **{f"decomp_{k}": v for k, v in decomp_arrays.items()},
        )
        log.info(f"  {mode}: mean reward = {reward_matrix.mean():.3f}")

    # ===================================================================
    # Step 6: Render videos
    # ===================================================================
    log.info("\n--- Step 6: Rendering videos ---")

    # Render solo videos for a subset of segments
    n_render = min(6, n_segments)
    for si in range(n_render):
        for mode in cfg.inference.modes:
            is_mimic = mode == "mimic_mjx"
            if is_mimic:
                inf_fn = inf_fn_mimic
                mode_params = mimic_params
                mode_ppo = mimic_ppo
                mode_rnn = False
                model_type = "mimic_mjx"
            else:
                inf_fn = inf_fn_code2act
                mode_params = code2act_params
                mode_ppo = ppo_networks
                mode_rnn = use_rnn
                model_type = "code2act"

            key = jax.random.PRNGKey(seed + si)
            result = run_rollout(
                env, inf_fn, mode_params, mode_ppo, mode_rnn, key,
                max_steps=max_steps,
                reset_clip_idx=si,
                jit_reset=jit_reset, jit_step=jit_step,
                model_type=model_type,
                ignore_done=True,
            )

            try:
                from experiments.shared.ghost_rendering import render_solo_video

                # Use KPMS codes from step 3 as the code bar (not decoder's
                # internal codes — mimic-mjx has none, and we want to show
                # the syllable sequence that was extracted from the new data).
                kpms_codes_for_seg = codes[si][:len(result["qpos"]) - 1]

                solo_path = output_dir / f"seg{si}_{mode}.mp4"
                render_solo_video(
                    env,
                    result["qpos"][:-1],
                    kpms_codes_for_seg,
                    solo_path,
                    fps=int(cfg.rendering.fps),
                    num_codes=num_codes,
                    title=f"Seg {si} ({mode})",
                )
                log.info(f"    Solo video: {solo_path}")
                if wandb_enabled:
                    wandb.log(
                        {f"generalization/seg{si}/{mode}": wandb.Video(str(solo_path), format="mp4")},
                        commit=False,
                    )
            except Exception as e:
                log.warning(f"    Solo video failed for seg{si}/{mode}: {e}")

    # ===================================================================
    # Step 7: Reward decomposition plot
    # ===================================================================
    log.info("\n--- Step 7: Generating plots ---")

    fig = plot_reward_decomposition(
        {m: {k: v.mean(axis=0) for k, v in curves.items()} for m, curves in mode_decomp.items()},
        max_steps,
    )
    fig.savefig(output_dir / "reward_decomposition.png", dpi=300)
    if wandb_enabled:
        wandb.log({"generalization/reward_decomposition": fig_to_image(fig)}, commit=False)
    plt.close(fig)

    # Mean reward bar chart
    set_nature_style()
    fig_bar, ax_bar = plt.subplots(figsize=(3.0, 2.5))
    modes_present = [m for m in cfg.inference.modes if m in mode_decomp]
    means = [mode_decomp[m]["total"].mean() for m in modes_present]
    stds = [mode_decomp[m]["total"].std(axis=1).mean() for m in modes_present]
    colors = [MODE_COLORS.get(m, "#999999") for m in modes_present]
    labels = [MODE_LABELS.get(m, m) for m in modes_present]
    ax_bar.bar(labels, means, yerr=stds, color=colors, capsize=3, edgecolor="none")
    ax_bar.set_ylabel("Mean Episode Reward")
    ax_bar.set_title("Generalization Performance")
    plt.tight_layout()
    fig_bar.savefig(output_dir / "mean_reward_comparison.png", dpi=300)
    if wandb_enabled:
        wandb.log({"generalization/mean_reward": fig_to_image(fig_bar)}, commit=False)
    plt.close(fig_bar)

    # ===================================================================
    # Step 8: Copy decomp data to figures/data for plotting
    # ===================================================================
    import shutil

    figures_data_dir = MOSEQ_DIR / "figures" / "data"
    figures_data_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes_present:
        src = output_dir / f"generalization_{mode}.npz"
        dst = figures_data_dir / f"generalization_{mode}.npz"
        shutil.copy2(src, dst)
        log.info(f"  Copied {src.name} → figures/data/")

    # ===================================================================
    # Done
    # ===================================================================
    if wandb_enabled:
        wandb.log({}, commit=True)
        wandb.finish()

    log.info("\n=== Generalization Experiment Complete ===")
    log.info(f"  Output: {output_dir}")
    for mode in modes_present:
        log.info(f"  {mode}: mean reward = {mode_decomp[mode]['total'].mean():.3f}")


if __name__ == "__main__":
    main()
