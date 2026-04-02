"""Generate KPMS syllable codes for all reference clips.

Loads the best KPMS model from a sweep and extracts syllable labels for each
clip, producing an ``.npz`` file with train/test code arrays aligned to the
balanced split indices.

Also computes and saves diagnostic plots (code usage, duration distribution,
transition matrix) and summary metrics.

**CRITICAL**: This script sets ``jax_enable_x64 = True`` and MUST run in a
separate process from the RL training.

Usage::

    cd moseq_jax
    python -m codegen.generate_codes
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
sys.path.insert(0, str(MOSEQ_DIR))
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def generate_codes(
    sweep_results_path: str,
    balanced_split_path: str,
    output_path: str,
    wandb_project: str | None = None,
    reference_data_path: str | None = None,
    stac_xml_path: str | None = None,
) -> None:
    """Extract codes from the best KPMS model and save to ``.npz``.

    The output contains:
    - ``all_codes``: shape ``[N_balanced, T]`` int32
    - ``train_codes``: shape ``[N_train, T]`` int32
    - ``test_codes``: shape ``[N_test, T]`` int32
    - ``train_indices``: original clip indices for train set
    - ``test_indices``: original clip indices for test set

    Args:
        sweep_results_path: Path to ``sweep_results.json``.
        balanced_split_path: Path to balanced split JSON.
        output_path: Output ``.npz`` path.
        wandb_project: WandB project name. ``None`` disables logging.
        reference_data_path: Path to reference clips H5 (for reconstruction).
        stac_xml_path: Path to stac-mjx XML model (for FK).
    """
    import h5py

    # Load sweep results
    with open(sweep_results_path) as f:
        results = json.load(f)

    best = results["best_model"]
    if best is None:
        raise ValueError("No best model found in sweep results")

    project_dir = best["project_dir"]
    model_name = best["model_name"]
    log.info(f"Best model: {model_name} at {project_dir}")

    # WandB init
    wandb_enabled = False
    if wandb_project is not None:
        try:
            import wandb

            wandb.init(
                project=wandb_project,
                name=f"kpms_codegen_{datetime.now().strftime('%y%m%d_%H%M%S')}",
                config={"best_model": best},
            )
            wandb_enabled = True
        except Exception as e:
            log.warning(f"Failed to init WandB: {e}")

    # Load balanced split
    with open(balanced_split_path) as f:
        splits = json.load(f)
    train_indices = splits["balanced"]["train_indices"]
    test_indices = splits["balanced"]["test_indices"]
    all_indices = sorted(set(train_indices) | set(test_indices))

    # Load syllable labels from pre-saved results.h5
    results_h5_path = os.path.join(project_dir, model_name, "results.h5")
    with h5py.File(results_h5_path, "r") as f:
        rec_names = sorted(f.keys())
        all_labels = [f[rn]["syllable"][:] for rn in rec_names]
    log.info(f"Loaded {len(all_labels)} recordings from {results_h5_path}")

    # Map balanced indices to recording order
    n_balanced = len(all_indices)
    if len(all_labels) != n_balanced:
        raise ValueError(f"Expected {n_balanced} recordings but got {len(all_labels)}")

    all_codes = np.array(all_labels, dtype=np.int32)
    log.info(f"All codes shape: {all_codes.shape}")
    log.info(f"Unique codes: {np.unique(all_codes).tolist()}")

    # Build index mapping: all_indices[i] -> position i in all_codes
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    train_positions = [idx_to_pos[idx] for idx in train_indices]
    test_positions = [idx_to_pos[idx] for idx in test_indices]

    train_codes = all_codes[train_positions]
    test_codes = all_codes[test_positions]

    log.info(f"Train codes: {train_codes.shape}, Test codes: {test_codes.shape}")

    # Save codes
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        all_codes=all_codes,
        train_codes=train_codes,
        test_codes=test_codes,
        train_indices=np.array(train_indices),
        test_indices=np.array(test_indices),
        kappa=np.float64(best.get("kappa", 0.0)),
        num_states=np.int32(best.get("n_states", int(np.max(all_codes)) + 1)),
        model_type=np.array(str(best.get("model_type", "unknown"))),
        mean_duration=np.float64(best.get("mean_duration", 0.0)),
    )
    log.info(f"Saved codes to {output_path}")

    n_unique_train = len(np.unique(train_codes))
    n_unique_test = len(np.unique(test_codes))
    log.info(
        f"Train: {n_unique_train} unique codes, Test: {n_unique_test} unique codes"
    )

    # --- Diagnostics (always computed, saved to output dir) ---
    plots_dir = output.parent / "codegen_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    num_states = best.get("n_states", int(np.max(all_codes)) + 1)

    # Code usage histogram
    code_counts = np.bincount(all_codes.ravel(), minlength=num_states)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(num_states), code_counts)
    ax.set_xlabel("Code")
    ax.set_ylabel("Count")
    ax.set_title("Code Usage Histogram")
    fig.tight_layout()
    fig.savefig(plots_dir / "code_usage.png", dpi=150)
    if wandb_enabled:
        wandb.log({"codegen/code_usage": wandb.Image(fig)}, commit=False)
    plt.close(fig)
    log.info(f"Code counts: {code_counts.tolist()}")

    # Duration histogram
    all_durations = []
    for codes in all_labels:
        changes = np.where(np.diff(codes) != 0)[0] + 1
        segments = np.split(codes, changes)
        all_durations.extend(len(s) for s in segments)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_durations, bins=50, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Syllable Duration Distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "duration_hist.png", dpi=150)
    if wandb_enabled:
        wandb.log({"codegen/duration_hist": wandb.Image(fig)}, commit=False)
    plt.close(fig)
    log.info(
        f"Duration stats: mean={np.mean(all_durations):.1f}, "
        f"std={np.std(all_durations):.1f}, "
        f"median={np.median(all_durations):.1f}"
    )

    # Transition matrix heatmap
    trans_matrix = np.zeros((num_states, num_states))
    for codes in all_labels:
        for i in range(len(codes) - 1):
            trans_matrix[codes[i], codes[i + 1]] += 1
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_probs = np.divide(
        trans_matrix,
        row_sums,
        where=row_sums > 0,
        out=np.zeros_like(trans_matrix),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(trans_probs, cmap="viridis", aspect="auto")
    ax.set_xlabel("Next Code")
    ax.set_ylabel("Current Code")
    ax.set_title("Transition Probability Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "transition_matrix.png", dpi=150)
    if wandb_enabled:
        wandb.log({"codegen/transition_matrix": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # Scalar summary
    summary_metrics = {
        "num_active_codes": int(np.sum(code_counts > 0)),
        "num_states": num_states,
        "num_train_clips": len(train_codes),
        "num_test_clips": len(test_codes),
        "mean_duration": float(np.mean(all_durations)),
    }

    # --- Reconstruction plot and 3D videos from best checkpoint ---
    if reference_data_path and stac_xml_path:
        try:
            import jax.numpy as jnp
            import keypoint_moseq as kpms
            from jax_moseq.models.keypoint_slds.alignment import estimate_coordinates
            from jax_moseq.utils import unbatch

            from moseq_jax.kpms.keypoint_loader import prepare_keypoints_for_kpms
            from moseq_jax.sweep.run_sweep import (
                _load_keypoints,
                _plot_reconstruction,
                _render_keypoint_video,
            )

            # Load keypoints via FK (same pipeline as sweep)
            keypoints, kp_names = _load_keypoints(
                reference_data_path, stac_xml_path, balanced_split_path
            )
            log.info(f"Loaded keypoints for reconstruction: {keypoints.shape}")

            # Load best checkpoint
            model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)

            # Re-format data to get data["Y"] and metadata
            coordinates, confidences = prepare_keypoints_for_kpms(keypoints)
            config = kpms.load_config(project_dir)
            data, metadata = kpms.format_data(coordinates, confidences, **config)

            # Original: data["Y"] flattened to [N, T, K*D]
            Y_data = data["Y"]
            if isinstance(Y_data, dict):
                orig_keys = sorted(Y_data.keys())
                original = np.stack([np.array(Y_data[k]) for k in orig_keys], axis=0)
            else:
                original = np.array(Y_data)
            if original.ndim == 4:
                n, t, k, d = original.shape
                original_flat = original.reshape(n, t, k * d)
            else:
                original_flat = original

            # Reconstruction: estimate_coordinates + unbatch
            Y_est = estimate_coordinates(
                jnp.array(model_ckpt["states"]["x"]),
                jnp.array(model_ckpt["states"]["v"]),
                jnp.array(model_ckpt["states"]["h"]),
                jnp.array(model_ckpt["params"]["Cd"]),
            )
            coords_dict = unbatch(np.array(Y_est), *metadata)
            rec_keys = sorted(coords_dict.keys())
            recon_list = []
            for rk in rec_keys:
                arr = np.array(coords_dict[rk])
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
                elif arr.ndim == 4:
                    arr = arr.reshape(arr.shape[0], -1)
                recon_list.append(arr)
            reconstructed_flat = np.stack(recon_list, axis=0)
            if reconstructed_flat.ndim == 4:
                n, t, k, d = reconstructed_flat.shape
                reconstructed_flat = reconstructed_flat.reshape(n, t, k * d)

            # Reconstruction MSE
            min_len = min(original_flat.shape[1], reconstructed_flat.shape[1])
            recon_mse = float(
                np.mean(
                    (original_flat[:, :min_len] - reconstructed_flat[:, :min_len]) ** 2
                )
            )
            summary_metrics["reconstruction_mse"] = recon_mse
            log.info(f"Reconstruction MSE: {recon_mse:.6f}")

            # Reconstruction timeseries plot (clip 0)
            fig = _plot_reconstruction(
                original_flat,
                reconstructed_flat,
                clip_idx=0,
                title_prefix=f"Best: {model_name}",
            )
            fig.savefig(plots_dir / "reconstruction_timeseries.png", dpi=150)
            if wandb_enabled:
                wandb.log({"codegen/reconstruction": wandb.Image(fig)}, commit=False)
            plt.close(fig)
            log.info("Saved reconstruction timeseries plot")

            # 3D video: original keypoints with syllable codes (clip 0)
            try:
                orig_video_path = str(plots_dir / "original_keypoints.mp4")
                _render_keypoint_video(
                    keypoints[0],
                    orig_video_path,
                    kp_names=kp_names,
                    codes=all_labels[0],
                    title="Original + syllable codes",
                )
                if wandb_enabled:
                    wandb.log(
                        {
                            "codegen/original_video": wandb.Video(
                                orig_video_path, fps=30, format="mp4"
                            )
                        },
                        commit=False,
                    )
                log.info(f"Saved original keypoint video: {orig_video_path}")
            except Exception as ve:
                log.warning(f"Original video rendering failed: {ve}")

            # 3D video: reconstructed keypoints with syllable codes (clip 0)
            # Use world-frame reconstruction from unbatch (per-recording)
            recon_worldframe = np.array(coords_dict[rec_keys[0]])
            try:
                recon_video_path = str(plots_dir / "reconstructed_keypoints.mp4")
                _render_keypoint_video(
                    recon_worldframe,
                    recon_video_path,
                    kp_names=kp_names,
                    codes=all_labels[0],
                    title="Reconstructed + syllable codes",
                )
                if wandb_enabled:
                    wandb.log(
                        {
                            "codegen/reconstructed_video": wandb.Video(
                                recon_video_path, fps=30, format="mp4"
                            )
                        },
                        commit=False,
                    )
                log.info(f"Saved reconstructed keypoint video: {recon_video_path}")
            except Exception as ve:
                log.warning(f"Reconstructed video rendering failed: {ve}")

        except Exception as e:
            log.warning(f"Reconstruction/video generation failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        if not reference_data_path or not stac_xml_path:
            log.info(
                "Skipping reconstruction (reference_data_path or stac_xml_path "
                "not provided)"
            )

    # Log summary
    log.info(f"Summary metrics: {summary_metrics}")
    summary_path = plots_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    log.info(f"Saved summary metrics to {summary_path}")

    if wandb_enabled:
        wandb.run.summary.update(
            {f"codegen/{k}": v for k, v in summary_metrics.items()}
        )
        wandb.finish()


def _load_config(config_path: str | None = None) -> dict:
    """Load config from YAML."""
    import yaml

    if config_path is None:
        config_path = str(MOSEQ_DIR / "configs" / "kpms_sweep.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate KPMS codes for RL training")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--sweep-results", type=str, default=None)
    parser.add_argument("--balanced-split", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    codegen_cfg = cfg.get("codegen", {})
    wandb_cfg = cfg.get("wandb", {})

    data_cfg = cfg.get("data", {})

    generate_codes(
        sweep_results_path=args.sweep_results or codegen_cfg["sweep_results"],
        balanced_split_path=args.balanced_split or codegen_cfg["balanced_split"],
        output_path=args.output or codegen_cfg["output"],
        wandb_project=args.wandb_project
        or (wandb_cfg.get("project") if wandb_cfg.get("enabled") else None),
        reference_data_path=data_cfg.get("reference_data_path"),
        stac_xml_path=data_cfg.get("stac_xml_path"),
    )


if __name__ == "__main__":
    main()
