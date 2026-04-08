"""Experiment 5: Round-trip code consistency.

Tests whether Code2Act behavior, re-encoded by the same KPMS model,
recovers the original input codes.  Compares three conditions:
  - Reference (ceiling): original mocap qpos
  - Mimic-MJX (oracle): oracle rollout qpos
  - Code2Act: decoder rollout qpos

Runs on two datasets: inference test set (250 frames) and
generalization set (1000 frames).

Usage:
    cd moseq_jax
    python -m experiments.run_roundtrip
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import logging
import sys
from pathlib import Path

import h5py
import hydra
import jax
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

from experiments.shared.keypoint_fk import setup_stac_model, qpos_to_keypoints_fk
from experiments.shared.plotting import set_nature_style, NATURE_COLORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CONDITIONS = ["reference", "mimic_mjx", "code2act"]
CONDITION_LABELS = {
    "reference": "Reference (ceiling)",
    "mimic_mjx": "Mimic-MJX (oracle)",
    "code2act": "Code2Act",
}
CONDITION_COLORS = {
    "reference": NATURE_COLORS["green"],
    "mimic_mjx": NATURE_COLORS["orange"],
    "code2act": NATURE_COLORS["blue"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_inference_qpos(
    cfg_ds: DictConfig,
) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    """Load qpos arrays and original codes for the inference test set.

    Returns:
        ``{condition: (qpos_per_clip, original_codes)}``
        where qpos_per_clip is a list of [T_i, 74] arrays and
        original_codes is [n_clips, clip_length].
    """
    codes_data = np.load(cfg_ds.codes_path)
    original_codes = codes_data["test_codes"]  # [148, 250]
    test_indices = codes_data["test_indices"]
    clip_length = int(cfg_ds.clip_length)

    result = {}

    # Reference: original mocap qpos from H5
    with h5py.File(cfg_ds.reference_h5, "r") as f:
        all_qpos = f["qpos"][:]  # [210500, 74]
    n_clips_total = all_qpos.shape[0] // clip_length
    all_qpos_clipped = all_qpos[: n_clips_total * clip_length].reshape(
        n_clips_total, clip_length, -1
    )
    ref_qpos = [all_qpos_clipped[idx] for idx in test_indices]
    result["reference"] = (ref_qpos, original_codes)

    # Code2Act rollout
    c2a = np.load(cfg_ds.code2act_path, allow_pickle=True)
    c2a_qpos = [np.asarray(c2a["qpos"][i], dtype=np.float32) for i in range(len(c2a["qpos"]))]
    result["code2act"] = (c2a_qpos, original_codes)

    # Mimic-MJX rollout
    mimic = np.load(cfg_ds.mimic_path, allow_pickle=True)
    mimic_qpos = [np.asarray(mimic["qpos"][i], dtype=np.float32) for i in range(len(mimic["qpos"]))]
    result["mimic_mjx"] = (mimic_qpos, original_codes)

    return result


def load_generalization_qpos(
    cfg_ds: DictConfig,
) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    """Load qpos arrays and original codes for the generalization set."""
    codes_data = np.load(cfg_ds.codes_path)
    original_codes = codes_data["codes"]  # [20, 1000]
    clip_length = int(cfg_ds.clip_length)
    n_clips = original_codes.shape[0]

    result = {}

    # Reference: segmented new data
    with h5py.File(cfg_ds.reference_h5, "r") as f:
        all_qpos = f["qpos"][:]  # [20000, 74]
    ref_clipped = all_qpos[: n_clips * clip_length].reshape(n_clips, clip_length, -1)
    ref_qpos = [ref_clipped[i] for i in range(n_clips)]
    result["reference"] = (ref_qpos, original_codes)

    # Code2Act
    c2a = np.load(cfg_ds.code2act_path, allow_pickle=True)
    c2a_qpos_raw = c2a["qpos"]
    c2a_qpos = [np.asarray(c2a_qpos_raw[i], dtype=np.float32) for i in range(len(c2a_qpos_raw))]
    result["code2act"] = (c2a_qpos, original_codes)

    # Mimic-MJX
    mimic = np.load(cfg_ds.mimic_path, allow_pickle=True)
    mimic_qpos_raw = mimic["qpos"]
    mimic_qpos = [np.asarray(mimic_qpos_raw[i], dtype=np.float32) for i in range(len(mimic_qpos_raw))]
    result["mimic_mjx"] = (mimic_qpos, original_codes)

    return result


# ---------------------------------------------------------------------------
# KPMS re-extraction
# ---------------------------------------------------------------------------


def run_kpms_on_qpos(
    qpos_per_clip: list[np.ndarray],
    original_codes: np.ndarray,
    clip_length: int,
    fk_h5: str,
    stac_xml: str,
    kpms_model_dir: str,
    kpms_model_name: str,
    kpms_num_iters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run FK + KPMS on qpos arrays and compare with original codes.

    Returns:
        ``(re_codes, valid_mask, coverage)``
        - re_codes: [n_clips, clip_length] re-extracted codes (-1 for invalid)
        - valid_mask: [n_clips, clip_length] bool
        - coverage: fraction of frames with valid comparison
    """
    n_clips = len(qpos_per_clip)

    # Truncate each clip to clip_length, track valid frames
    qpos_padded = np.zeros((n_clips, clip_length, 74), dtype=np.float32)
    valid_mask = np.zeros((n_clips, clip_length), dtype=bool)

    for i, q in enumerate(qpos_per_clip):
        # q may be (T+1, 74) from rollout (includes initial qpos)
        # or (T, 74) from reference — use first clip_length frames
        usable = min(q.shape[0], clip_length)
        qpos_padded[i, :usable] = q[:usable].astype(np.float32)
        valid_mask[i, :usable] = True

    coverage = valid_mask.sum() / valid_mask.size

    # Flatten for FK
    qpos_flat = qpos_padded.reshape(-1, 74)

    # FK
    jax.config.update("jax_enable_x64", True)
    log.info(f"    FK on {qpos_flat.shape[0]} frames...")
    mj_model, mj_data, site_ids, kp_names = setup_stac_model(fk_h5, stac_xml)
    keypoints = qpos_to_keypoints_fk(qpos_flat, mj_model, mj_data, site_ids)
    keypoints = keypoints.reshape(n_clips, clip_length, -1, 3)

    # KPMS inference
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    import keypoint_moseq as kpms

    model, _, _, _ = kpms.load_checkpoint(
        project_dir=kpms_model_dir, model_name=kpms_model_name,
    )
    config = kpms.load_config(kpms_model_dir)

    coordinates = {f"clip_{i}": keypoints[i] for i in range(n_clips)}

    fmt_config = dict(config)
    fmt_config["added_noise_level"] = 0.0
    data, metadata = kpms.format_data(coordinates, confidences=None, **fmt_config)

    init_kwargs = {
        k: v for k, v in config.items()
        if k in (
            "anterior_idxs", "posterior_idxs", "fix_heading", "whiten",
            "error_estimator", "conf_threshold", "PCA_fitting_num_frames",
            "trans_hypparams", "ar_hypparams", "obs_hypparams", "cen_hypparams",
        )
    }

    log.info(f"    KPMS inference ({kpms_num_iters} iters, {n_clips} clips)...")
    results = kpms.apply_model(
        model=model, data=data, metadata=metadata,
        num_iters=kpms_num_iters, ar_only=False,
        return_model=False, save_results=False, verbose=True,
        **init_kwargs,
    )

    # Extract re-extracted codes
    re_codes = np.full((n_clips, clip_length), -1, dtype=np.int32)
    for i in range(n_clips):
        syllables = results[f"clip_{i}"]["syllable"]
        usable = min(len(syllables), clip_length)
        re_codes[i, :usable] = syllables[:usable]

    jax.config.update("jax_enable_x64", False)

    return re_codes, valid_mask, coverage


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_accuracy(
    original: np.ndarray,
    reextracted: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """Frame-level accuracy (% match) over valid frames."""
    valid = valid_mask & (reextracted >= 0)
    if valid.sum() == 0:
        return 0.0
    return float((original[valid] == reextracted[valid]).mean())


def compute_confusion_matrix(
    original: np.ndarray,
    reextracted: np.ndarray,
    valid_mask: np.ndarray,
    num_codes: int,
) -> np.ndarray:
    """Confusion matrix [num_codes, num_codes]: original × re-extracted."""
    valid = valid_mask & (reextracted >= 0)
    cm = np.zeros((num_codes, num_codes), dtype=np.int64)
    orig_flat = original[valid]
    re_flat = reextracted[valid]
    for o, r in zip(orig_flat, re_flat):
        if 0 <= o < num_codes and 0 <= r < num_codes:
            cm[o, r] += 1
    return cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Nature-style confusion matrix heatmap."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    # Normalize rows to probabilities
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Re-extracted code")
    ax.set_ylabel("Original code")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(re-extracted | original)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_accuracy_comparison(
    results: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Bar chart: accuracy per condition, grouped by dataset."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    datasets = list(results.keys())
    n_ds = len(datasets)
    n_cond = len(CONDITIONS)
    bar_width = 0.22
    x = np.arange(n_ds)

    for ci, cond in enumerate(CONDITIONS):
        vals = [results[ds].get(cond, 0) for ds in datasets]
        offset = (ci - (n_cond - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            edgecolor="none",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=5.5,
            )

    ax.set_xticks(x)
    ds_labels = {"inference": "250-Frame\nTest Set", "generalization": "1000-Frame\nTest Set"}
    ax.set_xticklabels([ds_labels.get(ds, ds) for ds in datasets])
    ax.set_ylabel("Frame-Level Accuracy")
    ax.set_title("Round-Trip Code Consistency")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="roundtrip")
def main(cfg: DictConfig) -> None:
    log.info("=== Round-Trip Code Consistency Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kpms_model_dir = cfg.kpms.model_dir
    kpms_model_name = cfg.kpms.model_name
    kpms_num_iters = int(cfg.kpms.num_iters)
    stac_xml = cfg.stac_xml

    # Determine num_codes from KPMS config
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
    import keypoint_moseq as kpms_lib

    kpms_config = kpms_lib.load_config(kpms_model_dir)
    num_codes = kpms_config["trans_hypparams"]["num_states"]
    log.info(f"  KPMS model: {num_codes} codes")

    # Collect results across datasets
    all_accuracies: dict[str, dict[str, float]] = {}

    datasets = {
        "inference": (cfg.inference, load_generalization_qpos),
        "generalization": (cfg.generalization, load_generalization_qpos),
    }

    for ds_name, (ds_cfg, load_fn) in datasets.items():
        log.info(f"\n{'='*60}")
        log.info(f"  Dataset: {ds_name} (clip_length={ds_cfg.clip_length})")
        log.info(f"{'='*60}")

        ds_dir = output_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Load all conditions
        condition_data = load_fn(ds_cfg)
        clip_length = int(ds_cfg.clip_length)
        fk_h5 = str(ds_cfg.fk_h5)
        ds_accuracies = {}

        for cond in CONDITIONS:
            if cond not in condition_data:
                continue
            qpos_list, original_codes = condition_data[cond]
            log.info(f"\n  --- Condition: {CONDITION_LABELS[cond]} ---")
            log.info(f"    {len(qpos_list)} clips, clip_length={clip_length}")

            re_codes, valid_mask, coverage = run_kpms_on_qpos(
                qpos_list, original_codes, clip_length,
                fk_h5, stac_xml,
                kpms_model_dir, kpms_model_name, kpms_num_iters,
            )

            acc = compute_accuracy(original_codes, re_codes, valid_mask)
            cm = compute_confusion_matrix(original_codes, re_codes, valid_mask, num_codes)

            ds_accuracies[cond] = acc
            log.info(f"    Accuracy: {acc:.1%} (coverage: {coverage:.1%})")

            # Save
            np.savez_compressed(
                ds_dir / f"{cond}_roundtrip.npz",
                re_codes=re_codes, valid_mask=valid_mask,
                original_codes=original_codes, confusion_matrix=cm,
                accuracy=acc, coverage=coverage,
            )

            # Confusion matrix plot
            plot_confusion_matrix(
                cm,
                f"{CONDITION_LABELS[cond]} ({ds_name})",
                ds_dir / f"{cond}_confusion.png",
            )

        all_accuracies[ds_name] = ds_accuracies

    # Summary bar chart
    log.info("\n" + "=" * 60)
    log.info("  SUMMARY")
    log.info("=" * 60)
    for ds_name, accs in all_accuracies.items():
        log.info(f"  {ds_name}:")
        for cond, acc in accs.items():
            log.info(f"    {CONDITION_LABELS[cond]}: {acc:.1%}")

    plot_accuracy_comparison(all_accuracies, output_dir / "roundtrip_accuracy.png")

    # Save summary JSON
    summary = {
        ds: {cond: float(acc) for cond, acc in accs.items()}
        for ds, accs in all_accuracies.items()
    }
    with open(output_dir / "roundtrip_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n  All outputs saved to: {output_dir}")
    log.info("=== Round-Trip Experiment Complete ===")


if __name__ == "__main__":
    main()
