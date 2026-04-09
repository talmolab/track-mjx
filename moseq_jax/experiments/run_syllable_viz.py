"""Experiment 8: KPMS syllable 3D visualization.

Visualizes the KPMS syllable decomposition on the original reference data
using keypoint_moseq's built-in 3D trajectory plots, frequency/duration
statistics, and similarity dendrogram.  No decoder needed.

Usage:
    cd moseq_jax
    python -m experiments.run_syllable_viz

    # Skip dendrogram (faster):
    python -m experiments.run_syllable_viz dendrogram.enabled=false
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"

import json
import logging
import sys
from pathlib import Path

import h5py
import hydra
import matplotlib
import matplotlib.backends.backend_agg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

# Work around bokeh/numpy incompatibility (np.bool8 removed in numpy 2.x)
# by mocking the analysis submodule before importing keypoint_moseq.
sys.modules["keypoint_moseq.analysis"] = type(sys)("mock_analysis")
from keypoint_moseq import io as kpms_io  # noqa: E402
from keypoint_moseq import viz as kpms_viz  # noqa: E402

# Patch kpms_viz.rasterize_figure for matplotlib >= 3.8 where
# canvas.tostring_rgb() was removed.
_orig_rasterize = kpms_viz.rasterize_figure


def _patched_rasterize(fig):  # noqa: D103
    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    rgba = np.frombuffer(buf, dtype="uint8").reshape(
        canvas.get_width_height()[::-1] + (4,)
    )
    return rgba[:, :, :3].copy()


if not hasattr(matplotlib.backends.backend_agg.FigureCanvasAgg, "tostring_rgb"):
    kpms_viz.rasterize_figure = _patched_rasterize


# Patch colormap converter: numpy 2.x repr gives "np.uint8(255)" not "255"
_orig_cmap_to_plotly = kpms_viz.matplotlib_colormap_to_plotly


def _patched_cmap_to_plotly(cmap):  # noqa: D103
    cmap_obj = plt.colormaps[cmap]
    pl_entries = 255
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []
    for k in range(pl_entries):
        C = (np.array(cmap_obj(k * h)[:3]) * 255).astype(int)
        pl_colorscale.append([k * h, f"rgb({C[0]}, {C[1]}, {C[2]})"])
    return pl_colorscale


kpms_viz.matplotlib_colormap_to_plotly = _patched_cmap_to_plotly

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.clip_selection import load_balanced_splits
from experiments.shared.keypoint_fk import setup_stac_model, qpos_to_keypoints_fk
from experiments.shared.plotting import set_nature_style
from kpms.keypoint_loader import prepare_keypoints_for_kpms

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rodent skeleton definition (23 keypoints, alphabetical order)
# ---------------------------------------------------------------------------

RODENT_SKELETON = [
    # Spine chain
    ("Snout", "SpineF"),
    ("SpineF", "SpineM"),
    ("SpineM", "SpineL"),
    ("SpineL", "TailBase"),
    # Head
    ("Snout", "EarL"),
    ("Snout", "EarR"),
    # Left forelimb
    ("SpineF", "ShoulderL"),
    ("ShoulderL", "ElbowL"),
    ("ElbowL", "WristL"),
    ("WristL", "HandL"),
    # Right forelimb
    ("SpineF", "ShoulderR"),
    ("ShoulderR", "ElbowR"),
    ("ElbowR", "WristR"),
    ("WristR", "HandR"),
    # Left hindlimb
    ("SpineL", "HipL"),
    ("HipL", "KneeL"),
    ("KneeL", "AnkleL"),
    ("AnkleL", "FootL"),
    # Right hindlimb
    ("SpineL", "HipR"),
    ("HipR", "KneeR"),
    ("KneeR", "AnkleR"),
    ("AnkleR", "FootR"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_reference_keypoints(
    h5_path: str,
    balanced_split_path: str,
    n_frames_per_clip: int,
    fk_batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Load qpos from H5 and convert to 3D keypoints via FK.

    Args:
        h5_path: Path to rodent_reference_clips.h5.
        balanced_split_path: Path to rodent_balanced_splits.json.
        n_frames_per_clip: Frames per clip (250).
        fk_batch_size: FK batch size for GPU processing.

    Returns:
        ``(keypoints_4d, kp_names)`` where keypoints_4d is
        ``[N_balanced, T, K, 3]``.
    """
    log.info("Loading reference qpos ...")
    with h5py.File(h5_path, "r") as f:
        qpos_all = f["qpos"][:]  # [total_frames, nq]
    n_total_frames = qpos_all.shape[0]
    n_clips_total = n_total_frames // n_frames_per_clip
    log.info(f"  qpos: {qpos_all.shape} ({n_clips_total} clips)")

    log.info("Setting up stac model for FK ...")
    mj_model, mj_data, site_ids, kp_names = setup_stac_model(h5_path)
    log.info(f"  {len(kp_names)} keypoints: {kp_names[:5]} ...")

    log.info("Running forward kinematics ...")
    keypoints_flat = qpos_to_keypoints_fk(
        qpos_all, mj_model, mj_data, site_ids, batch_size=fk_batch_size,
    )  # [total_frames, K, 3]
    log.info(f"  FK output: {keypoints_flat.shape}")

    # Reshape to per-clip
    keypoints_all = keypoints_flat.reshape(n_clips_total, n_frames_per_clip, -1, 3)

    # Filter to balanced subset
    splits = load_balanced_splits(balanced_split_path)
    train_indices = splits["balanced"]["train_indices"]
    test_indices = splits["balanced"]["test_indices"]
    all_indices = sorted(set(train_indices) | set(test_indices))
    keypoints_balanced = keypoints_all[all_indices]
    log.info(
        f"  Balanced subset: {keypoints_balanced.shape} "
        f"({len(all_indices)} clips)"
    )

    # Scale from meters to millimeters.  keypoint_moseq's get_limits()
    # casts to int, so sub-1.0 coordinates collapse to zero.
    keypoints_balanced = keypoints_balanced * 1000.0
    log.info("  Scaled coordinates to millimeters (×1000)")

    return keypoints_balanced, kp_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None, config_path="configs", config_name="syllable_viz_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== KPMS Syllable 3D Visualization ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1: Load 3D keypoints via FK
    # ==================================================================
    log.info("\n--- Step 1: Load Reference Keypoints ---")
    keypoints_4d, kp_names = load_reference_keypoints(
        h5_path=cfg.data.reference_data_path,
        balanced_split_path=cfg.data.balanced_split_path,
        n_frames_per_clip=int(cfg.keypoints.n_frames_per_clip),
        fk_batch_size=int(cfg.keypoints.fk_batch_size),
    )

    # ==================================================================
    # Step 2: Build coordinates dict for keypoint_moseq
    # ==================================================================
    log.info("\n--- Step 2: Build Coordinates Dict ---")
    coordinates, _ = prepare_keypoints_for_kpms(keypoints_4d)
    log.info(f"  {len(coordinates)} recordings in coordinates dict")

    # ==================================================================
    # Step 3: Load KPMS results and config
    # ==================================================================
    log.info("\n--- Step 3: Load KPMS Results ---")
    project_dir = cfg.kpms_model.project_dir
    model_name = cfg.kpms_model.model_name

    results = kpms_io.load_results(
        project_dir=project_dir, model_name=model_name,
    )
    log.info(f"  Loaded results for {len(results)} recordings")

    # Scale centroids to match millimeter coordinates
    for rec_key in results:
        if "centroid" in results[rec_key]:
            results[rec_key]["centroid"] = (
                results[rec_key]["centroid"] * 1000.0
            )

    kpms_config = kpms_io.load_config(project_dir)
    bodyparts = kpms_config["bodyparts"]
    use_bodyparts = kpms_config.get("use_bodyparts", bodyparts)
    log.info(f"  Bodyparts: {len(bodyparts)} ({bodyparts[:3]} ...)")

    # Verify alignment
    coord_keys = sorted(coordinates.keys())
    result_keys = sorted(results.keys())
    assert coord_keys == result_keys, (
        f"Key mismatch: {len(coord_keys)} coords vs {len(result_keys)} results. "
        f"First diff: coords={coord_keys[:3]}, results={result_keys[:3]}"
    )
    log.info("  Coordinates and results keys aligned.")

    # ==================================================================
    # Step 4: Generate 3D trajectory plots (main output)
    # ==================================================================
    log.info("\n--- Step 4: Generate Trajectory Plots ---")
    traj_dir = output_dir / "trajectory_plots"
    traj_dir.mkdir(parents=True, exist_ok=True)

    kpms_viz.generate_trajectory_plots(
        coordinates=coordinates,
        results=results,
        output_dir=str(traj_dir),
        pre=float(cfg.trajectory_plots.pre),
        post=float(cfg.trajectory_plots.post),
        min_frequency=float(cfg.trajectory_plots.min_frequency),
        min_duration=int(cfg.trajectory_plots.min_duration),
        skeleton=RODENT_SKELETON,
        bodyparts=bodyparts,
        use_bodyparts=use_bodyparts,
        keypoint_colormap=cfg.trajectory_plots.keypoint_colormap,
        fps=int(cfg.trajectory_plots.fps),
        projection_planes=list(cfg.trajectory_plots.projection_planes),
        interactive=bool(cfg.trajectory_plots.interactive),
        save_gifs=bool(cfg.trajectory_plots.save_gifs),
        save_mp4s=bool(cfg.trajectory_plots.get("save_mp4s", False)),
        save_individually=bool(
            cfg.trajectory_plots.get("save_individually", True)
        ),
        density_sample=bool(cfg.trajectory_plots.get("density_sample", True)),
    )
    log.info(f"  Trajectory plots saved to {traj_dir}")

    # ==================================================================
    # Step 5: Syllable frequency histogram
    # ==================================================================
    log.info("\n--- Step 5: Syllable Frequencies ---")
    set_nature_style()
    fig, ax = kpms_viz.plot_syllable_frequencies(results=results)
    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"syllable_frequencies.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    log.info(f"  Saved syllable_frequencies.pdf/.png")

    # ==================================================================
    # Step 6: Duration distribution
    # ==================================================================
    log.info("\n--- Step 6: Duration Distribution ---")
    fig, ax = kpms_viz.plot_duration_distribution(results=results)
    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"duration_distribution.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)
    log.info(f"  Saved duration_distribution.pdf/.png")

    # ==================================================================
    # Step 7: Similarity dendrogram
    # ==================================================================
    if cfg.dendrogram.enabled:
        log.info("\n--- Step 7: Similarity Dendrogram ---")
        kpms_viz.plot_similarity_dendrogram(
            coordinates=coordinates,
            results=results,
            save_path=str(output_dir / "similarity_dendrogram"),
            metric=cfg.dendrogram.metric,
            figsize=tuple(cfg.dendrogram.figsize),
            skeleton=RODENT_SKELETON,
            bodyparts=bodyparts,
            use_bodyparts=use_bodyparts,
            fps=int(cfg.trajectory_plots.fps),
        )
        log.info(f"  Saved similarity_dendrogram")
    else:
        log.info("\n--- Step 7: Dendrogram skipped (disabled) ---")

    # ==================================================================
    # Step 8: Summary JSON
    # ==================================================================
    log.info("\n--- Step 8: Save Summary ---")
    all_syllables = np.concatenate(
        [results[k]["syllable"] for k in sorted(results.keys())]
    )
    unique_syllables = np.unique(all_syllables)

    summary = {
        "experiment": "syllable_viz",
        "n_recordings": len(coordinates),
        "n_keypoints": len(bodyparts),
        "n_frames_per_clip": int(cfg.keypoints.n_frames_per_clip),
        "n_active_syllables": int(len(unique_syllables)),
        "kpms_model": model_name,
        "kpms_project_dir": str(project_dir),
        "fps": int(cfg.trajectory_plots.fps),
        "skeleton_edges": len(RODENT_SKELETON),
        "trajectory_pre_sec": float(cfg.trajectory_plots.pre),
        "trajectory_post_sec": float(cfg.trajectory_plots.post),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Summary: {summary}")

    log.info("\n=== KPMS Syllable 3D Visualization Complete ===")
    log.info(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
