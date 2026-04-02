"""VQ-VAE Code Analysis Pipeline.

Main entry point for analyzing VQ-VAE code semantics:

1. Global transition matrix and stationary distribution
2. t-SNE/UMAP trajectory visualization
3. Pose gallery for popular codes
4. Kinematic profiles per code

Analysis requires pre-computed H5 rollout data. Use the inference module to
generate H5 files:
    python -m inference.run_inference checkpoint.path=/path/to/checkpoint

Usage:
    cd vqvae_jax
    python -m analysis.code_analysis

    # Override config values:
    python -m analysis.code_analysis \
        checkpoint.path=/path/to/checkpoint \
        data.h5_path=./outputs/rollout.h5

    # Enable WandB logging:
    python -m analysis.code_analysis wandb.enabled=true
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add paths for package imports
ANALYSIS_DIR = Path(__file__).parent
VQVAE_DIR = ANALYSIS_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(VQVAE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
from absl import logging
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from track_mjx.config import utils as config_utils

from .checkpoint_utils import load_vq_checkpoint, get_codebook, get_all_codebooks
from .tsne_trajectory_analysis import run_tsne_trajectory_analysis
from .inference_cache import InferenceResult
from .transition_context_analysis import (
    compute_global_transition_matrix,
    compute_stationary_distribution,
    compute_code_popularity,
    get_top_k_codes,
    render_kinematic_profiles,
    compute_transition_ngram_popularity,
    get_top_k_transitions,
    render_transition_pose_gallery,
)
from .utils import build_slider_html


def load_rollouts_from_h5(h5_path: str | Path) -> tuple[list[InferenceResult], dict]:
    """Load rollout data from H5 file and convert to InferenceResult format.

    Args:
        h5_path: Path to H5 file created by inference module.

    Returns:
        Tuple of (results, metadata).
    """
    sys.path.insert(0, str(VQVAE_DIR / "inference"))
    from inference.h5_utils import load_rollout_h5

    rollouts, metadata = load_rollout_h5(h5_path)

    results = []
    for rollout in rollouts:
        result = InferenceResult(
            clip_idx=rollout.clip_idx,
            code_indices=rollout.code_indices,
            qpos=rollout.qpos,
            qvel=rollout.qvel,
            rewards=rollout.rewards,
            states=None,
            rvq_indices=rollout.rvq_indices,
        )
        results.append(result)

    return results, metadata


def initialize_wandb_analysis(
    cfg: DictConfig, num_codes: int, h5_metadata: dict
) -> bool:
    """Initialize WandB for analysis session."""
    if not cfg.wandb.get("enabled", False):
        return False

    try:
        import wandb

        wandb.init(
            project=cfg.wandb.get("project", "vqvae-analysis"),
            entity=cfg.wandb.get("entity"),
            name=f"analysis_{datetime.now().strftime('%y%m%d_%H%M%S')}",
            config={
                "checkpoint_path": h5_metadata.get(
                    "checkpoint_path", cfg.checkpoint.path
                ),
                "checkpoint_step": h5_metadata.get(
                    "checkpoint_step", cfg.checkpoint.step
                ),
                "num_rollouts": h5_metadata.get("num_rollouts"),
                "num_codes": num_codes,
                "h5_path": cfg.data.h5_path,
            },
        )
        logging.info("WandB initialized for analysis session")
        return True

    except Exception as e:
        logging.warning(f"Failed to initialize WandB: {e}")
        return False


def generate_summary_report(
    output_dir: Path,
    cfg: DictConfig,
    num_codes: int,
    all_paths: dict[str, Any],
    h5_metadata: dict,
) -> str:
    """Generate a markdown summary report."""
    report_path = output_dir / "summary" / "analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = h5_metadata.get("checkpoint_path", cfg.checkpoint.path)
    checkpoint_step = h5_metadata.get("checkpoint_step", cfg.checkpoint.step)
    num_rollouts = h5_metadata.get("num_rollouts", "unknown")

    lines = [
        "# VQ-VAE Code Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- H5 data: `{cfg.data.h5_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Step: {checkpoint_step or 'latest'}",
        f"- Num rollouts: {num_rollouts}",
        f"- Num codes: {num_codes}",
        "",
        "## Output Files",
        "",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also save JSON summary
    json_path = output_dir / "summary" / "analysis_summary.json"
    summary = {
        "generated": datetime.now().isoformat(),
        "config": {
            "h5_path": cfg.data.h5_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": checkpoint_step,
            "num_rollouts": num_rollouts,
            "num_codes": num_codes,
        },
        "output_paths": all_paths,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return str(report_path)


@hydra.main(version_base=None, config_path="../configs", config_name="code_analysis")
def main(cfg: DictConfig):
    """Run VQ-VAE per-clip analysis pipeline."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("VQ-VAE Per-Clip Analysis Pipeline")
    print("=" * 60)

    # Create output directory
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Select H5 file based on analysis_split config
    analysis_split = cfg.data.get("analysis_split", "combined")
    if analysis_split == "train" and cfg.data.get("h5_path_train"):
        h5_path = cfg.data.h5_path_train
    elif analysis_split == "test" and cfg.data.get("h5_path_test"):
        h5_path = cfg.data.h5_path_test
    else:
        h5_path = cfg.data.h5_path

    if not Path(h5_path).exists():
        raise FileNotFoundError(
            f"H5 file not found: {h5_path}\n"
            "Please generate rollout data first using:\n"
            "  python -m inference.run_inference checkpoint.path=/path/to/checkpoint"
        )

    logging.info(f"\nLoading rollouts from H5: {h5_path}")
    results, h5_metadata = load_rollouts_from_h5(h5_path)
    num_codes = h5_metadata.get("num_codes", 64)
    logging.info(f"  Loaded {len(results)} rollouts, {num_codes} codes")

    # Load checkpoint for codebook
    ckpt = load_vq_checkpoint(cfg.checkpoint.path, step=cfg.checkpoint.step)
    vq_cfg = ckpt["cfg"]
    codebook = get_codebook(ckpt["policy"])
    num_codes = codebook.shape[0]

    # Create environment for rendering
    (_, cfg_dict, env_cfg_ml) = config_utils.prepare_config(cfg)
    reference_clips = ReferenceClips(
        data_path=vq_cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.data.get("clip_length", 250),
        keep_clips_idx=None,
    )
    env = imitation.Imitation(config=env_cfg_ml, clips=reference_clips)

    # Initialize WandB if enabled
    wandb_enabled = initialize_wandb_analysis(cfg, num_codes, h5_metadata)
    wandb_items: dict[str, Any] = {}

    # Get camera name
    env_suffix = getattr(env, "_suffix", "-rodent")
    camera_name = f"{cfg.render.camera}{env_suffix}"

    all_paths: dict[str, Any] = {}

    # === Section 1: Global Transition Matrix ===
    logging.info("\n" + "=" * 40)
    logging.info("Computing global transition matrix...")

    global_counts, global_fig = compute_global_transition_matrix(results, num_codes)

    # Save figure locally
    global_matrix_dir = output_dir / "global"
    global_matrix_dir.mkdir(parents=True, exist_ok=True)
    global_fig.savefig(
        global_matrix_dir / "transition_matrix.png", dpi=150, bbox_inches="tight"
    )

    if wandb_enabled:
        import wandb

        wandb_items["global/transition_matrix"] = wandb.Image(global_fig)

    import matplotlib.pyplot as plt

    plt.close(global_fig)

    all_paths["global"] = {
        "transition_matrix": str(global_matrix_dir / "transition_matrix.png"),
    }

    # === Section 1b: Stationary Distribution Analysis ===
    logging.info("\n" + "=" * 40)
    logging.info("Computing stationary distribution analysis...")

    # Get frame counts for comparison with stationary distribution
    frame_counts = compute_code_popularity(results, num_codes)

    stationary_results = compute_stationary_distribution(
        transition_counts=global_counts,
        frame_counts=frame_counts,
    )

    # Save figure locally (not to WandB)
    stationary_fig = stationary_results["figure"]
    stationary_fig.savefig(
        global_matrix_dir / "stationary_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close(stationary_fig)

    # Save stationary analysis JSON
    stationary_dist = stationary_results["stationary_dist"]

    # Find top codes by stationary probability
    top_stationary_idx = np.argsort(stationary_dist)[::-1][:10]
    top_stationary_codes = [
        {"code": int(i), "probability": float(stationary_dist[i])}
        for i in top_stationary_idx
    ]

    stationary_json = {
        "top_stationary_codes": top_stationary_codes,
    }
    with open(global_matrix_dir / "stationary_analysis.json", "w") as f:
        json.dump(stationary_json, f, indent=2)

    all_paths["global"]["stationary_distribution"] = str(
        global_matrix_dir / "stationary_distribution.png"
    )
    all_paths["global"]["stationary_json"] = str(
        global_matrix_dir / "stationary_analysis.json"
    )

    logging.info(
        f"  Top stationary codes: {[c['code'] for c in top_stationary_codes[:3]]}"
    )

    # === Section 1h: t-SNE Skill-Space Trajectory Analysis ===
    tsne_cfg = cfg.get("tsne_trajectory", {})
    if tsne_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Running t-SNE trajectory analysis...")

        tsne_dir = output_dir / "tsne_trajectory"

        # Use multi-depth codebooks for t-SNE when available
        tsne_all_codebooks = None
        all_cbs = get_all_codebooks(ckpt["policy"])
        if len(all_cbs) >= 2:
            tsne_all_codebooks = [np.array(cb) for cb in all_cbs]

        tsne_results = run_tsne_trajectory_analysis(
            results=results,
            codebook=np.array(codebook),
            output_dir=tsne_dir,
            cfg=(
                OmegaConf.to_container(tsne_cfg, resolve=True)
                if hasattr(tsne_cfg, "_metadata")
                else dict(tsne_cfg)
            ),
            env=env,
            camera=camera_name,
            width=cfg.render.get("width", 640),
            height=cfg.render.get("height", 480),
            fps=cfg.render.get("fps", 50),
            all_codebooks=tsne_all_codebooks,
        )

        if wandb_enabled:
            import wandb

            html_path = tsne_results.get("html_path")
            if html_path and Path(html_path).exists():
                wandb_items["tsne_trajectory/viewer"] = wandb.Html(
                    open(html_path).read()
                )

            static_html_path = tsne_results.get("static_html_path")
            if static_html_path and Path(static_html_path).exists():
                wandb_items["tsne_trajectory/static_viewer"] = wandb.Html(
                    open(static_html_path).read()
                )

            umap_html_path = tsne_results.get("umap_html_path")
            if umap_html_path and Path(umap_html_path).exists():
                wandb_items["tsne_trajectory/umap_viewer"] = wandb.Html(
                    open(umap_html_path).read()
                )

            umap_static_path = tsne_results.get("umap_static_html_path")
            if umap_static_path and Path(umap_static_path).exists():
                wandb_items["tsne_trajectory/umap_static_viewer"] = wandb.Html(
                    open(umap_static_path).read()
                )
            logging.info("  Logged t-SNE/UMAP trajectory viewers to WandB")

        all_paths["tsne_trajectory"] = {
            "html": tsne_results.get("html_path"),
            "static_html": tsne_results.get("static_html_path"),
            "umap_html": tsne_results.get("umap_html_path"),
            "umap_static_html": tsne_results.get("umap_static_html_path"),
            "json": tsne_results.get("json_path"),
        }

    # === Section 4: Pose Gallery (Popular Code Transitions) ===
    pose_gallery_cfg = cfg.get("pose_gallery", {})
    if pose_gallery_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Rendering pose gallery for popular transitions...")

        gallery_dir = output_dir / "pose_gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        top_k_gallery = pose_gallery_cfg.get("top_k_codes", 8)
        videos_per_code = pose_gallery_cfg.get("videos_per_code", 4)
        transition_length = pose_gallery_cfg.get("transition_length", 3)

        # Compute transition n-gram popularity
        ngram_counts = compute_transition_ngram_popularity(
            results, num_codes, n=transition_length
        )
        top_transitions = get_top_k_transitions(ngram_counts, top_k_gallery)

        pose_gallery_paths = {}
        gallery_video_paths: list[str] = []
        gallery_labels: list[str] = []

        for transition, count in top_transitions:
            arrow_label = "\u2192".join(str(c) for c in transition)
            logging.info(
                f"  Rendering pose gallery for transition "
                f"[{arrow_label}] ({count} occurrences)..."
            )
            fname = "_".join(str(c) for c in transition)
            video_path = gallery_dir / f"transition_{fname}_gallery.mp4"
            try:
                path = render_transition_pose_gallery(
                    results=results,
                    transition=transition,
                    num_codes=num_codes,
                    env=env,
                    output_path=video_path,
                    n_clips=videos_per_code,
                    camera=camera_name,
                    width=1280,
                    height=720,
                    fps=cfg.render.get("fps", 50),
                )
                if path:
                    pose_gallery_paths[arrow_label] = path
                    gallery_video_paths.append(str(path))
                    gallery_labels.append(
                        f"{arrow_label} ({count} occurrences)"
                    )
            except Exception as e:
                logging.warning(
                    f"    Failed to render pose gallery for transition "
                    f"[{arrow_label}]: {e}"
                )

        # Build slider HTML for pose gallery
        if wandb_enabled and gallery_video_paths:
            html = build_slider_html(
                gallery_video_paths,
                gallery_labels,
                "Pose Gallery",
                media_type="video",
            )
            wandb_items["pose_gallery/viewer"] = wandb.Html(html)

        all_paths["pose_gallery"] = pose_gallery_paths
        logging.info(f"  Rendered {len(pose_gallery_paths)} pose gallery videos")

    # === Section 5: Kinematic Profiles per Code ===
    if cfg.get("kinematic_profile", {}).get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Rendering kinematic profiles per code...")

        import matplotlib.pyplot as plt

        kin_dir = output_dir / "kinematic_profiles"
        kin_dir.mkdir(parents=True, exist_ok=True)

        joint_names = list(cfg.get("walker_config", {}).get("joint_names", []))
        frame_counts = compute_code_popularity(results, num_codes)
        top_codes = get_top_k_codes(
            frame_counts, cfg.get("pose_gallery", {}).get("top_k_codes", 5)
        )

        kin_paths = {}
        kin_png_paths: list[str] = []
        kin_labels: list[str] = []
        for code_idx, count in top_codes:
            logging.info(f"  Kinematic profile for code {code_idx} ({count} frames)...")
            fig = render_kinematic_profiles(results, code_idx, joint_names)
            fig_path = kin_dir / f"kinematic_code_{code_idx:03d}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            kin_paths[code_idx] = str(fig_path)
            kin_png_paths.append(str(fig_path))
            kin_labels.append(f"Code {code_idx} ({count} frames)")

        # Build slider HTML for kinematic profiles
        if wandb_enabled and kin_png_paths:
            import wandb

            html = build_slider_html(
                kin_png_paths,
                kin_labels,
                "Kinematic Profiles",
                media_type="image",
            )
            wandb_items["kinematic_profile/viewer"] = wandb.Html(html)

        all_paths["kinematic_profiles"] = kin_paths
        logging.info(f"  Rendered {len(kin_paths)} kinematic profile plots")

    # === Section 6: Generate Summary Report ===
    logging.info("\n" + "=" * 40)
    logging.info("Generating summary report...")
    report_path = generate_summary_report(
        output_dir, cfg, num_codes, all_paths, h5_metadata
    )
    all_paths["report"] = report_path

    # Single WandB log call with all accumulated items, then finish
    if wandb_enabled:
        import wandb

        if wandb_items and wandb.run is not None:
            wandb.log(wandb_items)
        wandb.finish()

    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to {output_dir}")
    print("=" * 60)
    if "global" in all_paths:
        print(f"\nGlobal transition matrix: {all_paths['global']['transition_matrix']}")
        if "stationary_distribution" in all_paths["global"]:
            print(
                f"Stationary distribution: {all_paths['global']['stationary_distribution']}"
            )
    if "tsne_trajectory" in all_paths:
        tp = all_paths["tsne_trajectory"]
        if tp.get("html"):
            print(f"t-SNE trajectory viewer: {tp['html']}")
        if tp.get("static_html"):
            print(f"t-SNE static viewer: {tp['static_html']}")
        if tp.get("umap_html"):
            print(f"UMAP trajectory viewer: {tp['umap_html']}")
        if tp.get("umap_static_html"):
            print(f"UMAP static viewer: {tp['umap_static_html']}")
    if "pose_gallery" in all_paths:
        n_gallery = len(all_paths["pose_gallery"])
        print(f"Pose gallery videos: {n_gallery} entries")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
