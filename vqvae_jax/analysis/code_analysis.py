"""VQ-VAE Code Analysis Pipeline.

Main entry point for analyzing VQ-VAE code semantics:

1. Per-clip analysis:
   - Transition matrices per clip
   - Interactive HTML viewer with slider navigation
   - Video rendering with code timeline overlays

2. Transition context analysis:
   - Compare predecessor/successor patterns for top K codes across clips
   - Measure consistency of code function across different contexts
   - Render transition videos (predecessor → code → successor)

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
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.config import utils as config_utils

from .checkpoint_utils import load_vq_checkpoint, get_codebook, get_all_codebooks
from .compositional_transition_analysis import (
    run_compositional_transition_analysis,
    run_qpos_code_determinism_analysis,
)
from .tsne_trajectory_analysis import run_tsne_trajectory_analysis
from .inference_cache import InferenceResult
from .mutual_information import run_mutual_information_analysis
from .rvq_analysis import run_rvq_analysis
from .per_clip_analysis import run_per_clip_analysis
from .transition_context_analysis import (
    run_transition_context_analysis,
    compute_global_transition_matrix,
    compute_stationary_distribution,
    compute_code_popularity,
    get_top_k_codes,
    render_code_pose_gallery,
)


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
                "per_clip": OmegaConf.to_container(cfg.per_clip, resolve=True),
            },
        )
        logging.info("WandB initialized for analysis session")
        return True

    except Exception as e:
        logging.warning(f"Failed to initialize WandB: {e}")
        return False


def log_to_wandb_immediately(key: str, value: Any, wandb_enabled: bool) -> None:
    """Log a single item to WandB immediately if enabled.

    Args:
        key: WandB metric key.
        value: Value to log (wandb.Html, wandb.Video, wandb.Image, etc.)
        wandb_enabled: Whether WandB is enabled.
    """
    if not wandb_enabled:
        return

    try:
        import wandb

        if wandb.run is not None:
            wandb.log({key: value})
    except Exception as e:
        logging.warning(f"Failed to log {key} to WandB: {e}")


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

    per_clip_paths = all_paths.get("per_clip", {})
    tc_paths = all_paths.get("per_clip_context", {})

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
        "## Per-Clip Analysis",
        "",
        f"- Analyzed {cfg.per_clip.get('num_clips', 10)} individual clips",
        f"- Video rendering: {cfg.per_clip.get('render_videos', True)}",
        "",
        "Each clip includes:",
        "- Transition matrix and probability heatmap",
        "- Video with code overlay bars",
        "",
    ]

    if cfg.get("transition_context", {}).get("enabled", False):
        tc_cfg = cfg.transition_context
        lines.extend(
            [
                "## Transition Context Analysis",
                "",
                f"- Analyzed top {tc_cfg.get('top_k', 10)} most frequent codes",
                f"- Compared predecessor/successor patterns across clips",
                f"- Rendered transition videos: {tc_cfg.get('render_videos', True)}",
                "",
                "For each code, measures:",
                "- Predecessor distribution similarity across clips",
                "- Successor distribution similarity across clips",
                "- Combined context similarity (indicates functional consistency)",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            "### Per-Clip Analysis",
            f"- Interactive HTML viewer: `{per_clip_paths.get('html', 'N/A')}`",
            f"- Per-clip stats JSON: `{per_clip_paths.get('json', 'N/A')}`",
            f"- Clip videos: `{output_dir}/per_clip/videos/`",
            "",
        ]
    )

    if tc_paths:
        lines.extend(
            [
                "### Per-Clip Context Analysis",
                f"- Conditional HTML viewer: `{tc_paths.get('conditional_html', 'N/A')}`",
                f"- Context stats JSON: `{tc_paths.get('json', 'N/A')}`",
                f"- Context videos: `{output_dir}/per_clip_context/videos/`",
                "",
            ]
        )

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
        "per_clip": OmegaConf.to_container(cfg.per_clip, resolve=True),
        "transition_context": OmegaConf.to_container(
            cfg.get("transition_context", {}), resolve=True
        ),
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

    # Get camera name
    env_suffix = getattr(env, "_suffix", "-rodent")
    camera_name = f"{cfg.render.camera}{env_suffix}"

    all_paths: dict[str, Any] = {}

    # === Section 1: Global Transition Matrix ===
    # Compute and log immediately (before other analyses for context)
    logging.info("\n" + "=" * 40)
    logging.info("Computing global transition matrix...")

    global_counts, global_fig = compute_global_transition_matrix(results, num_codes)

    # Save figure locally
    global_matrix_dir = output_dir / "global"
    global_matrix_dir.mkdir(parents=True, exist_ok=True)
    global_fig.savefig(
        global_matrix_dir / "transition_matrix.png", dpi=150, bbox_inches="tight"
    )

    # Log to WandB immediately
    if wandb_enabled:
        import wandb

        log_to_wandb_immediately(
            "global/transition_matrix", wandb.Image(global_fig), wandb_enabled
        )
        logging.info("  Logged global transition matrix to WandB")

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
    empirical_dist = stationary_results["empirical_dist"]

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

    # === Section 1c: Mutual Information Analysis ===
    mi_cfg = cfg.get("mutual_information", {})
    if mi_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Running mutual information analysis...")

        mi_dir = output_dir / "mutual_information"
        joint_names = list(cfg.get("walker_config", {}).get("joint_names", []))

        mi_paths = run_mutual_information_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=mi_dir,
            joint_names=joint_names,
            cfg=(
                OmegaConf.to_container(mi_cfg, resolve=True)
                if hasattr(mi_cfg, "_metadata")
                else dict(mi_cfg)
            ),
        )

        if wandb_enabled:
            import wandb

            for key, fig_path in mi_paths.items():
                log_to_wandb_immediately(
                    f"mutual_information/{key}",
                    wandb.Image(fig_path),
                    wandb_enabled,
                )
            logging.info("  Logged MI analysis figures to WandB")

        all_paths["mutual_information"] = mi_paths

    # === Section 1d: RVQ Analysis ===
    rvq_cfg = cfg.get("rvq_analysis", {})
    if rvq_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Running RVQ analysis...")

        rvq_dir = output_dir / "rvq_analysis"
        rvq_paths = run_rvq_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=rvq_dir,
            cfg=(
                OmegaConf.to_container(rvq_cfg, resolve=True)
                if hasattr(rvq_cfg, "_metadata")
                else dict(rvq_cfg)
            ),
        )

        if wandb_enabled and rvq_paths:
            import wandb

            for key, fig_path in rvq_paths.items():
                log_to_wandb_immediately(
                    f"rvq_analysis/{key}",
                    wandb.Image(fig_path),
                    wandb_enabled,
                )
            logging.info("  Logged RVQ analysis figures to WandB")

        all_paths["rvq_analysis"] = rvq_paths

    # === Section 1f: Qpos+Code Determinism Analysis ===
    determinism_cfg = cfg.get("qpos_code_determinism", {})
    if determinism_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Running qpos+code determinism analysis...")

        det_dir = output_dir / "qpos_code_determinism"
        det_results = run_qpos_code_determinism_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=det_dir,
            cfg=(
                OmegaConf.to_container(determinism_cfg, resolve=True)
                if hasattr(determinism_cfg, "_metadata")
                else dict(determinism_cfg)
            ),
        )

        if wandb_enabled:
            import wandb

            for key, fig_path in det_results.get("figure_paths", {}).items():
                log_to_wandb_immediately(
                    f"qpos_code_determinism/{key}",
                    wandb.Image(fig_path),
                    wandb_enabled,
                )
            logging.info("  Logged determinism figures to WandB")

        all_paths["qpos_code_determinism"] = det_results.get("figure_paths", {})
        all_paths["qpos_code_determinism"]["json"] = det_results.get("json_path")
        logging.info(f"  Total determinism pairs: {det_results.get('total_pairs', 0)}")

    # === Section 1g: Compositional Transition Analysis ===
    comp_cfg = cfg.get("compositional_transition", {})
    if comp_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Running compositional transition analysis...")

        comp_dir = output_dir / "compositional_transition"
        comp_results = run_compositional_transition_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=comp_dir,
            cfg=(
                OmegaConf.to_container(comp_cfg, resolve=True)
                if hasattr(comp_cfg, "_metadata")
                else dict(comp_cfg)
            ),
            env=env,
            camera=camera_name,
            width=cfg.render.get("width", 640),
            height=cfg.render.get("height", 480),
            fps=cfg.render.get("fps", 50),
        )

        if wandb_enabled:
            import wandb

            # Log HTML viewer
            html_path = comp_results.get("html_path")
            if html_path and Path(html_path).exists():
                log_to_wandb_immediately(
                    "compositional_transition/viewer",
                    wandb.Html(open(html_path).read()),
                    wandb_enabled,
                )
            logging.info("  Logged compositional analysis to WandB")

        all_paths["compositional_transition"] = {
            "html": comp_results.get("html_path"),
            "json": comp_results.get("json_path"),
        }
        logging.info(f"  Total determinism pairs: {comp_results.get('total_pairs', 0)}")

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
                log_to_wandb_immediately(
                    "tsne_trajectory/viewer",
                    wandb.Html(open(html_path).read()),
                    wandb_enabled,
                )

            static_html_path = tsne_results.get("static_html_path")
            if static_html_path and Path(static_html_path).exists():
                log_to_wandb_immediately(
                    "tsne_trajectory/static_viewer",
                    wandb.Html(open(static_html_path).read()),
                    wandb_enabled,
                )

            umap_html_path = tsne_results.get("umap_html_path")
            if umap_html_path and Path(umap_html_path).exists():
                log_to_wandb_immediately(
                    "tsne_trajectory/umap_viewer",
                    wandb.Html(open(umap_html_path).read()),
                    wandb_enabled,
                )

            umap_static_path = tsne_results.get("umap_static_html_path")
            if umap_static_path and Path(umap_static_path).exists():
                log_to_wandb_immediately(
                    "tsne_trajectory/umap_static_viewer",
                    wandb.Html(open(umap_static_path).read()),
                    wandb_enabled,
                )
            logging.info("  Logged t-SNE/UMAP trajectory viewers to WandB")

        all_paths["tsne_trajectory"] = {
            "html": tsne_results.get("html_path"),
            "static_html": tsne_results.get("static_html_path"),
            "umap_html": tsne_results.get("umap_html_path"),
            "umap_static_html": tsne_results.get("umap_static_html_path"),
            "json": tsne_results.get("json_path"),
        }

    # === Section 2: Per-Clip Analysis ===
    if cfg.per_clip.get("enabled", True):
        logging.info("\n" + "=" * 40)
        logging.info("Running per-clip analysis...")

        per_clip_cfg = cfg.per_clip
        per_clip_results = run_per_clip_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=output_dir / "per_clip",
            num_clips=per_clip_cfg.get("num_clips", 10),
            n_communities=per_clip_cfg.get("n_communities", None),
            render_videos=per_clip_cfg.get("render_videos", True),
            env=env,
            camera=camera_name,
            width=cfg.render.get("width", 640),
            height=cfg.render.get("height", 480),
            fps=cfg.render.get("fps", 50),
        )

        all_paths["per_clip"] = {
            "html": per_clip_results["html_path"],
            "json": per_clip_results["json_path"],
            "videos": per_clip_results.get("video_paths", {}),
        }

        # Log per-clip results to WandB immediately
        if wandb_enabled:
            import wandb

            html_path = per_clip_results["html_path"]
            if html_path and Path(html_path).exists():
                log_to_wandb_immediately(
                    "per_clip/interactive_viewer",
                    wandb.Html(open(html_path).read()),
                    wandb_enabled,
                )
            # Log per-clip videos
            video_paths = per_clip_results.get("video_paths", {})
            for name, path in video_paths.items():
                if path and Path(path).exists():
                    log_to_wandb_immediately(
                        f"per_clip/videos/{name}",
                        wandb.Video(path, format="mp4"),
                        wandb_enabled,
                    )
            logging.info("  Logged per-clip analysis to WandB")

    # === Section 3: Per-Clip Context Analysis ===
    if cfg.get("transition_context", {}).get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Running per-clip context analysis...")

        tc_cfg = cfg.transition_context
        tc_results = run_transition_context_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=output_dir / "per_clip_context",
            top_k=tc_cfg.get("top_k", 10),
            min_clips_for_comparison=tc_cfg.get("min_clips", 3),
            render_videos=tc_cfg.get("render_videos", True),
            env=env,
            camera=camera_name,
            width=cfg.render.get("width", 640),
            height=cfg.render.get("height", 480),
            fps=cfg.render.get("fps", 50),
            max_videos_per_code=tc_cfg.get("max_videos_per_code", 4),
            conditional_cfg=tc_cfg.get("conditional"),
        )

        all_paths["per_clip_context"] = {
            "json": tc_results["json_path"],
        }

        # Log conditional transition HTML to WandB immediately
        if tc_results.get("conditional_html_path"):
            cond_html = tc_results["conditional_html_path"]
            all_paths["per_clip_context"]["conditional_html"] = cond_html
            if wandb_enabled:
                import wandb

                if Path(cond_html).exists():
                    log_to_wandb_immediately(
                        "conditional_transition/viewer",
                        wandb.Html(open(cond_html).read()),
                        wandb_enabled,
                    )
                logging.info("  Logged conditional transition analysis to WandB")

    # === Section 4: Pose Gallery (Popular Code Start Positions) ===
    pose_gallery_cfg = cfg.get("transition_context", {}).get("pose_gallery", {})
    if pose_gallery_cfg.get("enabled", False):
        logging.info("\n" + "=" * 40)
        logging.info("Rendering pose gallery for popular codes...")

        gallery_dir = output_dir / "pose_gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        top_k_gallery = pose_gallery_cfg.get("top_k_codes", 8)
        videos_per_code = pose_gallery_cfg.get("videos_per_code", 6)
        context_frames = pose_gallery_cfg.get("context_frames", 15)

        # Get top K codes
        frame_counts = compute_code_popularity(results, num_codes)
        top_codes = get_top_k_codes(frame_counts, top_k_gallery)

        pose_gallery_paths = {}
        for code_idx, count in top_codes:
            logging.info(
                f"  Rendering pose gallery for code {code_idx} ({count} frames)..."
            )
            video_path = gallery_dir / f"code_{code_idx:03d}_gallery.mp4"

            try:
                path = render_code_pose_gallery(
                    results=results,
                    code_idx=code_idx,
                    num_codes=num_codes,
                    env=env,
                    output_path=video_path,
                    n_clips=videos_per_code,
                    context_frames=context_frames,
                    camera=camera_name,
                    width=cfg.render.get("width", 640),
                    height=cfg.render.get("height", 480),
                    fps=cfg.render.get("fps", 50),
                )
                if path:
                    pose_gallery_paths[code_idx] = path
                    # Log each gallery video to WandB immediately
                    if wandb_enabled:
                        import wandb

                        log_to_wandb_immediately(
                            f"pose_gallery/code_{code_idx}",
                            wandb.Video(path, format="mp4"),
                            wandb_enabled,
                        )
            except Exception as e:
                logging.warning(
                    f"    Failed to render pose gallery for code {code_idx}: {e}"
                )

        all_paths["pose_gallery"] = pose_gallery_paths
        logging.info(f"  Rendered {len(pose_gallery_paths)} pose gallery videos")
        if wandb_enabled:
            logging.info("  Logged pose gallery videos to WandB")

    # === Section 6: Generate Summary Report ===
    logging.info("\n" + "=" * 40)
    logging.info("Generating summary report...")
    report_path = generate_summary_report(
        output_dir, cfg, num_codes, all_paths, h5_metadata
    )
    all_paths["report"] = report_path

    # Finish WandB session
    if wandb_enabled:
        import wandb

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
    if "per_clip" in all_paths:
        print(f"Per-clip viewer: {all_paths['per_clip']['html']}")
    if "compositional_transition" in all_paths and all_paths[
        "compositional_transition"
    ].get("html"):
        print(
            f"Compositional tree viewer: "
            f"{all_paths['compositional_transition']['html']}"
        )
    if "per_clip_context" in all_paths and all_paths["per_clip_context"].get(
        "conditional_html"
    ):
        print(
            f"Per-clip context viewer: {all_paths['per_clip_context']['conditional_html']}"
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
        print(f"Pose gallery videos: {len(all_paths['pose_gallery'])} codes")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
