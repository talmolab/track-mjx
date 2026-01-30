"""VQ-VAE Code Analysis Pipeline.

Main entry point for analyzing VQ-VAE code semantics on a per-clip basis:
- Per-clip transition matrices and community detection
- Interactive HTML viewer with slider navigation
- Video rendering with code and community timeline overlays

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

from .checkpoint_utils import load_vq_checkpoint, get_codebook
from .inference_cache import InferenceResult
from .per_clip_analysis import run_per_clip_analysis


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
                "checkpoint_path": h5_metadata.get("checkpoint_path", cfg.checkpoint.path),
                "checkpoint_step": h5_metadata.get("checkpoint_step", cfg.checkpoint.step),
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


def log_analysis_to_wandb(
    output_dir: Path,
    all_paths: dict[str, Any],
) -> None:
    """Log analysis results to WandB."""
    try:
        import wandb

        if wandb.run is None:
            return

        # Log per-clip HTML viewer
        html_path = all_paths.get("html")
        if html_path and Path(html_path).exists():
            wandb.log({
                "per_clip/interactive_viewer": wandb.Html(open(html_path).read())
            })

        # Log per-clip videos
        video_paths = all_paths.get("videos", {})
        for name, path in video_paths.items():
            if path and Path(path).exists():
                wandb.log({f"clips/{name}": wandb.Video(path, format="mp4")})

        logging.info("Logged analysis results to WandB")

    except Exception as e:
        logging.warning(f"Failed to log to WandB: {e}")


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
        "# VQ-VAE Per-Clip Analysis Report",
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
        f"- Communities per clip: {cfg.per_clip.get('n_communities', 'auto-detect')}",
        f"- Video rendering: {cfg.per_clip.get('render_videos', True)}",
        "",
        "Each clip includes:",
        "- Transition matrix and probability heatmap",
        "- Community detection via spectral clustering",
        "- Transition graph with community-colored nodes",
        "- Dual timeline showing code and community colors",
        "- Video with code/community overlay bars",
        "",
        "## Output Files",
        "",
        f"- Interactive HTML viewer: `{all_paths.get('html', 'N/A')}`",
        f"- Per-clip stats JSON: `{all_paths.get('json', 'N/A')}`",
        f"- Clip videos: `{output_dir}/videos/`",
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
        "per_clip": OmegaConf.to_container(cfg.per_clip, resolve=True),
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

    # Load rollouts from H5 file
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

    # Run per-clip analysis (the only analysis mode)
    logging.info("\n" + "=" * 40)
    logging.info("Running per-clip analysis...")

    per_clip_cfg = cfg.per_clip
    per_clip_results = run_per_clip_analysis(
        results=results,
        num_codes=num_codes,
        output_dir=output_dir,
        num_clips=per_clip_cfg.get("num_clips", 10),
        n_communities=per_clip_cfg.get("n_communities", None),
        render_videos=per_clip_cfg.get("render_videos", True),
        env=env,
        camera=camera_name,
        width=cfg.render.get("width", 640),
        height=cfg.render.get("height", 480),
        fps=cfg.render.get("fps", 50),
    )

    all_paths = {
        "html": per_clip_results["html_path"],
        "json": per_clip_results["json_path"],
        "videos": per_clip_results.get("video_paths", {}),
    }

    # Generate summary report
    logging.info("\n" + "=" * 40)
    logging.info("Generating summary report...")
    report_path = generate_summary_report(
        output_dir, cfg, num_codes, all_paths, h5_metadata
    )
    all_paths["report"] = report_path

    # Log to WandB
    if wandb_enabled:
        log_analysis_to_wandb(output_dir, all_paths)
        import wandb
        wandb.finish()

    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to {output_dir}")
    print("=" * 60)
    print(f"\nInteractive viewer: {all_paths['html']}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    main()
