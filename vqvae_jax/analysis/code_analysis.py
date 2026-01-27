"""VQ-VAE Code Analysis Pipeline.

Main entry point for analyzing VQ-VAE code semantics through:
- Transition matrix and chain analysis
- Code segment extraction and duration statistics
- Kinematic feature correlation
- DTW-based segment alignment

Usage:
    cd vqvae_jax
    python -m analysis.code_analysis

    # Override config values:
    python -m analysis.code_analysis \
        checkpoint.path=/path/to/checkpoint \
        experiments.transition.enabled=true

    # Force re-run inference (ignore cache):
    python -m analysis.code_analysis inference.force_rerun=true
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
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.config import utils as config_utils

from .checkpoint_utils import (
    load_vq_checkpoint,
    load_vq_inference_fn,
    load_vq_inference_fn_with_stickiness,
    get_codebook,
)
from .inference_cache import (
    InferenceResult,
    compute_cache_key,
    get_cache_path,
    save_inference_cache,
    load_inference_cache,
)
from .transition_analysis import run_transition_analysis
from .segment_analysis import run_segment_analysis
from .kinematic_analysis import run_kinematic_analysis
from .alignment_analysis import run_alignment_analysis


def run_inference(
    env: Any,
    inference_fn: Any,
    num_clips: int,
    max_steps: int,
    seed: int,
    store_states: bool = True,
    use_stickiness: bool = False,
) -> list[InferenceResult]:
    """Run VQ-VAE inference on multiple clips.

    Args:
        env: Environment with reset/step methods.
        inference_fn: VQ-VAE inference function. If use_stickiness=False,
            signature is (obs, rng) -> (action, extras). If use_stickiness=True,
            signature is (obs, rng, prev_indices) -> (action, extras).
        num_clips: Number of clips to process.
        max_steps: Maximum steps per clip.
        seed: Random seed.
        store_states: Whether to store environment states for rendering.
        use_stickiness: If True, track previous code indices and pass to
            inference_fn for stickiness bias.

    Returns:
        List of InferenceResult objects.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    results = []
    rng = jax.random.PRNGKey(seed)

    for clip_idx in range(num_clips):
        logging.info(f"Running inference on clip {clip_idx}/{num_clips}...")

        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        states = [state] if store_states else None
        code_indices = []
        qpos_list = []
        qvel_list = []
        rewards = []
        prev_indices = None  # Track previous code for stickiness

        for step in range(max_steps):
            # Get observation and flatten to dict format expected by policy
            obs = state.obs
            flat_obs = flatten_obs_dict(obs)

            # Run inference (with or without stickiness)
            # Pass the dict directly - policy_network.apply expects a dict
            rng, action_rng = jax.random.split(rng)
            if use_stickiness:
                action, extras = inference_fn(flat_obs, action_rng, prev_indices)
            else:
                action, extras = inference_fn(flat_obs, action_rng)

            # Extract code index and update prev_indices for next step
            code_idx = int(extras["indices"])
            code_indices.append(code_idx)
            prev_indices = jnp.array(code_idx)  # For next iteration

            # Extract qpos/qvel
            if hasattr(state, "data"):
                qpos_list.append(np.array(state.data.qpos))
                qvel_list.append(np.array(state.data.qvel))
            elif hasattr(state, "pipeline_state"):
                qpos_list.append(np.array(state.pipeline_state.q))
                qvel_list.append(np.array(state.pipeline_state.qd))

            # Step environment
            next_state = jit_step(state, action)

            rewards.append(float(next_state.reward))
            if store_states:
                states.append(next_state)

            if next_state.done:
                break

            state = next_state

        # Create result
        result = InferenceResult(
            clip_idx=clip_idx,
            code_indices=np.array(code_indices),
            qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
            qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
            rewards=np.array(rewards),
            states=states,
        )
        results.append(result)

    return results


def run_or_load_inference(
    env: Any,
    inference_fn: Any,
    checkpoint_path: str,
    step: int | None,
    num_clips: int,
    max_steps: int,
    seed: int,
    cache_dir: str | Path,
    force_rerun: bool = False,
    store_states: bool = False,
    use_stickiness: bool = False,
) -> list[InferenceResult]:
    """Run inference or load from cache.

    Note: States are never cached (too large). If store_states=True and cache
    exists, we load cached data then re-run inference to populate states.

    Args:
        env: Environment.
        inference_fn: VQ-VAE inference function.
        checkpoint_path: Path to checkpoint.
        step: Checkpoint step.
        num_clips: Number of clips.
        max_steps: Maximum steps.
        seed: Random seed.
        cache_dir: Directory for cache files.
        force_rerun: If True, ignore cache and re-run.
        store_states: Whether to store states for rendering.
        use_stickiness: If True, use stickiness-aware inference with prev_indices.

    Returns:
        List of InferenceResult objects.
    """
    cache_dir = Path(cache_dir)
    cache_key = compute_cache_key(checkpoint_path, step, num_clips, seed, use_stickiness)
    cache_path = get_cache_path(cache_dir, cache_key)

    if not force_rerun and cache_path.exists():
        # Try to load from cache
        cached = load_inference_cache(cache_path)
        if cached is not None:
            results, metadata = cached
            logging.info(f"Loaded {len(results)} results from cache")

            if not store_states:
                # Don't need states, return cached results
                return results

            # Need states for rendering - re-run inference but keep cached data
            logging.info("Running inference to collect states for rendering...")
            results_with_states = run_inference(
                env=env,
                inference_fn=inference_fn,
                num_clips=num_clips,
                max_steps=max_steps,
                seed=seed,
                store_states=True,
                use_stickiness=use_stickiness,
            )
            return results_with_states

    # No cache - run inference
    logging.info(f"Running inference on {num_clips} clips...")
    results = run_inference(
        env=env,
        inference_fn=inference_fn,
        num_clips=num_clips,
        max_steps=max_steps,
        seed=seed,
        store_states=store_states,
        use_stickiness=use_stickiness,
    )

    # Save to cache (without states - too large)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "step": step,
        "num_clips": num_clips,
        "seed": seed,
        "max_steps": max_steps,
    }
    save_inference_cache(cache_path, results, metadata)

    return results


def generate_summary_report(
    output_dir: Path,
    cfg: DictConfig,
    num_codes: int,
    all_paths: dict[str, Any],
) -> str:
    """Generate a markdown summary report.

    Args:
        output_dir: Base output directory.
        cfg: Configuration.
        num_codes: Number of codes in codebook.
        all_paths: Dictionary of all generated file paths.

    Returns:
        Path to summary report.
    """
    report_path = output_dir / "summary" / "analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# VQ-VAE Code Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
        f"- Checkpoint: `{cfg.checkpoint.path}`",
        f"- Step: {cfg.checkpoint.step or 'latest'}",
        f"- Num clips: {cfg.inference.num_clips}",
        f"- Num codes: {num_codes}",
        "",
        "## Experiments Run",
        "",
    ]

    experiments = cfg.experiments
    if experiments.transition.enabled:
        lines.append("### Transition Analysis")
        lines.append("")
        lines.append("- Computed transition matrix and probabilities")
        lines.append(f"- Found top {experiments.transition.top_k_chains} chains")
        lines.append("- Classified code roles (entry, exit, hub, steady-state)")
        lines.append("")

    if experiments.segment_grid.enabled or experiments.duration.enabled:
        lines.append("### Segment Analysis")
        lines.append("")
        lines.append("- Extracted contiguous code segments")
        lines.append("- Computed duration statistics")
        if experiments.segment_grid.enabled:
            lines.append("- Rendered per-code segment videos")
        lines.append("")

    if experiments.kinematic.enabled:
        lines.append("### Kinematic Analysis")
        lines.append("")
        lines.append("- Extracted kinematic features per code")
        lines.append(f"- Features: {', '.join(experiments.kinematic.features)}")
        lines.append(f"- Clustering method: {experiments.kinematic.clustering_method}")
        lines.append("")

    if experiments.alignment.enabled:
        lines.append("### Alignment Analysis")
        lines.append("")
        lines.append("- Aligned segments using DTW")
        lines.append(f"- DTW feature: {experiments.alignment.dtw_feature}")
        lines.append("- Rendered aligned comparison videos")
        lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("See the following directories for outputs:")
    lines.append("")
    lines.append(f"- Cache: `{output_dir}/cache/`")
    lines.append(f"- Transitions: `{output_dir}/transitions/`")
    lines.append(f"- Segments: `{output_dir}/segments/`")
    lines.append(f"- Durations: `{output_dir}/durations/`")
    lines.append(f"- Kinematics: `{output_dir}/kinematics/`")
    lines.append(f"- Alignment: `{output_dir}/alignment/`")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also save JSON summary
    json_path = output_dir / "summary" / "analysis_summary.json"
    summary = {
        "generated": datetime.now().isoformat(),
        "config": {
            "checkpoint_path": cfg.checkpoint.path,
            "checkpoint_step": cfg.checkpoint.step,
            "num_clips": cfg.inference.num_clips,
            "num_codes": num_codes,
        },
        "experiments": {
            "transition": experiments.transition.enabled,
            "segment_grid": experiments.segment_grid.enabled,
            "duration": experiments.duration.enabled,
            "kinematic": experiments.kinematic.enabled,
            "alignment": experiments.alignment.enabled,
        },
        "output_paths": all_paths,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return str(report_path)


@hydra.main(version_base=None, config_path="../configs", config_name="code_analysis")
def main(cfg: DictConfig):
    """Run VQ-VAE code analysis pipeline."""
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("VQ-VAE Code Analysis Pipeline")
    print("=" * 60)

    # Create output directory
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Load checkpoint
    logging.info("\nLoading checkpoint...")
    ckpt = load_vq_checkpoint(
        cfg.checkpoint.path,
        step=cfg.checkpoint.step,
    )
    vq_cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]
    step = ckpt["step"]

    codebook = get_codebook(policy_params)
    num_codes = codebook.shape[0]
    latent_dim = codebook.shape[1]
    logging.info(f"  Codebook: {num_codes} codes, {latent_dim} dims")
    logging.info(f"  Checkpoint step: {step}")

    # Create stickiness-aware inference function
    # This properly applies the stickiness bias during inference
    stickiness_bias = vq_cfg.network_config.get("stickiness_bias", 0.0)
    use_stickiness = stickiness_bias > 0

    if use_stickiness:
        logging.info(f"  Stickiness bias: {stickiness_bias} (ENABLED)")
        inference_fn, _ = load_vq_inference_fn_with_stickiness(
            vq_cfg, policy_params, deterministic=True, get_activation=True
        )
    else:
        logging.info(f"  Stickiness bias: {stickiness_bias} (disabled)")
        inference_fn = load_vq_inference_fn(
            vq_cfg, policy_params, deterministic=True, get_activation=True
        )

    # Create environment
    logging.info("\nCreating environment...")
    (_, cfg_dict, env_cfg_ml) = config_utils.prepare_config(cfg)

    reference_clips = ReferenceClips(
        data_path=vq_cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.inference.get("clip_length", 250),
        keep_clips_idx=None,  # Load all clips, select during inference
    )
    env = imitation.Imitation(config=env_cfg_ml, clips=reference_clips)

    # Get camera name
    env_suffix = getattr(env, "_suffix", "-rodent")
    camera_name = f"{cfg.render.camera}{env_suffix}"

    # Run or load inference
    logging.info("\nRunning/loading inference...")
    need_states = (
        cfg.experiments.segment_grid.enabled or cfg.experiments.alignment.enabled
    )
    results = run_or_load_inference(
        env=env,
        inference_fn=inference_fn,
        checkpoint_path=cfg.checkpoint.path,
        step=cfg.checkpoint.step,
        num_clips=cfg.inference.num_clips,
        max_steps=cfg.inference.max_steps,
        seed=cfg.inference.seed,
        cache_dir=output_dir / "cache",
        force_rerun=cfg.inference.force_rerun,
        store_states=need_states,
        use_stickiness=use_stickiness,
    )

    all_paths = {}

    # Run transition analysis
    if cfg.experiments.transition.enabled:
        logging.info("\n" + "=" * 40)
        logging.info("Running transition analysis...")
        trans_paths = run_transition_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=output_dir / "transitions",
            min_chain_prob=cfg.experiments.transition.min_chain_prob,
            top_k_chains=cfg.experiments.transition.top_k_chains,
        )
        all_paths["transitions"] = trans_paths

    # Run segment analysis
    if cfg.experiments.segment_grid.enabled or cfg.experiments.duration.enabled:
        logging.info("\n" + "=" * 40)
        logging.info("Running segment analysis...")
        seg_results = run_segment_analysis(
            env=env,
            results=results,
            num_codes=num_codes,
            output_dir=output_dir / "segments",
            min_segment_frames=cfg.experiments.segment_grid.min_segment_frames,
            max_segments_per_code=cfg.experiments.segment_grid.max_segments_per_code,
            render_videos=cfg.experiments.segment_grid.enabled,
            camera=camera_name,
            fps=cfg.render.fps,
        )
        all_paths["segments"] = seg_results["paths"]

        # Copy duration plot to durations directory if enabled
        if cfg.experiments.duration.enabled:
            duration_dir = output_dir / "durations"
            duration_dir.mkdir(parents=True, exist_ok=True)
            all_paths["durations"] = seg_results["paths"]

    # Run kinematic analysis
    if cfg.experiments.kinematic.enabled:
        logging.info("\n" + "=" * 40)
        logging.info("Running kinematic analysis...")
        kin_paths = run_kinematic_analysis(
            results=results,
            num_codes=num_codes,
            output_dir=output_dir / "kinematics",
            clustering_method=cfg.experiments.kinematic.clustering_method,
        )
        all_paths["kinematics"] = kin_paths

    # Run alignment analysis
    if cfg.experiments.alignment.enabled:
        logging.info("\n" + "=" * 40)
        logging.info("Running alignment analysis...")
        align_results = run_alignment_analysis(
            env=env,
            results=results,
            num_codes=num_codes,
            output_dir=output_dir / "alignment",
            min_segment_length=cfg.experiments.alignment.min_segment_length,
            max_pairs_per_code=cfg.experiments.alignment.max_pairs_per_code,
            dtw_feature=cfg.experiments.alignment.dtw_feature,
            render_videos=True,
            camera=camera_name,
            fps=cfg.render.fps,
        )
        all_paths["alignment"] = align_results["paths"]

    # Generate summary report
    logging.info("\n" + "=" * 40)
    logging.info("Generating summary report...")
    report_path = generate_summary_report(output_dir, cfg, num_codes, all_paths)
    all_paths["report"] = report_path

    print("\n" + "=" * 60)
    print(f"Analysis complete! Results saved to {output_dir}")
    print("=" * 60)
    print(f"\nSummary report: {report_path}")


if __name__ == "__main__":
    main()
