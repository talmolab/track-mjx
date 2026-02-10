"""VQ-VAE inference pipeline for generating rollout data.

This module provides the standalone inference pipeline that runs VQ-VAE
inference on reference clips and saves results to H5 format for downstream
analysis.

Usage:
    cd vqvae_jax
    python -m inference.run_inference checkpoint.path=/path/to/checkpoint

    # With options:
    python -m inference.run_inference \
        checkpoint.path=/path/to/checkpoint \
        inference.num_clips=100 \
        inference.store_z_e=true \
        output.path=./outputs/rollout.h5
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add paths for package imports
INFERENCE_DIR = Path(__file__).parent
VQVAE_DIR = INFERENCE_DIR.parent
REPO_ROOT = VQVAE_DIR.parent
sys.path.insert(0, str(VQVAE_DIR))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from omegaconf import DictConfig
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.config import utils as config_utils

from .h5_utils import RolloutData, save_rollout_h5

# Import checkpoint utils from analysis module
sys.path.insert(0, str(VQVAE_DIR / "analysis"))
from analysis.checkpoint_utils import (
    load_vq_checkpoint,
    load_vq_inference_fn,
    load_vq_inference_fn_with_stickiness,
    get_codebook,
    get_all_codebooks,
)


def run_inference(
    env: Any,
    inference_fn: Any,
    num_clips: int,
    max_steps: int,
    seed: int,
    store_z_e: bool = False,
    use_stickiness: bool = False,
    rvq_depth: int = 1,
) -> list[RolloutData]:
    """Run VQ-VAE inference on multiple clips.

    Args:
        env: Environment with reset/step methods.
        inference_fn: VQ-VAE inference function. If use_stickiness=False,
            signature is (obs, rng) -> (action, extras). If use_stickiness=True,
            signature is (obs, rng, prev_indices) -> (action, extras).
        num_clips: Number of clips to process.
        max_steps: Maximum steps per clip.
        seed: Random seed.
        store_z_e: Whether to store encoder outputs (z_e) before quantization.
        use_stickiness: If True, track previous code indices and pass to
            inference_fn for stickiness bias.
        rvq_depth: Number of RVQ depth levels for index tracking.

    Returns:
        List of RolloutData objects.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    results = []
    rng = jax.random.PRNGKey(seed)

    for clip_idx in range(num_clips):
        logging.info(f"Running inference on clip {clip_idx + 1}/{num_clips}...")

        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        code_indices = []
        rvq_indices_per_depth = [[] for _ in range(rvq_depth)]
        qpos_list = []
        qvel_list = []
        rewards = []
        z_e_list = [] if store_z_e else None
        prev_indices = None  # Track previous code for stickiness

        for step in range(max_steps):
            # Get observation and flatten to dict format expected by policy
            obs = state.obs
            flat_obs = flatten_obs_dict(obs)

            # Run inference (with or without stickiness)
            rng, action_rng = jax.random.split(rng)
            if use_stickiness:
                action, extras = inference_fn(flat_obs, action_rng, prev_indices)
            else:
                action, extras = inference_fn(flat_obs, action_rng)

            # Extract primary (L0) code index
            code_idx = int(extras["indices"])
            code_indices.append(code_idx)

            # Extract and track per-depth indices for RVQ
            if "all_indices" in extras:
                all_indices = extras["all_indices"]
                # Update prev_indices with full tuple for multi-level stickiness
                prev_indices = all_indices
                for d in range(rvq_depth):
                    if isinstance(all_indices, tuple) and d < len(all_indices):
                        rvq_indices_per_depth[d].append(int(all_indices[d]))
                    elif d == 0:
                        rvq_indices_per_depth[d].append(code_idx)
            else:
                prev_indices = jnp.array(code_idx)
                rvq_indices_per_depth[0].append(code_idx)

            # Store z_e if requested
            if store_z_e and "z_e" in extras:
                z_e_list.append(np.array(extras["z_e"]))

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

            if next_state.done:
                break

            state = next_state

        # Build rvq_indices tuple (None for depth=1 to save space)
        rvq_indices = None
        if rvq_depth > 1 and rvq_indices_per_depth[0]:
            rvq_indices = tuple(
                np.array(rvq_indices_per_depth[d]) for d in range(rvq_depth)
            )

        # Create result
        z_e = np.stack(z_e_list) if z_e_list else None
        result = RolloutData(
            clip_idx=clip_idx,
            code_indices=np.array(code_indices),
            qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
            qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
            rewards=np.array(rewards),
            z_e=z_e,
            rvq_indices=rvq_indices,
        )
        results.append(result)

    return results


def run_inference_pipeline(cfg: DictConfig) -> str:
    """Run the complete inference pipeline and save to H5.

    Args:
        cfg: Hydra configuration with checkpoint, inference, and output sections.

    Returns:
        Path to the saved H5 file.
    """
    logging.set_verbosity(logging.INFO)

    print("=" * 60)
    print("VQ-VAE Inference Pipeline")
    print("=" * 60)

    # Validate required config
    if cfg.checkpoint.path is None:
        raise ValueError("checkpoint.path is required")

    # Load checkpoint
    logging.info("\nLoading checkpoint...")
    ckpt = load_vq_checkpoint(
        cfg.checkpoint.path,
        step=cfg.checkpoint.step,
    )
    vq_cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]
    step = ckpt["step"]

    codebooks = get_all_codebooks(policy_params)
    num_codes = codebooks[0].shape[0]
    latent_dim = codebooks[0].shape[1]
    rvq_depth = len(codebooks)
    logging.info(
        f"  Codebook: {num_codes} codes, {latent_dim} dims, {rvq_depth} depth(s)"
    )
    logging.info(f"  Checkpoint step: {step}")

    # Create stickiness-aware inference function if needed
    stickiness_bias = vq_cfg.network_config.get("stickiness_bias", 0.0)
    try:
        use_stickiness = any(float(b) > 0 for b in stickiness_bias)
    except TypeError:
        use_stickiness = float(stickiness_bias) > 0

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
        keep_clips_idx=None,  # Load all clips
    )
    env = imitation.Imitation(config=env_cfg_ml, clips=reference_clips)

    # Run inference
    logging.info("\nRunning inference...")
    logging.info(f"  Num clips: {cfg.inference.num_clips}")
    logging.info(f"  Max steps: {cfg.inference.max_steps}")
    logging.info(f"  Seed: {cfg.inference.seed}")
    logging.info(f"  Store z_e: {cfg.inference.store_z_e}")

    results = run_inference(
        env=env,
        inference_fn=inference_fn,
        num_clips=cfg.inference.num_clips,
        max_steps=cfg.inference.max_steps,
        seed=cfg.inference.seed,
        store_z_e=cfg.inference.store_z_e,
        use_stickiness=use_stickiness,
        rvq_depth=rvq_depth,
    )

    # Prepare metadata
    metadata = {
        "checkpoint_path": str(cfg.checkpoint.path),
        "checkpoint_step": step,
        "num_clips": cfg.inference.num_clips,
        "max_steps": cfg.inference.max_steps,
        "seed": cfg.inference.seed,
        "store_z_e": cfg.inference.store_z_e,
        "use_stickiness": use_stickiness,
        "stickiness_bias": str(stickiness_bias),
        "num_codes": num_codes,
        "latent_dim": latent_dim,
        "rvq_depth": rvq_depth,
        "timestamp": datetime.now().isoformat(),
    }

    # Save to H5
    output_path = Path(cfg.output.path)
    logging.info(f"\nSaving results to {output_path}...")
    save_rollout_h5(output_path, results, metadata)

    # Print summary
    total_frames = sum(len(r.code_indices) for r in results)
    logging.info(f"\nInference complete!")
    logging.info(f"  Total rollouts: {len(results)}")
    logging.info(f"  Total frames: {total_frames}")
    logging.info(f"  Output file: {output_path}")

    print("\n" + "=" * 60)
    print(f"Results saved to {output_path}")
    print("=" * 60)

    return str(output_path)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    """Hydra entry point for inference pipeline."""
    run_inference_pipeline(cfg)


if __name__ == "__main__":
    main()
