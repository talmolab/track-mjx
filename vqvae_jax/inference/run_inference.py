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

sys.path.insert(0, str(VQVAE_DIR))
from ref_direct_imitation import RefDirectImitation

from .h5_utils import RolloutData, save_rollout_h5

# Import checkpoint utils from analysis module
sys.path.insert(0, str(VQVAE_DIR / "analysis"))
from analysis.checkpoint_utils import (
    load_vq_checkpoint,
    load_vq_chunked_inference_fn,
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


def run_inference_chunked(
    env: Any,
    inference_fn: Any,
    initial_chunk_state_fn: Any,
    num_clips: int,
    max_steps: int,
    seed: int,
    store_z_e: bool = False,
    rvq_depth: int = 2,
) -> list[RolloutData]:
    """Run VQ-VAE inference with code-chunked (Semi-MDP) rollouts.

    The inference_fn carries chunk_state (held_d0_idx, tau) through the
    rollout, applying temporal commitment on the D0 code.

    Args:
        env: Environment with reset/step methods.
        inference_fn: Chunked inference function with signature
            (obs, chunk_state, rng) -> (action, extras, new_chunk_state).
        initial_chunk_state_fn: Callable returning initial chunk_state tuple.
        num_clips: Number of clips to process.
        max_steps: Maximum steps per clip.
        seed: Random seed.
        store_z_e: Whether to store encoder outputs (z_e) before quantization.
        rvq_depth: Number of RVQ depth levels for index tracking.

    Returns:
        List of RolloutData objects with tau field populated.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    results = []
    rng = jax.random.PRNGKey(seed)

    for clip_idx in range(num_clips):
        logging.info(
            f"Running chunked inference on clip {clip_idx + 1}/{num_clips}..."
        )

        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        code_indices = []
        rvq_indices_per_depth = [[] for _ in range(rvq_depth)]
        qpos_list = []
        qvel_list = []
        rewards = []
        tau_list = []
        z_e_list = [] if store_z_e else None
        chunk_state = initial_chunk_state_fn()

        for step in range(max_steps):
            obs = flatten_obs_dict(state.obs)

            rng, action_rng = jax.random.split(rng)
            action, extras, chunk_state = inference_fn(
                obs, chunk_state, action_rng
            )

            # Extract primary (L0) code index
            code_idx = int(extras["indices"])
            code_indices.append(code_idx)

            # Track tau (timer value)
            tau_val = int(extras.get("tau", 0))
            tau_list.append(tau_val)

            # Extract per-depth indices for RVQ
            all_indices = extras.get("all_indices")
            if all_indices is not None:
                for d in range(rvq_depth):
                    if isinstance(all_indices, tuple) and d < len(all_indices):
                        rvq_indices_per_depth[d].append(int(all_indices[d]))
                    elif d == 0:
                        rvq_indices_per_depth[d].append(code_idx)
            else:
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

        # Build rvq_indices tuple
        rvq_indices = None
        if rvq_depth > 1 and rvq_indices_per_depth[0]:
            rvq_indices = tuple(
                np.array(rvq_indices_per_depth[d]) for d in range(rvq_depth)
            )

        z_e = np.stack(z_e_list) if z_e_list else None
        result = RolloutData(
            clip_idx=clip_idx,
            code_indices=np.array(code_indices),
            qpos=np.stack(qpos_list) if qpos_list else np.zeros((0, 0)),
            qvel=np.stack(qvel_list) if qvel_list else np.zeros((0, 0)),
            rewards=np.array(rewards),
            z_e=z_e,
            rvq_indices=rvq_indices,
            tau=np.array(tau_list),
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

    # Detect mode from checkpoint config
    use_code_chunking = bool(
        vq_cfg.network_config.get("use_code_chunking", False)
    )
    use_continuous_latent = bool(
        vq_cfg.network_config.get("use_continuous_latent", False)
    )
    commitment_horizon = int(
        vq_cfg.network_config.get("code_commitment_horizon", 10)
    )
    use_ref_direct = bool(
        vq_cfg.network_config.get("use_ref_direct_encoder", False)
    )
    logging.info(f"  use_code_chunking: {use_code_chunking}")
    logging.info(f"  use_continuous_latent: {use_continuous_latent}")
    logging.info(f"  use_ref_direct_encoder: {use_ref_direct}")

    # Create appropriate inference function based on mode
    stickiness_bias = vq_cfg.network_config.get("stickiness_bias", 0.0)
    try:
        use_stickiness = any(float(b) > 0 for b in stickiness_bias)
    except TypeError:
        use_stickiness = float(stickiness_bias) > 0

    chunked_inference_fn = None
    initial_chunk_state_fn = None

    if use_code_chunking:
        logging.info(
            f"  Code chunking ENABLED (H={commitment_horizon}), "
            f"using chunked inference"
        )
        chunked_inference_fn, initial_chunk_state_fn = (
            load_vq_chunked_inference_fn(
                vq_cfg,
                policy_params,
                commitment_horizon=commitment_horizon,
                deterministic=True,
            )
        )
        inference_fn = None  # Not used in chunked mode
    elif use_stickiness:
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

    # Split-aware clip selection
    data_split = cfg.inference.get("data_split", "all")
    train_ratio = None
    if data_split in ("train", "test"):
        # Must use training config's clip_length and keep_clips_idx so split
        # indices map to the same physical clips as during training.
        reference_clips = ReferenceClips(
            data_path=vq_cfg.env_config.reference_data_path,
            n_frames_per_clip=vq_cfg.env_config.clip_length,
            keep_clips_idx=vq_cfg.env_config.get("keep_clips_idx", None),
        )
        train_ratio = float(vq_cfg.train_setup.get("train_subset_ratio", 1.0))
        train_seed = int(vq_cfg.train_setup.train_config.get("seed", 0))
        key_split, _ = jax.random.split(jax.random.PRNGKey(train_seed))
        train_clips, test_clips = reference_clips.split(
            train_ratio=train_ratio, seed=key_split
        )
        clips = train_clips if data_split == "train" else test_clips
        num_clips = clips.qpos.shape[0]
        logging.info(f"  Data split: {data_split} ({num_clips} clips)")
    else:
        reference_clips = ReferenceClips(
            data_path=vq_cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.inference.get("clip_length", 250),
            keep_clips_idx=None,
        )
        clips = reference_clips
        num_clips = cfg.inference.num_clips

    EnvClass = RefDirectImitation if use_ref_direct else imitation.Imitation
    env = EnvClass(config=env_cfg_ml, clips=clips)

    # Run inference
    logging.info("\nRunning inference...")
    logging.info(f"  Num clips: {num_clips}")
    logging.info(f"  Max steps: {cfg.inference.max_steps}")
    logging.info(f"  Seed: {cfg.inference.seed}")
    logging.info(f"  Store z_e: {cfg.inference.store_z_e}")

    if use_code_chunking:
        results = run_inference_chunked(
            env=env,
            inference_fn=chunked_inference_fn,
            initial_chunk_state_fn=initial_chunk_state_fn,
            num_clips=num_clips,
            max_steps=cfg.inference.max_steps,
            seed=cfg.inference.seed,
            store_z_e=cfg.inference.store_z_e,
            rvq_depth=rvq_depth,
        )
    else:
        results = run_inference(
            env=env,
            inference_fn=inference_fn,
            num_clips=num_clips,
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
        "num_clips": num_clips,
        "max_steps": cfg.inference.max_steps,
        "seed": cfg.inference.seed,
        "store_z_e": cfg.inference.store_z_e,
        "use_stickiness": use_stickiness,
        "stickiness_bias": str(stickiness_bias),
        "num_codes": num_codes,
        "latent_dim": latent_dim,
        "rvq_depth": rvq_depth,
        "timestamp": datetime.now().isoformat(),
        "data_split": data_split,
        "train_subset_ratio": train_ratio,
        "use_continuous_latent": use_continuous_latent,
        "use_code_chunking": use_code_chunking,
        "commitment_horizon": commitment_horizon if use_code_chunking else None,
    }

    # Save to H5 (auto-suffix path for train/test splits)
    output_path = Path(cfg.output.path)
    if data_split != "all":
        output_path = output_path.with_stem(f"{output_path.stem}_{data_split}")
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
