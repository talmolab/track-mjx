#!/usr/bin/env python3
"""Collect rollout dataset from MLP prior model for discriminator training.

This script generates qpos trajectories from:
1. Original reference data
2. Encoder-decoder simulation rollouts
3. Prior simulation rollouts with varying logvar values

Output H5 contains:
- original_qpos: Reference data (num_clips, num_steps, 74)
- encoder_decoder_qpos: Encoder-decoder rollouts
- prior_qpos_logvar_-4: Prior rollouts with logvar=-4 (std~0.14)
- prior_qpos_logvar_-2: Prior rollouts with logvar=-2 (std~0.37)
- prior_qpos_logvar_0: Prior rollouts with logvar=0 (std=1.0)
- prior_qpos_deterministic: Prior rollouts using mean (no sampling)
- prior_qpos_predicted_logvar: Prior rollouts using network-predicted logvar
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from brax.training import distribution
from brax.training.acme import running_statistics
from jax import random
from ml_collections import ConfigDict
from mujoco import mjx
from omegaconf import OmegaConf
from tqdm import tqdm
from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent import wrappers as vnl_wrappers
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent.mlp_prior import prior_networks
from track_mjx.agent.mlp_prior.prior_rollout_eval import (
    create_prior_policy,
    extract_prior_decoder_params,
)
from track_mjx.agent.ff_ppo import intention_network


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect rollout data from MLP prior model for discriminator training"
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to MLP prior checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for output H5 file (e.g., rollout_dataset.h5)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of clips to process in parallel (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--data-path-override",
        type=str,
        default=None,
        help="Override reference data path from checkpoint config",
    )

    return parser.parse_args()


def load_mlp_prior_checkpoint(
    checkpoint_path: str, step: int | None = None
) -> Tuple[Any, Tuple]:
    """Load an MLP prior checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.
        step: Specific step to load. If None, loads latest.

    Returns:
        Tuple of (config, policy_params)
    """
    step_prefix = "PriorNetwork"
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        print(f"Loading checkpoint from {checkpoint_path} at step {step}")

        # Load config
        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
        cfg = OmegaConf.create(cfg)

        # Get prior layer sizes from config
        prior_hidden_layer_sizes = tuple(
            cfg.network_config.get(
                "prior_layer_sizes", cfg.network_config.encoder_layer_sizes
            )
        )

        # Create abstract policy for restoration
        abstract_policy = prior_networks.create_abstract_prior_policy(
            cfg=OmegaConf.to_container(cfg),
            prior_hidden_layer_sizes=prior_hidden_layer_sizes,
        )

        # Load policy
        policy = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    return cfg, policy


def fix_config_paths(cfg: Any, data_path_override: str | None = None) -> Any:
    """Fix paths in config that may reference /tmp or other locations.

    Args:
        cfg: OmegaConf configuration.
        data_path_override: Optional override for reference_data_path.

    Returns:
        Updated configuration.
    """
    OLD_PATH_PREFIX = "/tmp/track-mjx"
    NEW_PATH_PREFIX = str(Path(__file__).parent.parent.parent)

    def fix_path(path_str):
        if path_str and path_str.startswith(OLD_PATH_PREFIX):
            return path_str.replace(OLD_PATH_PREFIX, NEW_PATH_PREFIX, 1)
        return path_str

    OmegaConf.set_struct(cfg.env_config, False)

    if hasattr(cfg.env_config, "walker_xml_path"):
        OmegaConf.update(
            cfg.env_config, "walker_xml_path", fix_path(cfg.env_config.walker_xml_path)
        )
    if hasattr(cfg.env_config, "arena_xml_path"):
        OmegaConf.update(
            cfg.env_config, "arena_xml_path", fix_path(cfg.env_config.arena_xml_path)
        )

    if data_path_override:
        OmegaConf.update(cfg.env_config, "reference_data_path", data_path_override)
    elif hasattr(cfg.env_config, "reference_data_path"):
        OmegaConf.update(
            cfg.env_config,
            "reference_data_path",
            fix_path(cfg.env_config.reference_data_path),
        )

    OmegaConf.set_struct(cfg.env_config, True)
    return cfg


def create_environment(cfg: Any) -> Tuple[Any, ReferenceClips]:
    """Create the imitation environment from config.

    Args:
        cfg: Configuration from checkpoint.

    Returns:
        Tuple of (environment, reference_clips).
    """
    env_cfg = ConfigDict(OmegaConf.to_container(cfg.env_config, resolve=True))

    reference_clips = ReferenceClips(
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.get("keep_clips_idx", None),
    )

    env = vnl_wrappers.FlattenObsWrapper(
        imitation.Imitation(config=env_cfg, clips=reference_clips)
    )

    return env, reference_clips


def create_encoder_decoder_policy(
    policy_params: Tuple,
    cfg: Any,
) -> Callable:
    """Create encoder-decoder policy using frozen encoder and decoder.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).
        cfg: Configuration from checkpoint.

    Returns:
        Policy function (obs, rng) -> (action, extras).
    """
    normalizer_params, network_params = policy_params
    encoder_params = network_params["params"]["encoder"]
    decoder_params = network_params["params"]["decoder"]

    return prior_networks.make_encoder_decoder_inference_fn(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        normalizer_params=normalizer_params,
        encoder_hidden_layer_sizes=tuple(cfg.network_config.encoder_layer_sizes),
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        latent_size=cfg.network_config.intention_size,
        action_size=cfg.network_config.action_size,
        reference_obs_size=cfg.network_config.reference_obs_size,
        proprioceptive_obs_size=(
            cfg.network_config.observation_size - cfg.network_config.reference_obs_size
        ),
        deterministic=True,
    )


def create_prior_policy_fn(
    policy_params: Tuple,
    cfg: Any,
    fixed_logvar: Optional[float],
    deterministic: bool,
) -> Callable:
    """Create prior policy using prior and decoder networks.

    Args:
        policy_params: Tuple of (normalizer_params, network_params).
        cfg: Configuration from checkpoint.
        fixed_logvar: Fixed log-variance for sampling. If None, uses predicted logvar.
        deterministic: If True, use mean instead of sampling.

    Returns:
        Policy function (obs, rng) -> (action, extras).
    """
    prior_params, decoder_params, normalizer_params = extract_prior_decoder_params(
        policy_params
    )

    proprioceptive_obs_size = (
        cfg.network_config.observation_size - cfg.network_config.reference_obs_size
    )

    # Create proprioceptive-only normalizer
    proprio_normalizer_params = running_statistics.RunningStatisticsState(
        count=normalizer_params.count,
        mean=normalizer_params.mean[-proprioceptive_obs_size:],
        summed_variance=normalizer_params.summed_variance[-proprioceptive_obs_size:],
        std=normalizer_params.std[-proprioceptive_obs_size:],
    )

    prior_hidden_layer_sizes = tuple(
        cfg.network_config.get(
            "prior_layer_sizes", cfg.network_config.encoder_layer_sizes
        )
    )

    return create_prior_policy(
        prior_network_params=prior_params,
        decoder_network_params=decoder_params,
        normalizer_params=proprio_normalizer_params,
        intention_latent_size=cfg.network_config.intention_size,
        action_size=cfg.network_config.action_size,
        proprioceptive_obs_size=proprioceptive_obs_size,
        decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
        prior_hidden_layer_sizes=prior_hidden_layer_sizes,
        preprocess_observations_fn=running_statistics.normalize,
        fixed_logvar=fixed_logvar,
        deterministic=deterministic,
    )


def create_batched_rollout_fn(
    env: Any,
    policy_fn: Callable,
    num_steps: int,
    proprioceptive_obs_size: int,
    is_prior_policy: bool,
) -> Callable:
    """Create a batched rollout function for multiple clips.

    Args:
        env: The environment.
        policy_fn: Policy function (obs, rng) -> (action, extras).
        num_steps: Number of physics steps per rollout.
        proprioceptive_obs_size: Size of proprioceptive observations.
        is_prior_policy: If True, pass only proprioceptive obs to policy.

    Returns:
        A function that takes (initial_qpos_batch, clip_indices, rng_keys) and returns qpos trajectories.
    """
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    mjx_model = env.mjx_model

    def single_rollout(
        initial_qpos: jax.Array, clip_idx: int, rng_key: jax.Array
    ) -> jax.Array:
        """Run single rollout from initial_qpos, return qpos trajectory."""
        reset_key, rollout_key = random.split(rng_key)

        # Reset environment to get proper state structure at specific clip
        state = jit_reset(reset_key, clip_idx=clip_idx, start_frame=0)

        # Replace qpos with initial_qpos and set qvel to zeros
        data = state.data.replace(qpos=initial_qpos)
        data = data.replace(qvel=jnp.zeros(mjx_model.nv))
        data = mjx.forward(mjx_model, data)
        state = state.replace(data=data)

        # Collect initial qpos
        qpos_list = [initial_qpos]

        def step_fn(carry, _):
            state, key = carry
            key, action_key = random.split(key)

            # Get observations for policy
            if is_prior_policy:
                # Prior policy uses only proprioceptive observations
                obs = state.obs[..., -proprioceptive_obs_size:]
            else:
                # Encoder-decoder uses full observations
                obs = state.obs

            # Get action from policy
            action, _ = policy_fn(obs, action_key)

            # Step environment
            next_state = jit_step(state, action)

            return (next_state, key), next_state.data.qpos

        # Run rollout for num_steps - 1 (we already have initial qpos)
        initial_carry = (state, rollout_key)
        _, qpos_trajectory = jax.lax.scan(
            step_fn, initial_carry, None, length=num_steps - 1
        )

        # Prepend initial qpos
        qpos_trajectory = jnp.concatenate(
            [initial_qpos[None, :], qpos_trajectory], axis=0
        )

        return qpos_trajectory  # (num_steps, 74)

    # Vmap over batch of clips
    batched_rollout = jax.jit(jax.vmap(single_rollout))

    return batched_rollout


def collect_rollouts(
    env: Any,
    reference_clips: ReferenceClips,
    policy_fn: Callable,
    num_clips: int,
    num_steps: int,
    batch_size: int,
    proprioceptive_obs_size: int,
    is_prior_policy: bool,
    rng_key: jax.Array,
    desc: str = "Collecting rollouts",
) -> np.ndarray:
    """Collect rollouts for all clips.

    Args:
        env: Environment.
        reference_clips: ReferenceClips object.
        policy_fn: Policy function.
        num_clips: Total number of clips.
        num_steps: Steps per rollout.
        batch_size: Clips per batch.
        proprioceptive_obs_size: Size of proprioceptive observations.
        is_prior_policy: If True, pass only proprioceptive obs to policy.
        rng_key: Random key.
        desc: Progress bar description.

    Returns:
        qpos array of shape (num_clips, num_steps, qpos_dim).
    """
    batched_rollout = create_batched_rollout_fn(
        env, policy_fn, num_steps, proprioceptive_obs_size, is_prior_policy
    )

    all_qpos = []
    num_batches = (num_clips + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=desc):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_clips)
        actual_batch_size = end_idx - start_idx

        # Get initial qpos for this batch (first frame of each clip)
        initial_qpos_batch = jnp.stack(
            [
                reference_clips.at(clip=i, frame=0).qpos
                for i in range(start_idx, end_idx)
            ]
        )

        # Clip indices for this batch
        clip_indices = jnp.arange(start_idx, end_idx)

        # Generate random keys for this batch
        rng_key, batch_key = random.split(rng_key)
        batch_keys = random.split(batch_key, actual_batch_size)

        # Run batched rollouts
        qpos_batch = batched_rollout(initial_qpos_batch, clip_indices, batch_keys)

        # Transfer to CPU and append
        all_qpos.append(np.array(qpos_batch))

    return np.concatenate(all_qpos, axis=0)


def get_original_qpos(reference_clips: ReferenceClips) -> np.ndarray:
    """Extract original qpos from reference clips.

    Args:
        reference_clips: ReferenceClips object.

    Returns:
        qpos array of shape (num_clips, num_steps, qpos_dim).
    """
    # reference_clips.qpos already has shape (num_clips, num_steps, qpos_dim)
    return np.array(reference_clips.qpos)


def save_results(
    output_path: str,
    original_qpos: np.ndarray,
    results: Dict[str, np.ndarray],
    checkpoint_path: str,
    num_clips: int,
    num_steps: int,
) -> None:
    """Save all rollout data to H5 file.

    Args:
        output_path: Path to output H5 file.
        original_qpos: Original reference qpos.
        results: Dictionary with rollout arrays.
        checkpoint_path: Path to source checkpoint.
        num_clips: Number of clips.
        num_steps: Steps per clip.
    """
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Original reference data
        f.create_dataset("original_qpos", data=original_qpos, dtype="float32")

        # Rollout results
        for name, data in results.items():
            dataset_name = f"{name}_qpos"
            f.create_dataset(dataset_name, data=data, dtype="float32")

        # Metadata
        f.attrs["num_clips"] = num_clips
        f.attrs["num_steps"] = num_steps
        f.attrs["qpos_dim"] = original_qpos.shape[-1]
        f.attrs["checkpoint_path"] = checkpoint_path
        f.attrs["logvars"] = [-4.0, -2.0, 0.0]

    print(f"Saved results to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("MLP Prior Rollout Dataset Collection")
    print("=" * 60)

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}")
    cfg, policy_params = load_mlp_prior_checkpoint(
        args.checkpoint_path, args.checkpoint_step
    )

    # Fix paths in config
    cfg = fix_config_paths(cfg, args.data_path_override)

    print(f"  Observation size: {cfg.network_config.observation_size}")
    print(f"  Action size: {cfg.network_config.action_size}")
    print(f"  Intention size: {cfg.network_config.intention_size}")
    print(f"  Reference obs size: {cfg.network_config.reference_obs_size}")

    # Create environment
    print("\nCreating environment...")
    env, reference_clips = create_environment(cfg)

    # reference_clips.qpos is already (num_clips, num_steps, qpos_dim)
    num_clips = reference_clips.qpos.shape[0]
    num_steps = reference_clips.qpos.shape[1]
    proprioceptive_obs_size = (
        cfg.network_config.observation_size - cfg.network_config.reference_obs_size
    )

    print(f"  Number of clips: {num_clips}")
    print(f"  Steps per clip: {num_steps}")
    print(f"  Proprioceptive obs size: {proprioceptive_obs_size}")

    # Initialize random key
    rng = random.PRNGKey(args.seed)

    results = {}

    # 1. Get original qpos
    print("\n" + "=" * 60)
    print("Extracting original reference qpos...")
    print("=" * 60)
    original_qpos = get_original_qpos(reference_clips)
    print(f"  Shape: {original_qpos.shape}")

    # 2. Encoder-decoder rollouts
    print("\n" + "=" * 60)
    print("Collecting encoder-decoder rollouts...")
    print("=" * 60)
    enc_dec_policy = create_encoder_decoder_policy(policy_params, cfg)
    rng, key = random.split(rng)
    t_start = time.time()
    results["encoder_decoder"] = collect_rollouts(
        env,
        reference_clips,
        enc_dec_policy,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        is_prior_policy=False,
        rng_key=key,
        desc="Encoder-decoder",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['encoder_decoder'].shape}")

    # 3. Prior rollouts with logvar=-4
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (logvar=-4, std~0.14)...")
    print("=" * 60)
    prior_policy_lv4 = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=-4.0, deterministic=False
    )
    rng, key = random.split(rng)
    t_start = time.time()
    results["prior_logvar_-4"] = collect_rollouts(
        env,
        reference_clips,
        prior_policy_lv4,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (logvar=-4)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['prior_logvar_-4'].shape}")

    # 4. Prior rollouts with logvar=-2
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (logvar=-2, std~0.37)...")
    print("=" * 60)
    prior_policy_lv2 = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=-2.0, deterministic=False
    )
    rng, key = random.split(rng)
    t_start = time.time()
    results["prior_logvar_-2"] = collect_rollouts(
        env,
        reference_clips,
        prior_policy_lv2,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (logvar=-2)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['prior_logvar_-2'].shape}")

    # 5. Prior rollouts with logvar=0
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (logvar=0, std=1.0)...")
    print("=" * 60)
    prior_policy_lv0 = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=0.0, deterministic=False
    )
    rng, key = random.split(rng)
    t_start = time.time()
    results["prior_logvar_0"] = collect_rollouts(
        env,
        reference_clips,
        prior_policy_lv0,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (logvar=0)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['prior_logvar_0'].shape}")

    # 6. Prior rollouts deterministic
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (deterministic, mean only)...")
    print("=" * 60)
    prior_policy_det = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=0.0, deterministic=True
    )
    rng, key = random.split(rng)
    t_start = time.time()
    results["prior_deterministic"] = collect_rollouts(
        env,
        reference_clips,
        prior_policy_det,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (deterministic)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['prior_deterministic'].shape}")

    # 7. Prior rollouts with predicted logvar
    print("\n" + "=" * 60)
    print("Collecting prior rollouts (predicted logvar, per-dimension)...")
    print("=" * 60)
    prior_policy_predicted = create_prior_policy_fn(
        policy_params, cfg, fixed_logvar=None, deterministic=False
    )
    rng, key = random.split(rng)
    t_start = time.time()
    results["prior_predicted_logvar"] = collect_rollouts(
        env,
        reference_clips,
        prior_policy_predicted,
        num_clips,
        num_steps,
        args.batch_size,
        proprioceptive_obs_size,
        is_prior_policy=True,
        rng_key=key,
        desc="Prior (predicted logvar)",
    )
    print(f"  Completed in {time.time() - t_start:.1f}s")
    print(f"  Shape: {results['prior_predicted_logvar'].shape}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    save_results(
        args.output_path,
        original_qpos,
        results,
        args.checkpoint_path,
        num_clips,
        num_steps,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output file: {args.output_path}")
    print(f"Datasets:")
    print(f"  - original_qpos: {original_qpos.shape}")
    for name, data in results.items():
        print(f"  - {name}_qpos: {data.shape}")
    print("\nDone!")


if __name__ == "__main__":
    main()
