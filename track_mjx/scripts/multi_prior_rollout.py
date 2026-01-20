#!/usr/bin/env python3
"""Multi-variance prior rollout script.

Runs prior-only rollouts across multiple variance settings and starting poses,
saves videos and rollout data (qpos, qvel) to an H5 file.

Example usage:
    python scripts/multi_prior_rollout.py \
        --checkpoint-path /path/to/checkpoint \
        --output-dir /path/to/output \
        --fixed-logvar-values -4.0 -2.0 -1.0 0.0 \
        --clip-configs 0,0 1,0 2,0 \
        --max-steps 1000

The script will create:
    - Videos: {output_dir}/{variance}_{pose}_rollout{idx}.mp4
    - Data: {output_dir}/rollout_data.h5 containing qpos/qvel for all rollouts
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set environment variables before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import h5py
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import random
from ml_collections import ConfigDict
from mujoco import mjx
from omegaconf import OmegaConf

from brax.training import distribution
from brax.training.acme import running_statistics

from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips

from track_mjx.agent.mlp_prior import prior_networks
from track_mjx.agent.mlp_prior.prior_rollout_eval import (
    check_termination_nan,
    compute_world_zaxis_termination,
    extract_prior_decoder_params,
)
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    flatten_obs_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-variance prior rollouts and save videos/data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to mlp_prior checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save videos and H5 data",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Specific checkpoint step to load (None = latest)",
    )

    # Variance settings
    parser.add_argument(
        "--fixed-logvar-values",
        type=float,
        nargs="+",
        default=[-4.0, -2.0, -1.0, 0.0],
        help="List of fixed logvar values to test",
    )
    parser.add_argument(
        "--include-deterministic",
        action="store_true",
        default=True,
        help="Include deterministic (mean-only) rollouts",
    )
    parser.add_argument(
        "--include-predicted-logvar",
        action="store_true",
        default=True,
        help="Include rollouts with network-predicted logvar",
    )

    # Starting pose settings
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        default=True,
        help="Include rollouts from neutral pose",
    )
    parser.add_argument(
        "--clip-configs",
        type=str,
        nargs="+",
        default=["0,0", "1,0", "2,0"],
        help="Clip configurations as 'clip_idx,start_frame' pairs",
    )

    # Rollout settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per rollout",
    )
    parser.add_argument(
        "--num-rollouts-per-config",
        type=int,
        default=1,
        help="Number of rollouts per (variance, starting_pose) combination",
    )

    # Rendering settings
    parser.add_argument(
        "--render-fps",
        type=int,
        default=50,
        help="FPS for rendered videos",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="close_profile-rodent",
        help="Camera name for rendering",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=480,
        help="Video height in pixels",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=640,
        help="Video width in pixels",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip video rendering (only save H5 data)",
    )
    parser.add_argument(
        "--num-rollouts-to-render",
        type=int,
        default=1,
        help="Number of rollouts to render as videos (per variance/pose config). "
        "Set to 0 to skip rendering, or -1 to render all.",
    )

    # Path fixing (for checkpoints saved on different machines)
    parser.add_argument(
        "--old-path-prefix",
        type=str,
        default="/tmp/track-mjx",
        help="Old path prefix to replace in checkpoint config",
    )
    parser.add_argument(
        "--new-path-prefix",
        type=str,
        default="/home/mila/a/aidan.sirbu/track-mjx",
        help="New path prefix for checkpoint config",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, step: Optional[int] = None):
    """Load an mlp_prior checkpoint."""
    step_prefix = "PriorNetwork"
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=step_prefix)

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()
        print(f"Loading checkpoint from {checkpoint_path} at step {step}")

        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
        cfg = OmegaConf.create(cfg)

        prior_hidden_layer_sizes = tuple(
            cfg.network_config.get(
                "prior_layer_sizes", cfg.network_config.encoder_layer_sizes
            )
        )

        abstract_policy = prior_networks.create_abstract_prior_policy(
            cfg=OmegaConf.to_container(cfg),
            prior_hidden_layer_sizes=prior_hidden_layer_sizes,
        )

        policy = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    return cfg, policy, step


def fix_paths(cfg, old_prefix: str, new_prefix: str):
    """Fix paths in config."""

    def fix_path(path_str):
        if path_str and path_str.startswith(old_prefix):
            return path_str.replace(old_prefix, new_prefix, 1)
        return path_str

    OmegaConf.set_struct(cfg.env_config, False)
    OmegaConf.update(
        cfg.env_config, "walker_xml_path", fix_path(cfg.env_config.walker_xml_path)
    )
    OmegaConf.update(
        cfg.env_config, "arena_xml_path", fix_path(cfg.env_config.arena_xml_path)
    )
    OmegaConf.update(
        cfg.env_config,
        "reference_data_path",
        fix_path(cfg.env_config.reference_data_path),
    )
    OmegaConf.set_struct(cfg.env_config, True)
    return cfg


def create_environment(cfg):
    """Create the imitation environment from config."""
    env_cfg = ConfigDict(OmegaConf.to_container(cfg.env_config, resolve=True))
    if hasattr(env_cfg, "nconmax"):
        env_cfg.naconmax = env_cfg.nconmax * 1

    reference_clips = ReferenceClips(
        data_path=cfg.env_config.reference_data_path,
        n_frames_per_clip=cfg.env_config.clip_length,
        keep_clips_idx=cfg.env_config.get("keep_clips_idx", None),
    )

    env = imitation.Imitation(config=env_cfg, clips=reference_clips)
    return env


def create_neutral_state(env, rng_key):
    """Create state initialized to neutral pose."""
    state = env.reset(rng_key)
    mjx_model = env.mjx_model
    neutral_qpos = jnp.array(env.mj_model.qpos0)
    neutral_qpos = neutral_qpos.at[2].set(0)
    data = state.data.replace(qpos=neutral_qpos)
    data = data.replace(qvel=jnp.zeros(mjx_model.nv))
    data = mjx.forward(mjx_model, data)
    state = state.replace(data=data)
    return state


def create_clip_state(env, clip_idx: int, start_frame: int, rng_key):
    """Create state initialized from a specific clip and frame."""
    state = env.reset(rng_key)
    unwrapped_env = env
    while hasattr(unwrapped_env, "_env"):
        unwrapped_env = unwrapped_env._env

    reference = unwrapped_env.reference_clips.at(clip=clip_idx, frame=start_frame)
    mjx_model = env.mjx_model
    data = state.data.replace(qpos=reference.qpos)
    data = data.replace(qvel=jnp.zeros(mjx_model.nv))
    data = mjx.forward(mjx_model, data)

    info = dict(state.info)
    info["start_frame"] = start_frame
    info["reference_clip"] = clip_idx
    state = state.replace(data=data, info=info)
    return state


def reparameterize(rng: jax.Array, mean: jax.Array, logvar: jax.Array) -> jax.Array:
    """Sample from a Gaussian using reparameterization trick."""
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def create_prior_policy(
    policy_params: Tuple,
    action_size: int,
    proprioceptive_obs_size: int,
    intention_latent_size: int,
    decoder_hidden_layer_sizes: Tuple[int, ...],
    prior_hidden_layer_sizes: Tuple[int, ...],
    fixed_logvar: Optional[float] = -2.0,
    deterministic: bool = False,
):
    """Create a policy function that uses only prior and decoder."""
    prior_params, decoder_params, normalizer_params = extract_prior_decoder_params(
        policy_params
    )

    if isinstance(normalizer_params, DictRunningStatisticsState):
        proprio_normalizer_params = normalizer_params.proprioception
    else:
        proprio_normalizer_params = running_statistics.RunningStatisticsState(
            count=normalizer_params.count,
            mean=normalizer_params.mean[-proprioceptive_obs_size:],
            summed_variance=normalizer_params.summed_variance[
                -proprioceptive_obs_size:
            ],
            std=normalizer_params.std[-proprioceptive_obs_size:],
        )

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    prior_module = prior_networks.Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=intention_latent_size,
    )

    decoder_module = prior_networks.Decoder(
        layer_sizes=list(decoder_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
    )

    def policy_fn(obs: jax.Array, rng_key: jax.Array) -> Tuple[jax.Array, dict]:
        key_sample, key_action = random.split(rng_key)
        normalized_obs = running_statistics.normalize(obs, proprio_normalizer_params)

        prior_mean, prior_logvar = prior_module.apply(
            {"params": prior_params}, normalized_obs
        )

        if fixed_logvar is not None:
            logvar_for_sampling = jnp.full_like(prior_mean, fixed_logvar)
        else:
            logvar_for_sampling = prior_logvar

        if deterministic:
            z = prior_mean
        else:
            z = reparameterize(key_sample, prior_mean, logvar_for_sampling)

        decoder_input = jnp.concatenate([z, normalized_obs], axis=-1)
        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)
        action = parametric_action_distribution.mode(logits)

        return action, {
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "intention": z,
        }

    return policy_fn


def create_vectorized_rollout_fn(env, policy_fn, max_steps: int, starting_state):
    """Create a vectorized rollout function."""
    jit_step = jax.jit(env.step)

    def single_rollout_fn(rng_key: jax.Array):
        state = starting_state

        def step_fn(carry, _):
            state, key, nan_terminated = carry
            key, key_action = random.split(key)

            if hasattr(state.obs, "get") or isinstance(state.obs, dict):
                flat_obs = flatten_obs_dict(state.obs)
                proprio = flat_obs["proprioception"]
            else:
                proprio = state.obs

            action, _ = policy_fn(proprio, key_action)
            next_state = jit_step(state, action)

            step_nan = check_termination_nan(next_state.data)
            new_nan_terminated = jnp.logical_or(nan_terminated, step_nan)

            return (next_state, key, new_nan_terminated), next_state

        initial_carry = (state, rng_key, jnp.array(False))
        (_, _, nan_terminated), all_states = jax.lax.scan(
            step_fn, initial_carry, None, length=max_steps
        )

        upside_down_flags = compute_world_zaxis_termination(env, all_states.data)
        any_upside_down = jnp.any(upside_down_flags)
        first_upside_down_step = jnp.argmax(upside_down_flags)

        terminated = jnp.logical_or(nan_terminated, any_upside_down)
        step_count = jnp.where(any_upside_down, first_upside_down_step + 1, max_steps)

        return step_count, terminated, all_states

    vmapped_rollout = jax.jit(jax.vmap(single_rollout_fn))
    return vmapped_rollout, jit_step


def render_rollout(
    env, states, num_steps: int, camera: str, height: int = 480, width: int = 640
):
    """Render a rollout to frames."""
    states_list = []
    for i in range(num_steps):
        state_i = jax.tree_util.tree_map(lambda x: x[i], states)
        states_list.append(state_i)
    frames = env.render(states_list, camera=camera, height=height, width=width)
    return frames


def parse_clip_configs(clip_config_strs: List[str]) -> List[Tuple[int, int]]:
    """Parse clip config strings like '0,0' into (clip_idx, start_frame) tuples."""
    configs = []
    for s in clip_config_strs:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid clip config: {s}. Expected 'clip_idx,start_frame'"
            )
        configs.append((int(parts[0]), int(parts[1])))
    return configs


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("MULTI-VARIANCE PRIOR ROLLOUT")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"Fixed logvar values: {args.fixed_logvar_values}")
    print(f"Include deterministic: {args.include_deterministic}")
    print(f"Include predicted logvar: {args.include_predicted_logvar}")
    print(f"Include neutral pose: {args.include_neutral}")
    print(f"Clip configs: {args.clip_configs}")
    print(f"Render videos: {not args.no_render}")
    print("=" * 70)

    # Load checkpoint
    cfg, policy_params, ckpt_step = load_checkpoint(
        args.checkpoint_path, args.checkpoint_step
    )

    # Fix paths
    cfg = fix_paths(cfg, args.old_path_prefix, args.new_path_prefix)

    # Create environment
    env = create_environment(cfg)

    # Get proprioceptive observation size
    if "obs_sizes" in cfg.network_config:
        proprio_obs_size = cfg.network_config.obs_sizes["proprioception"]
    elif "proprioceptive_obs_size" in cfg.network_config:
        proprio_obs_size = cfg.network_config.proprioceptive_obs_size
    else:
        reference_obs_size = cfg.network_config.get("reference_obs_size", 0)
        proprio_obs_size = cfg.network_config.observation_size - reference_obs_size

    print(f"Environment created successfully")
    print(f"  Action size: {cfg.network_config.action_size}")
    print(f"  Proprioceptive obs size: {proprio_obs_size}")

    # Build variance configurations
    variance_configs = []

    if args.include_deterministic:
        variance_configs.append(
            {"name": "deterministic", "deterministic": True, "fixed_logvar": None}
        )

    if args.include_predicted_logvar:
        variance_configs.append(
            {"name": "predicted_logvar", "deterministic": False, "fixed_logvar": None}
        )

    for logvar in args.fixed_logvar_values:
        variance_configs.append(
            {
                "name": f"logvar_{logvar:.1f}",
                "deterministic": False,
                "fixed_logvar": logvar,
            }
        )

    # Build starting pose configurations
    starting_pose_configs = []

    if args.include_neutral:
        starting_pose_configs.append(
            {
                "name": "neutral",
                "mode": "neutral",
                "clip_idx": None,
                "start_frame": None,
            }
        )

    clip_configs = parse_clip_configs(args.clip_configs)
    for clip_idx, start_frame in clip_configs:
        starting_pose_configs.append(
            {
                "name": f"clip{clip_idx}_frame{start_frame}",
                "mode": "clip",
                "clip_idx": clip_idx,
                "start_frame": start_frame,
            }
        )

    print(f"\nVariance configurations ({len(variance_configs)}):")
    for vc in variance_configs:
        print(f"  - {vc['name']}")

    print(f"\nStarting pose configurations ({len(starting_pose_configs)}):")
    for sp in starting_pose_configs:
        print(f"  - {sp['name']}")

    total_configs = len(variance_configs) * len(starting_pose_configs)
    print(f"\nTotal rollout combinations: {total_configs}")

    # Get network config parameters
    action_size = cfg.network_config.action_size
    intention_size = cfg.network_config.intention_size
    decoder_layers = tuple(cfg.network_config.decoder_layer_sizes)
    prior_layers = tuple(
        cfg.network_config.get(
            "prior_layer_sizes", cfg.network_config.encoder_layer_sizes
        )
    )

    # Initialize RNG
    rng = random.PRNGKey(args.seed)

    # Storage for all results
    all_results = {}

    # Prepare H5 file
    h5_path = output_dir / "rollout_data.h5"

    total_start_time = time.time()
    config_count = 0

    with h5py.File(h5_path, "w") as h5f:
        # Save metadata
        meta_grp = h5f.create_group("metadata")
        meta_grp.attrs["checkpoint_path"] = args.checkpoint_path
        meta_grp.attrs["checkpoint_step"] = ckpt_step
        meta_grp.attrs["max_steps"] = args.max_steps
        meta_grp.attrs["seed"] = args.seed
        meta_grp.attrs["timestamp"] = datetime.now().isoformat()
        meta_grp.attrs["num_rollouts_per_config"] = args.num_rollouts_per_config

        # Loop through all variance configurations
        for var_cfg in variance_configs:
            var_name = var_cfg["name"]
            all_results[var_name] = {}

            print(f"\n{'=' * 60}")
            print(f"Variance: {var_name}")
            print(f"{'=' * 60}")

            # Create policy for this variance configuration
            policy = create_prior_policy(
                policy_params=policy_params,
                action_size=action_size,
                proprioceptive_obs_size=proprio_obs_size,
                intention_latent_size=intention_size,
                decoder_hidden_layer_sizes=decoder_layers,
                prior_hidden_layer_sizes=prior_layers,
                fixed_logvar=var_cfg["fixed_logvar"],
                deterministic=var_cfg["deterministic"],
            )

            # Create H5 group for this variance setting
            var_grp = h5f.create_group(var_name)
            var_grp.attrs["deterministic"] = var_cfg["deterministic"]
            if var_cfg["fixed_logvar"] is not None:
                var_grp.attrs["fixed_logvar"] = var_cfg["fixed_logvar"]
            else:
                var_grp.attrs["fixed_logvar"] = (
                    "predicted" if not var_cfg["deterministic"] else "N/A"
                )

            # Loop through all starting poses
            for pose_cfg in starting_pose_configs:
                pose_name = pose_cfg["name"]
                config_count += 1

                print(
                    f"\n  [{config_count}/{total_configs}] Starting pose: {pose_name}"
                )

                # Split RNG
                rng, rng_init, rng_rollout = random.split(rng, 3)

                # Create starting state
                if pose_cfg["mode"] == "neutral":
                    starting_state = create_neutral_state(env, rng_init)
                else:
                    starting_state = create_clip_state(
                        env, pose_cfg["clip_idx"], pose_cfg["start_frame"], rng_init
                    )

                # Create vectorized rollout function
                vmapped_rollout, _ = create_vectorized_rollout_fn(
                    env=env,
                    policy_fn=policy,
                    max_steps=args.max_steps,
                    starting_state=starting_state,
                )

                # Determine number of rollouts (1 for deterministic, N for stochastic)
                num_rollouts = (
                    1 if var_cfg["deterministic"] else args.num_rollouts_per_config
                )

                # Generate rollout keys
                rollout_keys = random.split(rng_rollout, num_rollouts)

                # Run rollouts
                t_start = time.time()
                step_counts, terminated_flags, all_states = vmapped_rollout(
                    rollout_keys
                )
                step_counts = jnp.array(step_counts)
                step_counts.block_until_ready()
                t_elapsed = time.time() - t_start

                print(f"    Rollouts: {num_rollouts}, Time: {t_elapsed:.2f}s")
                print(
                    f"    Steps: {int(step_counts[0])} (terminated: {bool(terminated_flags[0])})"
                )

                # Create H5 subgroup for this pose
                pose_grp = var_grp.create_group(pose_name)
                pose_grp.attrs["mode"] = pose_cfg["mode"]
                pose_grp.attrs["num_rollouts"] = num_rollouts
                if pose_cfg["clip_idx"] is not None:
                    pose_grp.attrs["clip_idx"] = pose_cfg["clip_idx"]
                    pose_grp.attrs["start_frame"] = pose_cfg["start_frame"]

                # Save rollout data to H5
                # Shape: (num_rollouts, max_steps, dim)
                qpos_data = np.array(all_states.data.qpos)
                qvel_data = np.array(all_states.data.qvel)
                step_counts_np = np.array(step_counts)
                terminated_np = np.array(terminated_flags)

                pose_grp.create_dataset("qpos", data=qpos_data, compression="gzip")
                pose_grp.create_dataset("qvel", data=qvel_data, compression="gzip")
                pose_grp.create_dataset("step_counts", data=step_counts_np)
                pose_grp.create_dataset("terminated", data=terminated_np)

                # Render and save videos (if enabled)
                # Determine how many rollouts to render
                if args.no_render or args.num_rollouts_to_render == 0:
                    num_to_render = 0
                elif args.num_rollouts_to_render < 0:
                    num_to_render = num_rollouts  # Render all
                else:
                    num_to_render = min(args.num_rollouts_to_render, num_rollouts)

                for rollout_idx in range(num_to_render):
                    rollout_states = jax.tree_util.tree_map(
                        lambda x: x[rollout_idx], all_states
                    )
                    frames = render_rollout(
                        env=env,
                        states=rollout_states,
                        num_steps=args.max_steps,
                        camera=args.camera_name,
                        height=args.render_height,
                        width=args.render_width,
                    )

                    video_filename = (
                        f"{var_name}_{pose_name}_rollout{rollout_idx:02d}.mp4"
                    )
                    video_path = output_dir / video_filename
                    with imageio.get_writer(
                        str(video_path), fps=args.render_fps
                    ) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    print(f"    Saved: {video_filename}")

                # Store results summary
                all_results[var_name][pose_name] = {
                    "step_counts": step_counts_np.tolist(),
                    "terminated": terminated_np.tolist(),
                }

    total_elapsed = time.time() - total_start_time

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} minutes)")
    print(f"H5 data saved to: {h5_path}")
    if not args.no_render:
        print(f"Videos saved to: {output_dir}")

    print(f"\n{'=' * 70}")
    print("RESULTS TABLE")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Variance':<20}"
    for pose_cfg in starting_pose_configs:
        header += f" {pose_cfg['name']:<15}"
    print(header)
    print("-" * 70)

    # Data rows
    for var_name in all_results:
        row = f"{var_name:<20}"
        for pose_cfg in starting_pose_configs:
            pose_name = pose_cfg["name"]
            steps = all_results[var_name][pose_name]["step_counts"]
            avg_steps = np.mean(steps)
            row += f" {avg_steps:<15.0f}"
        print(row)

    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
