"""Rollout script for VAE intention network checkpoints.

Loads a checkpoint, runs rollouts on all clips, and saves to HDF5.

Because of the checkpoint naming, change naconmax=self._config.naconmax to naconmax=self._config.nconmax

Usage:
    # Rollout all clips and save to HDF5
    python rollout.py --checkpoint model_checkpoints/260114_020711_262993 --all-clips

    # Rollout single clip
    python rollout.py --checkpoint model_checkpoints/260114_020711_262993 --clip 0

    # With video rendering
    python rollout.py --clip 0 --render
"""

import os

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

import argparse
import logging
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from track_mjx.agent.checkpointing import load_checkpoint_for_eval, load_inference_fn
from track_mjx.agent.observation_utils import flatten_obs_dict
from track_mjx.analysis.rollout import create_environment

logging.basicConfig(level=logging.INFO)


def remap_config_paths(cfg, walker_xml: str | None, arena_xml: str | None, reference_data: str | None):
    """Remap paths in config that may be from a different system.

    Automatically detects vnl_playground package location and remaps paths
    from other systems (e.g., Harvard cluster) to local paths.

    Args:
        cfg: Configuration dict to modify in place.
        walker_xml: Override path for walker XML.
        arena_xml: Override path for arena XML.
        reference_data: Override path for reference clips H5.
    """
    env_cfg = cfg.env_config

    # Get vnl_playground package location for auto-remapping
    vnl_pkg_dir = None
    try:
        import vnl_playground
        vnl_pkg_dir = Path(vnl_playground.__file__).parent
        logging.info(f"Found vnl_playground at: {vnl_pkg_dir}")
    except ImportError:
        logging.warning("vnl_playground not found, cannot auto-remap paths")

    # Check for paths from another system (Harvard cluster)
    remote_prefixes = ["/n/holylfs06", "/n/holylabs"]

    def is_remote_path(p: str) -> bool:
        return any(p.startswith(prefix) for prefix in remote_prefixes)

    def auto_remap(key: str, current_path: str) -> str | None:
        """Try to auto-remap a remote path to local."""
        if not is_remote_path(current_path):
            return None

        # Extract relative path from vnl-playground
        if "vnl-playground/" in current_path or "vnl_playground/" in current_path:
            if vnl_pkg_dir is None:
                return None
            # Extract path after vnl_playground/
            for marker in ["vnl-playground/vnl_playground/", "vnl_playground/"]:
                if marker in current_path:
                    rel_path = current_path.split(marker)[-1]
                    local_path = vnl_pkg_dir / rel_path
                    if local_path.exists():
                        return str(local_path)
                    break

        # Extract relative path from track-mjx for reference data
        if "track-mjx/" in current_path:
            rel_path = current_path.split("track-mjx/")[-1]
            local_path = Path(__file__).parent / rel_path
            if local_path.exists():
                return str(local_path)

        return None

    paths_to_check = [
        ("walker_xml_path", walker_xml),
        ("arena_xml_path", arena_xml),
        ("reference_data_path", reference_data),
    ]

    for key, override in paths_to_check:
        current_path = env_cfg.get(key, "")
        if override:
            env_cfg[key] = override
            logging.info(f"Overriding {key}: {override}")
        elif is_remote_path(current_path):
            # Try auto-remap
            local_path = auto_remap(key, current_path)
            if local_path:
                env_cfg[key] = local_path
                logging.info(f"Auto-remapped {key}: {local_path}")
            else:
                logging.error(
                    f"Config contains path from another system: {key}={current_path}\n"
                    f"Please provide --{key.replace('_', '-')} to override."
                )
                raise ValueError(f"Cannot access path: {current_path}")


def run_single_rollout(
    env,
    jit_inference_fn,
    jit_reset,
    jit_step,
    clip_idx: int,
    seed: int,
    num_steps: int,
) -> dict:
    """Run a single rollout for one clip.

    Args:
        env: Environment instance.
        jit_inference_fn: JIT-compiled inference function.
        jit_reset: JIT-compiled reset function.
        jit_step: JIT-compiled step function.
        clip_idx: Clip index to rollout.
        seed: Random seed.
        num_steps: Number of steps to run.

    Returns:
        Dictionary with rollout data for this clip.
    """
    # Initialize
    rollout_key = jax.random.PRNGKey(seed + clip_idx)
    rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

    state = jit_reset(reset_rng, clip_idx=clip_idx, start_frame=0)

    # Storage
    qposes = [np.array(state.data.qpos)]
    intentions = []
    ctrls = []
    obs_list = []

    for step in range(num_steps):
        _, act_rng = jax.random.split(act_rng)

        # Store observation (flatten nested dict obs to single array)
        flat_obs = flatten_obs_dict(state.obs)
        obs_flat = np.concatenate([
            np.array(flat_obs["task_obs"]),
            np.array(flat_obs["proprioception"]),
        ])
        obs_list.append(obs_flat)

        # Run inference
        action, extras = jit_inference_fn(state.obs, act_rng)

        # Extract latent intention
        if "latent_mean" in extras:
            intentions.append(np.array(extras["latent_mean"]))

        ctrls.append(np.array(action))

        # Step environment
        state = jit_step(state, action)
        qposes.append(np.array(state.data.qpos))

    return {
        "qpos": np.stack(qposes),          # [T+1, qpos_dim]
        "intention": np.stack(intentions), # [T, latent_dim]
        "ctrl": np.stack(ctrls),           # [T, action_dim]
        "obs": np.stack(obs_list),         # [T, obs_dim]
    }


def run_all_clips_to_h5(
    checkpoint_path: str,
    output_path: str,
    seed: int = 42,
    walker_xml: str | None = None,
    arena_xml: str | None = None,
    reference_data: str | None = None,
) -> None:
    """Run rollouts on all clips and save to HDF5.

    Output structure:
        /intention  (num_clips, T, latent_dim)
        /qpos       (num_clips, T+1, qpos_dim)
        /ctrl       (num_clips, T, action_dim)
        /obs        (num_clips, T, obs_dim)

    Args:
        checkpoint_path: Path to checkpoint directory.
        output_path: Path to output HDF5 file.
        seed: Base random seed.
        walker_xml: Override path for walker XML.
        arena_xml: Override path for arena XML.
        reference_data: Override path for reference clips H5.
    """
    # Convert to absolute path (Orbax requires absolute paths)
    checkpoint_path = str(Path(checkpoint_path).resolve())

    # 1. Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = load_checkpoint_for_eval(checkpoint_path)
    cfg = ckpt["cfg"]
    policy = ckpt["policy"]

    # Remap paths if needed
    remap_config_paths(cfg, walker_xml, arena_xml, reference_data)

    arch_name = cfg.network_config.get("arch_name", "intention")
    intention_size = cfg.network_config.intention_size
    logging.info(f"Architecture: {arch_name}")
    logging.info(f"Intention size: {intention_size}")

    # 2. Create inference function
    inference_fn = load_inference_fn(
        cfg,
        policy,
        deterministic=True,
        get_activation=True,
    )

    # 3. Create environment
    logging.info("Creating environment...")
    env = create_environment(cfg)

    # Get number of clips (hardcoded for rodent dataset)
    num_clips = 842
    logging.info(f"Number of clips: {num_clips}")

    # Calculate steps per clip
    mocap_dt = 1.0 / cfg.env_config.mocap_hz
    steps_per_frame = int(mocap_dt / cfg.env_config.ctrl_dt)
    num_steps = cfg.env_config.clip_length * steps_per_frame - 1
    logging.info(f"Steps per clip: {num_steps}")

    # JIT compile functions
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Run first clip to get dimensions
    logging.info("Running first clip to determine dimensions...")
    first_result = run_single_rollout(
        env, jit_inference_fn, jit_reset, jit_step,
        clip_idx=0, seed=seed, num_steps=num_steps,
    )

    qpos_dim = first_result["qpos"].shape[1]
    latent_dim = first_result["intention"].shape[1]
    ctrl_dim = first_result["ctrl"].shape[1]
    obs_dim = first_result["obs"].shape[1]

    logging.info(f"Dimensions: qpos={qpos_dim}, latent={latent_dim}, ctrl={ctrl_dim}, obs={obs_dim}")

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create HDF5 file and datasets
    logging.info(f"Creating HDF5 file: {output_path}")
    with h5py.File(output_path, "w") as f:
        # Create datasets with final shapes
        ds_intention = f.create_dataset(
            "intention",
            shape=(num_clips, num_steps, latent_dim),
            dtype=np.float32,
        )
        ds_qpos = f.create_dataset(
            "qpos",
            shape=(num_clips, num_steps + 1, qpos_dim),
            dtype=np.float32,
        )
        ds_ctrl = f.create_dataset(
            "ctrl",
            shape=(num_clips, num_steps, ctrl_dim),
            dtype=np.float32,
        )
        ds_obs = f.create_dataset(
            "obs",
            shape=(num_clips, num_steps, obs_dim),
            dtype=np.float32,
        )

        # Store first clip
        ds_intention[0] = first_result["intention"]
        ds_qpos[0] = first_result["qpos"]
        ds_ctrl[0] = first_result["ctrl"]
        ds_obs[0] = first_result["obs"]

        # Run remaining clips
        logging.info(f"Running rollouts for {num_clips} clips...")
        for clip_idx in tqdm(range(1, num_clips), desc="Rollouts"):
            result = run_single_rollout(
                env, jit_inference_fn, jit_reset, jit_step,
                clip_idx=clip_idx, seed=seed, num_steps=num_steps,
            )

            ds_intention[clip_idx] = result["intention"]
            ds_qpos[clip_idx] = result["qpos"]
            ds_ctrl[clip_idx] = result["ctrl"]
            ds_obs[clip_idx] = result["obs"]

        # Store metadata as attributes
        f.attrs["checkpoint_path"] = checkpoint_path
        f.attrs["num_clips"] = num_clips
        f.attrs["num_steps"] = num_steps
        f.attrs["seed"] = seed
        f.attrs["intention_size"] = latent_dim
        f.attrs["qpos_dim"] = qpos_dim
        f.attrs["ctrl_dim"] = ctrl_dim
        f.attrs["obs_dim"] = obs_dim

    logging.info(f"\nSaved to {output_path}")
    logging.info(f"  /intention  {(num_clips, num_steps, latent_dim)}")
    logging.info(f"  /qpos       {(num_clips, num_steps + 1, qpos_dim)}")
    logging.info(f"  /ctrl       {(num_clips, num_steps, ctrl_dim)}")
    logging.info(f"  /obs        {(num_clips, num_steps, obs_dim)}")


def run_single_clip(
    checkpoint_path: str,
    clip_idx: int = 0,
    seed: int = 42,
    output_dir: str = "outputs/rollouts",
    render: bool = False,
    camera: str = "close_profile-rodent",
    walker_xml: str | None = None,
    arena_xml: str | None = None,
    reference_data: str | None = None,
) -> dict:
    """Run rollout for a single clip and save to npz.

    Args:
        checkpoint_path: Path to checkpoint directory.
        clip_idx: Reference clip index to track.
        seed: Random seed.
        output_dir: Directory to save outputs.
        render: Whether to render video.
        camera: Camera name for rendering.
        walker_xml: Override path for walker XML.
        arena_xml: Override path for arena XML.
        reference_data: Override path for reference clips H5.

    Returns:
        Dictionary with rollout data.
    """
    # Convert to absolute path (Orbax requires absolute paths)
    checkpoint_path = str(Path(checkpoint_path).resolve())

    # 1. Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = load_checkpoint_for_eval(checkpoint_path)
    cfg = ckpt["cfg"]
    policy = ckpt["policy"]

    # Remap paths if needed
    remap_config_paths(cfg, walker_xml, arena_xml, reference_data)

    logging.info(f"Architecture: {cfg.network_config.get('arch_name', 'intention')}")
    logging.info(f"Intention size: {cfg.network_config.intention_size}")

    # 2. Create inference function
    inference_fn = load_inference_fn(
        cfg,
        policy,
        deterministic=True,
        get_activation=True,
    )

    # 3. Create environment
    logging.info("Creating environment...")
    env = create_environment(cfg)

    # Calculate steps
    mocap_dt = 1.0 / cfg.env_config.mocap_hz
    steps_per_frame = int(mocap_dt / cfg.env_config.ctrl_dt)
    num_steps = cfg.env_config.clip_length * steps_per_frame - 1

    logging.info(f"Running rollout on clip {clip_idx} with seed {seed}...")
    logging.info(f"Steps: {num_steps}")

    # JIT compile
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Run rollout (need states for rendering)
    rollout_key = jax.random.PRNGKey(seed)
    rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

    state = jit_reset(reset_rng, clip_idx=clip_idx, start_frame=0)

    rollout_states = [state]
    qposes = [np.array(state.data.qpos)]
    intentions = []
    ctrls = []
    rewards = []

    for step in range(num_steps):
        _, act_rng = jax.random.split(act_rng)

        action, extras = jit_inference_fn(state.obs, act_rng)

        if "latent_mean" in extras:
            intentions.append(np.array(extras["latent_mean"]))

        ctrls.append(np.array(action))

        state = jit_step(state, action)
        rollout_states.append(state)
        qposes.append(np.array(state.data.qpos))
        rewards.append(float(state.reward))

        if step % 50 == 0:
            logging.info(f"  Step {step}/{num_steps}, reward: {state.reward:.4f}")

    # Compile results
    results = {
        "qpos": np.stack(qposes),
        "intention": np.stack(intentions) if intentions else None,
        "ctrl": np.stack(ctrls),
        "rewards": np.array(rewards),
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(checkpoint_path).name
    save_path = output_path / f"rollout_{checkpoint_name}_clip{clip_idx}_seed{seed}.npz"

    np.savez(save_path, **{k: v for k, v in results.items() if v is not None})
    logging.info(f"Saved to {save_path}")

    # Summary
    logging.info("\n=== Rollout Summary ===")
    logging.info(f"Total reward: {sum(rewards):.4f}")
    logging.info(f"Qpos shape: {results['qpos'].shape}")
    if results["intention"] is not None:
        logging.info(f"Intention shape: {results['intention'].shape}")

    # Render
    if render:
        logging.info("Rendering video...")
        try:
            import imageio

            frames = env.render(rollout_states, camera=camera)
            video_path = output_path / f"rollout_{checkpoint_name}_clip{clip_idx}_seed{seed}.mp4"
            with imageio.get_writer(str(video_path), fps=50) as writer:
                for frame in frames:
                    writer.append_data(frame)
            logging.info(f"Saved video to {video_path}")
        except Exception as e:
            logging.warning(f"Failed to render video: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run rollout with VAE checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_checkpoints/260114_020711_262993",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Rollout all clips and save to HDF5",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=0,
        help="Clip index to rollout (ignored if --all-clips)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (for --all-clips: .h5 file, otherwise: directory)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render video (only for single clip mode)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="close_profile-rodent",
        help="Camera for rendering (e.g., close_profile-rodent, close_profile-ghost)",
    )
    parser.add_argument(
        "--walker-xml-path",
        type=str,
        default=None,
        help="Override path for walker XML (e.g., vnl_playground/tasks/rodent/xmls/rodent.xml)",
    )
    parser.add_argument(
        "--arena-xml-path",
        type=str,
        default=None,
        help="Override path for arena XML (e.g., vnl_playground/tasks/rodent/xmls/arena.xml)",
    )
    parser.add_argument(
        "--reference-data-path",
        type=str,
        default=None,
        help="Override path for reference clips H5 (e.g., data/rodent/rodent_reference_clips.h5)",
    )

    args = parser.parse_args()

    if args.all_clips:
        # Rollout all clips to HDF5
        output_path = args.output or f"outputs/rollouts/rollout_{Path(args.checkpoint).name}.h5"
        run_all_clips_to_h5(
            checkpoint_path=args.checkpoint,
            output_path=output_path,
            seed=args.seed,
            walker_xml=args.walker_xml_path,
            arena_xml=args.arena_xml_path,
            reference_data=args.reference_data_path,
        )
    else:
        # Single clip rollout
        output_dir = args.output or "outputs/rollouts"
        run_single_clip(
            checkpoint_path=args.checkpoint,
            clip_idx=args.clip,
            seed=args.seed,
            output_dir=output_dir,
            render=args.render,
            camera=args.camera,
            walker_xml=args.walker_xml_path,
            arena_xml=args.arena_xml_path,
            reference_data=args.reference_data_path,
        )


if __name__ == "__main__":
    main()
