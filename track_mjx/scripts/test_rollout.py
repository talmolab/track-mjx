"""Test script for rollout.py with registry-based environment creation.

Loads a checkpoint, creates an environment via rollout.create_environment,
and generates a single rollout to verify the pipeline works end-to-end.

Usage:
    python track_mjx/scripts/test_rollout.py --checkpoint 260131_223134_344901
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse

import jax
from omegaconf import OmegaConf

from track_mjx.agent import checkpointing
from track_mjx.analysis import rollout
from track_mjx.config import utils


def main():
    parser = argparse.ArgumentParser(description="Test rollout.py pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="260131_223134_344901",
        help="Checkpoint run ID or path",
    )
    parser.add_argument("--clip_idx", type=int, default=0, help="Reference clip index")
    args = parser.parse_args()

    # Resolve checkpoint path
    if os.path.isabs(args.checkpoint):
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.abspath(f"./model_checkpoints/{args.checkpoint}")
    print(f"Checkpoint path: {ckpt_path}")

    # Load config from checkpoint
    cfg_dict = checkpointing.load_config_from_checkpoint(ckpt_path)
    cfg = OmegaConf.create(cfg_dict)

    # Prepare config (resolves paths, creates env_cfg_ml)
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)
    print(f"Env name: {cfg.env_config.env_name}")
    print(f"Reference data: {cfg.env_config.reference_data_path}")
    print(f"Clip length: {cfg.env_config.clip_length}")

    # Create environment using updated rollout.create_environment
    print("Creating environment...")
    env = rollout.create_environment(cfg)
    print(f"Environment created: {type(env).__name__}")

    # Verify observation structure
    state = env.reset(jax.random.PRNGKey(0))
    print(f"Observation keys: {list(state.obs.keys())}")
    for key in state.obs:
        inner = state.obs[key]
        if isinstance(inner, dict):
            leaf_shapes = jax.tree.map(lambda x: x.shape, inner)
            print(f"  {key}: {leaf_shapes}")
        else:
            print(f"  {key}: {inner.shape}")

    # Load policy
    print("Loading policy...")
    ckpt = checkpointing.load_policy(ckpt_path, cfg)
    inference_fn = checkpointing.load_inference_fn(cfg, ckpt)
    print("Policy loaded")

    # Create rollout generator and generate a rollout
    print(f"Generating rollout for clip_idx={args.clip_idx}...")
    generate_rollout = rollout.create_rollout_generator(
        cfg, env, inference_fn, log_metrics=True
    )
    result = generate_rollout(clip_idx=args.clip_idx, seed=42)

    # Print output shapes
    print("\nRollout output shapes:")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v.shape}")
        else:
            print(f"  {key}: {value.shape}")

    print("\nTest passed!")


if __name__ == "__main__":
    main()
