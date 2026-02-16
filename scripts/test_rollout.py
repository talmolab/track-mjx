"""Test script for rollout.py with both feedforward and recurrent architectures.

Loads checkpoints, creates environments, and generates rollouts to verify the
pipeline works end-to-end. Tests single rollouts, vmapped rollouts, and
activation logging for both architectures.

Usage:
    # Test a single checkpoint (auto-detects architecture)
    python scripts/test_rollout.py --checkpoint 260131_223134_344901

    # Test multiple checkpoints
    python scripts/test_rollout.py \
        --checkpoint 260212_184836_101120 \
        --checkpoint 260210_013247_287627
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from track_mjx.agent import checkpointing
from track_mjx.analysis import rollout
from track_mjx.config import utils


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve a checkpoint ID or path to an absolute path."""
    if os.path.isabs(checkpoint):
        return checkpoint
    return os.path.abspath(f"./model_checkpoints/{checkpoint}")


def print_tree_shapes(tree, prefix="", max_depth=2, _depth=0):
    """Print shapes of all arrays in a pytree."""
    if _depth >= max_depth:
        leaves = jax.tree_util.tree_leaves(tree)
        if leaves:
            print(f"{prefix}({len(leaves)} leaves)")
        return

    if isinstance(tree, dict):
        for k, v in tree.items():
            if hasattr(v, "shape"):
                print(f"{prefix}{k}: {v.shape} ({v.dtype})")
            elif isinstance(v, (dict, list, tuple)):
                print(f"{prefix}{k}:")
                print_tree_shapes(v, prefix + "  ", max_depth, _depth + 1)
            else:
                print(f"{prefix}{k}: {type(v).__name__}")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            if hasattr(v, "shape"):
                print(f"{prefix}[{i}]: {v.shape} ({v.dtype})")
            elif isinstance(v, (dict, list, tuple)):
                print(f"{prefix}[{i}]:")
                print_tree_shapes(v, prefix + "  ", max_depth, _depth + 1)
            else:
                print(f"{prefix}[{i}]: {type(v).__name__}")
    elif hasattr(tree, "shape"):
        print(f"{prefix}{tree.shape} ({tree.dtype})")
    else:
        print(f"{prefix}{type(tree).__name__}")


def test_checkpoint(ckpt_id: str) -> bool:
    """Test rollout pipeline for a single checkpoint.

    Returns True if all tests pass, False otherwise.
    """
    ckpt_path = resolve_checkpoint_path(ckpt_id)
    print(f"\n{'='*60}")
    print(f"Testing checkpoint: {ckpt_id}")
    print(f"Path: {ckpt_path}")
    print(f"{'='*60}")

    # Load config
    cfg_dict = checkpointing.load_config_from_checkpoint(ckpt_path)
    cfg = OmegaConf.create(cfg_dict)
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    arch_name = cfg.network_config.get("arch_name", "intention")
    is_recurrent = arch_name == "recurrent_intention"
    print(f"Architecture: {arch_name}")
    print(f"Env name: {cfg.env_config.env_name}")

    # Create environment
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

    # Load policy
    print("Loading policy...")
    ckpt = checkpointing.load_policy(ckpt_path, cfg)
    inference_fn = checkpointing.load_inference_fn(
        cfg, ckpt, deterministic=True, get_activation=True
    )

    # Load init_hidden_fn if recurrent
    init_hidden_fn = checkpointing.load_init_hidden_fn(cfg)
    if is_recurrent:
        assert init_hidden_fn is not None, "Expected init_hidden_fn for recurrent arch"
        print(f"RNN type: {cfg.network_config.rnn_type}")
        print(f"RNN hidden sizes: {list(cfg.network_config.rnn_hidden_sizes)}")
    else:
        assert init_hidden_fn is None, "Expected no init_hidden_fn for FF arch"
    print("Policy loaded")

    # =========================================================================
    # Test 1: Single rollout with activations and metrics
    # =========================================================================
    print("\n--- Test 1: Single rollout (activations + metrics) ---")
    generate_rollout = rollout.create_rollout_generator(
        cfg,
        env,
        inference_fn,
        init_hidden_fn=init_hidden_fn,
        log_activations=True,
        log_metrics=True,
    )
    result = generate_rollout(clip_idx=0, seed=42)

    print("\nRollout output keys:")
    for key, value in result.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: (dict)")
        elif isinstance(value, list):
            print(f"  {key}: (list, len={len(value)})")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Validate core fields
    assert "qposes_ref" in result, "Missing qposes_ref"
    assert "qposes_rollout" in result, "Missing qposes_rollout"
    assert "ctrl" in result, "Missing ctrl"
    assert "state_rewards" in result, "Missing state_rewards"
    assert result["qposes_ref"].shape == result["qposes_rollout"].shape
    print(f"  qposes shape: {result['qposes_rollout'].shape}")
    print(f"  ctrl shape: {result['ctrl'].shape}")
    print(f"  rewards shape: {result['state_rewards'].shape}")

    # Validate activations
    assert "activations" in result, "Missing activations"
    act = result["activations"]
    assert "encoder" in act, "Missing encoder in activations"
    assert "decoder" in act, "Missing decoder in activations"
    assert "egocentric_obs" in act, "Missing egocentric_obs in activations"
    assert "traj_obs" in act, "Missing traj_obs in activations"
    assert "intention" in act, "Missing intention in activations"
    print("\nActivation structure:")
    print_tree_shapes(act, prefix="  ")

    # Validate recurrent-specific fields
    if is_recurrent:
        assert "hidden_states" in act, "Missing hidden_states in activations"
        assert "final_hidden" in result, "Missing final_hidden in result"
        print("\nFinal hidden structure:")
        print_tree_shapes(result["final_hidden"], prefix="  ")

    # Validate metrics
    assert "rollout_metrics" in result, "Missing rollout_metrics"
    print(f"\nMetrics keys: {list(result['rollout_metrics'].keys())}")

    # Check for NaN
    nan_count = 0
    total_leaves = 0
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            nan_count += int(jnp.any(jnp.isnan(leaf)))
            total_leaves += 1
    print(f"\nNaN check: {nan_count}/{total_leaves} arrays contain NaN")

    # =========================================================================
    # Test 2: Vmapped rollouts
    # =========================================================================
    print("\n--- Test 2: Vmapped rollouts ---")
    clip_indices = jnp.arange(3)
    vmap_rollout = jax.jit(jax.vmap(generate_rollout))
    batch_result = vmap_rollout(clip_indices)

    print("Batched output shapes:")
    for key, value in batch_result.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            first_leaf = jax.tree_util.tree_leaves(value)[0]
            if hasattr(first_leaf, "shape"):
                print(f"  {key}: (dict, first leaf: {first_leaf.shape})")

    # Validate batched shapes
    assert batch_result["qposes_rollout"].shape[0] == len(clip_indices)
    assert batch_result["ctrl"].shape[0] == len(clip_indices)
    print(f"  Batch dimension correct: {len(clip_indices)}")

    if is_recurrent:
        assert "final_hidden" in batch_result, "Missing final_hidden in batched result"

    print(f"\nAll tests PASSED for {ckpt_id} ({arch_name})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test rollout.py pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        help="Checkpoint run ID or path (can specify multiple times)",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = ["260131_223134_344901"]

    all_passed = True
    for ckpt in args.checkpoint:
        try:
            passed = test_checkpoint(ckpt)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\nFAILED for {ckpt}: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    if all_passed:
        print(f"\n{'='*60}")
        print(f"ALL TESTS PASSED ({len(args.checkpoint)} checkpoints)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("SOME TESTS FAILED")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
