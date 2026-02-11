#!/usr/bin/env python3
"""End-to-end validation of the vision PPO training pipeline.

This script validates the full pipeline works without running a full training
job. It tests:
  1. Environment creation (RunGapVision with small config)
  2. VisionRenderer initialization on GPU via mujoco_warp
  3. PPO network construction (CNN encoder + MLP decoder + value network)
  4. generate_unroll_with_vision producing correct shapes and non-zero images
  5. A JIT'd SGD step through the full loss/gradient pipeline

Usage:
    cd /home/talmolab/Desktop/SalkResearch/track-mjx
    python scripts/test_vision_training.py
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
from brax.training import types
from ml_collections import config_dict

from mujoco_playground import wrapper as playground_wrappers
from vnl_playground import registry

# Work around mujoco_warp version detection bug: if the installed mujoco
# reports as a dev build but doesn't have flexedge_J_rownnz on MjModel,
# force the fallback (non-bleeding-edge) code path.  We must patch BEFORE
# VisionRenderer is imported because it calls put_model() at init time.
if not hasattr(mujoco.MjModel, "flexedge_J_rownnz"):
    # Patch the standalone mujoco_warp if it's on sys.path
    try:
        import mujoco_warp._src.io as _mjw_io
        if getattr(_mjw_io, "BLEEDING_EDGE_MUJOCO", False):
            _mjw_io.BLEEDING_EDGE_MUJOCO = False
            print("[patch] Set mujoco_warp BLEEDING_EDGE_MUJOCO = False "
                  "(flexedge_J_rownnz not on MjModel)")
    except ImportError:
        pass

    # Also try patching after vision.py's _import_mujoco_warp() runs by
    # deferring via a module-level hook isn't feasible, so we set the
    # MUJOCO_WARP_PATH env variable to ensure the standalone version loads
    # and we can pre-patch it.
    mjw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "mujoco_warp",
    )
    if os.path.isdir(mjw_path):
        os.environ.setdefault("MUJOCO_WARP_PATH", mjw_path)
        # Evict any cached vendored module and reimport the standalone one
        sys.modules.pop("mujoco_warp", None)
        sys.modules.pop("mujoco_warp._src", None)
        sys.modules.pop("mujoco_warp._src.io", None)
        if mjw_path not in sys.path:
            sys.path.insert(0, mjw_path)
        import mujoco_warp._src.io as _mjw_io
        _mjw_io.BLEEDING_EDGE_MUJOCO = False

from vnl_playground.tasks.rodent.vision import VisionRenderer

from track_mjx.agent.ff_ppo import ppo_networks as ff_networks
from track_mjx.agent.ff_ppo import vision_ppo
from track_mjx.agent.ff_ppo import losses
from track_mjx.agent import gradients
from track_mjx.agent.observation_utils import (
    get_obs_sizes,
    init_dict_normalizer,
)

# ---------------------------------------------------------------------------
# Test parameters (small for quick validation)
# ---------------------------------------------------------------------------
NUM_ENVS = 4
VISION_WIDTH = 32
VISION_HEIGHT = 32
GRAYSCALE = True
UNROLL_LENGTH = 5
CAMERA_NAME = "egocentric-rodent"


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(condition: bool, message: str) -> None:
    """Assert a condition with a descriptive message."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise AssertionError(f"Validation failed: {message}")


def main():
    t0 = time.time()
    print("Vision PPO Training Pipeline -- End-to-End Validation")
    print(f"JAX devices: {jax.devices()}")

    # ------------------------------------------------------------------
    # Step 1: Create environment
    # ------------------------------------------------------------------
    section("Step 1: Create RunGapVision environment")

    # Start from the default config and override vision-specific settings
    env_cfg = registry.get_default_config("RodentRunGapVision")
    env_cfg.vision_width = VISION_WIDTH
    env_cfg.vision_height = VISION_HEIGHT
    env_cfg.grayscale = GRAYSCALE
    env_cfg.vision_camera_name = CAMERA_NAME
    env = registry.load("RodentRunGapVision", config=env_cfg, flatten_obs=False)
    print(f"  Environment created: {type(env).__name__}")
    print(f"  Action size: {env.action_size}")
    print(f"  Vision shape: {env.vision_shape}")
    check(env.action_size > 0, "Action size is positive")
    check(env.vision_shape == (VISION_HEIGHT, VISION_WIDTH, 1 if GRAYSCALE else 3),
          f"Vision shape matches config: {env.vision_shape}")

    # ------------------------------------------------------------------
    # Step 2: Wrap for training and reset
    # ------------------------------------------------------------------
    section("Step 2: Wrap environment and reset")

    wrapped_env = playground_wrappers.wrap_for_brax_training(
        env, episode_length=200, action_repeat=1, full_reset=False
    )

    key = jax.random.PRNGKey(0)
    key, key_env = jax.random.split(key)
    key_envs = jax.random.split(key_env, NUM_ENVS)
    # The brax wrappers (VmapWrapper -> EpisodeWrapper -> AutoResetWrapper)
    # already handle batching internally, so we just call reset with the
    # batch of keys.
    reset_fn = jax.jit(wrapped_env.reset)
    env_state = reset_fn(key_envs)

    print(f"  Wrapped env type: {type(wrapped_env).__name__}")
    print(f"  Observation keys: {list(env_state.obs.keys())}")
    print(f"  Reward shape: {env_state.reward.shape}")
    check("proprioception" in env_state.obs, "Obs has 'proprioception' key")
    check("vision" in env_state.obs, "Obs has 'vision' key")
    check(env_state.reward.shape == (NUM_ENVS,),
          f"Reward shape is ({NUM_ENVS},)")

    # ------------------------------------------------------------------
    # Step 3: Create VisionRenderer
    # ------------------------------------------------------------------
    section("Step 3: Create VisionRenderer")

    renderer = VisionRenderer(
        mj_model=env.mj_model,
        nworld=NUM_ENVS,
        camera_name=CAMERA_NAME,
        width=VISION_WIDTH,
        height=VISION_HEIGHT,
    )
    print(f"  Renderer created: {VISION_WIDTH}x{VISION_HEIGHT}, "
          f"camera={CAMERA_NAME}, nworld={NUM_ENVS}")
    check(renderer.width == VISION_WIDTH, "Renderer width matches config")
    check(renderer.height == VISION_HEIGHT, "Renderer height matches config")

    # Quick render test
    renderer.sync_state(env_state.data)
    rgb, _ = renderer.render()
    print(f"  Test render shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"  Test render mean pixel value: {rgb.mean():.2f}")
    check(rgb.shape == (NUM_ENVS, VISION_HEIGHT, VISION_WIDTH, 3),
          "Render output shape is correct")
    check(rgb.mean() > 0, "Rendered image is non-zero (not all black)")

    # ------------------------------------------------------------------
    # Step 4: Create PPO networks
    # ------------------------------------------------------------------
    section("Step 4: Create PPO networks")

    obs_sizes = get_obs_sizes(env_state.obs)
    print(f"  Observation sizes: {obs_sizes}")

    vision_channels = 1 if GRAYSCALE else 3
    vision_shape = (VISION_HEIGHT, VISION_WIDTH, vision_channels)

    ppo_network = ff_networks.make_vision_ppo_networks(
        obs_sizes=obs_sizes,
        action_size=wrapped_env.action_size,
        vision_shape=vision_shape,
    )
    print(f"  Policy network type: {type(ppo_network.policy_network).__name__}")
    print(f"  Value network type: {type(ppo_network.value_network).__name__}")

    # Create policy
    make_policy = ff_networks.make_inference_fn(ppo_network)

    # Init params
    key, key_policy, key_value = jax.random.split(key, 3)
    init_params = losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    normalizer_params = init_dict_normalizer(env_state.obs)

    # Count parameters
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    n_policy_params = count_params(init_params.policy)
    n_value_params = count_params(init_params.value)
    print(f"  Policy params: {n_policy_params:,}")
    print(f"  Value params: {n_value_params:,}")
    check(n_policy_params > 0, "Policy network has parameters")
    check(n_value_params > 0, "Value network has parameters")

    # Test policy forward pass
    policy = make_policy((normalizer_params, init_params.policy))
    key, test_key = jax.random.split(key)
    test_actions, test_extras = policy(env_state.obs, test_key)
    print(f"  Test action shape: {test_actions.shape}")
    check(test_actions.shape == (NUM_ENVS, wrapped_env.action_size),
          "Policy produces correct action shape")
    check("log_prob" in test_extras, "Policy extras contain 'log_prob'")
    check("raw_action" in test_extras, "Policy extras contain 'raw_action'")

    # ------------------------------------------------------------------
    # Step 5: Run generate_unroll_with_vision
    # ------------------------------------------------------------------
    section("Step 5: Run generate_unroll_with_vision")

    key, unroll_key = jax.random.split(key)
    t_unroll = time.time()
    final_state, data = vision_ppo.generate_unroll_with_vision(
        wrapped_env,
        env_state,
        policy,
        unroll_key,
        unroll_length=UNROLL_LENGTH,
        renderer=renderer,
        grayscale=GRAYSCALE,
        extra_fields=("truncation",),
    )
    unroll_time = time.time() - t_unroll
    print(f"  Unroll completed in {unroll_time:.2f}s")

    # Validate shapes
    print(f"  Transition observation keys: {list(data.observation.keys())}")
    print(f"  Reward shape: {data.reward.shape}")
    print(f"  Action shape: {data.action.shape}")
    print(f"  Discount shape: {data.discount.shape}")

    check(data.reward.shape == (UNROLL_LENGTH, NUM_ENVS),
          f"Reward shape is ({UNROLL_LENGTH}, {NUM_ENVS})")
    check(data.action.shape == (UNROLL_LENGTH, NUM_ENVS, wrapped_env.action_size),
          f"Action shape is ({UNROLL_LENGTH}, {NUM_ENVS}, {wrapped_env.action_size})")
    check(data.discount.shape == (UNROLL_LENGTH, NUM_ENVS),
          f"Discount shape is ({UNROLL_LENGTH}, {NUM_ENVS})")

    # Validate vision observations
    vision_obs = data.observation["vision"]
    expected_vision_shape = (UNROLL_LENGTH, NUM_ENVS, VISION_HEIGHT, VISION_WIDTH,
                             1 if GRAYSCALE else 3)
    print(f"  Vision obs shape: {vision_obs.shape}")
    print(f"  Vision obs dtype: {vision_obs.dtype}")
    vision_mean = float(jnp.mean(vision_obs))
    print(f"  Vision obs mean: {vision_mean:.4f}")
    check(vision_obs.shape == expected_vision_shape,
          f"Vision shape is {expected_vision_shape}")
    check(vision_mean > 0, "Vision observations are non-zero (images rendered)")

    # Validate rewards are finite
    rewards_finite = bool(jnp.all(jnp.isfinite(data.reward)))
    reward_mean = float(jnp.mean(data.reward))
    print(f"  Reward mean: {reward_mean:.4f}")
    check(rewards_finite, "All rewards are finite")

    # Validate truncation field is present
    check("truncation" in data.extras["state_extras"],
          "Truncation field present in state_extras")

    # Validate policy extras
    check("log_prob" in data.extras["policy_extras"],
          "log_prob present in policy_extras")
    check("raw_action" in data.extras["policy_extras"],
          "raw_action present in policy_extras")

    # ------------------------------------------------------------------
    # Step 6: Run one SGD step (full loss/gradient pipeline)
    # ------------------------------------------------------------------
    section("Step 6: Run SGD step (loss + gradient pipeline)")

    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=1e-4, weight_decay=0.0, eps=1e-5),
    )

    loss_fn = functools.partial(
        losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=1e-4,
        latent_kl_weight=1e-3,
        latent_ar1_weight=1e-3,
        discounting=0.97,
        reward_scaling=1.0,
        gae_lambda=0.95,
        clipping_epsilon=0.3,
        normalize_advantage=True,
        vf_coefficient=0.5,
    )

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=None, has_aux=True, clip_threshold=10.0,
    )

    # Reshape data for SGD: (UNROLL_LENGTH, NUM_ENVS, ...) -> (NUM_ENVS, UNROLL_LENGTH, ...)
    sgd_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Run one gradient step
    key, key_loss = jax.random.split(key)
    optimizer_state = optimizer.init(init_params)

    print("  Running JIT compilation + gradient step...")
    t_sgd = time.time()
    (loss_val, metrics), new_params, new_opt_state = gradient_update_fn(
        init_params, normalizer_params, sgd_data, key_loss, jnp.int32(0),
        optimizer_state=optimizer_state, params=init_params,
    )
    sgd_time = time.time() - t_sgd
    print(f"  SGD step completed in {sgd_time:.2f}s")

    loss_finite = bool(jnp.isfinite(loss_val))
    print(f"  Loss value: {float(loss_val):.4f}")
    check(loss_finite, "Loss is finite")

    print(f"  Metrics:")
    for metric_name, metric_val in sorted(metrics.items()):
        val = float(metric_val)
        print(f"    {metric_name}: {val:.6f}")
        check(np.isfinite(val), f"Metric '{metric_name}' is finite")

    # Verify parameters were updated
    old_policy_flat = jax.tree_util.tree_leaves(init_params.policy)
    new_policy_flat = jax.tree_util.tree_leaves(new_params.policy)
    params_changed = any(
        not jnp.allclose(old, new) for old, new in zip(old_policy_flat, new_policy_flat)
    )
    check(params_changed, "Policy parameters were updated by SGD step")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("VALIDATION COMPLETE")
    total_time = time.time() - t0
    print(f"  All checks passed!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"    Environment setup: {unroll_time:.1f}s (includes JIT)")
    print(f"    SGD step: {sgd_time:.1f}s (includes JIT compilation)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  VALIDATION FAILED")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)
