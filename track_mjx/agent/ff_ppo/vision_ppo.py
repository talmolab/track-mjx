"""Vision-augmented PPO training with interleaved mujoco_warp rendering.

Workaround 1: Replaces brax's ``acting.generate_unroll`` (which uses
``jax.lax.scan``) with a Python-level for-loop that interleaves GPU-based
rendering via mujoco_warp after each physics step.  This is necessary because
the mujoco_warp renderer operates outside the JAX computation graph and cannot
be called from inside ``lax.scan``.

The key function is ``generate_unroll_with_vision``, which follows the same
contract as ``brax.training.acting.generate_unroll`` -- it returns
``(final_state, data)`` where ``data`` is a ``types.Transition`` with leaves
shaped ``(unroll_length, num_envs, ...)``.

The ``train()`` function that consumes this unroll will be added in a
subsequent task.
"""

import functools
import time
from typing import Any, Callable, Sequence, Tuple

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import logging
from brax import base, envs
from brax.training import acting, pmap, types
from brax.training.acme import running_statistics
from brax.training.types import Params, PRNGKey

from mujoco_playground import wrapper as mp_wrapper
from optax.transforms import freeze

from track_mjx.agent import checkpointing, gradients, network_masks
from track_mjx.agent.ff_ppo import losses, ppo_networks
from track_mjx.agent.observation_utils import (
    DictRunningStatisticsState,
    get_obs_sizes,
    init_dict_normalizer,
    update_dict_normalizer,
)

# ---------------------------------------------------------------------------
# Type aliases (mirrors ppo.py)
# ---------------------------------------------------------------------------
InferenceParams = tuple[DictRunningStatisticsState, Params]
Metrics = types.Metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


# ---------------------------------------------------------------------------
# Helper: RGB to grayscale
# ---------------------------------------------------------------------------

def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to single-channel grayscale in [0, 1].

    Uses the standard luminance weights (ITU-R BT.601):
        Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

    Args:
        rgb: NumPy array of shape ``(N, H, W, 3)`` with dtype ``uint8``.

    Returns:
        NumPy array of shape ``(N, H, W, 1)`` with dtype ``float32``,
        values in ``[0, 1]``.
    """
    # Normalize to [0, 1] first, then apply luminance weights
    rgb_float = rgb.astype(np.float32) / 255.0
    gray = (
        0.2989 * rgb_float[..., 0:1]
        + 0.5870 * rgb_float[..., 1:2]
        + 0.1140 * rgb_float[..., 2:3]
    )
    return gray


# ---------------------------------------------------------------------------
# Core: generate_unroll_with_vision
# ---------------------------------------------------------------------------

def generate_unroll_with_vision(
    env: envs.Env,
    env_state: envs.State,
    policy: types.Policy,
    key: PRNGKey,
    unroll_length: int,
    renderer: Any,
    grayscale: bool = True,
    extra_fields: Sequence[str] = (),
) -> Tuple[envs.State, types.Transition]:
    """Collect a trajectory of ``unroll_length`` steps with vision rendering.

    This mirrors :func:`brax.training.acting.generate_unroll` but uses a
    Python ``for`` loop instead of ``jax.lax.scan`` so that we can call the
    mujoco_warp renderer (a non-JAX operation) between each physics step.

    At every step the function:
        1. Syncs the physics state to the renderer
           (``renderer.sync_state(env_state.data)``).
        2. Renders an egocentric RGB image
           (``renderer.render()``).
        3. Optionally converts RGB to grayscale.
        4. Injects the image into ``env_state.obs["vision"]`` so the policy
           can condition on it.
        5. Queries the policy for an action.
        6. Advances the environment by one step.
        7. Records the transition.

    Args:
        env: A brax-wrapped environment.
        env_state: Current batched environment state.
        policy: A callable ``(obs, key) -> (action, extras)``.
        key: JAX PRNG key.
        unroll_length: Number of environment steps to collect.
        renderer: A mujoco_warp renderer object that exposes
            ``sync_state(mjx_data)`` and ``render() -> (rgb, depth)``.
        grayscale: If True, convert rendered RGB to single-channel grayscale.
        extra_fields: Additional fields to extract from ``env_state.info``
            (e.g., ``("truncation",)``).

    Returns:
        A tuple ``(final_state, data)`` where:
            - ``final_state`` is the environment state after ``unroll_length``
              steps.
            - ``data`` is a :class:`~brax.training.types.Transition`
              whose leaf arrays have shape ``(unroll_length, num_envs, ...)``.
    """
    transitions: list[types.Transition] = []

    current_state = env_state
    current_key = key

    for _ in range(unroll_length):
        current_key, step_key = jax.random.split(current_key)

        # --- 1. Render vision from the current physics state ---------------
        renderer.sync_state(current_state.data)
        rgb, _ = renderer.render()  # (num_envs, H, W, 3) uint8

        # --- 2. Process the image ------------------------------------------
        if grayscale:
            vision = rgb_to_grayscale(rgb)  # (num_envs, H, W, 1) float32
        else:
            vision = rgb.astype(np.float32) / 255.0  # (num_envs, H, W, 3) float32

        # --- 3. Inject vision into the observation dict --------------------
        # Convert to JAX array so it can flow into the policy network.
        vision_jax = jnp.asarray(vision)

        # Build a new observation dict with vision replaced.
        obs_with_vision = {**current_state.obs, "vision": vision_jax}

        # --- 4. Query the policy -------------------------------------------
        actions, policy_extras = policy(obs_with_vision, step_key)

        # --- 5. Step the environment ---------------------------------------
        next_state = env.step(current_state, actions)

        # --- 6. Record the transition (same format as acting.actor_step) ---
        state_extras = {x: next_state.info[x] for x in extra_fields}
        transition = types.Transition(
            observation=obs_with_vision,
            action=actions,
            reward=next_state.reward,
            discount=1 - next_state.done,
            next_observation=next_state.obs,
            extras={
                "policy_extras": policy_extras,
                "state_extras": state_extras,
            },
        )
        transitions.append(transition)

        current_state = next_state

    # Stack transitions along a new leading time axis: (unroll_length, num_envs, ...)
    data = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *transitions)

    return current_state, data
