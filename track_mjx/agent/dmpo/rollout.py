"""MJX rollout for DMPO.

Stores raw (pre-tanh) actions in the trajectory, so the MPO loss sees
unbounded Gaussian samples. Apply tanh on the way to the env via `bind`.

Output shape conforms to flashbax's TrajectoryBuffer.add expectation:
[add_batch_size, T, ...] per leaf.
"""
import jax
import jax.numpy as jnp
from track_mjx.agent.dmpo.action_utils import bind


def collect_rollout(env, policy_fn, rng, num_envs: int, num_steps: int):
    """Roll out num_envs parallel envs for num_steps timesteps.

    Args:
      env: object with .reset(rng) and .step(state, action). Both must be
        jittable / vmappable. The state must be a registered pytree (jax
        treats it as such automatically for dataclasses with jax types).
      policy_fn: callable (obs, key) -> raw_action (unbounded Gaussian sample).
        Caller provides this; typically wraps a network.apply over policy params.
      rng: PRNGKey for action sampling and env reset.
      num_envs: number of parallel envs.
      num_steps: trajectory length.

    Returns:
      trajectory: dict with keys observation/action/reward/discount/next_observation,
        each shaped [num_envs, num_steps, ...]. Action is the *raw* (pre-tanh)
        Gaussian sample.
      final_state: final per-env state after num_steps (so the caller can resume).
    """
    rng, k_reset = jax.random.split(rng)
    reset_keys = jax.random.split(k_reset, num_envs)
    state = jax.vmap(env.reset)(reset_keys)

    def step_fn(carry, _):
        state, rng = carry
        rng, k_act = jax.random.split(rng)
        keys = jax.random.split(k_act, num_envs)
        raw_action = jax.vmap(policy_fn)(state.obs, keys)
        bound_action = bind(raw_action)
        new_state, reward = jax.vmap(env.step)(state, bound_action)
        transition = {
            "observation": state.obs,
            "action": raw_action,           # store raw, NOT bound
            "reward": reward,
            "discount": (1.0 - new_state.done).astype(jnp.float32),
            "next_observation": new_state.obs,
        }
        return (new_state, rng), transition

    (final_state, _), traj = jax.lax.scan(
        step_fn, (state, rng), None, length=num_steps,
    )
    # scan stacks along axis 0 (time); flashbax wants [B, T, ...] not [T, B, ...].
    traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
    return traj, final_state
