"""Extract the four jitted, batched, timed callables from a real mimic-mjx env.

Components (per flybody Table 5, with control+physics merged into one MuJoCo column):
  - policy    : inference_fn(obs, rng)                      -> network forward pass
  - mujoco    : mjx_env.step(model, data, action, n_sub)   -> forward dynamics + integrate
  - rl_env    : _get_obs + _get_reward + _is_done           -> task overhead
  - full_step : env.step(state, action)                     -> true fused control step (RL env incl. mujoco)

All callables are vmapped over the leading ``num_envs`` axis with a shared (non-DR) model.
"""

from typing import Any, Callable

import jax
from mujoco_playground._src import mjx_env

from track_mjx.analysis import rollout


def unwrap(env: Any) -> Any:
    """Strip wrappers down to the underlying task env."""
    e = env
    while hasattr(e, "_env"):
        e = e._env
    return e


def build_env_and_state(cfg: Any, num_envs: int, seed: int = 0):
    """Create the env and a batched initial State (vmapped reset over num_envs keys)."""
    env = rollout.create_environment(cfg.env_config)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    state = jax.jit(jax.vmap(env.reset))(keys)
    return env, state


def build_control_step(env: Any, inference_fn: Callable, rng) -> Callable:
    """Return a (batched) carry->carry control step: policy then env.step.

    The rng is fixed (timing only); deterministic policy ignores it anyway.
    """
    def control_step(state):
        action, _ = inference_fn(state.obs, rng)
        return env.step(state, action)

    return jax.vmap(control_step)


def build_timed_callables(cfg: Any, env: Any, state: Any, inference_fn: Callable):
    """Return {name: (jitted_callable, args_tuple)} for the four components."""
    base = unwrap(env)
    model = base.mjx_model
    n_substeps = int(round(float(cfg.env_config.ctrl_dt) / float(cfg.env_config.sim_dt)))
    rng = jax.random.PRNGKey(0)

    # Policy
    policy = jax.jit(inference_fn)
    action, _ = policy(state.obs, rng)
    action = jax.block_until_ready(action)

    # MuJoCo step (control + physics), batched over data+action, shared model
    def _mjx(data, act):
        return mjx_env.step(model, data, act, n_substeps)

    mujoco = jax.jit(jax.vmap(_mjx))

    # Full control step (fused) — also gives a representative stepped state for rl_env
    full_step = jax.jit(jax.vmap(env.step))
    next_state = full_step(state, action)
    next_state = jax.block_until_ready(next_state)

    # RL-env overhead: obs + reward + done on a stepped (data, info, metrics)
    def _rl(data, info, metrics):
        obs = base._get_obs(data, info)
        reward = base._get_reward(data, info, metrics)
        done = base._is_done(data, info, metrics)
        return obs, reward, done

    rl_env = jax.jit(jax.vmap(_rl))

    return {
        "policy": (policy, (state.obs, rng)),
        "mujoco": (mujoco, (state.data, action)),
        "rl_env": (rl_env, (next_state.data, next_state.info, next_state.metrics)),
        "full_step": (full_step, (state, action)),
    }
