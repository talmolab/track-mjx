"""Rollout generation utilities for VNL imitation learning environments.

This module provides functions to create VNL environments and generate policy
rollouts with optional logging of activations, metrics, and sensor data.
Rollouts can be used for evaluation, visualization, or data collection.

Supports both feedforward and recurrent policies.
"""

from typing import Any, Callable

import jax
from mujoco_playground._src import mjx_env
from jax import numpy as jnp
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf
from vnl_playground import registry
from vnl_playground.tasks import wrappers as vnl_wrappers


def create_environment(env_config: dict[str, Any] | DictConfig) -> mjx_env.MjxEnv:
    """Create a VNL imitation learning environment from a configuration.

    Uses the vnl-playground registry to create the environment with reference
    clips, returning nested dictionary observations:
        {'state': {'task_obs': ..., 'proprioception': ...},
         'privileged_state': {'task_obs': ..., 'proprioception': ...}}

    Args:
        env_config: Environment configuration only (not the full training
            config). Must contain env_name, reference_data_path, clip_length,
            and optionally keep_clips_idx.

    Returns:
        A Brax-compatible environment with nested dictionary observations.

    Example:
        >>> cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)
        >>> env = create_environment(cfg.env_config)
        >>> state = env.reset(jax.random.PRNGKey(0))
        >>> print(state.obs.keys())  # dict_keys(['state', 'privileged_state'])
    """
    if isinstance(env_config, DictConfig):
        env_cfg_dict = OmegaConf.to_container(env_config, resolve=True)
    else:
        env_cfg_dict = dict(env_config)

    env_cfg_ml = config_dict.ConfigDict(env_cfg_dict)

    env_name = env_cfg_ml.env_name
    default_config = registry.get_default_config(env_name)
    reference_clips = registry.load_reference_clips(
        env_name,
        data_path=env_cfg_ml.reference_data_path,
        n_frames_per_clip=env_cfg_ml.clip_length,
        keep_clips_idx=env_cfg_ml.get("keep_clips_idx", None),
        joint_names=env_cfg_ml.get("joints", default_config.get("joints", None)),
        body_names=env_cfg_ml.get("bodies", default_config.get("bodies", None)),
    )
    return vnl_wrappers.TrackMjxObsWrapper(
        registry.load(
            env_name, config=env_cfg_ml, clips=reference_clips, flatten_obs=False
        )
    )


def create_rollout_generator(
    cfg: dict[str, Any] | DictConfig,
    env: mjx_env.MjxEnv,
    inference_fn: Callable,
    model: str | None = None,
    log_full_states: bool = False,
    log_activations: bool = False,
    log_metrics: bool = False,
    log_sensor_data: bool = False,
    init_hidden_fn: Callable[[int], Any] | None = None,
) -> Callable[[int | None, int], dict[str, Any]]:
    """Legacy rollout generator matching the pre-VNL track-mjx logic.

    This keeps the old rollout semantics:
    - reset at `start_frame=0`
    - build rollout states by prepending the initial state to scan outputs
    - build reference qposes by repeating each mocap frame `steps_per_frame`
    - return a jitted rollout function

    The implementation is adapted to the current VNL `mjx_env.State` structure.
    Recurrent behavior is inferred from `init_hidden_fn`; `model` is retained
    only for backward-compatible call sites.

    Args:
        log_full_states: If True, include the stacked rollout states in the
            result dict under ``rollout_states``.
    """
    is_recurrent = init_hidden_fn is not None
    if model is None:
        model = "recurrent" if is_recurrent else "mlp"
    elif is_recurrent and model == "mlp":
        raise ValueError(
            "init_hidden_fn was provided, but model='mlp'. "
            "Use model='recurrent'/'lstm' or omit model."
        )
    elif not is_recurrent and model in ("lstm", "recurrent"):
        raise ValueError(
            f"model='{model}' requires init_hidden_fn, but none was provided."
        )

    mocap_dt = 1.0 / cfg.env_config.mocap_hz
    steps_per_frame = mocap_dt / cfg.env_config.ctrl_dt
    num_steps = int(cfg.env_config.clip_length * steps_per_frame) - 1

    unwrapped_env = env
    while hasattr(unwrapped_env, "_env"):
        unwrapped_env = unwrapped_env._env

    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    def generate_rollout(clip_idx: int | None = None, seed: int = 42) -> dict[str, Any]:
        rollout_key = jax.random.PRNGKey(seed)
        rollout_key, reset_rng, act_rng = jax.random.split(rollout_key, 3)

        init_state = jit_reset(reset_rng, clip_idx=clip_idx, start_frame=0)

        def _step_fn_mlp(carry, _):
            state, rng = carry
            rng, new_rng = jax.random.split(rng)
            ctrl, extras = jit_inference_fn(state.obs, rng)
            next_state = jit_step(state, ctrl)

            joint_force = next_state.data.cfrc_ext if log_sensor_data else None
            sensor_reading = next_state.data.sensordata if log_sensor_data else None
            activations = extras["activations"] if log_activations else None
            return (next_state, new_rng), (
                next_state,
                ctrl,
                activations,
                joint_force,
                sensor_reading,
            )

        def _step_fn_recurrent(carry, _):
            state, rng, hidden = carry
            rng, new_rng = jax.random.split(rng)
            ctrl, extras, new_hidden = jit_inference_fn(state.obs, hidden, rng)
            next_state = jit_step(state, ctrl)

            joint_force = next_state.data.cfrc_ext if log_sensor_data else None
            sensor_reading = next_state.data.sensordata if log_sensor_data else None
            activations = extras["activations"] if log_activations else None

            return (next_state, new_rng, new_hidden), (
                next_state,
                ctrl,
                hidden,
                activations,
                joint_force,
                sensor_reading,
            )

        if not is_recurrent:
            init_carry = (init_state, act_rng)
            (
                (_, _),
                (
                    states,
                    ctrls,
                    activations,
                    joint_forces,
                    sensor_readings,
                ),
            ) = jax.lax.scan(_step_fn_mlp, init_carry, None, length=num_steps)
        else:
            hidden = jax.tree.map(lambda x: x[0], init_hidden_fn(1))
            init_carry = (init_state, act_rng, hidden)
            (
                (_, _, _),
                (
                    states,
                    ctrls,
                    _stacked_hidden,
                    activations,
                    joint_forces,
                    sensor_readings,
                ),
            ) = jax.lax.scan(_step_fn_recurrent, init_carry, None, length=num_steps)

        def prepend(element, arr):
            if arr.ndim == 0:
                return arr
            return jnp.concatenate([element[None], arr])

        rollout_states = jax.tree.map(prepend, init_state, states)

        ref_clip_idx = jnp.asarray(init_state.info["reference_clip"]).astype(int)
        ref_qpos = unwrapped_env.reference_clips.qpos[ref_clip_idx]

        start_frame = jnp.asarray(rollout_states.info["start_frame"])
        times = jnp.asarray(rollout_states.data.time)
        frame_indices = jnp.round(
            times * float(env._config.mocap_hz) + start_frame
        ).astype(jnp.int32)
        ref_qpos = jnp.asarray(env.reference_clips.qpos[ref_clip_idx])
        qposes_ref = ref_qpos[frame_indices]

        result = {
            "qposes_ref": qposes_ref,
            "qposes_rollout": rollout_states.data.qpos,
            "ctrl": ctrls,
            "state_rewards": rollout_states.reward,
        }

        if log_full_states:
            result["rollout_states"] = rollout_states

        if log_metrics:
            rollout_metrics = {}
            metric_names = OmegaConf.select(
                cfg, "logging_config.rollout_metrics", default=[]
            )
            for metric_name in metric_names:
                rollout_metrics[f"{metric_name}s"] = rollout_states.metrics[metric_name]
            result["rollout_metrics"] = rollout_metrics

        if log_activations and activations is not None:
            result["activations"] = activations

        if log_sensor_data:
            if joint_forces is not None:
                result["joint_forces"] = joint_forces
            if sensor_readings is not None:
                result["sensor_readings"] = sensor_readings

        return result

    return jax.jit(generate_rollout)
