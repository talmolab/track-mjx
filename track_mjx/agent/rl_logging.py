"""Logging functions for reinforcement learning tasks."""

from pathlib import Path
import jax
import wandb
import imageio
import numpy as np
from jax import numpy as jp
from typing import Callable, Any
from omegaconf import DictConfig

from track_mjx.agent import custom_losses


def log_metric_to_wandb(metric_name: str, data: jp.ndarray, title: str = ""):
    if isinstance(data[0], tuple):
        frames, values = zip(*data)
    else:
        frames, values = data
    table = wandb.Table(
        data=[[x, y] for x, y in zip(frames, values)],
        columns=["frame", metric_name],
    )
    wandb.log(
        {
            f"eval/rollout_{metric_name}": wandb.plot.line(
                table,
                "frame",
                metric_name,
                title=title or f"{metric_name} for each rollout frame",
            )
        },
        commit=False,
    )


def rl_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    renderer,
    mj_model,
    mj_data,
    scene_option,
    current_step: int,  # all args above this line are passed in by functools.partial
    jit_logging_inference_fn,
    params: custom_losses.PPONetworkParams,
    policy_params_fn_key: jax.random.PRNGKey,
) -> None:
    """Logs metrics and videos for reinforcement learning evaluation rollouts.

    Uses deterministic action selection for consistent evaluation.

    Args:
        env: Environment instance
        jit_reset: Jitted reset function
        jit_step: Jitted step function
        cfg: Configuration
        model_path: Path to save model artifacts
        renderer: MuJoCo renderer
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        scene_option: MuJoCo scene options
        current_step: Current training step
        jit_logging_inference_fn: Jitted policy inference function
        params: Policy parameters
        policy_params_fn_key: PRNG key
    """
    env_config = cfg["env_config"]
    deterministic_eval = cfg.get("train_setup", {}).get("deterministic_eval", True)

    # Create a deterministic wrapper around the inference function that ignores the RNG
    def deterministic_inference_fn(params, obs, _unused_rng):
        # Use a fixed RNG to ensure consistent behavior
        fixed_rng = jax.random.PRNGKey(0)
        action, extras = jit_logging_inference_fn(params, obs, fixed_rng)

        # If extras contains distribution parameters (e.g., mean, log_std),
        # we can directly use the mean instead of the sampled action
        if isinstance(extras, dict) and "mean" in extras:
            action = extras["mean"]

        return action, extras

    # Choose which inference function to use based on config
    inference_fn = (
        deterministic_inference_fn if deterministic_eval else jit_logging_inference_fn
    )

    # Performance optimization: Only run debug logging if explicitly enabled
    debug_mode = cfg.get("logging_config", {}).get("debug_mode", False)

    # Create a JAX compiled rollout function for better performance
    @jax.jit
    def run_rollout_step(state, rng):
        """Compiled step function combining inference and environment step."""
        next_rng, act_rng = jax.random.split(rng)
        action, extras = inference_fn(params, state.obs, act_rng)
        clipped_action = jp.clip(action, -0.1, 0.1)
        next_state = jit_step(state, clipped_action)
        return next_state, next_rng, clipped_action

    # Initialize
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)
    state = jit_reset(reset_rng)
    max_frames = cfg.train_setup.get("episode_length", 200)

    # Optimized rollout collection
    rollout = [state]
    controls = []

    # Safely print debug info only when outside of JIT context
    if debug_mode:
        # Only run this debugging code when explicitly enabled
        try:
            state = jit_reset(reset_rng)
            if hasattr(state, "pipeline_state"):
                first_obs = state.obs
                print(f"Initial observation shape: {first_obs.shape}")

                # Access target information safely after JIT completion
                if "target_position" in state.info:
                    target_pos = state.info["target_position"]
                    print(f"Initial target position from info: {target_pos}")

                # Get fingertip position safely (without JIT)
                fingertip_pos = env._get_fingertip_position(state.pipeline_state)
                print(f"Initial fingertip position: {fingertip_pos}")
        except Exception as e:
            print(f"Debug info extraction failed: {e}")

    # Run rollout with compiled function
    for i in range(max_frames):
        state, act_rng, ctrl = run_rollout_step(state, act_rng)
        controls.append(ctrl)
        rollout.append(state)

        # Check for termination
        if state.done or jp.any(jp.isnan(state.pipeline_state.qpos)):
            if jp.any(jp.isnan(state.pipeline_state.qpos)):
                print("NaN detected in qpos, terminating rollout")
            break

    # Performance monitoring
    if debug_mode:
        import time

        start_time = time.time()
        rollout_time = time.time() - start_time
        print(
            f"Rollout collection took {rollout_time:.4f} seconds for {len(rollout)} frames"
        )
        print(f"Effective SPS: {len(rollout)/rollout_time:.2f}")

    rewards = [s.metrics["reward"] for s in rollout]
    distance_rewards = [s.metrics["distance_reward"] for s in rollout]
    target_distances = [s.metrics["target_distance"] for s in rollout]

    ctrl_costs = [s.metrics.get("ctrl_cost", 0.0) for s in rollout]
    ctrl_diff_costs = [s.metrics.get("ctrl_diff_cost", 0.0) for s in rollout]
    energy_costs = [s.metrics.get("energy_cost", 0.0) for s in rollout]
    jerk_costs = [s.metrics.get("jerk_cost", 0.0) for s in rollout]

    log_metric_to_wandb(
        "rewards", list(enumerate(rewards)), title="Rewards for each rollout frame"
    )
    log_metric_to_wandb(
        "distance_rewards",
        list(enumerate(distance_rewards)),
        title="Distance rewards for each rollout frame",
    )
    log_metric_to_wandb(
        "target_distances",
        list(enumerate(target_distances)),
        title="Target distances for each rollout frame",
    )
    log_metric_to_wandb(
        "ctrl_costs",
        list(enumerate(ctrl_costs)),
        title="Control costs for each rollout frame",
    )
    log_metric_to_wandb(
        "ctrl_diff_costs",
        list(enumerate(ctrl_diff_costs)),
        title="Control difference costs for each rollout frame",
    )
    log_metric_to_wandb(
        "energy_costs",
        list(enumerate(energy_costs)),
        title="Energy costs for each rollout frame",
    )
    log_metric_to_wandb(
        "jerk_costs",
        list(enumerate(jerk_costs)),
        title="Jerk costs for each rollout frame",
    )

    qposes_rollout = np.array([s.pipeline_state.qpos for s in rollout])
    video_path = f"{model_path}/{current_step}.mp4"

    print(f"Using camera: {cfg.env_config.render_camera_name}")

    print("\nTARGET POSITION VERIFICATION:")
    # Only access this data after the rollout is complete and we have concrete values
    target_pos_from_info = rollout[-1].info.get("target_position") if rollout else None
    target_size = (
        rollout[-1].info.get("current_target_size", 0.001) if rollout else 0.001
    )

    if target_pos_from_info is not None:
        try:
            print(f"Final target position from info: {target_pos_from_info}")
            final_geom_xpos = rollout[-1].pipeline_state.geom_xpos[env._target_idx]
            print(f"Final geom_xpos: {final_geom_xpos}")
            print(f"Current target size: {target_size}")
        except Exception as e:
            print(f"Error accessing final target position: {e}")

    target_geom_idx = None
    for i in range(mj_model.ngeom):
        if mj_model.geom(i).name == "target":
            target_geom_idx = i
            print(f"Found target geom at index: {i}")
            # Make target appropriately sized and highly visible
            mj_model.geom(i).size[0] = target_size  # Slightly larger for visibility
            mj_model.geom(i).rgba = [1.0, 0.0, 0.0, 1.0]  # Bright red, fully opaque
            print(f"Target geom body ID: {mj_model.geom_bodyid[i]}")
            break

    if target_geom_idx is None:
        print("ERROR: Could not find target geom in model!")

    with imageio.get_writer(video_path, fps=int((1.0 / env.dt))) as video:
        for i, qpos in enumerate(qposes_rollout):
            for j in range(mj_model.nq):
                if j < len(qpos):
                    mj_data.qpos[j] = qpos[j]

            import mujoco

            mujoco.mj_forward(mj_model, mj_data)

            # Force target position update on every frame
            if target_geom_idx is not None and target_pos_from_info is not None:
                mj_data.geom_xpos[target_geom_idx] = np.array(
                    target_pos_from_info, dtype=np.float64
                )
                # Verify position was updated
                if i == 0:
                    print(
                        f"Target geom_xpos after forced update: {mj_data.geom_xpos[target_geom_idx]}"
                    )

            if i == 0 and target_geom_idx is not None:
                try:
                    print(
                        f"First frame - target geom_pos: {mj_model.geom(target_geom_idx).pos}"
                    )
                    print(
                        f"First frame - target geom_xpos (after update): {mj_data.geom_xpos[target_geom_idx]}"
                    )
                except Exception as e:
                    print(f"Error accessing target position data: {e}")

            renderer.update_scene(
                mj_data,
                camera=cfg.env_config.render_camera_name,
                scene_option=scene_option,
            )
            pixels = renderer.render()
            video.append_data(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
