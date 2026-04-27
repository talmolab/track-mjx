"""LatentMimic Phase-2 wandb rollout logging.

Mirrors track_mjx.agent.wandb_logging.rollout_logging_fn but tailored to the
LatentMimic obs/info schema (no `latent_mean` extras; logs r_mimic / mimic_kl
from state.info instead).
"""
import logging
from typing import Any, Callable

import imageio
import jax
import jax.numpy as jnp
import mujoco
import wandb
from omegaconf import DictConfig

from track_mjx.agent.wandb_logging import log_lineplot_to_wandb


def latent_mimic_rollout_logging_fn(
    env,
    jit_reset: Callable,
    jit_step: Callable,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    jit_logging_inference_fn: Callable,
    params: Any,
    policy_params_fn_key: jax.Array,
    render_video: bool = True,
    ppo_network: Any = None,
):
    """Run an evaluation rollout, log scalars + r_mimic curves + a video.

    Args:
        env: LatentMimicEnvWrapper-wrapped imitation env (must expose render()
            via the underlying base env).
        jit_reset: JIT-compiled env.reset.
        jit_step: JIT-compiled env.step.
        cfg: Full Hydra config (env_config, render_config, etc.).
        model_path: Directory for saved video files (the orbax checkpoint dir
            is reused so videos live next to checkpoints).
        current_step: Training step (used for the video filename).
        jit_logging_inference_fn: (params, obs, rng) -> (action, extras).
        params: PPO network params for the policy.
        policy_params_fn_key: JAX PRNG key.
        render_video: If True, render a video clip and log to wandb.
        ppo_network: Unused for latent_mimic (kept for signature parity with
            track_mjx.agent.wandb_logging.rollout_logging_fn).
    """
    del ppo_network  # unused; kept for signature parity
    _, reset_rng, act_rng = jax.random.split(policy_params_fn_key, 3)
    state = jit_reset(reset_rng)

    rollout = [state]
    r_mimic_curve = []
    kl_curve = []
    z_target_means = []

    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        ctrl, _ = jit_logging_inference_fn(params, state.obs, act_rng)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl
        state = jit_step(state, ctrl)
        rollout.append(state)
        r_mimic_curve.append(float(state.info.get("r_mimic", 0.0)))
        kl_curve.append(float(state.info.get("mimic_kl", 0.0)))
        z_target_means.append(state.obs["z_target"])

    # Per-step curves
    log_lineplot_to_wandb(
        name="eval/rollout_r_mimic",
        metric_name="r_mimic",
        data=list(enumerate(r_mimic_curve)),
        title="r_mimic per rollout step",
    )
    log_lineplot_to_wandb(
        name="eval/rollout_kl",
        metric_name="mimic_kl",
        data=list(enumerate(kl_curve)),
        title="KL(z_target||z_sim) per rollout step",
    )

    # z_target stats
    if len(z_target_means) > 0:
        z_arr = jnp.stack(z_target_means)
        z_mean_per_dim = jnp.mean(z_arr, axis=0)
        z_std_per_dim = jnp.std(z_arr, axis=0)
        for i in range(min(z_mean_per_dim.shape[0], 16)):  # cap to first 16 for clarity
            wandb.log(
                {
                    f"latents/z_target_mean{i}": float(z_mean_per_dim[i]),
                    f"latents/z_target_std{i}": float(z_std_per_dim[i]),
                },
                commit=False,
            )

    # Aggregate scalars
    wandb.log(
        {
            "eval/rollout_r_mimic_mean": float(jnp.mean(jnp.array(r_mimic_curve))),
            "eval/rollout_kl_mean": float(jnp.mean(jnp.array(kl_curve))),
            "eval/episode_length": len(r_mimic_curve),
        },
        commit=False,
    )

    if render_video:
        _log_rollout_video(env, cfg, model_path, current_step, rollout)


def _log_rollout_video(env, cfg, model_path, current_step, rollout):
    """Render rollout video and log to wandb.

    Many things can go wrong with rendering on a headless box (env mismatch,
    missing GL, etc.); on any failure we log a warning and continue training.
    """
    render_fps = cfg.render_config.render_fps
    video_path = f"{model_path}/{current_step}.mp4"
    try:
        with imageio.get_writer(video_path, fps=render_fps) as writer:
            video = env.render(
                rollout,
                camera=f"{cfg.render_config.render_camera_name}{getattr(env, '_suffix', '')}",
                height=480,
                width=640,
            )
            for frame in video:
                writer.append_data(frame)
        wandb.log(
            {"videos/rollout": wandb.Video(video_path, format="mp4")},
            commit=False,
        )
    except mujoco.FatalError as e:
        logging.warning(f"Rendering video failed with MuJoCo error: {e}")
    except Exception as e:
        logging.warning(f"Rendering video failed: {e}")
