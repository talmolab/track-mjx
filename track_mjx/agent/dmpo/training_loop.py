"""Reusable training loop for DMPO.

Extracted from ``track_mjx/agent/dmpo/train_dmpo.py`` so both
``track_mjx.train_dmpo`` (imitation) and ``vnl_playground.train_dmpo``
(downstream) can share it. Knows nothing about which env it's training —
env/nets/spec are arguments. Eval rendering, wandb logging, and
checkpointing are delegated to caller-provided callbacks.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import jax

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.train_dmpo_chunk import make_train_chunk
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

log = logging.getLogger(__name__)


def run(
    *,
    env: Any,
    nets: Any,
    optimizers: Any,
    rb: Any,
    cfg: DMPOConfig,
    K: int,
    iters_per_chunk: int,
    rng: jax.Array,
    state: Any,
    env_state: Any,
    rb_state: Any,
    warmup_done: bool = False,
    max_chunks: Optional[int] = None,
    eval_callback: Optional[Callable[[Any, int, jax.Array], None]] = None,
    wandb_log_callback: Optional[Callable[[dict, int], None]] = None,
    ckpt_mgr: Any = None,
    ckpt_save_callback: Optional[Callable[[Any, int], None]] = None,
    cfg_dict: Optional[dict] = None,
    log_every_steps: Optional[int] = None,
    eval_every_steps: Optional[int] = None,
    num_timesteps: Optional[int] = None,
    extra_state_extras: tuple = (),
) -> tuple[Any, Any, Any, dict]:
    """Run the DMPO training loop until ``num_timesteps`` env steps.

    Args:
      env, nets, optimizers, rb, cfg, K, iters_per_chunk: per the DMPO
        algorithm; built by the caller's env/network setup.
      rng: starting PRNGKey.
      state, env_state, rb_state: starting training state, env state
        (None on first call → reset path), and replay state.
      warmup_done: True to skip the warm-up branch entirely (used by
        tests that pre-fill the replay).
      max_chunks: if set, stop after this many chunks (used by tests).
        If None, use ``num_timesteps`` from cfg / argument.
      eval_callback(state, total_env_steps, rng) → None: called at
        eval cadence (env-step boundaries). Caller does its own
        eval rollout / video render / wandb upload.
      wandb_log_callback(metrics_dict, total_env_steps) → None: called
        at eval cadence with the most recent chunk's mean-over-N
        training metrics. Caller decides whether to upload.
      ckpt_mgr: an orbax CheckpointManager or None.
      ckpt_save_callback(state, env_step) → None: called at eval cadence
        when ckpt_mgr is non-None.
      cfg_dict: snapshotted hydra config dict for checkpoint metadata.
      log_every_steps, eval_every_steps, num_timesteps: pulled from
        ``cfg`` if not provided. Argument form lets test override.

    Returns:
      (final_state, final_env_state, final_rb_state, last_train_metrics)
    """
    log_every = log_every_steps if log_every_steps is not None else cfg.log_every_steps
    eval_every = eval_every_steps if eval_every_steps is not None else cfg.eval_every_steps
    timesteps = num_timesteps if num_timesteps is not None else cfg.num_timesteps

    fused_step = make_fused_train_step(
        env, nets, optimizers, rb, cfg, K=K,
        extra_state_extras=extra_state_extras,
    )
    train_chunk = make_train_chunk(fused_step, n_iters=iters_per_chunk)

    env_steps_per_chunk = cfg.num_envs * cfg.unroll_length * iters_per_chunk
    last_train_metrics: dict = {}
    total_env_steps = 0
    last_eval_step = 0
    chunks_completed = 0
    first_step = True
    second_step = True
    t0 = time.time()

    while True:
        # Stop conditions.
        if max_chunks is not None and chunks_completed >= max_chunks:
            break
        if max_chunks is None and total_env_steps >= timesteps:
            break

        rng, k_step = jax.random.split(rng)

        # Warm-up branch — single fused_step at a time. See Task 3 review
        # comment on warmup-SGD landmine; the lax.select gate inside
        # fused_step makes this safe.
        if not warmup_done:
            if first_step:
                log.info("Compiling fused_step (reset path)...")
                t1 = time.time()
                state, env_state, rb_state, metrics = fused_step(
                    state, env_state, rb_state, k_step
                )
                jax.block_until_ready(metrics["policy_loss"])
                log.info("fused_step (reset) compiled in %.1fs", time.time() - t1)
                first_step = False
            else:
                state, env_state, rb_state, metrics = fused_step(
                    state, env_state, rb_state, k_step
                )
            last_train_metrics = metrics
            total_env_steps += cfg.num_envs * cfg.unroll_length
            if not bool(rb.can_sample(rb_state)):
                log.info("warming replay: %d env steps (need ~%d)",
                         total_env_steps, cfg.min_replay_size)
                continue
            warmup_done = True
            log.info("replay warm-up complete at %d env steps", total_env_steps)
            continue

        # Steady state — one chunk = N fused_step iters in one dispatch.
        if second_step:
            log.info("Compiling train_chunk (n_iters=%d)...", iters_per_chunk)
            t1 = time.time()
            state, env_state, rb_state, metrics = train_chunk(
                state, env_state, rb_state, k_step
            )
            jax.block_until_ready(metrics["policy_loss"])
            log.info("train_chunk compiled in %.1fs", time.time() - t1)
            second_step = False
        else:
            state, env_state, rb_state, metrics = train_chunk(
                state, env_state, rb_state, k_step
            )
        last_train_metrics = metrics
        total_env_steps += env_steps_per_chunk
        chunks_completed += 1

        elapsed = max(time.time() - t0, 1e-6)
        log.info(
            "chunk env_steps=%d steps_per_sec=%.0f policy_loss=%.4g critic_loss=%.4g",
            int(total_env_steps),
            total_env_steps / elapsed,
            float(last_train_metrics.get("policy_loss", 0.0)),
            float(last_train_metrics.get("critic_loss", 0.0)),
        )

        # Eval cadence — emits training metrics to wandb, runs eval
        # rollout / render via callbacks, saves checkpoint.
        if total_env_steps - last_eval_step >= eval_every:
            if wandb_log_callback is not None and last_train_metrics:
                payload = {f"train/{k}": float(v) for k, v in last_train_metrics.items()}
                payload["env_steps"] = int(total_env_steps)
                payload["steps_per_sec"] = total_env_steps / elapsed
                payload["num_updates_per_rollout"] = K
                wandb_log_callback(payload, total_env_steps)
            rng, k_eval = jax.random.split(rng)
            if eval_callback is not None:
                eval_callback(state, total_env_steps, k_eval)
            if ckpt_save_callback is not None:
                ckpt_save_callback(state, total_env_steps)
            last_eval_step = total_env_steps

    return state, env_state, rb_state, last_train_metrics
