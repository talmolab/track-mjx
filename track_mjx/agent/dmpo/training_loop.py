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
from typing import TYPE_CHECKING, Any

import jax

from track_mjx.agent.dmpo.config import DMPOConfig, realized_ratios
from track_mjx.agent.dmpo.train_dmpo_chunk import make_train_chunk
from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

if TYPE_CHECKING:
    from collections.abc import Callable

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
    max_chunks: int | None = None,
    eval_callback: Callable[[Any, int, jax.Array], None] | None = None,
    wandb_log_callback: Callable[[dict, int], None] | None = None,
    ckpt_mgr: Any = None,
    ckpt_save_callback: Callable[[Any, int], None] | None = None,
    cfg_dict: dict | None = None,
    log_every_steps: int | None = None,
    eval_every_steps: int | None = None,
    num_timesteps: int | None = None,
    extra_state_extras: tuple = (),
    start_env_steps: int = 0,
    frozen_behavior_params: Any = None,
    dual_prior: Any = None,
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
        when ckpt_mgr is non-None, and once more at loop exit when the last
        save was not at the final env-step count.
      cfg_dict: snapshotted hydra config dict for checkpoint metadata.
      log_every_steps, eval_every_steps, num_timesteps: pulled from
        ``cfg`` if not provided. Argument form lets test override.
      frozen_behavior_params: optional frozen policy params for behavior
        mixing, passed to ``make_fused_train_step``.
      dual_prior: optional ``dual_prior.DualPriorExplore`` for scheduled
        behavior-prior exploration, passed to ``make_fused_train_step``.

    Returns:
      (final_state, final_env_state, final_rb_state, last_train_metrics)
    """
    eval_every = (
        eval_every_steps if eval_every_steps is not None else cfg.eval_every_steps
    )
    timesteps = num_timesteps if num_timesteps is not None else cfg.num_timesteps

    # Recurrent policy head: the fused step / chunk builds grow a
    # policy_hidden carry (see make_fused_train_step). The hidden is
    # TRANSIENT — never checkpointed, starts None here on fresh AND resumed
    # runs alike (the rollout zero-inits when None; on resume env_state is
    # also None, so episodes restart from reset and zero hidden is the
    # correct pairing, not an approximation).
    recurrent = getattr(nets, "recurrent_meta", None) is not None
    policy_hidden = None

    fused_step = make_fused_train_step(
        env,
        nets,
        optimizers,
        rb,
        cfg,
        K=K,
        extra_state_extras=extra_state_extras,
        frozen_behavior_params=frozen_behavior_params,
        dual_prior=dual_prior,
    )
    train_chunk = make_train_chunk(
        fused_step, n_iters=iters_per_chunk, recurrent=recurrent
    )

    env_steps_per_chunk = cfg.num_envs * cfg.unroll_length * iters_per_chunk
    last_train_metrics: dict = {}
    # Resume support. Checkpoints are SAVED at `step=total_env_steps`
    # (`ckpt_save_callback(state, total_env_steps)`), so the checkpoint's own
    # step number IS the env-step count -- the caller passes `mgr.latest_step()`
    # here after a successful restore. Previously this was hard-coded to 0, so a
    # resumed run restored the params but restarted the env-step counter: it
    # would train `num_timesteps` MORE steps, re-save checkpoints under
    # already-used step numbers, and log a wandb x-axis that jumped backwards.
    # Default 0 keeps every fresh run bit-identical.
    total_env_steps = int(start_env_steps)
    last_eval_step = int(start_env_steps)
    # Env-step count of the newest checkpoint. A resumed run starts at the step
    # it restored from, which already exists on disk and must not be re-saved.
    last_ckpt_step = int(start_env_steps)
    chunks_completed = 0
    first_step = True
    second_step = True
    t0 = time.time()
    # Wall-clock spent in eval / checkpoint callbacks, so the train-vs-eval
    # share can be read from the log.
    cum_eval_wall_s = 0.0
    cum_ckpt_wall_s = 0.0

    while True:
        # Stop conditions.
        if max_chunks is not None and chunks_completed >= max_chunks:
            break
        if max_chunks is None and total_env_steps >= timesteps:
            break

        rng, k_step = jax.random.split(rng)

        # Warm-up branch — single fused_step at a time. SGD must not run
        # while the replay buffer cannot yet be sampled; the lax.select
        # gate inside fused_step makes this safe.
        if not warmup_done:
            if first_step:
                log.info("Compiling fused_step (reset path)...")
                t1 = time.time()
                if recurrent:
                    state, env_state, policy_hidden, rb_state, metrics = fused_step(
                        state, env_state, policy_hidden, rb_state, k_step
                    )
                else:
                    state, env_state, rb_state, metrics = fused_step(
                        state, env_state, rb_state, k_step
                    )
                jax.block_until_ready(metrics["policy_loss"])
                log.info("fused_step (reset) compiled in %.1fs", time.time() - t1)
                first_step = False
            else:
                if recurrent:
                    state, env_state, policy_hidden, rb_state, metrics = fused_step(
                        state, env_state, policy_hidden, rb_state, k_step
                    )
                else:
                    state, env_state, rb_state, metrics = fused_step(
                        state, env_state, rb_state, k_step
                    )
            last_train_metrics = metrics
            total_env_steps += cfg.num_envs * cfg.unroll_length
            if not bool(rb.can_sample(rb_state)):
                log.info(
                    "warming replay: %d env steps (need ~%d)",
                    total_env_steps,
                    cfg.min_replay_size,
                )
                continue
            warmup_done = True
            log.info("replay warm-up complete at %d env steps", total_env_steps)
            continue

        # Steady state — one chunk = N fused_step iters in one dispatch.
        if recurrent and policy_hidden is None:
            # Only reachable with warmup_done=True (tests pre-filling the
            # replay): the warm-up branch above otherwise materializes the
            # hidden. The scan carry needs a concrete pytree (structure
            # cannot change None -> tuple mid-scan), and zeros is the same
            # fresh-reset pairing the rollout itself would use.
            policy_hidden = nets.recurrent_meta.init_hidden(cfg.num_envs)
        if second_step:
            log.info("Compiling train_chunk (n_iters=%d)...", iters_per_chunk)
            t1 = time.time()
            if recurrent:
                state, env_state, policy_hidden, rb_state, metrics = train_chunk(
                    state, env_state, policy_hidden, rb_state, k_step
                )
            else:
                state, env_state, rb_state, metrics = train_chunk(
                    state, env_state, rb_state, k_step
                )
            jax.block_until_ready(metrics["policy_loss"])
            log.info("train_chunk compiled in %.1fs", time.time() - t1)
            second_step = False
        else:
            if recurrent:
                state, env_state, policy_hidden, rb_state, metrics = train_chunk(
                    state, env_state, policy_hidden, rb_state, k_step
                )
            else:
                state, env_state, rb_state, metrics = train_chunk(
                    state, env_state, rb_state, k_step
                )
        last_train_metrics = metrics
        total_env_steps += env_steps_per_chunk
        chunks_completed += 1

        elapsed = max(time.time() - t0, 1e-6)
        # Throughput must count only the steps THIS process ran; on a resumed
        # run `total_env_steps` starts at the checkpoint's value while `elapsed`
        # starts at 0, so the cumulative ratio would report a fictitious spike.
        session_sps = (total_env_steps - int(start_env_steps)) / elapsed
        log.info(
            "chunk env_steps=%d steps_per_sec=%.0f policy_loss=%.4g critic_loss=%.4g",
            int(total_env_steps),
            session_sps,
            float(last_train_metrics.get("policy_loss", 0.0)),
            float(last_train_metrics.get("critic_loss", 0.0)),
        )
        if "schedule/dual_prior_p" in last_train_metrics:
            # Dual-prior arms only, so every other arm's log stays line-identical.
            log.info(
                "dual_prior env_steps=%d p=%.4f gate_frac=%.4f",
                int(total_env_steps),
                float(last_train_metrics["schedule/dual_prior_p"]),
                float(last_train_metrics["schedule/dual_prior_gate_frac"]),
            )

        # Eval cadence — emits training metrics to wandb, runs eval
        # rollout / render via callbacks, saves checkpoint.
        if total_env_steps - last_eval_step >= eval_every:
            if wandb_log_callback is not None and last_train_metrics:
                payload = {
                    f"train/{k}": float(v) for k, v in last_train_metrics.items()
                }
                payload["env_steps"] = int(total_env_steps)
                payload["steps_per_sec"] = session_sps
                # PPO-convention alias so the two projects' throughput panels
                # overlay. NOTE the known track-mjx caveat: PPO's `training/sps`
                # is inflated by num_resets_per_eval, whereas this one is a plain
                # env_steps/wall-clock. This number is the honest one; do not
                # "correct" it upward to match a PPO curve.
                payload["training/sps"] = session_sps
                payload["num_updates_per_rollout"] = K
                # L0: cumulative callback wall-clock up to the PREVIOUS eval (this eval runs below)
                payload["timing/eval_wall_s_cum"] = cum_eval_wall_s
                payload["timing/ckpt_wall_s_cum"] = cum_ckpt_wall_s
                payload["timing/eval_ckpt_frac"] = (
                    cum_eval_wall_s + cum_ckpt_wall_s
                ) / elapsed
                # Learner-throughput ratios, MEASURED rather than configured (the
                # configured samples_per_insert knob is inverted in the live entry
                # points; see DMPOConfig.sgd_steps_per_rollout). The Ray run that
                # solves this task realizes 3.236 samples/insert; this port 0.5.
                payload.update(
                    {
                        f"replay/{k}": float(v)
                        for k, v in realized_ratios(cfg, K).items()
                    }
                )
                wandb_log_callback(payload, total_env_steps)
            rng, k_eval = jax.random.split(rng)
            t_eval = time.time()
            if eval_callback is not None:
                eval_callback(state, total_env_steps, k_eval)
            eval_wall_s = time.time() - t_eval
            t_ckpt = time.time()
            if ckpt_save_callback is not None:
                ckpt_save_callback(state, total_env_steps)
                last_ckpt_step = total_env_steps
            ckpt_wall_s = time.time() - t_ckpt
            cum_eval_wall_s += eval_wall_s
            cum_ckpt_wall_s += ckpt_wall_s
            wall_total = max(time.time() - t0, 1e-6)
            log.info(
                "timing env_steps=%d eval_wall_s=%.1f ckpt_wall_s=%.1f | cumulative eval=%.0fs ckpt=%.0fs "
                "of %.0fs session wall (eval+ckpt share %.1f%%)",
                int(total_env_steps),
                eval_wall_s,
                ckpt_wall_s,
                cum_eval_wall_s,
                cum_ckpt_wall_s,
                wall_total,
                100.0 * (cum_eval_wall_s + cum_ckpt_wall_s) / wall_total,
            )
            last_eval_step = total_env_steps

    # Checkpoints are otherwise written only at eval cadence, so a budget that is
    # not a multiple of the cadence would leave the final stretch of training
    # unsaved. Write one more at the exact final step, unless the last periodic
    # save already landed there.
    if ckpt_save_callback is not None and last_ckpt_step != total_env_steps:
        ckpt_save_callback(state, total_env_steps)
        log.info("final checkpoint saved at env_steps=%d", int(total_env_steps))

    return state, env_state, rb_state, last_train_metrics
