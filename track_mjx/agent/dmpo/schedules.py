"""Warm-start transition schedules, derived inside jit from the SGD counter.

Why from ``state.steps`` and not the host env-step counter: total_env_steps is
a Python int in training_loop.py that never enters the jitted fused step, and
widening the fused-step signature would break its documented contract
(train_dmpo_chunk.py) and force retraces. ``state.steps`` is already traced,
already checkpointed, and already drives one schedule (the kl_anchor w-decay,
learner.py). The env-step estimate ``steps * num_envs * unroll / K`` ignores
the replay warm-up offset (SGD is gated off while env steps accrue for ~2
rollouts) — <0.5% of any schedule length used here.

All functions are pure jnp on traced scalars; with the config knobs at their
defaults they are never called (the callers branch at trace time).
"""
from __future__ import annotations

import jax.numpy as jnp

from track_mjx.agent.dmpo.config import DMPOConfig


def env_steps_estimate(steps, cfg: DMPOConfig, K: int):
    """Estimated env steps as a traced f32 scalar, from the SGD counter."""
    per_sgd = cfg.num_envs * cfg.unroll_length / float(K)
    return steps.astype(jnp.float32) * per_sgd


def reward_anneal_lambda(t, cfg: DMPOConfig):
    """Dense-component weight: linear 1 -> 0 over [0, reward_anneal_env_steps]."""
    n = float(cfg.reward_anneal_env_steps)
    if n <= 0:
        # Degenerate config: a set sparse key with no schedule means
        # "sparse-only immediately" (lambda == 0 everywhere).
        return jnp.float32(0.0)
    return jnp.clip(1.0 - t / n, 0.0, 1.0).astype(jnp.float32)


def behavior_mix_frac(t, cfg: DMPOConfig):
    """Frozen-policy env fraction: init until hold, linear to 0 at end."""
    init = float(cfg.behavior_mix_init)
    hold = float(cfg.behavior_mix_hold_env_steps)
    end = float(cfg.behavior_mix_end_env_steps)
    if end <= hold:
        # No decay window configured: hard step from init to 0 at `hold`.
        return jnp.where(t < hold, init, 0.0).astype(jnp.float32)
    decay = 1.0 - (t - hold) / (end - hold)
    return (init * jnp.clip(decay, 0.0, 1.0)).astype(jnp.float32)
