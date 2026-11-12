"""Scheduled behavior-prior exploration for DMPO (SCAMPER's ``prior_schedule`` transfer mode).

SCAMPER (Sirbu et al., NeurIPS 2026, App. H.3.2 / Table 10) explores a sparse-reward task with
the **behavior prior** early in training and hands control to **residual control of the motion
prior** later. In its fixed-schedule variant the binary gate is not a policy output; it follows a
schedule over training:

    latent = gate * (mu_motion + residual) + (1 - gate) * (mu_behavior + sigma * eps)

with P(gate = 1) = p(t) ramping from ``p_start`` to ``p_end``. Table 10 reports success only when
control passes early (warmup 0.05, ramp 0.25 -> 9.94 treats) and failure for later switches or
none at all. The gate is a two-state Markov chain with stationary probability p and switching
timescale ``bout_tau`` env steps, so the rat moves in bouts instead of chattering at 100 Hz.
These match ``scamper/agent/task_transfer/arena_forage/wrappers.py::PriorScheduleDecoderHighLevelWrapper``.

**How this maps onto DMPO.** DMPO's policy is the distribution over MOTOR actions that the frozen
decoder emits at ``z = mu_motion + residual``; the critic is Q(s, a_motor). So the behavior-prior
bouts are simply a different *behavior policy* that writes motor actions into replay:

* the rollout executes ``a ~ N(decoder(mu_motion + residual))`` where gate = 1, and
  ``a ~ N(decoder(mu_behavior + sigma * eps))`` where gate = 0, and stores the EXECUTED action;
* the critic learns Q(s, a) from both kinds of transitions;
* the MPO E-step and M-step only ever sample and fit the motion-prior policy, at replayed states.

The learner is therefore unchanged. Two caveats, both inherent to the approach and recorded in
the training README:

* the n-step returns are not off-policy corrected, so while p < 1 the critic's target is the
  return of the gate MIXTURE, not of the motion-prior policy alone;
* evaluation (deterministic ``.mode()``) always runs the motion-prior policy, i.e. SCAMPER's
  ``eval_progress: 1.0``.

**One deviation from SCAMPER.** SCAMPER carries the gate across episodes in the env state. Here
the gate is re-drawn from its stationary distribution, Bernoulli(p), at the start of every
rollout (``unroll_length`` steps), then follows the Markov chain inside the rollout. P(gate = 1)
is exactly p at every step either way; only bouts longer than one rollout get an extra chance to
switch at the rollout boundary. This keeps the fused-step signature unchanged (no new carry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "DualPriorExplore",
    "gate_prob",
    "init_gate",
    "make_behavior_prior_explore_fn",
    "step_gate",
]


class DualPriorExplore(NamedTuple):
    """Everything the rollout needs for scheduled behavior-prior exploration.

    Attributes:
        explore_fn: ``(policy_params, obs, key) -> raw_action`` for ONE env (the rollout vmaps it).
            ``obs`` is the normalized observation dict; the result is a pre-tanh motor sample.
        p_start: P(gate = 1) before and at the start of the ramp.
        p_end: P(gate = 1) after the ramp.
        warmup_env_steps: env steps held at ``p_start``.
        ramp_env_steps: env steps spent ramping ``p_start`` -> ``p_end``.
        ramp_shape: ``"linear"`` or ``"cosine"``.
        bout_tau: switching timescale of the gate chain, in env steps (>= 1).
    """

    explore_fn: Callable
    p_start: float
    p_end: float
    warmup_env_steps: float
    ramp_env_steps: float
    ramp_shape: str
    bout_tau: float

    def validate(self) -> DualPriorExplore:
        """Raise on an out-of-range schedule; return self for chaining."""
        if not (0.0 <= self.p_start <= 1.0 and 0.0 <= self.p_end <= 1.0):
            raise ValueError(
                f"dual_prior p_start/p_end must be in [0, 1], got {self.p_start}/{self.p_end}"
            )
        if self.ramp_env_steps <= 0.0:
            raise ValueError(
                f"dual_prior ramp_env_steps must be > 0, got {self.ramp_env_steps}"
            )
        if self.warmup_env_steps < 0.0:
            raise ValueError(
                f"dual_prior warmup_env_steps must be >= 0, got {self.warmup_env_steps}"
            )
        if self.ramp_shape not in ("linear", "cosine"):
            raise ValueError(
                f"dual_prior ramp_shape must be 'linear' or 'cosine', got {self.ramp_shape!r}"
            )
        if self.bout_tau < 1.0:
            raise ValueError(
                f"dual_prior bout_tau must be >= 1 env step, got {self.bout_tau}"
            )
        return self


def gate_prob(t_env: jax.Array | float, sched: DualPriorExplore) -> jax.Array:
    """P(gate = 1), i.e. of motion-prior control, at env step ``t_env`` (traced f32 scalar)."""
    frac = jnp.clip(
        (jnp.asarray(t_env, jnp.float32) - sched.warmup_env_steps)
        / sched.ramp_env_steps,
        0.0,
        1.0,
    )
    if sched.ramp_shape == "cosine":
        frac = 0.5 - 0.5 * jnp.cos(jnp.pi * frac)
    return jnp.float32(sched.p_start + (sched.p_end - sched.p_start) * frac)


def init_gate(key: jax.Array, p: jax.Array, num_envs: int) -> jax.Array:
    """Per-env gate drawn from the chain's stationary distribution, Bernoulli(p); f32 0/1."""
    return (jax.random.uniform(key, (num_envs,)) < p).astype(jnp.float32)


def step_gate(
    prev_gate: jax.Array, p: jax.Array, bout_tau: float, key: jax.Array
) -> jax.Array:
    """Advance the two-state gate chain one step.

    P(0 -> 1) = p / tau and P(1 -> 0) = (1 - p) / tau, so the stationary P(gate = 1) is p, with
    mean bouts of tau / (1 - p) steps under motion-prior control and tau / p under the behavior
    prior. At p = 0 the chain is pinned to the behavior prior, at p = 1 to the motion prior.
    """
    p_on = jnp.clip(p / bout_tau, 0.0, 1.0)
    p_off = jnp.clip((1.0 - p) / bout_tau, 0.0, 1.0)
    switch_prob = jnp.where(prev_gate > 0.5, p_off, p_on)
    switched = jax.random.uniform(key, prev_gate.shape) < switch_prob
    return jnp.where(switched, 1.0 - prev_gate, prev_gate)


def make_behavior_prior_explore_fn(
    *,
    behavior_prior_params: dict,
    behavior_prior_layer_sizes: Sequence[int],
    decoder_layer_sizes: Sequence[int],
    latent_size: int,
    action_size: int,
    behavior_logvar: float | None,
) -> Callable:
    """Build ``explore_fn(policy_params, obs, key) -> raw_action`` for one env.

    ``z = mu_behavior(proprio) + sigma * eps``, with ``sigma = exp(behavior_logvar / 2)`` (SCAMPER's
    fixed log-variance; ``None`` uses the behavior prior's own predicted log-variance). The action
    is then sampled from the decoder's Gaussian at ``[z, proprio]`` exactly as the DMPO policy
    samples it (``loc = mu``, ``scale = softplus(log_std) + 1e-3``), so the only difference from
    a policy step is where ``z`` comes from.

    The decoder parameters are read from ``policy_params['params']['decoder']`` at call time, so
    the explore path always decodes with the policy's own (frozen) decoder. The behavior prior's
    parameters are frozen and captured here.

    Args:
        behavior_prior_params: inner flax params of SCAMPER's ``rgm`` network.
        behavior_prior_layer_sizes: its hidden sizes (``rgm_layer_sizes``).
        decoder_layer_sizes: the decoder's hidden sizes (without the output layer).
        latent_size: latent dimensionality (16).
        action_size: motor action dimensionality.
        behavior_logvar: fixed log-variance of the latent sample, or ``None``.

    Returns:
        The per-env explore function.
    """
    # Imported here so rollout.py (which needs only the gate helpers) does not
    # pull in the network stack.
    from track_mjx.agent.dmpo.networks_kl_anchor import _DecoderModule, _PriorModule

    bp_module = _PriorModule(
        layer_sizes=tuple(behavior_prior_layer_sizes), latents=int(latent_size)
    )
    decoder_module = _DecoderModule(
        layer_sizes=(*tuple(decoder_layer_sizes), 2 * int(action_size))
    )
    bp_variables = {"params": behavior_prior_params}

    def explore_fn(policy_params, obs, key):
        proprio = obs["proprioception"]
        mu_b, logvar_b = bp_module.apply(bp_variables, proprio)
        if behavior_logvar is None:
            std_b = jnp.exp(0.5 * logvar_b)
        else:
            std_b = jnp.exp(0.5 * jnp.float32(behavior_logvar))
        k_z, k_a = jax.random.split(key)
        z = mu_b + std_b * jax.random.normal(k_z, mu_b.shape, mu_b.dtype)
        out = decoder_module.apply(
            {"params": policy_params["params"]["decoder"]},
            jnp.concatenate([z, proprio], axis=-1),
        )
        mu = out[..., :action_size]
        scale = jax.nn.softplus(out[..., action_size:]) + 1e-3
        return mu + scale * jax.random.normal(k_a, mu.shape, mu.dtype)

    return explore_fn
