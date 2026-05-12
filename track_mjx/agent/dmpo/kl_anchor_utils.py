"""KL utilities for the DMPO kl-anchor mode.

`pretanh_gaussian_kl` is a closed-form KL between two diagonal Gaussians
parameterized by (mu, log_std). After the 2026-05-06 KL-in-loss port,
this is consumed by ``_policy_loss_fn`` to add the anchor term
``-alpha * mean(exp(-w * KL))`` to the policy loss. Returns PER-SAMPLE
KL (shape ``(...,)``); callers aggregate (e.g., ``mean``) downstream.

`linear_decay_schedule` produces a callable `f(step) -> float` that
linearly decays from `init` to `floor` over the first `decay_frac` of
`total_steps`, then stays at `floor`.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def pretanh_gaussian_kl(
    mu_p: jnp.ndarray,
    log_std_p: jnp.ndarray,
    mu_q: jnp.ndarray,
    log_std_q: jnp.ndarray,
) -> jnp.ndarray:
    """KL(N(mu_p, sigma_p^2) || N(mu_q, sigma_q^2)), summed over the action
    dimension. Returns PER-SAMPLE KL of shape ``(...,)`` matching the leading
    (batch / time) dimensions of the inputs. Callers aggregate (mean, sum, …)
    downstream.

    Critical: this MUST return per-sample KL — `_policy_loss_fn` computes
    ``mean(exp(-w * KL))`` (Jensen-correct anchor reward), not
    ``exp(-w * mean(KL))`` (Jensen-biased). See SCAMPER reference at
    SCAMPER/scamper/agent/task_transfer/kl_anchor/kl_utils.py:14-37.
    """
    sigma_p_sq = jnp.exp(2.0 * log_std_p)
    sigma_q_sq = jnp.exp(2.0 * log_std_q)
    log_ratio = log_std_q - log_std_p
    term = (sigma_p_sq + (mu_p - mu_q) ** 2) / (2.0 * sigma_q_sq) - 0.5
    return jnp.sum(log_ratio + term, axis=-1)


def linear_decay_schedule(
    init: float, floor: float, decay_frac: float, total_steps: int
) -> Callable[[int], jnp.ndarray]:
    """Return f(step) -> scalar that linearly decays init->floor over decay_frac, then floors."""
    decay_steps = max(1, int(decay_frac * total_steps))

    def f(step):
        progress = jnp.minimum(jnp.float32(step) / jnp.float32(decay_steps), 1.0)
        return jnp.float32(init) + progress * jnp.float32(floor - init)

    return f
