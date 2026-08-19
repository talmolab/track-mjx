"""The MPO temperature runs away when Q is flat, and the floor is now settable.

Background, measured on arm_m1_sparse_gaponly (sparse gap-crossing reward):

    step    log_temperature   kl_q_rel   critic_loss   crossings/ep
    10.1M      -4.06           0.966       0.191          0.221
    20.8M      -9.31           0.441       0.0047         0.260
    31.4M     -18.00 (FLOOR)   0.00002     0.000002       0.030

`kl_q_rel` is the realized KL(q||pi) over epsilon. Once it reaches ~0 the E-step
returns the policy it was given, so the M-step regresses the policy onto itself
and learning stops. The dense reference arm_i1_nstep100_proprio sits at
log_temperature -0.54 with kl_q_rel ~1.0 for its whole 300M.

These tests pin (1) WHY the runaway happens, so the mechanism cannot be quietly
edited away, and (2) that exposing the floor did not disturb any existing arm.
"""

import jax
import jax.numpy as jnp
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.losses import (
    MPOParams,
    clip_mpo_params,
    compute_weights_and_temperature_loss,
)

EPSILON = 0.1
N_ACTIONS, BATCH = 20, 8


def _params(log_t, log_am=0.0, log_as=0.0, penalty=None):
    return MPOParams(
        log_temperature=jnp.full([1], log_t),
        log_alpha_mean=jnp.full([1], log_am),
        log_alpha_stddev=jnp.full([1], log_as),
        log_penalty_temperature=None if penalty is None else jnp.full([1], penalty),
    )


# --------------------------------------------------------------------------
# 1. The runaway itself.
# --------------------------------------------------------------------------

def test_flat_q_gives_a_constant_positive_temperature_gradient():
    """With Q identical across sampled actions, d(loss)/d(t) == epsilon for ALL t.

    loss = t * (epsilon + mean(logsumexp(Q/t)) - log N).
    If every Q is equal then logsumexp(Q/t) = Q/t + log N EXACTLY, so
    loss = t*epsilon + mean(Q) and the derivative is epsilon -- independent of t
    and never negative. Nothing stops the decay; it runs to the clip floor.
    """
    q_flat = jnp.full((N_ACTIONS, BATCH), 3.0)

    def loss(t):
        _, lt = compute_weights_and_temperature_loss(q_flat, EPSILON, t)
        return jnp.mean(lt)

    grad = jax.grad(loss)
    # 1e-4 is the smallest t at which float32 still resolves the gradient: the
    # loss is t*(Q/t + epsilon), so by t~1e-6 the Q/t term is ~1e6 and the
    # epsilon contribution is below the representable difference. See
    # test_gradient_underflows_at_the_floor for the other side of that.
    for t in (10.0, 1.0, 1e-2, 1e-4):
        assert float(grad(jnp.array(t))) == pytest.approx(EPSILON, rel=1e-2), (
            f"at t={t} the gradient should be exactly epsilon; if this fails the "
            "runaway mechanism has changed and the sparse-arm analysis is stale"
        )


def test_gradient_is_numerically_junk_near_the_floor_and_never_turns_negative():
    """Below t ~ 1e-5 the float32 gradient is meaningless -- and still points DOWN.

    The loss is t * (mean(logsumexp(Q/t)) + epsilon - log N). As t shrinks, Q/t
    grows without bound and the epsilon term is lost to cancellation. Measured
    with Q flat at 3.0, N=20:

        t = 3.35e-04 (log -8)   grad = +0.0996   <- correct, == epsilon
        t = 6.14e-06 (log -12)  grad = +0.0313   <- already wrong by 3x
        t = 1.13e-07 (log -16)  grad = +6.0      <- junk
        t = 1.52e-08 (log -18)  grad = +80.0     <- junk, 800x epsilon

    The sign is what matters: it never goes negative, so there is no numerical
    route back up once the temperature is pinned. That is why the floor has to
    be raised rather than waited out, and it is consistent with
    arm_m1_sparse_gaponly sitting at exactly -18.000000 once it arrived.
    """
    q_flat = jnp.full((N_ACTIONS, BATCH), 3.0)

    def loss(t):
        _, lt = compute_weights_and_temperature_loss(q_flat, EPSILON, t)
        return jnp.mean(lt)

    g_ok = float(jax.grad(loss)(jax.nn.softplus(jnp.array(-8.0))))
    assert g_ok == pytest.approx(EPSILON, rel=1e-2), "at log_t=-8 float32 is still fine"

    g_floor = float(jax.grad(loss)(jax.nn.softplus(jnp.array(-18.0))))
    assert g_floor > 0.0, "the gradient must still point DOWN -- no way back up"
    assert g_floor > 10 * EPSILON, (
        f"expected the float32 gradient to be junk at the floor, got {g_floor}"
    )


def test_spread_q_restores_a_downward_gradient_at_small_t():
    """The restoring force that flat Q removes: epsilon - log N < 0 as t -> 0."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (N_ACTIONS, BATCH))

    def loss(t):
        _, lt = compute_weights_and_temperature_loss(q, EPSILON, t)
        return jnp.mean(lt)

    g = float(jax.grad(loss)(jnp.array(1e-4)))
    assert g < 0, "with spread Q a tiny temperature must be pushed back UP"
    # small-t limit is epsilon - log N (the mean(max Q) term stops depending on t)
    assert g == pytest.approx(EPSILON - jnp.log(N_ACTIONS), abs=0.05)


def test_flat_q_weights_are_uniform_at_every_temperature():
    """Why a HIGHER floor is safe: at flat Q the floor changes nothing at all."""
    q_flat = jnp.full((N_ACTIONS, BATCH), -2.0)
    for t in (1.0, 1e-8):
        w, _ = compute_weights_and_temperature_loss(q_flat, EPSILON, jnp.array(t))
        assert jnp.allclose(w, 1.0 / N_ACTIONS, atol=1e-6)


def test_tiny_q_spread_becomes_argmax_only_at_a_low_floor():
    """Why a LOW floor is dangerous: 1/t amplifies critic noise into an argmax.

    This is the concrete harm the floor guards against -- and the reason scaling
    the reward UP is equivalent to lowering the floor, since the floor is in
    absolute temperature while Q scales with the reward weight.
    """
    q = jnp.zeros((N_ACTIONS, BATCH)).at[3].set(1e-6)      # 1e-6 of "noise"
    w_hi, _ = compute_weights_and_temperature_loss(q, EPSILON, jnp.exp(jnp.array(-4.0)))
    w_lo, _ = compute_weights_and_temperature_loss(q, EPSILON, jnp.exp(jnp.array(-18.0)))
    assert float(w_hi[3].mean()) == pytest.approx(1.0 / N_ACTIONS, abs=1e-3), (
        "at a sane temperature 1e-6 of noise must not move the weights"
    )
    assert float(w_lo[3].mean()) > 0.99, (
        "at the -18 floor the same 1e-6 of noise takes essentially all the weight"
    )


# --------------------------------------------------------------------------
# 2. The knob.
# --------------------------------------------------------------------------

def test_default_is_bit_identical_to_the_old_behaviour():
    p = _params(-50.0, -50.0, -50.0, penalty=-50.0)
    c = clip_mpo_params(p)
    assert float(c.log_temperature[0]) == -18.0
    assert float(c.log_alpha_mean[0]) == -18.0
    assert float(c.log_alpha_stddev[0]) == -18.0
    assert float(c.log_penalty_temperature[0]) == -18.0


@pytest.mark.parametrize("floor", [-8.0, -4.0, 0.0])
def test_explicit_floor_applies_to_both_temperatures(floor):
    c = clip_mpo_params(_params(-50.0, penalty=-50.0), floor)
    assert float(c.log_temperature[0]) == floor
    assert float(c.log_penalty_temperature[0]) == floor


def test_floor_does_not_touch_the_alpha_duals():
    """alpha_mean / alpha_stddev keep the -18 Acme floor; only temperature moves.

    They are a different constraint (the M-step KL trust region) and are not
    implicated in the runaway, so widening them would be an unrelated change.
    """
    c = clip_mpo_params(_params(-50.0, -50.0, -50.0), -4.0)
    assert float(c.log_temperature[0]) == -4.0
    assert float(c.log_alpha_mean[0]) == -18.0
    assert float(c.log_alpha_stddev[0]) == -18.0


def test_floor_never_raises_a_healthy_temperature():
    """The dense reference lives at -0.54; a -8 floor must leave it alone."""
    c = clip_mpo_params(_params(-0.537), -8.0)
    assert float(c.log_temperature[0]) == pytest.approx(-0.537)


def test_config_default_preserves_every_existing_arm():
    assert DMPOConfig().min_log_temperature == -18.0


def test_penalty_temperature_stays_absent_when_it_was_absent():
    """action_penalization=False arms have no penalty dual; do not invent one."""
    assert clip_mpo_params(_params(-50.0), -4.0).log_penalty_temperature is None
