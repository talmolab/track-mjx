import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo.kl_anchor_utils import (
    pretanh_gaussian_kl,
    linear_decay_schedule,
)


def test_pretanh_gaussian_kl_zero_when_identical():
    mu = jnp.array([[0.1, -0.2, 0.3]])
    log_std = jnp.array([[-1.0, -0.5, 0.0]])
    kl = pretanh_gaussian_kl(mu, log_std, mu, log_std)
    # Per-sample shape (1,); compare elementwise.
    np.testing.assert_allclose(np.asarray(kl), [0.0], atol=1e-6)


def test_pretanh_gaussian_kl_positive_when_different():
    mu_p = jnp.array([[0.1, -0.2, 0.3]])
    log_std_p = jnp.array([[-1.0, -0.5, 0.0]])
    mu_q = jnp.array([[0.5, 0.5, 0.5]])
    log_std_q = jnp.array([[-0.5, -1.0, 0.5]])
    kl = pretanh_gaussian_kl(mu_p, log_std_p, mu_q, log_std_q)
    assert float(kl[0]) > 0


def test_pretanh_gaussian_kl_matches_analytic_scalar():
    mu_p, sigma_p = 0.3, 0.5
    mu_q, sigma_q = -0.1, 1.2
    expected = (
        np.log(sigma_q / sigma_p)
        + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
        - 0.5
    )
    kl = pretanh_gaussian_kl(
        jnp.array([[mu_p]]),
        jnp.array([[np.log(sigma_p)]]),
        jnp.array([[mu_q]]),
        jnp.array([[np.log(sigma_q)]]),
    )
    # Per-sample shape (1,); compare element 0.
    np.testing.assert_allclose(float(kl[0]), expected, atol=1e-6)


def test_linear_decay_schedule_endpoints():
    sched = linear_decay_schedule(init=1.0, floor=0.05, decay_frac=0.3, total_steps=1000)
    assert abs(float(sched(0)) - 1.0) < 1e-6
    assert abs(float(sched(300)) - 0.05) < 1e-6
    assert abs(float(sched(500)) - 0.05) < 1e-6
    mid = float(sched(150))
    assert 0.5 < mid < 0.55


def test_pretanh_gaussian_kl_returns_per_sample_not_scalar_mean():
    """Multi-batch input must return shape (B,) — per-sample KL.

    The KL-in-loss policy term computes mean(exp(-w * KL)) (Jensen-correct);
    if the helper averaged over the batch first, the math would silently
    desync from the SCAMPER reference (Jensen-biased).
    """
    # Two batch elements: first is identical (KL=0), second is different (KL>0).
    mu_p = jnp.array([[0.0, 0.0], [0.5, -0.3]])
    log_std_p = jnp.array([[0.0, 0.0], [0.1, 0.2]])
    mu_q = jnp.array([[0.0, 0.0], [-0.5, 0.3]])
    log_std_q = jnp.array([[0.0, 0.0], [-0.1, -0.2]])
    kl = pretanh_gaussian_kl(mu_p, log_std_p, mu_q, log_std_q)
    # Shape must be (2,) — per-sample. NOT scalar.
    assert kl.shape == (2,), kl.shape
    # First sample is identical → KL = 0; second is different → KL > 0.
    np.testing.assert_allclose(float(kl[0]), 0.0, atol=1e-6)
    assert float(kl[1]) > 0
