import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def env_spec():
    return {"obs_size": 53, "action_size": 37}


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def fixed_batch():
    """Deterministic mini-batch for parity tests. Shapes: [B, T, D]."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, _ = jax.random.split(key, 5)
    B, T = 4, 6
    return {
        "observation": jax.random.normal(k1, (B, T, 53)),
        "action": jax.random.uniform(k2, (B, T, 37), minval=-0.99, maxval=0.99),
        "reward": jax.random.normal(k3, (B, T)),
        "discount": jnp.ones((B, T)),
        "next_observation": jax.random.normal(k4, (B, T, 53)),
    }
