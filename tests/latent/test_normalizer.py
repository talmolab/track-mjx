import numpy as np
import jax.numpy as jnp

from track_mjx.agent.latent_ppo.data.normalizer import (
    FeatureNormalizer,
    fit_normalizer,
)


def test_fit_normalizer_stats_match_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 8)).astype(np.float32) * 3.0 + 1.0
    norm = fit_normalizer(x)
    np.testing.assert_allclose(np.asarray(norm.mean), x.mean(0), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(norm.std), x.std(0), rtol=1e-5)


def test_normalize_then_denormalize_is_identity():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    norm = fit_normalizer(x)
    z = norm.apply(jnp.asarray(x))
    x2 = norm.invert(z)
    np.testing.assert_allclose(np.asarray(x2), x, atol=1e-5)


def test_normalizer_handles_zero_variance_dim():
    x = np.zeros((10, 3), dtype=np.float32)
    x[:, 0] = 1.0  # constant column => std=0
    norm = fit_normalizer(x)
    # Should not produce NaNs after division
    z = norm.apply(jnp.asarray(x))
    assert jnp.all(jnp.isfinite(z))
