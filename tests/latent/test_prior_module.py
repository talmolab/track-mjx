"""Tests against the real Phase 1 v8 checkpoint at latent_prior_v8/best."""
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

V8_BEST = Path(
    os.path.expandvars(
        "$HOME/Desktop/SalkResearch/track-mjx/checkpoints/latent_prior_v8/best"
    )
)


@pytest.fixture(scope="module")
def prior():
    if not V8_BEST.exists():
        pytest.skip(f"v8 best ckpt not present at {V8_BEST}")
    from track_mjx.agent.latent_ppo.prior_module import FrozenLatentPrior

    return FrozenLatentPrior.from_dir(V8_BEST)


def test_prior_loads_v8_metadata(prior):
    assert prior.use_qvel is False  # v8 dropped qvel
    assert prior.n_joints == 67
    assert prior.feat_dim == 74  # 3 + 4 + 67
    assert prior.latent_dim == 60
    assert prior.window_len == 10
    assert prior.horizon == 5


def test_encode_window_shape(prior):
    rng = jax.random.PRNGKey(0)
    window = jax.random.normal(rng, (3, prior.window_len, prior.feat_dim))
    mean, logvar = prior.encode(window)
    assert mean.shape == (3, prior.latent_dim)
    assert logvar.shape == (3, prior.latent_dim)


def test_predict_shape(prior):
    z = jnp.zeros((2, prior.latent_dim))
    out = prior.predict(z)
    assert out.shape == (2, prior.horizon, prior.feat_dim)


def test_normalize_changes_input(prior):
    raw = jnp.ones((1, prior.window_len, prior.feat_dim))
    norm = prior.normalize(raw)
    # Normalizer is non-identity (mean!=0, std!=1) -- outputs differ from input.
    assert norm.shape == raw.shape
    assert not jnp.allclose(norm, raw)


def test_encode_then_predict_roundtrip(prior):
    """Realistic flow: encode a real window, predict future, get sensible shapes."""
    rng = jax.random.PRNGKey(1)
    window = jax.random.normal(rng, (4, prior.window_len, prior.feat_dim))
    mean, _ = prior.encode(window)
    pred = prior.predict(mean)
    assert pred.shape == (4, prior.horizon, prior.feat_dim)
    # Output should not be all zeros (predictor is not collapsed)
    assert float(jnp.std(pred)) > 1e-3
