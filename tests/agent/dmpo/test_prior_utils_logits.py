"""Smoke test for make_decoder_logits_fn — must return raw pre-tanh logits."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np


def test_make_decoder_logits_fn_returns_raw_logits():
    """The new fn returns logits of shape (..., 2*action_size) — concat of
    [mu_pretanh, log_std_pretanh] — without applying tanh.

    Uses non-zero input + non-zero params so that |mu| > 1 in some dim,
    which is the regime where tanh would be observable. A buggy
    implementation that applied tanh to mu would fail the allclose.
    """
    from brax.training.acme import running_statistics

    from track_mjx.agent.dmpo.obs_utils import DictRunningStatisticsState
    from track_mjx.agent.dmpo.prior_utils import make_decoder_logits_fn
    from track_mjx.agent.ff_ppo.intention_network import Decoder as ScamperDecoder

    proprio_size = 8
    latent_size = 4
    action_size = 6
    decoder_hidden = (16,)
    rng = jax.random.key(0)
    decoder = ScamperDecoder(layer_sizes=[*decoder_hidden, 2 * action_size])
    decoder_params = decoder.init(rng, jnp.zeros((1, latent_size + proprio_size)))[
        "params"
    ]
    proprio_norm = running_statistics.RunningStatisticsState(
        mean=jnp.zeros((proprio_size,)),
        std=jnp.ones((proprio_size,)),
        count=jnp.array(1_000_000.0),
        summed_variance=jnp.zeros((proprio_size,)),
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )
    target_norm = running_statistics.RunningStatisticsState(
        mean=jnp.zeros((1,)),
        std=jnp.ones((1,)),
        count=jnp.array(1_000_000.0),
        summed_variance=jnp.zeros((1,)),
        std_eps=1e-6,
        mode=running_statistics.NormalizationMode.WELFORD,
    )
    normalizer_params = DictRunningStatisticsState(
        imitation_target=target_norm, proprioception=proprio_norm
    )
    cfg = {
        "network_config": {
            "action_size": action_size,
            "intention_size": latent_size,
            "decoder_layer_sizes": list(decoder_hidden),
        }
    }

    fn = make_decoder_logits_fn(decoder_params, normalizer_params, cfg)
    # Non-zero input — drives the decoder into a regime where mu may exceed
    # |1|, exposing any accidental tanh application.
    latent_proprio = (
        jax.random.normal(jax.random.key(7), (3, latent_size + proprio_size)) * 3.0
    )
    logits, extras = fn(latent_proprio)

    # Shape check: (3, 2 * action_size).
    assert logits.shape == (3, 2 * action_size), logits.shape
    # Verify logits == decoder.apply directly (no postprocessing).
    expected, _ = decoder.apply({"params": decoder_params}, latent_proprio)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected), atol=1e-6)
    # Confirm the test actually traverses the |mu| > 1 regime — otherwise a
    # buggy tanh would not be observable. Bump seed/scale if this trips.
    mu = expected[..., :action_size]
    assert float(jnp.max(jnp.abs(mu))) > 1.0, (
        f"Test does not exercise the |mu|>1 regime where tanh would be "
        f"observable; max(|mu|)={float(jnp.max(jnp.abs(mu))):.4f}. Increase "
        f"input scale or change seed."
    )
    # Lock the extras dict contract so callers can rely on it.
    np.testing.assert_array_equal(np.asarray(extras["logits"]), np.asarray(logits))
