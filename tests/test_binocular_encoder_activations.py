"""Tests for per-layer CNN activation extraction in BinocularVisionEncoder."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from track_mjx.agent.ff_ppo.binocular_vision_encoder import BinocularVisionEncoder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEATURE_SIZE = 32
CHANNELS = (4, 8, 16, 32)
MONO_CHANNELS = 1
BATCH = 2
H, W = 32, 32
# Input shape: (batch, H, W, 2*mono_channels)
INPUT_SHAPE = (BATCH, H, W, 2 * MONO_CHANNELS)


def make_encoder(shared_weights: bool = True) -> BinocularVisionEncoder:
    return BinocularVisionEncoder(
        feature_size=FEATURE_SIZE,
        channels=CHANNELS,
        mono_channels=MONO_CHANNELS,
        shared_weights=shared_weights,
    )


def init_params(encoder: BinocularVisionEncoder, images: jnp.ndarray):
    rng = jax.random.PRNGKey(0)
    return encoder.init(rng, images)


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_default_returns_single_array(self):
        """Default call returns a single jnp.ndarray with shape (batch, 2*feature_size)."""
        encoder = make_encoder()
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)
        out = encoder.apply(params, images)
        assert isinstance(out, jnp.ndarray), "Default output should be jnp.ndarray"
        assert out.shape == (BATCH, 2 * FEATURE_SIZE), (
            f"Expected shape ({BATCH}, {2 * FEATURE_SIZE}), got {out.shape}"
        )

    def test_get_activation_false_same(self):
        """Explicit get_activation=False produces output identical to default."""
        encoder = make_encoder()
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)
        out_default = encoder.apply(params, images)
        out_false = encoder.apply(params, images, get_activation=False)
        assert isinstance(out_false, jnp.ndarray), (
            "get_activation=False output should be jnp.ndarray"
        )
        np.testing.assert_array_equal(out_default, out_false)


# ---------------------------------------------------------------------------
# TestBinocularActivations
# ---------------------------------------------------------------------------


class TestBinocularActivations:
    def test_returns_tuple(self):
        """get_activation=True returns a tuple of length 2."""
        encoder = make_encoder()
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)
        result = encoder.apply(params, images, get_activation=True)
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, f"Tuple should have length 2, got {len(result)}"

    def test_features_unchanged(self):
        """Features from get_activation=True match the default (no-activation) output."""
        encoder = make_encoder()
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)
        out_default = encoder.apply(params, images)
        features, _ = encoder.apply(params, images, get_activation=True)
        np.testing.assert_array_equal(out_default, features)

    def test_left_right_keys(self):
        """The activations dict has exactly 'left' and 'right' keys."""
        encoder = make_encoder()
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)
        _, acts = encoder.apply(params, images, get_activation=True)
        assert isinstance(acts, dict), "Activations should be a dict"
        assert set(acts.keys()) == {"left", "right"}, (
            f"Expected keys {{'left', 'right'}}, got {set(acts.keys())}"
        )

    def test_per_eye_conv_layers(self):
        """Each eye's activation dict has conv_0..conv_3 and fc with correct shapes.

        For 32x32 input with channels=(4,8,16,32) and strides=(1,1,2,2):
          conv_0: (batch, 30, 30, 4)   stride=1, VALID, 3x3 -> 32-2=30
          conv_1: (batch, 28, 28, 8)   stride=1, VALID, 3x3 -> 30-2=28
          conv_2: (batch, 13, 13, 16)  stride=2, VALID, 3x3 -> (28-2)//2=13
          conv_3: (batch,  6,  6, 32)  stride=2, VALID, 3x3 -> (13-2)//2≈5? let's check
          fc:     (batch, feature_size)
        """
        encoder = make_encoder()
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)
        _, acts = encoder.apply(params, images, get_activation=True)

        for eye in ("left", "right"):
            eye_acts = acts[eye]
            assert set(eye_acts.keys()) == {"conv_0", "conv_1", "conv_2", "conv_3", "fc"}, (
                f"Eye '{eye}' missing expected layer keys. Got: {set(eye_acts.keys())}"
            )
            # conv_0: stride 1, VALID, 3x3 kernel on 32x32 -> 30x30
            assert eye_acts["conv_0"].shape == (BATCH, 30, 30, 4), (
                f"conv_0 shape mismatch for eye '{eye}': {eye_acts['conv_0'].shape}"
            )
            # conv_1: stride 1, VALID, 3x3 kernel on 30x30 -> 28x28
            assert eye_acts["conv_1"].shape == (BATCH, 28, 28, 8), (
                f"conv_1 shape mismatch for eye '{eye}': {eye_acts['conv_1'].shape}"
            )
            # conv_2: stride 2, VALID, 3x3 kernel on 28x28 -> 13x13
            assert eye_acts["conv_2"].shape == (BATCH, 13, 13, 16), (
                f"conv_2 shape mismatch for eye '{eye}': {eye_acts['conv_2'].shape}"
            )
            # conv_3: stride 2, VALID, 3x3 kernel on 13x13 -> 6x6
            assert eye_acts["conv_3"].shape == (BATCH, 6, 6, 32), (
                f"conv_3 shape mismatch for eye '{eye}': {eye_acts['conv_3'].shape}"
            )
            # fc: (batch, feature_size)
            assert eye_acts["fc"].shape == (BATCH, FEATURE_SIZE), (
                f"fc shape mismatch for eye '{eye}': {eye_acts['fc'].shape}"
            )

    def test_shared_weights_same_for_same_input(self):
        """With identical left/right images + shared_weights=True, activations match."""
        encoder = make_encoder(shared_weights=True)
        # Build input where left and right channels are identical
        single_eye = jax.random.uniform(jax.random.PRNGKey(42), (BATCH, H, W, MONO_CHANNELS))
        images = jnp.concatenate([single_eye, single_eye], axis=-1)

        params = init_params(encoder, images)
        _, acts = encoder.apply(params, images, get_activation=True)

        for layer_key in ("conv_0", "conv_1", "conv_2", "conv_3", "fc"):
            np.testing.assert_allclose(
                acts["left"][layer_key],
                acts["right"][layer_key],
                atol=1e-5,
                err_msg=f"Shared-weight activations differ for {layer_key}",
            )


# ---------------------------------------------------------------------------
# TestIndependentWeights
# ---------------------------------------------------------------------------


class TestIndependentWeights:
    def test_independent_returns_activations(self):
        """shared_weights=False also works and returns the correct structure."""
        encoder = make_encoder(shared_weights=False)
        images = jnp.ones(INPUT_SHAPE)
        params = init_params(encoder, images)

        # Default: no activations, correct shape
        out = encoder.apply(params, images)
        assert out.shape == (BATCH, 2 * FEATURE_SIZE), (
            f"Expected shape ({BATCH}, {2 * FEATURE_SIZE}), got {out.shape}"
        )

        # With activations: correct structure
        result = encoder.apply(params, images, get_activation=True)
        assert isinstance(result, tuple) and len(result) == 2
        features, acts = result
        assert set(acts.keys()) == {"left", "right"}
        for eye in ("left", "right"):
            assert set(acts[eye].keys()) == {"conv_0", "conv_1", "conv_2", "conv_3", "fc"}
        np.testing.assert_array_equal(out, features)
