"""Tests for per-layer CNN activation extraction in VisionEncoder."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder

# Shared encoder config for all tests
ENCODER_KWARGS = dict(
    feature_size=32,
    channels=(4, 8, 16, 32),
    strides=(1, 1, 2, 2),
    padding="VALID",
)

BATCH_SIZE = 2
IMG_SHAPE = (32, 32, 1)


@pytest.fixture
def encoder_and_params():
    """Create encoder and initialize params with a batch of images."""
    encoder = VisionEncoder(**ENCODER_KWARGS)
    rng = jax.random.PRNGKey(0)
    images = jnp.ones((BATCH_SIZE, *IMG_SHAPE))
    params = encoder.init(rng, images)
    return encoder, params, images


class TestVisionEncoderBackwardCompat:
    """Ensure default call signature is unchanged."""

    def test_default_returns_single_array(self, encoder_and_params):
        """Calling without get_activation returns a plain jnp.ndarray."""
        encoder, params, images = encoder_and_params
        out = encoder.apply(params, images)
        assert isinstance(out, jnp.ndarray), "Default call must return jnp.ndarray"
        assert out.shape == (BATCH_SIZE, 32), f"Expected (2, 32), got {out.shape}"

    def test_get_activation_false_same_as_default(self, encoder_and_params):
        """Explicit get_activation=False is identical to omitting the argument."""
        encoder, params, images = encoder_and_params
        default_out = encoder.apply(params, images)
        explicit_out = encoder.apply(params, images, get_activation=False)
        assert isinstance(explicit_out, jnp.ndarray)
        np.testing.assert_array_equal(default_out, explicit_out)


class TestVisionEncoderActivations:
    """Tests for the get_activation=True code path."""

    def test_get_activation_returns_tuple(self, encoder_and_params):
        """get_activation=True returns a tuple of length 2."""
        encoder, params, images = encoder_and_params
        out = encoder.apply(params, images, get_activation=True)
        assert isinstance(out, tuple), "get_activation=True must return a tuple"
        assert len(out) == 2, f"Expected tuple of length 2, got {len(out)}"

    def test_features_unchanged(self, encoder_and_params):
        """The features returned with get_activation=True match the default output."""
        encoder, params, images = encoder_and_params
        default_features = encoder.apply(params, images)
        features, _ = encoder.apply(params, images, get_activation=True)
        np.testing.assert_array_equal(
            default_features, features,
            err_msg="Features must be identical with or without get_activation",
        )

    def test_activation_dict_keys(self, encoder_and_params):
        """The activation dict has exactly conv_0..conv_3 and fc."""
        encoder, params, images = encoder_and_params
        _, activations = encoder.apply(params, images, get_activation=True)
        expected_keys = {"conv_0", "conv_1", "conv_2", "conv_3", "fc"}
        assert set(activations.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(activations.keys())}"
        )

    def test_conv_spatial_shapes(self, encoder_and_params):
        """Conv activation spatial shapes match expected for 32x32 VALID padding."""
        encoder, params, images = encoder_and_params
        _, activations = encoder.apply(params, images, get_activation=True)

        # 32x32 -> 30x30 -> 28x28 -> 13x13 -> 6x6
        expected = {
            "conv_0": (BATCH_SIZE, 30, 30, 4),
            "conv_1": (BATCH_SIZE, 28, 28, 8),
            "conv_2": (BATCH_SIZE, 13, 13, 16),
            "conv_3": (BATCH_SIZE, 6, 6, 32),
        }
        for key, shape in expected.items():
            assert activations[key].shape == shape, (
                f"{key}: expected shape {shape}, got {activations[key].shape}"
            )

    def test_fc_shape(self, encoder_and_params):
        """FC activation has shape (batch, feature_size)."""
        encoder, params, images = encoder_and_params
        _, activations = encoder.apply(params, images, get_activation=True)
        assert activations["fc"].shape == (BATCH_SIZE, 32), (
            f"Expected fc shape (2, 32), got {activations['fc'].shape}"
        )

    def test_unbatched_input(self):
        """Single unbatched image (32,32,1) → scalar leading dims."""
        encoder = VisionEncoder(**ENCODER_KWARGS)
        rng = jax.random.PRNGKey(1)
        image = jnp.ones(IMG_SHAPE)  # (32, 32, 1)
        params = encoder.init(rng, image)

        features, activations = encoder.apply(params, image, get_activation=True)

        assert features.shape == (32,), f"Expected (32,), got {features.shape}"
        assert activations["conv_0"].shape == (30, 30, 4), (
            f"Expected (30, 30, 4), got {activations['conv_0'].shape}"
        )
        assert activations["fc"].shape == (32,), (
            f"Expected (32,), got {activations['fc'].shape}"
        )
