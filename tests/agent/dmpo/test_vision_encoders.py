"""Tests for the DMPO-private vision encoders and activation lookup."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo import vision_encoders
from track_mjx.agent.dmpo.vision_encoders import (
    BinocularVisionEncoder,
    VisionEncoder,
    get_activation_fn,
)

_KEY = jax.random.key(0)


def _images(*shape, seed=0):
    """Random images in [0, 1) with the given shape."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, 1.0, shape), dtype=jnp.float32)


# --------------------------------------------------------------------------
# Construction-time validation of the conv stack
# --------------------------------------------------------------------------


def test_channels_strides_length_mismatch_raises():
    with pytest.raises(ValueError) as exc:
        BinocularVisionEncoder(channels=(4, 8), strides=(1, 2, 2))
    message = str(exc.value)
    assert "len(channels)=2" in message
    assert "len(strides)=3" in message


def test_monocular_channels_strides_length_mismatch_raises():
    with pytest.raises(ValueError) as exc:
        VisionEncoder(channels=(2, 4, 8), strides=(1, 2))
    message = str(exc.value)
    assert "len(channels)=3" in message
    assert "len(strides)=2" in message


def test_non_positive_strides_and_empty_channels_raise():
    with pytest.raises(ValueError, match="strides"):
        VisionEncoder(channels=(2, 4), strides=(1, 0))
    with pytest.raises(ValueError, match="channels"):
        VisionEncoder(channels=())
    with pytest.raises(ValueError, match="channels"):
        BinocularVisionEncoder(channels=(4, 0))


def test_default_strides_follow_channel_count():
    """Unspecified strides are the leading entries of the default pattern."""
    x = _images(2, 16, 16, 1)
    implicit = VisionEncoder(feature_size=6, channels=(2, 4))
    explicit = VisionEncoder(feature_size=6, channels=(2, 4), strides=(1, 1))
    params = implicit.init(_KEY, x)
    np.testing.assert_array_equal(implicit.apply(params, x), explicit.apply(params, x))

    full = VisionEncoder(feature_size=6, channels=(2, 4, 8, 16))
    pinned = VisionEncoder(feature_size=6, channels=(2, 4, 8, 16), strides=(1, 1, 2, 2))
    x32 = _images(2, 32, 32, 1)
    full_params = full.init(_KEY, x32)
    np.testing.assert_array_equal(
        full.apply(full_params, x32), pinned.apply(full_params, x32)
    )


def test_more_channels_than_default_strides_needs_explicit_strides():
    with pytest.raises(ValueError, match="strides"):
        VisionEncoder(channels=(2, 4, 8, 16, 32))
    enc = VisionEncoder(
        feature_size=4, channels=(2, 4, 8, 16, 32), strides=(1, 1, 1, 2, 2)
    )
    x = _images(1, 32, 32, 1)
    assert enc.apply(enc.init(_KEY, x), x).shape == (1, 4)


def test_unknown_padding_raises():
    with pytest.raises(ValueError, match="padding"):
        VisionEncoder(padding="REFLECT")


# --------------------------------------------------------------------------
# Input validation at call time
# --------------------------------------------------------------------------


def test_binocular_channel_count_checked():
    enc = BinocularVisionEncoder(feature_size=8, channels=(4, 8), mono_channels=1)
    bad = _images(2, 16, 16, 3)
    with pytest.raises(ValueError) as exc:
        enc.init(_KEY, bad)
    message = str(exc.value)
    assert "mono_channels=1" in message
    assert "expected 2" in message
    assert "got 3" in message


def test_binocular_rgb_channel_count_accepted():
    enc = BinocularVisionEncoder(feature_size=8, channels=(4, 8), mono_channels=3)
    x = _images(2, 16, 16, 6)
    assert enc.apply(enc.init(_KEY, x), x).shape == (2, 16)
    with pytest.raises(ValueError, match="mono_channels=3"):
        enc.init(_KEY, _images(2, 16, 16, 2))


def test_image_rank_checked():
    enc = VisionEncoder(feature_size=4, channels=(2,))
    with pytest.raises(ValueError, match="images"):
        enc.init(_KEY, jnp.zeros((16, 16)))


def test_image_too_small_for_conv_stack_raises():
    # Four VALID 3x3 convs with strides (1, 1, 2, 2) shrink 8x8 to nothing.
    enc = VisionEncoder(feature_size=4, channels=(2, 4, 8, 16))
    with pytest.raises(ValueError, match="too small"):
        enc.init(_KEY, jnp.zeros((1, 8, 8, 1)))
    # SAME padding keeps every layer non-empty at the same input size.
    same = VisionEncoder(feature_size=4, channels=(2, 4, 8, 16), padding="SAME")
    assert same.apply(
        same.init(_KEY, jnp.zeros((1, 8, 8, 1))), jnp.zeros((1, 8, 8, 1))
    ).shape == (1, 4)


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


def test_dropping_vision_features_changes_output():
    enc = BinocularVisionEncoder(feature_size=8, channels=(4, 8, 16, 32))
    x = _images(3, 32, 32, 2)
    params = enc.init(_KEY, x)
    with_vision = enc.apply(params, x)
    without_vision = enc.apply(params, jnp.zeros_like(x))
    assert with_vision.shape == (3, 16)
    assert float(jnp.max(jnp.abs(with_vision - without_vision))) > 1e-6

    mono = VisionEncoder(feature_size=8)
    x_mono = _images(3, 32, 32, 1)
    mono_params = mono.init(_KEY, x_mono)
    diff = mono.apply(mono_params, x_mono) - mono.apply(
        mono_params, jnp.zeros_like(x_mono)
    )
    assert float(jnp.max(jnp.abs(diff))) > 1e-6


def test_leading_batch_dims_are_preserved():
    enc = VisionEncoder(feature_size=5, channels=(2, 4))
    x = _images(2, 3, 16, 16, 1)
    params = enc.init(_KEY, x)
    batched = enc.apply(params, x)
    assert batched.shape == (2, 3, 5)
    np.testing.assert_allclose(
        batched[1, 2], enc.apply(params, x[1, 2]), rtol=1e-5, atol=1e-6
    )


def test_get_activation_returns_per_layer_maps():
    enc = BinocularVisionEncoder(feature_size=8, channels=(4, 8))
    x = _images(2, 16, 16, 2)
    params = enc.init(_KEY, x)
    features, acts = enc.apply(params, x, get_activation=True)
    np.testing.assert_array_equal(features, enc.apply(params, x))
    assert set(acts) == {"left", "right"}
    for eye in ("left", "right"):
        assert set(acts[eye]) == {"conv_0", "conv_1", "fc"}
        assert acts[eye]["conv_0"].shape == (2, 14, 14, 4)
        assert acts[eye]["fc"].shape == (2, 8)


def test_shared_weights_use_one_cnn_and_independent_use_two():
    x = _images(2, 16, 16, 2)
    shared = BinocularVisionEncoder(feature_size=8, channels=(4, 8))
    independent = BinocularVisionEncoder(
        feature_size=8, channels=(4, 8), shared_weights=False
    )
    shared_params = shared.init(_KEY, x)["params"]
    independent_params = independent.init(_KEY, x)["params"]
    assert set(shared_params) == {"shared_cnn"}
    assert set(independent_params) == {"left_cnn", "right_cnn"}

    # Identical images in both eyes give identical halves with shared weights.
    same_eyes = jnp.concatenate([x[..., :1], x[..., :1]], axis=-1)
    out = shared.apply({"params": shared_params}, same_eyes)
    np.testing.assert_array_equal(out[..., :8], out[..., 8:])


def test_param_tree_names_are_stable():
    """Checkpoints and the warm-start graft address these exact paths."""
    x = _images(1, 16, 16, 2)
    params = BinocularVisionEncoder(feature_size=8, channels=(4, 8)).init(_KEY, x)
    cnn = params["params"]["shared_cnn"]
    assert set(cnn) == {"conv_0", "conv_1", "fc_project", "LayerNorm_0"}
    assert cnn["conv_0"]["kernel"].shape == (3, 3, 1, 4)
    assert cnn["conv_1"]["kernel"].shape == (3, 3, 4, 8)
    # 16 -> 14 -> 12 spatially, then flatten 12 * 12 * 8 -> feature_size.
    assert cnn["fc_project"]["kernel"].shape == (12 * 12 * 8, 8)


# --------------------------------------------------------------------------
# Activation lookup
# --------------------------------------------------------------------------


def test_get_activation_fn_known_names():
    assert float(get_activation_fn("silu")(jnp.array(1.0))) > 0
    assert float(get_activation_fn("relu")(jnp.array(-1.0))) == 0
    assert get_activation_fn("TanH") is get_activation_fn("tanh")


def test_get_activation_fn_unknown_raises():
    with pytest.raises(ValueError) as exc:
        get_activation_fn("swishy")
    message = str(exc.value)
    assert "swishy" in message
    for name in ("elu", "gelu", "relu", "silu", "tanh"):
        assert name in message


# --------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------


def test_no_private_references():
    source = inspect.getsource(vision_encoders)
    # Assembled from pieces so this file does not itself contain the phrases.
    assert "Opt" + "ion " not in source
    assert "vis" + "_net" not in source
