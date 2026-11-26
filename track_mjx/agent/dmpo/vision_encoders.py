"""CNN vision encoders private to the DMPO agent.

Compact convolutional encoders that map egocentric camera images to a
fixed-size feature vector:

- ``VisionEncoder`` encodes one image (grayscale or RGB).
- ``BinocularVisionEncoder`` encodes a channel-stacked left/right image pair,
  either with one set of weights shared by both eyes (a Siamese encoder, the
  default) or with an independent encoder per eye.

``get_activation_fn`` resolves the activation names used in network configs.

The Flax submodule and parameter names (``conv_{i}``, ``fc_project``,
``shared_cnn``, ``left_cnn``, ``right_cnn``, ...) are part of the checkpoint
format and of the warm-start graft; do not rename them.

Misconfigurations that would otherwise degrade silently are rejected with a
``ValueError`` instead: a ``channels`` / ``strides`` length mismatch, an input
with the wrong number of channels, and an image too small for the conv stack.
"""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn

# Strides used when a caller does not pass any: one entry per conv layer, so
# the default covers at most this many layers.
_DEFAULT_STRIDES: tuple[int, ...] = (1, 1, 2, 2)
_KERNEL_SIZE = 3
_PADDINGS = ("VALID", "SAME")

_ACTIVATION_MAP: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "silu": nn.silu,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "gelu": nn.gelu,
    "elu": nn.elu,
}


def get_activation_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Resolve an activation function name to a callable.

    Args:
        name: Activation function name (case-insensitive). Supported:
            "silu", "relu", "tanh", "gelu", "elu".

    Returns:
        The corresponding Flax/JAX activation function.

    Raises:
        ValueError: If name is not recognized.
    """
    fn = _ACTIVATION_MAP.get(name.lower())
    if fn is None:
        raise ValueError(
            f"Unknown activation {name!r}. Supported: {sorted(_ACTIVATION_MAP.keys())}"
        )
    return fn


def _resolve_conv_stack(
    channels: Sequence[int], strides: Sequence[int] | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate a conv stack and return ``(channels, strides)`` as tuples.

    Args:
        channels: Output channels of each conv layer.
        strides: Stride of each conv layer, or None for the default pattern
            ``(1, 1, 2, 2)`` truncated to ``len(channels)``.

    Returns:
        The channels and the resolved strides, with equal lengths.

    Raises:
        ValueError: If ``channels`` is empty or holds a non-positive entry, if
            ``strides`` holds a non-positive entry, if their lengths differ, or
            if ``strides`` is None and there are more channels than default
            strides.
    """
    channels = tuple(channels)
    if not channels:
        raise ValueError("channels must list at least one conv layer.")
    if any(ch < 1 for ch in channels):
        raise ValueError(f"channels must all be positive, got {channels}.")

    if strides is None:
        if len(channels) > len(_DEFAULT_STRIDES):
            raise ValueError(
                f"strides must be given explicitly when channels has more than "
                f"{len(_DEFAULT_STRIDES)} layers; got len(channels)={len(channels)} "
                f"and strides=None."
            )
        return channels, _DEFAULT_STRIDES[: len(channels)]

    strides = tuple(strides)
    if len(strides) != len(channels):
        raise ValueError(
            "channels and strides must have the same length, got "
            f"len(channels)={len(channels)} and len(strides)={len(strides)}."
        )
    if any(s < 1 for s in strides):
        raise ValueError(f"strides must all be positive, got {strides}.")
    return channels, strides


def _check_spatial_size(
    height: int, width: int, strides: Sequence[int], padding: str
) -> None:
    """Raise if any conv layer would have an empty spatial output.

    Args:
        height: Input image height.
        width: Input image width.
        strides: Stride of each conv layer.
        padding: "VALID" or "SAME".

    Raises:
        ValueError: If the image is too small for the conv stack.
    """
    h, w = height, width
    for i, stride in enumerate(strides):
        if padding == "VALID":
            h = (h - _KERNEL_SIZE) // stride + 1 if h >= _KERNEL_SIZE else 0
            w = (w - _KERNEL_SIZE) // stride + 1 if w >= _KERNEL_SIZE else 0
        else:
            h = -(-h // stride)
            w = -(-w // stride)
        if h < 1 or w < 1:
            raise ValueError(
                f"Image of size {height}x{width} is too small for the conv "
                f"stack: layer conv_{i} ({_KERNEL_SIZE}x{_KERNEL_SIZE}, stride "
                f"{stride}, {padding} padding) would have an empty output."
            )


class VisionEncoder(nn.Module):
    """Convolutional encoder for egocentric camera images.

    N conv layers with per-layer strides and 3x3 kernels, each followed by a
    ReLU, then flatten and a dense projection (with LayerNorm and ReLU) to a
    fixed feature size.

    For 32x32 input with defaults: 32->30->28->13->6, flatten (6*6*16=576)->8.

    Attributes:
        feature_size: Output feature vector dimension.
        channels: Channel sizes for each conv layer.
        strides: Stride for each conv layer (one int per layer, applied to both
            spatial dimensions). Must have the same length as channels. None
            selects the default pattern ``(1, 1, 2, 2)``, truncated to the
            number of layers; it is only available for up to four layers.
        padding: Padding mode for conv layers ("VALID" or "SAME").
    """

    feature_size: int = 8
    channels: Sequence[int] = (2, 4, 8, 16)
    strides: Sequence[int] | None = None
    padding: str = "VALID"

    def __post_init__(self):
        super().__post_init__()
        _resolve_conv_stack(self.channels, self.strides)
        if self.padding not in _PADDINGS:
            raise ValueError(
                f"padding must be one of {_PADDINGS}, got {self.padding!r}."
            )

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        get_activation: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Encode images to feature vectors.

        Args:
            images: Input images, shape [..., H, W, C] with values in [0, 1].
            get_activation: If True, also return a dict of per-layer activations.
                Default False preserves the original return type exactly.

        Returns:
            If get_activation is False (default): feature vectors of shape
            [..., feature_size].
            If get_activation is True: a tuple (features, activations) where
            features has shape [..., feature_size] and activations is a dict
            with keys "conv_0".."conv_N" (post-ReLU spatial maps, shape
            [..., H', W', C']) and "fc" (shape [..., feature_size]).

        Raises:
            ValueError: If ``images`` has fewer than three dimensions or is too
                small for the conv stack.
        """
        if images.ndim < 3:
            raise ValueError(
                f"images must have shape [..., H, W, C], got shape {images.shape}."
            )
        channels, strides = _resolve_conv_stack(self.channels, self.strides)
        h, w, c = images.shape[-3], images.shape[-2], images.shape[-1]
        _check_spatial_size(h, w, strides, self.padding)

        # Handle batched input: remember leading dims
        leading_shape = images.shape[:-3]
        x = images.reshape(-1, h, w, c)  # Flatten to (batch, H, W, C)

        layer_activations: dict[str, jnp.ndarray] = {}

        for i, (ch, stride) in enumerate(zip(channels, strides)):
            x = nn.Conv(
                features=ch,
                kernel_size=(_KERNEL_SIZE, _KERNEL_SIZE),
                strides=(stride, stride),
                padding=self.padding,
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)
            if get_activation:
                # Restore leading dims for the spatial map: (..., H', W', C')
                layer_activations[f"conv_{i}"] = x.reshape(*leading_shape, *x.shape[1:])

        # Flatten spatial dims
        x = x.reshape(x.shape[0], -1)

        # Project to feature size
        x = nn.Dense(self.feature_size, name="fc_project")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Restore leading dims
        x = x.reshape(*leading_shape, self.feature_size)

        if get_activation:
            layer_activations["fc"] = x
            return x, layer_activations

        return x


class BinocularVisionEncoder(nn.Module):
    """CNN encoder for binocular (stereo) vision with configurable weight sharing.

    Splits the channel-stacked input into left and right eye images,
    processes each through CNN(s), and concatenates the resulting feature
    vectors. Both modes produce the same output shape: [..., 2 * feature_size].

    When ``shared_weights=True`` (default): a single VisionEncoder is applied to
    both eyes, a Siamese architecture. It enforces symmetric feature extraction,
    matches the homologous left/right visual circuits of animals, and costs the
    parameters of one monocular CNN.

    When ``shared_weights=False``: two independent VisionEncoder instances
    process each eye with separate weights. Twice the CNN parameters, but no
    symmetry prior.

    Input: (H, W, 2*C) -- channel-stacked binocular image.
    Output: (2 * feature_size,) -- concatenated monocular features.

    Attributes:
        feature_size: Feature vector dimension per eye. Total output
            dimension is 2 * feature_size.
        channels: Channel sizes for each conv layer in the CNN(s).
        mono_channels: Number of channels per eye (1 for grayscale, 3 for RGB).
        shared_weights: If True, use the shared-weight Siamese architecture. If
            False, use independent per-eye CNNs.
        strides: Stride for each conv layer, as for ``VisionEncoder``. None
            selects its default pattern ``(1, 1, 2, 2)``, truncated to the
            number of layers.
    """

    feature_size: int = 32
    channels: Sequence[int] = (4, 8, 16, 32)
    mono_channels: int = 1
    shared_weights: bool = True
    strides: Sequence[int] | None = None

    def __post_init__(self):
        super().__post_init__()
        _resolve_conv_stack(self.channels, self.strides)
        if self.mono_channels < 1:
            raise ValueError(
                f"mono_channels must be positive, got {self.mono_channels}."
            )

    def setup(self):
        if self.shared_weights:
            # Single CNN, applied to both eyes (Siamese)
            self.shared_cnn = VisionEncoder(
                feature_size=self.feature_size,
                channels=self.channels,
                strides=self.strides,
            )
        else:
            # Two independent CNNs
            self.left_cnn = VisionEncoder(
                feature_size=self.feature_size,
                channels=self.channels,
                strides=self.strides,
            )
            self.right_cnn = VisionEncoder(
                feature_size=self.feature_size,
                channels=self.channels,
                strides=self.strides,
            )

    def __call__(
        self,
        binocular_images: jnp.ndarray,
        get_activation: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, dict[str, jnp.ndarray]]]:
        """Encode binocular images to feature vectors.

        Args:
            binocular_images: Shape [..., H, W, 2*C] channel-stacked stereo input.
                Channels 0:C are left eye, channels C:2C are right eye.
            get_activation: If True, also return per-eye per-layer activations.
                Default False preserves the original return type exactly.

        Returns:
            If get_activation is False (default): feature vectors of shape
            [..., 2 * feature_size]. Layout: [left_features, right_features].

            If get_activation is True: a tuple (features, activations) where
            features has shape [..., 2 * feature_size] and activations is a
            dict with keys "left" and "right", each mapping to the
            layer_acts_dict from VisionEncoder (keys "conv_0".."conv_N", "fc").

        Raises:
            ValueError: If the last axis of ``binocular_images`` is not
                ``2 * mono_channels``.
        """
        c = self.mono_channels
        if binocular_images.shape[-1] != 2 * c:
            raise ValueError(
                f"Binocular input must stack two eyes of mono_channels={c} "
                f"channels: expected {2 * c} channels, got "
                f"{binocular_images.shape[-1]} (input shape "
                f"{binocular_images.shape})."
            )
        left = binocular_images[..., :c]  # [..., H, W, C]
        right = binocular_images[..., c:]  # [..., H, W, C]

        if self.shared_weights:
            left_out = self.shared_cnn(left, get_activation=get_activation)
            right_out = self.shared_cnn(right, get_activation=get_activation)
        else:
            left_out = self.left_cnn(left, get_activation=get_activation)
            right_out = self.right_cnn(right, get_activation=get_activation)

        if get_activation:
            left_features, left_acts = left_out
            right_features, right_acts = right_out
            features = jnp.concatenate([left_features, right_features], axis=-1)
            return features, {"left": left_acts, "right": right_acts}

        return jnp.concatenate([left_out, right_out], axis=-1)
