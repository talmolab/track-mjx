"""CNN vision encoder for processing egocentric camera images.

Provides a compact convolutional encoder that maps raw camera images (RGB
or grayscale) to a fixed-size feature vector, compatible with the intention
network's latent space.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class VisionEncoder(nn.Module):
    """Convolutional encoder for egocentric camera images.

    Architecture matches the reference VisNetRodent (vis_net.py): N conv layers
    with per-layer strides and VALID padding, followed by flatten and dense
    projection to a fixed feature size.

    For 32x32 input with defaults: 32->30->28->13->6, flatten (6*6*16=576)->8.

    Attributes:
        feature_size: Output feature vector dimension.
        channels: Channel sizes for each conv layer.
        strides: Stride for each conv layer (one int per layer, applied to both
            spatial dimensions). Must have same length as channels.
        padding: Padding mode for conv layers ("VALID" or "SAME").
    """

    feature_size: int = 8
    channels: Sequence[int] = (2, 4, 8, 16)
    strides: Sequence[int] = (1, 1, 2, 2)
    padding: str = "VALID"

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
        """
        # Handle batched input: remember leading dims
        leading_shape = images.shape[:-3]
        h, w, c = images.shape[-3], images.shape[-2], images.shape[-1]
        x = images.reshape(-1, h, w, c)  # Flatten to (batch, H, W, C)

        layer_activations: dict[str, jnp.ndarray] = {}

        for i, (ch, stride) in enumerate(zip(self.channels, self.strides)):
            x = nn.Conv(
                features=ch,
                kernel_size=(3, 3),
                strides=(stride, stride),
                padding=self.padding,
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)
            if get_activation:
                # Restore leading dims for the spatial map: (..., H', W', C')
                layer_activations[f"conv_{i}"] = x.reshape(
                    *leading_shape, *x.shape[1:]
                )

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
