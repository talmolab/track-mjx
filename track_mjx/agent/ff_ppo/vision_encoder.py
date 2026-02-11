"""CNN vision encoder for processing egocentric camera images.

Provides a compact convolutional encoder that maps raw RGB images to
a fixed-size feature vector, compatible with the intention network's
latent space.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class VisionEncoder(nn.Module):
    """Convolutional encoder for egocentric camera images.

    Architecture: 3 conv layers with stride-2 downsampling, followed by
    a flatten and dense projection to a fixed feature size.

    For 64x64 input: 64 -> 32 -> 16 -> 8, then flatten (8*8*64=4096) -> feature_size.

    Attributes:
        feature_size: Output feature vector dimension.
        channels: Channel sizes for each conv layer.
    """

    feature_size: int = 128
    channels: Sequence[int] = (32, 64, 64)

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        """Encode images to feature vectors.

        Args:
            images: Input images, shape [..., H, W, 3] with values in [0, 1].

        Returns:
            Feature vectors, shape [..., feature_size].
        """
        # Handle batched input: remember leading dims
        leading_shape = images.shape[:-3]
        h, w, c = images.shape[-3], images.shape[-2], images.shape[-1]
        x = images.reshape(-1, h, w, c)  # Flatten to (batch, H, W, C)

        for i, ch in enumerate(self.channels):
            x = nn.Conv(
                features=ch,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                name=f"conv_{i}",
            )(x)
            x = nn.relu(x)

        # Flatten spatial dims
        x = x.reshape(x.shape[0], -1)

        # Project to feature size
        x = nn.Dense(self.feature_size, name="fc_project")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Restore leading dims
        x = x.reshape(*leading_shape, self.feature_size)
        return x
