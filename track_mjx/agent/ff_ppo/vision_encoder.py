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
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        """Encode images to feature vectors.

        Args:
            images: Input images, shape [..., H, W, C] with values in [0, 1].

        Returns:
            Feature vectors, shape [..., feature_size].
        """
        # Handle batched input: remember leading dims
        leading_shape = images.shape[:-3]
        h, w, c = images.shape[-3], images.shape[-2], images.shape[-1]
        x = images.reshape(-1, h, w, c)  # Flatten to (batch, H, W, C)

        for i, (ch, stride) in enumerate(zip(self.channels, self.strides)):
            x = nn.Conv(
                features=ch,
                kernel_size=(3, 3),
                strides=(stride, stride),
                padding=self.padding,
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
