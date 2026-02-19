"""Binocular vision encoder with configurable weight sharing.

Processes left and right eye images through CNN(s), then concatenates
features. Supports two modes controlled by the ``shared_weights`` attribute:

- ``shared_weights=True`` (Option C, Siamese): A single VisionEncoder
  processes both eyes with shared weights, enforcing symmetric feature
  extraction. This matches the biological constraint that left/right V1
  use homologous circuits. Parameter count = 1x monocular CNN.

- ``shared_weights=False`` (Option B, Independent): Two separate
  VisionEncoder instances process each eye independently with their own
  weights. Maximum flexibility but 2x parameters and no symmetry prior.

Both modes produce the same output shape: [..., 2 * feature_size].

Input: (H, W, 2*C) -- channel-stacked binocular image
Output: (2 * feature_size,) -- concatenated monocular features
"""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from track_mjx.agent.ff_ppo.vision_encoder import VisionEncoder


class BinocularVisionEncoder(nn.Module):
    """CNN encoder for binocular (stereo) vision with configurable weight sharing.

    Splits the channel-stacked input into left and right eye images,
    processes each through CNN(s), and concatenates the resulting feature
    vectors.

    When ``shared_weights=True`` (default, Option C): A single VisionEncoder
    is applied to both eyes -- Siamese architecture with shared weights.
    Biologically plausible (homologous V1 circuits) and parameter efficient.

    When ``shared_weights=False`` (Option B): Two independent VisionEncoder
    instances process each eye with separate weights. Double the CNN
    parameters but maximum flexibility for asymmetric feature extraction.

    Attributes:
        feature_size: Feature vector dimension per eye. Total output
            dimension is 2 * feature_size.
        channels: Channel sizes for each conv layer in the CNN(s).
        mono_channels: Number of channels per eye (1 for grayscale, 3 for RGB).
        shared_weights: If True, use shared-weight Siamese architecture
            (Option C). If False, use independent dual-CNN (Option B).
    """

    feature_size: int = 32
    channels: Sequence[int] = (4, 8, 16, 32)
    mono_channels: int = 1
    shared_weights: bool = True

    def setup(self):
        if self.shared_weights:
            # Option C: Single CNN, applied to both eyes (Siamese)
            self.shared_cnn = VisionEncoder(
                feature_size=self.feature_size,
                channels=self.channels,
            )
        else:
            # Option B: Two independent CNNs
            self.left_cnn = VisionEncoder(
                feature_size=self.feature_size,
                channels=self.channels,
            )
            self.right_cnn = VisionEncoder(
                feature_size=self.feature_size,
                channels=self.channels,
            )

    def __call__(self, binocular_images: jnp.ndarray) -> jnp.ndarray:
        """Encode binocular images to feature vectors.

        Args:
            binocular_images: Shape [..., H, W, 2*C] channel-stacked stereo input.
                Channels 0:C are left eye, channels C:2C are right eye.

        Returns:
            Feature vectors, shape [..., 2 * feature_size].
            Layout: [left_features, right_features].
        """
        c = self.mono_channels
        left = binocular_images[..., :c]    # [..., H, W, C]
        right = binocular_images[..., c:]   # [..., H, W, C]

        if self.shared_weights:
            left_features = self.shared_cnn(left)    # [..., feature_size]
            right_features = self.shared_cnn(right)  # [..., feature_size]
        else:
            left_features = self.left_cnn(left)      # [..., feature_size]
            right_features = self.right_cnn(right)   # [..., feature_size]

        return jnp.concatenate([left_features, right_features], axis=-1)
