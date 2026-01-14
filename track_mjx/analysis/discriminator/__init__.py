"""Discriminator training pipeline for motion clip classification.

This module provides a self-contained pipeline for training a binary classifier
to distinguish between different sources of motion clips (e.g., original reference
data vs prior-generated rollouts).
"""

from track_mjx.analysis.discriminator.data_loading import (
    MotionClipDataset,
    create_train_test_split,
    load_h5_dataset,
    load_h5_metadata,
)
from track_mjx.analysis.discriminator.discriminator_network import (
    Discriminator,
    make_discriminator_network,
)
from track_mjx.analysis.discriminator.discriminator_train import (
    DiscriminatorParams,
    TrainingState,
    train,
)

__all__ = [
    "Discriminator",
    "DiscriminatorParams",
    "MotionClipDataset",
    "TrainingState",
    "create_train_test_split",
    "load_h5_dataset",
    "load_h5_metadata",
    "make_discriminator_network",
    "train",
]
