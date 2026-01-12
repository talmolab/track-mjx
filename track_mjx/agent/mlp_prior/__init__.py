"""
mlp_prior: Training pipeline for prior network alignment to pretrained encoder.

This module trains a prior network to match the encoder distributions from a
pretrained ff_ppo checkpoint. The encoder and decoder remain frozen; only
the prior is trained using KL divergence loss.
"""

from track_mjx.agent.mlp_prior.prior_train import train
from track_mjx.agent.mlp_prior.prior_networks import (
    load_frozen_encoder_decoder,
    make_prior_networks,
    create_combined_checkpoint_params,
)
from track_mjx.agent.mlp_prior.losses import (
    compute_encoder_prior_kl_loss,
    compute_prior_training_loss,
    create_ramp_schedule,
)
from track_mjx.agent.mlp_prior.prior_rollout_eval import MultiModePriorRolloutEvaluator

__all__ = [
    "train",
    "load_frozen_encoder_decoder",
    "make_prior_networks",
    "create_combined_checkpoint_params",
    "compute_encoder_prior_kl_loss",
    "compute_prior_training_loss",
    "create_ramp_schedule",
    "MultiModePriorRolloutEvaluator",
]
