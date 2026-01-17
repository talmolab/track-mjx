"""VQ-VAE Prior Distillation module.

This module trains a Prior network to predict VQ-VAE encoder outputs
from proprioceptive observations only, enabling "freeloop" generation
without reference trajectories.
"""

from .vq_prior_losses import (
    VQPriorDistillNetworkParams,
    compute_vq_prior_distill_loss,
    compute_mse_alignment_loss,
    compute_ar1_loss,
    create_ar_schedule,
)
from .vq_prior_networks import (
    VQPrior,
    VQPriorNetworks,
    make_vq_prior_network,
    make_vq_prior_networks,
    make_prior_inference_fn,
    make_freeloop_policy_fn,
)
from .vq_prior_distill import (
    VQPriorDistillTrainingState,
    load_frozen_vqvae,
    create_frozen_encoder,
    create_frozen_decoder,
    create_frozen_vqvae_policy,
    train,
)
from .vq_prior_rollout import (
    run_freeloop_rollout,
    VQPriorFreelloopEvaluator,
    log_freeloop_to_wandb,
)

__all__ = [
    # Losses
    "VQPriorDistillNetworkParams",
    "compute_vq_prior_distill_loss",
    "compute_mse_alignment_loss",
    "compute_ar1_loss",
    "create_ar_schedule",
    # Networks
    "VQPrior",
    "VQPriorNetworks",
    "make_vq_prior_network",
    "make_vq_prior_networks",
    "make_prior_inference_fn",
    "make_freeloop_policy_fn",
    # Training
    "VQPriorDistillTrainingState",
    "load_frozen_vqvae",
    "create_frozen_encoder",
    "create_frozen_decoder",
    "create_frozen_vqvae_policy",
    "train",
    # Evaluation
    "run_freeloop_rollout",
    "VQPriorFreelloopEvaluator",
    "log_freeloop_to_wandb",
]
