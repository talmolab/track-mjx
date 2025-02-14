"""
Helper functions for saving and loading checkpoint for Orbax checkpoint.
"""

import orbax.checkpoint as ocp
from track_mjx.agent.losses import PPONetworkParams
from track_mjx.agent.ppo import TrainingState


def checkpoint_save(
    ckpt_mgr: ocp.CheckpointManager,
    step: int,
    policy_params: PPONetworkParams,
    train_params: TrainingState,
):
    """
    Save the checkpoint. ocp will save both policy module (for easier rollout) and
    the entire training module (for continual training).

    Args:
        ckpt_mgr: The Orbax checkpoint manager.
        step: The current training step.
        policy_params: The parameters for the policy model.
        train_params: The parameters for the training model.
    """
    items_to_save = {
        "policy/policy_params": policy_params,
        "training/train_params": train_params,
    }

    ckpt_mgr.save(step=step, items=items_to_save)
