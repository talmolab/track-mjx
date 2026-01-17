"""VQ-VAE specific PPO training module.

This module provides a train function that uses VQ-VAE losses instead of
standard VAE KL divergence losses. It wraps the main PPO training by
temporarily patching the loss module.
"""

import functools
from typing import Any, Callable

from brax import envs
from brax.training.types import Metrics
import orbax.checkpoint as ocp
from mujoco_playground import wrapper as mp_wrapper

# Import the main ppo module
from track_mjx.agent.ff_ppo import ppo, losses as original_losses

# Import VQ-VAE specific modules
from vq_ppo_networks import make_vq_intention_ppo_networks
from vq_losses import compute_vq_ppo_loss, PPONetworkParams


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: dict[str, Any],
    checkpoint_to_restore: str | None = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: int | None = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 20,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: Callable[..., Any] = make_vq_intention_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Callable | None = None,
    get_activation: bool = True,
    freeze_decoder: bool = False,
    checkpoint_callback: Callable[[int], None] | None = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    # VQ-VAE specific parameters
    commitment_cost: float = 0.25,
    codebook_loss_weight: float = 1.0,
):
    """Train a VQ-VAE PPO agent.

    This is a wrapper around the main PPO train function that uses VQ-VAE
    specific loss computation instead of KL divergence.

    Args:
        All standard PPO args, plus:
        commitment_cost: Weight for commitment loss (beta in VQ-VAE paper).
        codebook_loss_weight: Weight for codebook loss.

    Returns:
        Tuple of (make_policy, params, metrics).
    """
    # Store original compute_ppo_loss
    original_compute_ppo_loss = original_losses.compute_ppo_loss

    # Create VQ-VAE loss function with same interface as ff_ppo losses.compute_ppo_loss
    def vq_compute_ppo_loss(
        params,
        normalizer_params,
        data,
        rng,
        step,
        ppo_network,
        entropy_cost=1e-4,
        latent_kl_weight=1e-3,  # Ignored in VQ-VAE
        latent_ar1_weight=1e-3,  # Ignored in VQ-VAE
        discounting=0.9,
        reward_scaling=1.0,
        gae_lambda=0.95,
        clipping_epsilon=0.3,
        normalize_advantage=True,
        vf_coefficient=0.5,
        latent_kl_schedule=None,  # Ignored in VQ-VAE
        latent_ar1_schedule=None,  # Ignored in VQ-VAE
    ):
        """VQ-VAE loss with same interface as compute_ppo_loss."""
        return compute_vq_ppo_loss(
            params=params,
            normalizer_params=normalizer_params,
            data=data,
            rng=rng,
            step=step,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            commitment_cost=commitment_cost,
            codebook_loss_weight=codebook_loss_weight,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
            vq_loss_schedule=None,
        )

    # Monkey-patch the loss function
    original_losses.compute_ppo_loss = vq_compute_ppo_loss

    try:
        # Run training with VQ-VAE loss
        result = ppo.train(
            environment=environment,
            num_timesteps=num_timesteps,
            episode_length=episode_length,
            ckpt_mgr=ckpt_mgr,
            config_dict=config_dict,
            checkpoint_to_restore=checkpoint_to_restore,
            action_repeat=action_repeat,
            num_envs=num_envs,
            max_devices_per_host=max_devices_per_host,
            num_eval_envs=num_eval_envs,
            learning_rate=learning_rate,
            entropy_cost=entropy_cost,
            latent_kl_weight=0.0,  # Not used in VQ-VAE
            latent_ar1_weight=0.0,  # Not used in VQ-VAE
            discounting=discounting,
            seed=seed,
            use_pmap_on_reset=use_pmap_on_reset,
            unroll_length=unroll_length,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            num_evals=num_evals,
            num_resets_per_eval=num_resets_per_eval,
            normalize_observations=normalize_observations,
            reward_scaling=reward_scaling,
            clipping_epsilon=clipping_epsilon,
            gae_lambda=gae_lambda,
            deterministic_eval=deterministic_eval,
            network_factory=network_factory,
            progress_fn=progress_fn,
            normalize_advantage=normalize_advantage,
            eval_env=eval_env,
            eval_env_test_set=eval_env_test_set,
            policy_params_fn=policy_params_fn,
            randomization_fn=randomization_fn,
            get_activation=get_activation,
            use_kl_schedule=False,  # Not used in VQ-VAE
            freeze_decoder=freeze_decoder,
            checkpoint_callback=checkpoint_callback,
            wrap_for_training=wrap_for_training,
        )
    finally:
        # Restore original loss function
        original_losses.compute_ppo_loss = original_compute_ppo_loss

    return result
