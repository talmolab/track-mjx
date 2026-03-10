"""MoSeq decoder PPO training module.

Wraps the main PPO training by temporarily patching the loss module and
inference functions to use the MoSeq decoder-only policy network.
Same monkey-patching pattern as ``vqvae_jax/vq_ppo.py``.
"""

import functools
import logging
from typing import Any, Callable

import jax
from brax import envs
from brax.training.types import Metrics
import orbax.checkpoint as ocp
from mujoco_playground import wrapper as mp_wrapper

from track_mjx.agent.ff_ppo import ppo, losses as original_losses
from track_mjx.agent.ff_ppo import ppo_networks as original_ppo_networks

from moseq_losses import compute_moseq_ppo_loss
from moseq_ppo_networks import (
    make_moseq_decoder_ppo_networks,
    make_moseq_inference_fn,
    make_moseq_logging_inference_fn,
)


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
    network_factory: Callable[..., Any] = make_moseq_decoder_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Callable | None = None,
    get_activation: bool = False,
    freeze_decoder: bool = False,
    checkpoint_callback: Callable[[int], None] | None = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    # MoSeq-specific (kept for interface parity with vq_ppo.train)
    num_codes: int = 32,
    code_embed_dim: int = 16,
):
    """Train a MoSeq decoder PPO agent.

    Monkey-patches the PPO loss and inference functions, delegates to
    ``ppo.train``, then restores originals.

    Returns:
        Tuple of ``(make_policy, params, metrics)``.
    """
    original_compute_ppo_loss = original_losses.compute_ppo_loss
    original_make_inference_fn = original_ppo_networks.make_inference_fn
    original_make_logging_inference_fn = original_ppo_networks.make_logging_inference_fn

    # Build wrapped loss with same signature as compute_ppo_loss
    def moseq_loss_wrapper(
        params,
        normalizer_params,
        data,
        rng,
        step,
        ppo_network,
        entropy_cost=1e-4,
        latent_kl_weight=1e-3,  # ignored
        latent_ar1_weight=1e-3,  # ignored
        discounting=0.9,
        reward_scaling=1.0,
        gae_lambda=0.95,
        clipping_epsilon=0.3,
        normalize_advantage=True,
        vf_coefficient=0.5,
        latent_kl_schedule=None,  # ignored
        latent_ar1_schedule=None,  # ignored
    ):
        return compute_moseq_ppo_loss(
            params=params,
            normalizer_params=normalizer_params,
            data=data,
            rng=rng,
            step=step,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
            vf_coefficient=vf_coefficient,
        )

    # Monkey-patch
    original_losses.compute_ppo_loss = moseq_loss_wrapper
    original_ppo_networks.make_inference_fn = make_moseq_inference_fn
    original_ppo_networks.make_logging_inference_fn = make_moseq_logging_inference_fn

    try:
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
            latent_kl_weight=0.0,
            latent_ar1_weight=0.0,
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
            use_kl_schedule=False,
            freeze_decoder=freeze_decoder,
            checkpoint_callback=checkpoint_callback,
            wrap_for_training=wrap_for_training,
        )
    finally:
        original_losses.compute_ppo_loss = original_compute_ppo_loss
        original_ppo_networks.make_inference_fn = original_make_inference_fn
        original_ppo_networks.make_logging_inference_fn = (
            original_make_logging_inference_fn
        )

    return result
