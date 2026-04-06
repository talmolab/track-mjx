"""MoSeq high-level RNN intention PPO training module.

Wraps the main PPO training by patching the loss and inference functions
to use the high-level intention RNN policy. The RNN carry infrastructure
threads hidden state through rollouts.

Same monkey-patching pattern as ``moseq_ppo.py``.
"""

import functools
import logging
from typing import Any, Callable

import jax.numpy as jnp
from brax import envs
from brax.training.types import Metrics
import orbax.checkpoint as ocp
from mujoco_playground import wrapper as mp_wrapper

from track_mjx.agent.ff_ppo import ppo, losses as original_losses
from track_mjx.agent.ff_ppo import ppo_networks as original_ppo_networks

from moseq_highlvl_losses import compute_moseq_highlvl_ppo_loss
from moseq_highlvl_network import (
    make_moseq_highlvl_rnn_networks,
    make_moseq_highlvl_inference_fn,
    make_moseq_highlvl_logging_inference_fn,
    make_moseq_highlvl_rnn_rollout_policy_fn,
)
from moseq_ppo import generate_unroll_rnn_moseq


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
    network_factory: Callable[..., Any] = make_moseq_highlvl_rnn_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Callable | None = None,
    get_activation: bool = False,
    checkpoint_callback: Callable[[int], None] | None = None,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
    # High-level RNN specific
    rnn_hidden_sizes: tuple[int, ...] = (256,),
):
    """Train a MoSeq high-level intention RNN agent.

    Monkey-patches PPO loss and inference functions, delegates to
    ``ppo.train``, then restores originals.
    """
    original_compute_ppo_loss = original_losses.compute_ppo_loss
    original_make_inference_fn = original_ppo_networks.make_inference_fn
    original_make_logging_inference_fn = (
        original_ppo_networks.make_logging_inference_fn
    )

    _rnn_hidden_sizes = tuple(rnn_hidden_sizes)

    def highlvl_loss_wrapper(
        params,
        normalizer_params,
        data,
        rng,
        step,
        ppo_network,
        entropy_cost=1e-4,
        latent_kl_weight=1e-3,
        latent_ar1_weight=1e-3,
        discounting=0.9,
        reward_scaling=1.0,
        gae_lambda=0.95,
        clipping_epsilon=0.3,
        normalize_advantage=True,
        vf_coefficient=0.5,
        latent_kl_schedule=None,
        latent_ar1_schedule=None,
    ):
        return compute_moseq_highlvl_ppo_loss(
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

    original_losses.compute_ppo_loss = highlvl_loss_wrapper
    original_ppo_networks.make_inference_fn = make_moseq_highlvl_inference_fn
    original_ppo_networks.make_logging_inference_fn = (
        make_moseq_highlvl_logging_inference_fn
    )

    logging.info("High-level RNN mode: hidden_sizes=%s", _rnn_hidden_sizes)

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
            checkpoint_callback=checkpoint_callback,
            wrap_for_training=wrap_for_training,
            generate_unroll_fn=generate_unroll_rnn_moseq,
            init_carry_state_fn=lambda n: [
                jnp.zeros((n, h)) for h in _rnn_hidden_sizes
            ],
            make_rollout_policy_fn=make_moseq_highlvl_rnn_rollout_policy_fn,
        )
    finally:
        original_losses.compute_ppo_loss = original_compute_ppo_loss
        original_ppo_networks.make_inference_fn = original_make_inference_fn
        original_ppo_networks.make_logging_inference_fn = (
            original_make_logging_inference_fn
        )

    return result
