"""VQ-VAE specific PPO training module.

This module provides a train function that uses VQ-VAE losses instead of
standard VAE KL divergence losses. It wraps the main PPO training by
temporarily patching the loss module.
"""

import functools
import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training import types
from brax.training.types import Metrics
import orbax.checkpoint as ocp
from mujoco_playground import wrapper as mp_wrapper

# Import the main ppo module
from track_mjx.agent.ff_ppo import ppo, losses as original_losses
from track_mjx.agent.ff_ppo import ppo_networks as original_ppo_networks

# Import VQ-VAE specific modules
from vq_ppo_networks import (
    make_vq_intention_ppo_networks,
    make_vq_inference_fn,
    make_vq_logging_inference_fn,
    make_vq_chunked_inference_fn,
)
from vq_losses import (
    compute_vq_ppo_loss,
    compute_vq_chunked_ppo_loss,
    PPONetworkParams,
    reinit_dead_codes,
)


# ---------------------------------------------------------------------------
# Chunked rollout helpers
# ---------------------------------------------------------------------------

Transition = types.Transition


def reset_chunk_state_on_done(
    chunk_state: tuple[jnp.ndarray, jnp.ndarray],
    done: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reset (held_d0_idx, tau) to (0, 0) where episodes ended.

    Args:
        chunk_state: Tuple of (held_d0_idx, tau), each shape [B].
        done: Boolean done signal, shape [B].

    Returns:
        Updated chunk_state with zeros where done==True.
    """
    held_d0, tau = chunk_state
    return (
        jnp.where(done, jnp.zeros_like(held_d0), held_d0),
        jnp.where(done, jnp.zeros_like(tau), tau),
    )


def generate_unroll_chunked(
    env,
    env_state,
    policy,
    chunk_state: tuple[jnp.ndarray, jnp.ndarray],
    key,
    unroll_length: int,
    extra_fields: tuple[str, ...] = (),
):
    """Generate an unroll with chunk state carry for Semi-MDP commitment.

    Drop-in replacement for brax.training.acting.generate_unroll when using
    code chunking. The policy must accept (obs, chunk_state, key) and return
    (action, extras, new_chunk_state).

    Args:
        env: Wrapped environment.
        env_state: Current environment state.
        policy: Chunked policy function with signature
            (obs, chunk_state, key) -> (action, extras, new_chunk_state).
        chunk_state: Tuple of (held_d0_idx, tau), each shape [B].
        key: JAX random key.
        unroll_length: Number of steps to unroll.
        extra_fields: Additional fields to extract from env state info.

    Returns:
        Tuple of (final_env_state, transitions, final_chunk_state).
    """

    def f(carry, unused_t):
        state, cs, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        actions, policy_extras, new_cs = policy(state.obs, cs, current_key)
        nstate = env.step(state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        new_cs = reset_chunk_state_on_done(new_cs, nstate.done)
        transition = Transition(
            observation=state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={"policy_extras": policy_extras, "state_extras": state_extras},
        )
        return (nstate, new_cs, next_key), transition

    (final_state, final_cs, _), data = jax.lax.scan(
        f, (env_state, chunk_state, key), (), length=unroll_length
    )
    return final_state, data, final_cs


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
    rvq_depth: int = 1,
    codebook_entropy_weight: float = 0.0,
    codebook_entropy_temperature: float = 1.0,
    dead_code_reinit: bool = False,
    dead_code_threshold: float = 0.01,
    num_codes: int = 32,
    reinit_data: dict | None = None,
    kl_weight: float = 0.0,
    # Code chunking (Semi-MDP temporal commitment)
    use_code_chunking: bool = False,
    code_commitment_horizon: int = 0,
):
    """Train a VQ-VAE PPO agent.

    This is a wrapper around the main PPO train function that uses VQ-VAE
    specific loss computation instead of KL divergence.

    Args:
        All standard PPO args, plus:
        commitment_cost: Weight for commitment loss (beta in VQ-VAE paper).
        codebook_loss_weight: Weight for codebook loss.
        codebook_entropy_weight: Weight for soft codebook entropy regularization.
        codebook_entropy_temperature: Temperature for soft code assignments.
        dead_code_reinit: Whether to reinitialize dead codebook entries.
        dead_code_threshold: Fraction of uniform usage below which a code is dead.
        num_codes: Number of codes per level.
        reinit_data: Mutable dict shared with rollout callback for dead code reinit.
            Expected keys after rollout: "z_e" (ndarray) and "all_indices" (tuple).

    Returns:
        Tuple of (make_policy, params, metrics).
    """
    # Store original compute_ppo_loss
    original_compute_ppo_loss = original_losses.compute_ppo_loss

    # Create VQ-VAE loss function with same interface as ff_ppo losses.compute_ppo_loss
    if use_code_chunking:

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
            """VQ-VAE chunked loss with same interface as compute_ppo_loss."""
            return compute_vq_chunked_ppo_loss(
                params=params,
                normalizer_params=normalizer_params,
                data=data,
                rng=rng,
                step=step,
                ppo_network=ppo_network,
                entropy_cost=entropy_cost,
                commitment_cost=commitment_cost,
                codebook_loss_weight=codebook_loss_weight,
                commitment_horizon=code_commitment_horizon,
                num_codes=num_codes,
                discounting=discounting,
                reward_scaling=reward_scaling,
                gae_lambda=gae_lambda,
                clipping_epsilon=clipping_epsilon,
                normalize_advantage=normalize_advantage,
                codebook_entropy_weight=codebook_entropy_weight,
                codebook_entropy_temperature=codebook_entropy_temperature,
                kl_weight=kl_weight,
            )
    else:

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
                rvq_depth=rvq_depth,
                codebook_entropy_weight=codebook_entropy_weight,
                codebook_entropy_temperature=codebook_entropy_temperature,
                kl_weight=kl_weight,
            )

    # Build post-eval hook for dead code reinit
    post_eval_fn = None
    if dead_code_reinit and reinit_data is not None:

        def _post_eval_params_fn(training_state, it):
            """Reinitialize dead codes using rollout data collected by callback."""
            z_e = reinit_data.get("z_e")
            all_indices = reinit_data.get("all_indices")
            if z_e is None or all_indices is None:
                return None

            try:
                from track_mjx.agent.ff_ppo.ppo import _unpmap

                unpmap_state = _unpmap(training_state)
                policy_params = unpmap_state.params.policy

                new_policy_params = reinit_dead_codes(
                    policy_params=policy_params,
                    z_e_samples=jnp.array(z_e),
                    all_indices=all_indices,
                    num_codes=num_codes,
                    rvq_depth=rvq_depth,
                    threshold=dead_code_threshold,
                    rng=jax.random.PRNGKey(it),
                )

                # Check if anything changed
                if new_policy_params is policy_params:
                    return None

                # Rebuild training state with new params on all devices
                new_params = unpmap_state.params.replace(policy=new_policy_params)
                new_state = unpmap_state.replace(params=new_params)
                new_state = jax.device_put_replicated(new_state, jax.local_devices())
                logging.info(f"Dead code reinit applied at iteration {it}")
                return new_state
            except Exception as e:
                logging.warning(f"Dead code reinit failed: {e}")
                return None

        post_eval_fn = _post_eval_params_fn

    # Monkey-patch the loss function
    original_losses.compute_ppo_loss = vq_compute_ppo_loss

    # When chunking, use the stateful chunked inference fn + custom unroll.
    # When not chunking, use standard VQ inference with monkey-patching.
    if use_code_chunking:
        # Chunked training: use generate_unroll_chunked with carry state.
        # The Evaluator and logging still use standard VQ inference fn (obs, key).
        # The training rollout uses chunked policy via make_rollout_policy_fn.
        original_make_inference_fn = original_ppo_networks.make_inference_fn
        original_ppo_networks.make_inference_fn = make_vq_inference_fn
        original_make_logging_inference_fn = (
            original_ppo_networks.make_logging_inference_fn
        )
        original_ppo_networks.make_logging_inference_fn = make_vq_logging_inference_fn

        _generate_unroll_fn = generate_unroll_chunked
        _init_carry_state_fn = lambda n: (
            jnp.zeros(n, dtype=jnp.int32),  # held_d0_idx
            jnp.zeros(n, dtype=jnp.int32),  # tau
        )
        _make_rollout_policy_fn = functools.partial(
            make_vq_chunked_inference_fn,
            commitment_horizon=code_commitment_horizon,
        )
    else:
        # Standard VQ: monkey-patch inference to handle 4-value return
        original_make_inference_fn = original_ppo_networks.make_inference_fn
        original_ppo_networks.make_inference_fn = make_vq_inference_fn
        original_make_logging_inference_fn = (
            original_ppo_networks.make_logging_inference_fn
        )
        original_ppo_networks.make_logging_inference_fn = make_vq_logging_inference_fn

        _generate_unroll_fn = None
        _init_carry_state_fn = None
        _make_rollout_policy_fn = None

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
            post_eval_params_fn=post_eval_fn,
            generate_unroll_fn=_generate_unroll_fn,
            init_carry_state_fn=_init_carry_state_fn,
            make_rollout_policy_fn=_make_rollout_policy_fn,
        )
    finally:
        # Restore original functions
        original_losses.compute_ppo_loss = original_compute_ppo_loss
        original_ppo_networks.make_inference_fn = original_make_inference_fn
        original_ppo_networks.make_logging_inference_fn = (
            original_make_logging_inference_fn
        )

    return result
