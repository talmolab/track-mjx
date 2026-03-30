"""MoSeq decoder PPO training module.

Wraps the main PPO training by temporarily patching the loss module and
inference functions to use the MoSeq decoder-only policy network.
Same monkey-patching pattern as ``vqvae_jax/vq_ppo.py``.

Supports both feedforward and recurrent (RNN) decoders.  When
``use_rnn_decoder=True``, the carry infrastructure in ``ppo.train``
is activated to thread RNN hidden state through rollouts and loss
computation.
"""

import functools
import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import types
from brax.training.types import Metrics
import orbax.checkpoint as ocp
from mujoco_playground import wrapper as mp_wrapper

from track_mjx.agent.ff_ppo import ppo, losses as original_losses
from track_mjx.agent.ff_ppo import ppo_networks as original_ppo_networks

from moseq_losses import compute_moseq_ppo_loss, compute_moseq_recurrent_ppo_loss
from moseq_ppo_networks import (
    make_moseq_decoder_ppo_networks,
    make_moseq_inference_fn,
    make_moseq_logging_inference_fn,
    make_moseq_recurrent_inference_fn,
    make_moseq_recurrent_logging_inference_fn,
    make_moseq_rnn_rollout_policy_fn,
)

# ---------------------------------------------------------------------------
# Carry-aware unroll for RNN decoder
# ---------------------------------------------------------------------------


def generate_unroll_rnn_moseq(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    hidden_state: list[jnp.ndarray],
    key,
    unroll_length: int,
    extra_fields: tuple = (),
) -> tuple[envs.State, types.Transition, list[jnp.ndarray]]:
    """Collect a trajectory with the RNN decoder policy.

    Args:
        env: Environment.
        env_state: Current environment state.
        policy: Carry-aware policy ``(obs, carry, key) -> (action, extras, new_carry)``.
        hidden_state: Initial RNN hidden states (list of ``[B, H]``).
        key: PRNG key.
        unroll_length: Number of timesteps to collect.
        extra_fields: Extra fields to extract from env state info.

    Returns:
        ``(final_state, transitions, final_hidden)``.
    """

    def step_fn(carry, _unused_t):
        state, hidden, current_key = carry
        current_key, next_key = jax.random.split(current_key)

        actions, policy_extras, new_hidden = policy(state.obs, hidden, current_key)
        nstate = env.step(state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        transition = types.Transition(
            observation=state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={"policy_extras": policy_extras, "state_extras": state_extras},
        )

        # Reset hidden state where episodes ended
        done_expanded = nstate.done[..., None]
        new_hidden = [jnp.where(done_expanded, 0.0, h) for h in new_hidden]

        return (nstate, new_hidden, next_key), transition

    (final_state, final_hidden, _), data = jax.lax.scan(
        step_fn, (env_state, hidden_state, key), (), length=unroll_length
    )
    return final_state, data, final_hidden


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


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
    # MoSeq-specific
    num_codes: int = 32,
    code_embed_dim: int = 16,
    kl_weight: float = 0.0,
    # RNN decoder
    use_rnn_decoder: bool = False,
    rnn_hidden_sizes: tuple[int, ...] = (256,),
    z_e_scale_fn: Callable[[int], float] | None = None,
    kl_weight_fn: Callable[[int], float] | None = None,
):
    """Train a MoSeq decoder PPO agent.

    Monkey-patches the PPO loss and inference functions, delegates to
    ``ppo.train``, then restores originals.

    Args:
        use_rnn_decoder: Use RNN decoder with carry-aware training.
        rnn_hidden_sizes: GRU hidden sizes per layer.
        z_e_scale_fn: Optional ``step -> z_e_scale`` function for scaling.
        kl_weight_fn: Optional ``step -> kl_weight`` function for KL scheduling.

    Returns:
        Tuple of ``(make_policy, params, metrics)``.
    """
    original_compute_ppo_loss = original_losses.compute_ppo_loss
    original_make_inference_fn = original_ppo_networks.make_inference_fn
    original_make_logging_inference_fn = original_ppo_networks.make_logging_inference_fn

    _kl_weight = kl_weight
    _z_e_scale_fn = z_e_scale_fn
    _kl_weight_fn = kl_weight_fn

    if use_rnn_decoder:
        # --- RNN mode: carry-aware loss + inference ---
        _rnn_hidden_sizes = tuple(rnn_hidden_sizes)

        def moseq_rnn_loss_wrapper(
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
            z_e_scale = 1.0
            if _z_e_scale_fn is not None:
                z_e_scale = _z_e_scale_fn(step)
            effective_kl = _kl_weight
            if _kl_weight_fn is not None:
                effective_kl = _kl_weight_fn(step)
            return compute_moseq_recurrent_ppo_loss(
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
                kl_weight=effective_kl,
                z_e_scale=z_e_scale,
            )

        original_losses.compute_ppo_loss = moseq_rnn_loss_wrapper
        original_ppo_networks.make_inference_fn = make_moseq_recurrent_inference_fn
        original_ppo_networks.make_logging_inference_fn = (
            make_moseq_recurrent_logging_inference_fn
        )

        # Carry callbacks for ppo.train
        _generate_unroll_fn = generate_unroll_rnn_moseq

        _init_carry_state_fn = lambda n: [jnp.zeros((n, h)) for h in _rnn_hidden_sizes]
        _make_rollout_policy_fn = make_moseq_rnn_rollout_policy_fn

        logging.info(
            "RNN decoder mode: hidden_sizes=%s, kl_weight=%s",
            _rnn_hidden_sizes,
            _kl_weight,
        )

    else:
        # --- Feedforward mode (original) ---
        def moseq_loss_wrapper(
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
                kl_weight=_kl_weight,
            )

        original_losses.compute_ppo_loss = moseq_loss_wrapper
        original_ppo_networks.make_inference_fn = make_moseq_inference_fn
        original_ppo_networks.make_logging_inference_fn = (
            make_moseq_logging_inference_fn
        )

        _generate_unroll_fn = None
        _init_carry_state_fn = None
        _make_rollout_policy_fn = None

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
            generate_unroll_fn=_generate_unroll_fn,
            init_carry_state_fn=_init_carry_state_fn,
            make_rollout_policy_fn=_make_rollout_policy_fn,
        )
    finally:
        original_losses.compute_ppo_loss = original_compute_ppo_loss
        original_ppo_networks.make_inference_fn = original_make_inference_fn
        original_ppo_networks.make_logging_inference_fn = (
            original_make_logging_inference_fn
        )

    return result
