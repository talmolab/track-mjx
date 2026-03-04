"""Temporal PPO training with recurrent decoder latent commitment."""

import functools
import time
from typing import Any, Callable, Tuple

import flax
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import logging
from brax import base, envs
from brax.training import pmap, types
from brax.training.types import Params, PRNGKey
from mujoco_playground import wrapper as mp_wrapper

from brax.training.acme import running_statistics

from track_mjx.agent import checkpointing, gradients
from track_mjx.agent.observation_utils import get_obs_shape, get_obs_sizes
from track_mjx.agent.temporal_ppo import losses, networks
from track_mjx.agent.temporal_ppo.types import TemporalPolicyCarry

InferenceParams = tuple[running_statistics.RunningStatisticsState, Params]
Metrics = types.Metrics

STEPS_IN_THOUSANDS = 1e3
_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Training state for temporal PPO."""

    optimizer_state: optax.OptState
    params: losses.TemporalPPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree: Any) -> Any:
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _safe_global_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return optax.global_norm(tree)


def _get_policy_param_subtree(grads_policy: Any, key: str) -> Any:
    params_tree = grads_policy["params"]
    if hasattr(params_tree, "get"):
        return params_tree.get(key, {})
    if key in params_tree:
        return params_tree[key]
    return {}


def actor_step_temporal(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    policy_carry: TemporalPolicyCarry,
    key: PRNGKey,
    temporal_network: networks.TemporalPPONetworks,
    train_step: jnp.ndarray | None,
    extra_fields: tuple = (),
) -> Tuple[envs.State, types.Transition, TemporalPolicyCarry]:
    """Collects one step with temporal policy."""
    actions, policy_extras, new_policy_carry = policy(
        env_state.obs, policy_carry, key, train_step=train_step
    )
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}

    _, _, eff_max = networks.compute_effective_horizons(
        boundary_mode=temporal_network.boundary_mode,
        macro_horizon=temporal_network.macro_horizon,
        min_macro_horizon=temporal_network.min_macro_horizon,
        max_macro_horizon=temporal_network.max_macro_horizon,
        horizon_ramp=temporal_network.horizon_ramp,
        horizon_ramp_steps=temporal_network.horizon_ramp_steps,
        train_step=train_step,
    )

    new_policy_carry = networks.reset_carry_on_done(
        new_policy_carry,
        nstate.done,
        cell_type=temporal_network.cell_type,
        reset_segment_step=eff_max,
    )

    transition = types.Transition(
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )

    return nstate, transition, new_policy_carry


def generate_unroll_temporal(
    env: envs.Env,
    env_state: envs.State,
    policy: Callable,
    policy_carry: TemporalPolicyCarry,
    key: PRNGKey,
    unroll_length: int,
    temporal_network: networks.TemporalPPONetworks,
    train_step: jnp.ndarray | None,
    extra_fields: tuple = (),
) -> Tuple[envs.State, types.Transition, TemporalPolicyCarry]:
    """Collects an unroll with temporal policy."""

    def f(carry, unused_t):
        state, carry_policy, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition, new_carry = actor_step_temporal(
            env,
            state,
            policy,
            carry_policy,
            current_key,
            temporal_network=temporal_network,
            train_step=train_step,
            extra_fields=extra_fields,
        )
        return (nstate, new_carry, next_key), transition

    (final_state, final_carry, _), data = jax.lax.scan(
        f,
        (env_state, policy_carry, key),
        (),
        length=unroll_length,
    )
    return final_state, data, final_carry


class TemporalEvaluator:
    """Evaluator for temporal policies."""

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[Params], Callable],
        temporal_network: networks.TemporalPPONetworks,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        self._key = key
        self._eval_walltime = 0.0
        self._network = temporal_network
        self._num_eval_envs = num_eval_envs

        eval_env = envs.training.EvalWrapper(eval_env)
        self._eval_state_to_donate = jax.jit(eval_env.reset)(
            jax.random.split(key, num_eval_envs)
        )

        def generate_eval_unroll(
            eval_env_state_donated: envs.State,
            policy_params: Params,
            unroll_key: PRNGKey,
        ) -> envs.State:
            del eval_env_state_donated
            reset_keys = jax.random.split(unroll_key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)

            policy_carry = temporal_network.policy_network.init_carry(num_eval_envs)

            final_state, _, _ = generate_unroll_temporal(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                policy_carry,
                unroll_key,
                unroll_length=episode_length // action_repeat,
                temporal_network=temporal_network,
                train_step=None,
            )
            return final_state

        self._generate_eval_unroll = jax.jit(
            generate_eval_unroll,
            donate_argnums=(0,),
            keep_unused=True,
        )
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params: Params,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
        data_split: str = "",
    ) -> Metrics:
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(
            self._eval_state_to_donate,
            policy_params,
            unroll_key,
        )
        self._eval_state_to_donate = eval_state
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t

        metrics = {}
        prefix = f"{data_split}/" if data_split else ""

        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"eval/{prefix}episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        metrics[f"eval/{prefix}avg_episode_length"] = np.mean(
            eval_metrics.episode_steps
        )
        metrics[f"eval/{prefix}epoch_eval_time"] = epoch_eval_time
        metrics[f"eval/{prefix}sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime += epoch_eval_time
        metrics = {
            f"eval/{prefix}walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }
        return metrics


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
    gate_entropy_cost: float = 1e-4,
    latent_entropy_cost: float = 0.0,
    latent_kl_weight: float = 1e-3,
    latent_ar1_weight: float = 1e-3,
    discounting: float = 0.9,
    discounting_gate: float | None = None,
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
    network_factory: Callable[
        ..., networks.TemporalPPONetworks
    ] = networks.make_temporal_intention_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    eval_env: envs.Env | None = None,
    eval_env_test_set: envs.Env | None = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: (
        Callable[[base.System, jnp.ndarray], tuple[base.System, base.System]] | None
    ) = None,
    use_kl_schedule: bool = True,
    kl_ramp_up_frac: float = 0.25,
    target_refresh_rate: float | None = None,
    lambda_refresh_rate: float = 0.0,
    checkpoint_callback: Callable[[int], None] | None = None,
    grad_clip_threshold: float = 20.0,
    wrap_for_training: Callable[..., mp_wrapper.Wrapper] = functools.partial(
        mp_wrapper.wrap_for_brax_training, full_reset=False
    ),
) -> tuple[Callable, InferenceParams, Metrics]:
    """Trains temporal PPO with fixed or learned boundary mode."""
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    def reset_fn_donated_env_state(env_state_donated, key_envs):
        return env.reset(key_envs)

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
        env_state = reset_fn_(key_envs)
        reset_fn = jax.pmap(
            reset_fn_donated_env_state,
            axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0,),
        )
    else:
        reset_fn_ = jax.jit(jax.vmap(env.reset))
        env_state = reset_fn_(key_envs)
        reset_fn = jax.jit(
            reset_fn_donated_env_state,
            donate_argnums=(0,),
            keep_unused=True,
        )

    obs_sizes = get_obs_sizes(env_state.obs)
    logging.info(f"Observation sizes: {obs_sizes}")

    config_dict["network_config"].update(
        {
            "obs_sizes": obs_sizes,
            "action_size": env.action_size,
            "normalize_observations": normalize_observations,
        }
    )

    temporal_network = network_factory(obs_sizes, env.action_size)
    make_policy = networks.make_inference_fn(temporal_network)
    make_logging_policy = networks.make_logging_inference_fn(temporal_network)
    jit_logging_inference_fn = jax.jit(make_logging_policy(deterministic=True))

    horizon_ref = (
        temporal_network.max_macro_horizon
        if temporal_network.boundary_mode == "learned"
        else temporal_network.macro_horizon
    )
    expected_refresh_pairs = max(unroll_length / float(max(horizon_ref, 1)) - 1.0, 0.0)
    if expected_refresh_pairs < 1.0:
        logging.warning(
            "Temporal unroll likely too short for stable latent temporal losses: "
            "unroll_length=%s, horizon_ref=%s, expected_refresh_pairs_per_unroll=%.2f",
            unroll_length,
            horizon_ref,
            expected_refresh_pairs,
        )

    envs_per_device = num_envs // device_count

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_threshold),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0, eps=1e-5),
    )

    latent_kl_schedule = None
    latent_ar1_schedule = None
    if use_kl_schedule:
        ramp_steps = int(num_evals * kl_ramp_up_frac)
        latent_kl_schedule = losses.create_ramp_schedule(
            max_value=latent_kl_weight,
            ramp_steps=ramp_steps,
            schedule="linear",
        )
        latent_ar1_schedule = losses.create_ramp_schedule(
            max_value=latent_ar1_weight,
            ramp_steps=ramp_steps,
            schedule="linear",
        )

    loss_fn = functools.partial(
        losses.compute_temporal_ppo_loss,
        temporal_ppo_network=temporal_network,
        entropy_cost=entropy_cost,
        gate_entropy_cost=gate_entropy_cost,
        latent_entropy_cost=latent_entropy_cost,
        latent_kl_weight=latent_kl_weight,
        latent_ar1_weight=latent_ar1_weight,
        discounting=discounting,
        discounting_gate=discounting_gate,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
        latent_kl_schedule=latent_kl_schedule,
        latent_ar1_schedule=latent_ar1_schedule,
        target_refresh_rate=target_refresh_rate,
        lambda_refresh_rate=lambda_refresh_rate,
    )

    init_params = losses.TemporalPPONetworkParams(
        policy=temporal_network.policy_network.init(key_policy),
        value=temporal_network.value_network.init(key_value),
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(get_obs_shape(env_state.obs)),
        env_steps=0,
    )

    if checkpoint_to_restore is not None:
        training_state = checkpointing.load_training_state(
            checkpoint_to_restore,
            training_state,
        )
        logging.info(f"Restored checkpoint from {checkpoint_to_restore}")

    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
    )

    def gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        it,
        optimizer_state,
    ):
        (loss_value, metrics), grads = loss_and_pgrad_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            it,
        )

        grad_norm = _safe_global_norm(grads)
        grad_clipped = (grad_norm > grad_clip_threshold).astype(jnp.float32)

        encoder_grad_norm = _safe_global_norm(
            _get_policy_param_subtree(grads.policy, "encoder")
        )
        decoder_grad_norm = _safe_global_norm(
            _get_policy_param_subtree(grads.policy, "decoder")
        )
        gate_grad_norm = _safe_global_norm(
            _get_policy_param_subtree(grads.policy, "gate_head")
        )
        value_grad_norm = _safe_global_norm(grads.value)
        policy_grad_norm = _safe_global_norm(grads.policy)

        metrics = {
            **metrics,
            "grad_norm": grad_norm,
            "grad_clipped": grad_clipped,
            "grad_norm_encoder": encoder_grad_norm,
            "grad_norm_decoder": decoder_grad_norm,
            "grad_norm_gate": gate_grad_norm,
            "grad_norm_value": value_grad_norm,
            "grad_norm_policy_total": policy_grad_norm,
            "expected_refresh_pairs_per_unroll": jnp.asarray(
                expected_refresh_pairs, dtype=jnp.float32
            ),
        }

        params_update, optimizer_state = optimizer.update(
            grads,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, params_update)
        return (loss_value, metrics), params, optimizer_state

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key_mb, it = carry
        key_mb, key_loss = jax.random.split(key_mb)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            it,
            optimizer_state,
        )
        return (optimizer_state, params, key_mb, it), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key_sgd, it = carry
        key_sgd, key_perm, key_grad = jax.random.split(key_sgd, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad, it),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key_sgd, it), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, TemporalPolicyCarry, PRNGKey, int],
        unused_t,
    ) -> Tuple[
        Tuple[TrainingState, envs.State, TemporalPolicyCarry, PRNGKey, int], Metrics
    ]:
        training_state, state, policy_carry, key_train, it = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key_train, 3)

        policy = make_policy(
            (training_state.normalizer_params, training_state.params.policy)
        )

        def f(carry_unroll, unused):
            current_state, current_carry, current_key = carry_unroll
            current_key, next_key = jax.random.split(current_key)
            initial_carry = current_carry
            next_state, data, new_carry = generate_unroll_temporal(
                env,
                current_state,
                policy,
                current_carry,
                current_key,
                unroll_length,
                temporal_network=temporal_network,
                train_step=it,
                extra_fields=("truncation",),
            )
            return (next_state, new_carry, next_key), (data, initial_carry)

        (state, policy_carry, _), (data, initial_policy_carry) = jax.lax.scan(
            f,
            (state, policy_carry, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            data,
        )

        initial_policy_carry = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            initial_policy_carry,
        )

        policy_extras = dict(data.extras["policy_extras"])
        policy_extras["initial_policy_hidden"] = initial_policy_carry.decoder_hidden
        policy_extras["initial_segment_step"] = initial_policy_carry.segment_step
        policy_extras["initial_current_latent"] = initial_policy_carry.current_latent
        policy_extras["initial_latent_mean"] = initial_policy_carry.current_latent_mean
        policy_extras["initial_latent_logvar"] = (
            initial_policy_carry.current_latent_logvar
        )

        data = types.Transition(
            observation=data.observation,
            action=data.action,
            reward=data.reward,
            discount=data.discount,
            next_observation=data.next_observation,
            extras={
                "policy_extras": policy_extras,
                "state_extras": data.extras["state_extras"],
            },
        )

        if normalize_observations:
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
        else:
            normalizer_params = training_state.normalizer_params

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=jnp.int32(
                training_state.env_steps
                + env_step_per_training_step / STEPS_IN_THOUSANDS
            ),
        )
        return (new_training_state, state, policy_carry, new_key, it), metrics

    def training_epoch(
        training_state: TrainingState,
        state: envs.State,
        policy_carry: TemporalPolicyCarry,
        key_epoch: PRNGKey,
        it: int,
    ) -> Tuple[TrainingState, envs.State, TemporalPolicyCarry, Metrics]:
        (training_state, state, policy_carry, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, policy_carry, key_epoch, it),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, policy_carry, loss_metrics

    training_epoch = jax.pmap(
        training_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1, 2),
    )

    training_walltime = 0.0

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        policy_carry: TemporalPolicyCarry,
        key_epoch: PRNGKey,
        it: int,
    ) -> Tuple[TrainingState, envs.State, TemporalPolicyCarry, Metrics]:
        nonlocal training_walltime
        t = time.time()

        training_state, env_state, policy_carry = _strip_weak_type(
            (training_state, env_state, policy_carry)
        )
        step = jnp.ones_like(training_state.env_steps) * it
        result = training_epoch(
            training_state,
            env_state,
            policy_carry,
            key_epoch,
            step,
        )
        training_state, env_state, policy_carry, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/expected_refresh_pairs_per_unroll": expected_refresh_pairs,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, policy_carry, metrics

    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use],
    )

    policy_carry = temporal_network.policy_network.init_carry(envs_per_device)
    policy_carry = jax.device_put_replicated(
        policy_carry,
        jax.local_devices()[:local_devices_to_use],
    )

    if eval_env is None:
        eval_env = environment
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=None,
    )

    evaluator = TemporalEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        temporal_network,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    evaluator_test_set = None
    if eval_env_test_set is not None:
        key_env, key_env_test_set = jax.random.split(key_env, 2)
        eval_env_test_set = wrap_for_training(
            eval_env_test_set,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=None,
        )
        evaluator_test_set = TemporalEvaluator(
            eval_env_test_set,
            functools.partial(make_policy, deterministic=deterministic_eval),
            temporal_network,
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=key_env_test_set,
        )

    start_it = 0
    if ckpt_mgr is not None:
        latest_step = ckpt_mgr.latest_step()
        if latest_step is not None:
            num_evals_after_init -= latest_step
            start_it = latest_step

    logging.info(
        f"Starting at iteration: {start_it} with {num_evals_after_init} evals left"
    )

    metrics = {}
    if process_id == 0 and num_evals > 1 and start_it == 0:
        policy_param = _unpmap(
            (training_state.normalizer_params, training_state.params.policy)
        )
        metrics = evaluator.run_evaluation(policy_param, training_metrics={})
        if evaluator_test_set is not None:
            metrics = evaluator_test_set.run_evaluation(
                policy_param,
                metrics,
                data_split="test_set",
            )
        progress_fn(start_it, metrics)

        if ckpt_mgr is not None:
            ckpt_mgr.save(
                step=0,
                args=ocp.args.Composite(
                    policy=ocp.args.StandardSave(policy_param),
                    train_state=ocp.args.StandardSave(_unpmap(training_state)),
                    config=ocp.args.JsonSave(config_dict),
                ),
            )
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(0)
                except (OSError, IOError) as e:
                    logging.error(
                        "Initial checkpoint callback failed with I/O error: %s. "
                        "Training will continue but checkpoint state may be incomplete.",
                        e,
                    )

    training_metrics = {}
    start_it += 1
    current_step = int(_unpmap(training_state.env_steps))

    for it in range(start_it, num_evals_after_init + start_it):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (
                training_state,
                env_state,
                policy_carry,
                training_metrics,
            ) = training_epoch_with_timing(
                training_state,
                env_state,
                policy_carry,
                epoch_keys,
                it,
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])

            if num_resets_per_eval > 0:
                env_state = reset_fn((training_state, env_state), key_envs)
                policy_carry = temporal_network.policy_network.init_carry(
                    envs_per_device
                )
                policy_carry = jax.device_put_replicated(
                    policy_carry,
                    jax.local_devices()[:local_devices_to_use],
                )

        if process_id == 0:
            policy_param = _unpmap(
                (training_state.normalizer_params, training_state.params.policy)
            )
            metrics = evaluator.run_evaluation(policy_param, training_metrics)
            if evaluator_test_set is not None:
                metrics = evaluator_test_set.run_evaluation(
                    policy_param,
                    metrics,
                    data_split="test_set",
                )

            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            render_video = (
                it % config_dict.get("render_config", {}).get("render_interval", 10)
                == 0
            )
            policy_params_fn(
                current_step=it,
                jit_logging_inference_fn=jit_logging_inference_fn,
                params=policy_param,
                policy_params_fn_key=policy_params_fn_key,
                render_video=render_video,
                ppo_network=temporal_network,
            )

            progress_fn(current_step, metrics)

            if ckpt_mgr is not None:
                checkpointing.save(
                    ckpt_mgr,
                    it,
                    policy_param,
                    _unpmap(training_state),
                    config_dict,
                    checkpoint_callback,
                )

    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params.policy))
    logging.info("total steps: %s", current_step)
    pmap.synchronize_hosts()
    return make_policy, params, metrics
