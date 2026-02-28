"""Temporal PPO networks with a recurrent decoder and latent commitment."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, types
from brax.training.types import PRNGKey
from flax import linen as nn

from brax.training.acme import running_statistics

from track_mjx.agent.ff_ppo.intention_network import Encoder, reparameterize
from track_mjx.agent.observation_utils import normalizer_select
from track_mjx.agent.temporal_ppo.types import (
    HiddenState,
    RNNCellType,
    TemporalBoundaryMode,
    TemporalPolicyCarry,
)

Initializer = Callable[..., Any]


_RNN_CELL_CLASSES: dict[RNNCellType, type[nn.RNNCellBase]] = {
    "simple": nn.SimpleCell,
    "gru": nn.GRUCell,
    "lstm": nn.LSTMCell,
}


def get_rnn_cell(
    cell_type: RNNCellType,
    hidden_size: int,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
) -> nn.RNNCellBase:
    """Returns the configured Flax RNN cell."""
    if cell_type not in _RNN_CELL_CLASSES:
        raise ValueError(
            f"Unsupported RNN cell type: {cell_type}. "
            f"Must be one of {list(_RNN_CELL_CLASSES.keys())}."
        )
    return _RNN_CELL_CLASSES[cell_type](features=hidden_size, kernel_init=kernel_init)


def init_hidden_state(
    cell_type: RNNCellType,
    hidden_size: int,
    batch_size: int,
) -> HiddenState:
    """Initializes one RNN layer hidden state."""
    zeros = jnp.zeros((batch_size, hidden_size))
    return (zeros, zeros) if cell_type == "lstm" else zeros


def _reset_single_hidden(
    hidden: HiddenState,
    done_expanded: jnp.ndarray,
    cell_type: RNNCellType,
) -> HiddenState:
    if cell_type == "lstm":
        c, h = hidden
        return (
            jnp.where(done_expanded, 0.0, c),
            jnp.where(done_expanded, 0.0, h),
        )
    return jnp.where(done_expanded, 0.0, hidden)


def reset_hidden_on_done(
    hidden: list[HiddenState],
    done: jnp.ndarray,
    cell_type: RNNCellType,
) -> list[HiddenState]:
    """Resets hidden states at episode boundaries."""
    done_expanded = done[..., None]
    return [_reset_single_hidden(h, done_expanded, cell_type) for h in hidden]


def _extract_top_hidden(
    hidden: list[HiddenState], cell_type: RNNCellType
) -> jnp.ndarray:
    """Gets the top-layer hidden activation used by the gate head."""
    top = hidden[-1]
    if cell_type == "lstm":
        return top[1]
    return top


def _split_policy_rng(
    key: jax.Array,
    obs_is_batched: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits policy RNG into encoder/noise/gate streams."""
    if key.ndim == 1:
        return tuple(jax.random.split(key, 3))

    if not obs_is_batched:
        return tuple(jax.random.split(key[0], 3))

    keys = jax.vmap(lambda k: jax.random.split(k, 3))(key)
    return keys[:, 0], keys[:, 1], keys[:, 2]


def _ramp_int(
    target: int,
    train_step: jnp.ndarray | None,
    enabled: bool,
    ramp_steps: int,
) -> jnp.ndarray:
    """Ramps an integer parameter from 1 to target."""
    if (not enabled) or ramp_steps <= 0 or train_step is None:
        return jnp.asarray(target, dtype=jnp.int32)

    target_f = jnp.asarray(target, dtype=jnp.float32)
    step_f = jnp.asarray(train_step, dtype=jnp.float32)
    frac = jnp.clip(step_f / float(ramp_steps), 0.0, 1.0)
    value = 1.0 + frac * (target_f - 1.0)
    return jnp.maximum(
        jnp.asarray(1, dtype=jnp.int32), jnp.asarray(jnp.round(value), dtype=jnp.int32)
    )


def compute_effective_horizons(
    boundary_mode: TemporalBoundaryMode,
    macro_horizon: int,
    min_macro_horizon: int,
    max_macro_horizon: int,
    horizon_ramp: bool,
    horizon_ramp_steps: int,
    train_step: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes effective (possibly ramped) horizons."""
    if boundary_mode == "fixed":
        eff_macro = _ramp_int(
            macro_horizon,
            train_step=train_step,
            enabled=horizon_ramp,
            ramp_steps=horizon_ramp_steps,
        )
        return eff_macro, eff_macro, eff_macro

    eff_min = _ramp_int(
        min_macro_horizon,
        train_step=train_step,
        enabled=horizon_ramp,
        ramp_steps=horizon_ramp_steps,
    )
    eff_max = _ramp_int(
        max_macro_horizon,
        train_step=train_step,
        enabled=horizon_ramp,
        ramp_steps=horizon_ramp_steps,
    )
    eff_min = jnp.minimum(eff_min, eff_max)
    return eff_min, eff_min, eff_max


def init_temporal_carry(
    *,
    cell_type: RNNCellType,
    rnn_hidden_sizes: Sequence[int],
    latent_size: int,
    batch_size: int,
    reset_segment_step: int,
) -> TemporalPolicyCarry:
    """Initializes temporal policy carry."""
    hidden = [
        init_hidden_state(cell_type, hidden_size, batch_size)
        for hidden_size in rnn_hidden_sizes
    ]
    return TemporalPolicyCarry(
        decoder_hidden=hidden,
        current_latent=jnp.zeros((batch_size, latent_size), dtype=jnp.float32),
        current_latent_mean=jnp.zeros((batch_size, latent_size), dtype=jnp.float32),
        current_latent_logvar=jnp.zeros((batch_size, latent_size), dtype=jnp.float32),
        segment_step=jnp.ones((batch_size,), dtype=jnp.int32)
        * jnp.asarray(reset_segment_step, dtype=jnp.int32),
    )


def reset_carry_on_done(
    carry: TemporalPolicyCarry,
    done: jnp.ndarray,
    *,
    cell_type: RNNCellType,
    reset_segment_step: jnp.ndarray,
) -> TemporalPolicyCarry:
    """Resets temporal carry on done and forces refresh next step."""
    done_expanded = done[..., None]
    return TemporalPolicyCarry(
        decoder_hidden=reset_hidden_on_done(carry.decoder_hidden, done, cell_type),
        current_latent=jnp.where(done_expanded, 0.0, carry.current_latent),
        current_latent_mean=jnp.where(done_expanded, 0.0, carry.current_latent_mean),
        current_latent_logvar=jnp.where(
            done_expanded, 0.0, carry.current_latent_logvar
        ),
        segment_step=jnp.where(
            done,
            jnp.asarray(reset_segment_step, dtype=jnp.int32),
            carry.segment_step,
        ),
    )


def bernoulli_log_prob(logits: jnp.ndarray, samples: jnp.ndarray) -> jnp.ndarray:
    """Computes Bernoulli log-prob for {0,1} samples."""
    samples = samples.astype(jnp.float32)
    return samples * jax.nn.log_sigmoid(logits) + (1.0 - samples) * jax.nn.log_sigmoid(
        -logits
    )


def bernoulli_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """Computes Bernoulli entropy."""
    probs = jax.nn.sigmoid(logits)
    eps = 1e-8
    return -(probs * jnp.log(probs + eps) + (1.0 - probs) * jnp.log(1.0 - probs + eps))


def gaussian_diag_log_prob(
    mean: jnp.ndarray,
    logvar: jnp.ndarray,
    sample: jnp.ndarray,
) -> jnp.ndarray:
    """Computes diagonal-Gaussian log-prob."""
    inv_var = jnp.exp(-logvar)
    log_two_pi = jnp.log(2.0 * jnp.pi)
    return -0.5 * jnp.sum(
        log_two_pi + logvar + jnp.square(sample - mean) * inv_var,
        axis=-1,
    )


def gaussian_diag_entropy(logvar: jnp.ndarray) -> jnp.ndarray:
    """Computes diagonal-Gaussian entropy."""
    log_two_pi_e = jnp.log(2.0 * jnp.pi * jnp.e)
    return 0.5 * jnp.sum(logvar + log_two_pi_e, axis=-1)


class TemporalDecoder(nn.Module):
    """RNN decoder that outputs motor policy logits."""

    output_size: int
    rnn_hidden_sizes: Sequence[int] = (256,)
    cell_type: RNNCellType = "gru"
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    def setup(self):
        self.rnn_cells = [
            get_rnn_cell(self.cell_type, hidden_size, self.kernel_init)
            for hidden_size in self.rnn_hidden_sizes
        ]
        self.action_head = nn.Dense(
            self.output_size,
            name="action_head",
            kernel_init=self.kernel_init,
        )

    @property
    def num_rnn_layers(self) -> int:
        return len(self.rnn_hidden_sizes)

    def __call__(
        self,
        x: jnp.ndarray,
        hidden: list[HiddenState],
        get_activation: bool = False,
    ) -> (
        tuple[jnp.ndarray, list[HiddenState], jnp.ndarray]
        | tuple[
            jnp.ndarray,
            list[HiddenState],
            jnp.ndarray,
            dict[str, jnp.ndarray],
        ]
    ):
        if get_activation:
            activations: dict[str, jnp.ndarray] = {}

        new_hidden = []
        rnn_input = x
        for i, (cell, h) in enumerate(zip(self.rnn_cells, hidden)):
            new_h, _ = cell(h, rnn_input)
            new_hidden.append(new_h)
            if self.cell_type == "lstm":
                rnn_input = new_h[1]
            else:
                rnn_input = new_h
            if get_activation:
                activations[f"rnn_layer_{i}"] = rnn_input

        motor_logits = self.action_head(rnn_input)
        if get_activation:
            return motor_logits, new_hidden, rnn_input, activations
        return motor_logits, new_hidden, rnn_input


class TemporalIntentionNetwork(nn.Module):
    """Temporal encoder-decoder policy with latent commitment."""

    output_size: int
    encoder_layers: Sequence[int]
    latents: int = 60
    rnn_hidden_sizes: Sequence[int] = (256,)
    cell_type: RNNCellType = "gru"
    boundary_mode: TemporalBoundaryMode = "fixed"
    macro_horizon: int = 16
    min_macro_horizon: int = 4
    max_macro_horizon: int = 64
    proprioception_noise_std: float = 0.0
    horizon_ramp: bool = False
    horizon_ramp_steps: int = 0
    eval_gate_threshold: float = 0.5

    def setup(self):
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = TemporalDecoder(
            output_size=self.output_size,
            rnn_hidden_sizes=self.rnn_hidden_sizes,
            cell_type=self.cell_type,
        )
        if self.boundary_mode == "learned":
            self.gate_head = nn.Dense(1, name="gate_head")

    def _effective_horizons(
        self, train_step: jnp.ndarray | None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return compute_effective_horizons(
            self.boundary_mode,
            macro_horizon=self.macro_horizon,
            min_macro_horizon=self.min_macro_horizon,
            max_macro_horizon=self.max_macro_horizon,
            horizon_ramp=self.horizon_ramp,
            horizon_ramp_steps=self.horizon_ramp_steps,
            train_step=train_step,
        )

    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        carry: TemporalPolicyCarry,
        key: jax.Array,
        deterministic: bool = False,
        gate_sample_override: jnp.ndarray | None = None,
        latent_override: jnp.ndarray | None = None,
        train_step: jnp.ndarray | None = None,
        get_activation: bool = False,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        TemporalPolicyCarry,
    ]:
        """Runs a single temporal policy step."""
        traj = obs["task_obs"]
        egocentric_obs = obs["proprioception"]
        obs_is_batched = traj.ndim >= 2

        encoder_rng, noise_rng, gate_rng = _split_policy_rng(key, obs_is_batched)

        if not deterministic and self.proprioception_noise_std > 0.0:
            if noise_rng.ndim == 1:
                noise = jax.random.normal(noise_rng, egocentric_obs.shape)
            elif not obs_is_batched:
                noise = jax.random.normal(noise_rng[0], egocentric_obs.shape)
            else:
                noise = jax.vmap(
                    lambda rng_key, obs_i: jax.random.normal(rng_key, obs_i.shape)
                )(noise_rng, egocentric_obs)
            egocentric_obs = egocentric_obs * (
                1.0 + self.proprioception_noise_std * noise
            )

        eff_min, _, eff_max = self._effective_horizons(train_step)

        if self.boundary_mode == "learned":
            prev_top_hidden = jax.lax.stop_gradient(
                _extract_top_hidden(carry.decoder_hidden, self.cell_type)
            )
            gate_logits = jnp.squeeze(self.gate_head(prev_top_hidden), axis=-1)
            gate_probs = jax.nn.sigmoid(gate_logits)

            if gate_sample_override is None:
                if deterministic:
                    gate_samples = (gate_probs > self.eval_gate_threshold).astype(
                        jnp.float32
                    )
                else:
                    if gate_rng.ndim == 1:
                        gate_samples = jax.random.bernoulli(
                            gate_rng, gate_probs
                        ).astype(jnp.float32)
                    elif not obs_is_batched:
                        gate_samples = jax.random.bernoulli(
                            gate_rng[0], gate_probs
                        ).astype(jnp.float32)
                    else:
                        gate_samples = jax.vmap(jax.random.bernoulli)(
                            gate_rng, gate_probs
                        ).astype(jnp.float32)
            else:
                gate_samples = gate_sample_override.astype(jnp.float32)

            gate_valid = (carry.segment_step >= eff_min) & (
                carry.segment_step < eff_max
            )
            refresh_from_gate = gate_valid & (gate_samples > 0.5)
            refresh_mask = (carry.segment_step >= eff_max) | refresh_from_gate
        else:
            gate_logits = jnp.zeros_like(carry.segment_step, dtype=jnp.float32)
            gate_probs = jnp.zeros_like(carry.segment_step, dtype=jnp.float32)
            gate_samples = jnp.zeros_like(carry.segment_step, dtype=jnp.float32)
            gate_valid = jnp.zeros_like(carry.segment_step, dtype=bool)
            refresh_mask = carry.segment_step >= eff_max

        if get_activation:
            (fresh_mean, fresh_logvar), encoder_activations = self.encoder(
                traj, get_activation=True
            )
        else:
            fresh_mean, fresh_logvar = self.encoder(traj, get_activation=False)

        if deterministic:
            fresh_z = fresh_mean
        else:
            fresh_z = reparameterize(encoder_rng, fresh_mean, fresh_logvar)

        refresh_expanded = refresh_mask[..., None]
        reused_z = jax.lax.stop_gradient(carry.current_latent)
        reused_mean = jax.lax.stop_gradient(carry.current_latent_mean)
        reused_logvar = jax.lax.stop_gradient(carry.current_latent_logvar)

        if latent_override is None:
            selected_z = jnp.where(refresh_expanded, fresh_z, reused_z)
        else:
            selected_z = latent_override.astype(fresh_z.dtype)
        selected_mean = jnp.where(refresh_expanded, fresh_mean, reused_mean)
        selected_logvar = jnp.where(refresh_expanded, fresh_logvar, reused_logvar)

        fresh_latent_log_prob = gaussian_diag_log_prob(
            fresh_mean, fresh_logvar, selected_z
        )
        latent_log_prob = jnp.where(refresh_mask, fresh_latent_log_prob, 0.0)
        fresh_latent_entropy = gaussian_diag_entropy(fresh_logvar)
        latent_entropy = jnp.where(refresh_mask, fresh_latent_entropy, 0.0)

        decoder_latent = jax.lax.stop_gradient(selected_z)
        decoder_input = jnp.concatenate([decoder_latent, egocentric_obs], axis=-1)
        if get_activation:
            (
                motor_logits,
                new_hidden,
                _,
                decoder_activations,
            ) = self.decoder(decoder_input, carry.decoder_hidden, get_activation=True)
        else:
            motor_logits, new_hidden, _ = self.decoder(
                decoder_input, carry.decoder_hidden, get_activation=False
            )

        next_segment_step = jnp.where(
            refresh_mask,
            jnp.ones_like(carry.segment_step, dtype=jnp.int32),
            carry.segment_step + 1,
        )

        new_carry = TemporalPolicyCarry(
            decoder_hidden=new_hidden,
            current_latent=selected_z,
            current_latent_mean=selected_mean,
            current_latent_logvar=selected_logvar,
            segment_step=next_segment_step,
        )

        if get_activation:
            _ = {
                "encoder": encoder_activations,
                "decoder": decoder_activations,
                "traj_obs": traj,
                "egocentric_obs": egocentric_obs,
                "intention": selected_z,
            }

        return (
            motor_logits,
            selected_mean,
            selected_logvar,
            selected_z,
            latent_log_prob,
            latent_entropy,
            gate_logits,
            gate_probs,
            gate_samples,
            gate_valid.astype(jnp.float32),
            refresh_mask.astype(jnp.float32),
            new_carry,
        )


@dataclasses.dataclass
class RecurrentNetwork:
    """Container for temporal policy network functions."""

    init: Callable[..., Any]
    apply: Callable[..., Any]
    apply_sequence: Callable[..., Any]
    init_carry: Callable[[int], TemporalPolicyCarry]


@dataclasses.dataclass
class TemporalValueNetwork:
    """Value network with committed-latent conditioning."""

    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class TemporalPPONetworks:
    """Container for temporal PPO network components."""

    policy_network: RecurrentNetwork
    value_network: TemporalValueNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    rnn_hidden_sizes: tuple[int, ...]
    cell_type: RNNCellType
    boundary_mode: TemporalBoundaryMode
    macro_horizon: int
    min_macro_horizon: int
    max_macro_horizon: int
    horizon_ramp: bool
    horizon_ramp_steps: int


class LatentConditionedValueMLP(nn.Module):
    """MLP value network on normalized observation plus committed latent."""

    hidden_layer_sizes: Sequence[int]
    condition_on_latent: bool = True

    @nn.compact
    def __call__(
        self,
        obs: Mapping[str, jnp.ndarray],
        latent: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.condition_on_latent:
            x = jnp.concatenate(
                [obs["task_obs"], obs["proprioception"], latent], axis=-1
            )
        else:
            x = jnp.concatenate([obs["task_obs"], obs["proprioception"]], axis=-1)

        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(hidden_size, name=f"hidden_{i}")(x)
            x = nn.silu(x)
            x = nn.LayerNorm()(x)

        value = nn.Dense(1, name="value_head")(x)
        return jnp.squeeze(value, axis=-1)


def make_temporal_value_network(
    obs_sizes: Mapping[str, int],
    latent_size: int,
    hidden_layer_sizes: Sequence[int],
    value_obs_key: str,
    condition_on_latent: bool,
) -> TemporalValueNetwork:
    """Creates a value network conditioned on committed latent."""
    value_module = LatentConditionedValueMLP(
        hidden_layer_sizes=hidden_layer_sizes,
        condition_on_latent=condition_on_latent,
    )

    dummy_obs = {
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"]), dtype=jnp.float32),
        "proprioception": jnp.zeros(
            (1, obs_sizes["proprioception"]), dtype=jnp.float32
        ),
    }
    dummy_latent = jnp.zeros((1, latent_size), dtype=jnp.float32)

    def init(key: jax.Array):
        return value_module.init(key, dummy_obs, dummy_latent)

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        value_params,
        obs: Mapping[str, Mapping[str, jnp.ndarray]],
        latent: jnp.ndarray,
    ) -> jnp.ndarray:
        value_normalizer = normalizer_select(processor_params, value_obs_key)
        normalized_obs = running_statistics.normalize(
            obs[value_obs_key], value_normalizer
        )
        return value_module.apply(value_params, normalized_obs, latent)

    return TemporalValueNetwork(init=init, apply=apply)


def make_inference_fn(
    temporal_ppo_networks: TemporalPPONetworks,
) -> Callable[..., types.Policy]:
    """Creates temporal policy factory for acting/evaluation."""

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
        get_activation: bool = False,
    ) -> Callable:
        del get_activation
        policy_network = temporal_ppo_networks.policy_network
        parametric_action_distribution = (
            temporal_ppo_networks.parametric_action_distribution
        )

        def policy(
            observations: types.Observation,
            carry: TemporalPolicyCarry,
            key_sample: PRNGKey,
            train_step: jnp.ndarray | None = None,
        ) -> tuple[types.Action, types.Extra, TemporalPolicyCarry]:
            key_action, key_network = jax.random.split(key_sample)

            obs_leaf = jax.tree_util.tree_leaves(observations)[0]
            if obs_leaf.ndim >= 2:
                batch_size = obs_leaf.shape[0]
                per_sample_keys = jax.random.split(key_network, batch_size)
            else:
                per_sample_keys = key_network

            (
                motor_logits,
                latent_mean,
                latent_logvar,
                latent_z,
                latent_log_prob,
                latent_entropy,
                gate_logits,
                gate_probs,
                gate_samples,
                gate_valid,
                refresh_mask,
                new_carry,
            ) = policy_network.apply(
                *params,
                observations,
                carry,
                per_sample_keys,
                deterministic=deterministic,
                train_step=train_step,
            )

            if deterministic:
                action = jnp.array(parametric_action_distribution.mode(motor_logits))
                extras: dict[str, jnp.ndarray] = {
                    "latent_mean": latent_mean,
                    "latent_logvar": latent_logvar,
                    "latent": latent_z,
                    "latent_log_prob": latent_log_prob,
                    "latent_entropy": latent_entropy,
                    "refresh_mask": refresh_mask,
                    "gate_prob": gate_probs,
                    "gate_sample": gate_samples,
                    "gate_valid": gate_valid,
                    "gate_logit": gate_logits,
                }
                return action, extras, new_carry

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                motor_logits, key_action
            )
            motor_log_prob = parametric_action_distribution.log_prob(
                motor_logits, raw_actions
            )
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )

            gate_log_prob = bernoulli_log_prob(gate_logits, gate_samples)

            extras = {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "latent": latent_z,
                "latent_log_prob": latent_log_prob,
                "latent_entropy": latent_entropy,
                "log_prob": motor_log_prob,
                "raw_action": raw_actions,
                "logits": motor_logits,
                "policy_rng": per_sample_keys,
                "gate_log_prob": gate_log_prob,
                "gate_sample": gate_samples,
                "gate_logit": gate_logits,
                "gate_prob": gate_probs,
                "gate_valid": gate_valid,
                "refresh_mask": refresh_mask,
            }
            return jnp.array(postprocessed_actions), extras, new_carry

        return policy

    return make_policy


def make_logging_inference_fn(
    temporal_ppo_networks: TemporalPPONetworks,
) -> Callable[[bool], Callable]:
    """Creates logging policy factory for temporal inference."""

    def make_logging_policy(deterministic: bool = False) -> Callable:
        policy_network = temporal_ppo_networks.policy_network
        parametric_action_distribution = (
            temporal_ppo_networks.parametric_action_distribution
        )

        def logging_policy(
            params: types.PolicyParams,
            observations: types.Observation,
            carry: TemporalPolicyCarry,
            key_sample: PRNGKey,
        ) -> tuple[types.Action, types.Extra, TemporalPolicyCarry]:
            key_action, key_network = jax.random.split(key_sample)

            (
                motor_logits,
                latent_mean,
                latent_logvar,
                latent_z,
                _,
                _,
                gate_logits,
                gate_probs,
                gate_samples,
                gate_valid,
                refresh_mask,
                new_carry,
            ) = policy_network.apply(
                *params,
                observations,
                carry,
                key_network,
                deterministic=deterministic,
                train_step=None,
            )

            if deterministic:
                action = parametric_action_distribution.mode(motor_logits)
            else:
                action = parametric_action_distribution.sample(motor_logits, key_action)

            extras = {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "latent": latent_z,
                "refresh_mask": refresh_mask,
                "gate_prob": gate_probs,
                "gate_sample": gate_samples,
                "gate_valid": gate_valid,
                "gate_logit": gate_logits,
            }
            return jnp.array(action), extras, new_carry

        return logging_policy

    return make_logging_policy


def make_temporal_intention_ppo_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    rnn_type: RNNCellType = "gru",
    rnn_hidden_sizes: Sequence[int] = (256,),
    boundary_mode: TemporalBoundaryMode = "fixed",
    macro_horizon: int = 16,
    min_macro_horizon: int = 4,
    max_macro_horizon: int = 64,
    eval_gate_threshold: float = 0.5,
    proprioception_noise_std: float = 0.0,
    value_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    condition_value_on_latent: bool = True,
    horizon_ramp: bool = False,
    horizon_ramp_steps: int = 0,
) -> TemporalPPONetworks:
    """Creates temporal intention PPO networks."""
    if boundary_mode not in ("fixed", "learned"):
        raise ValueError(
            f"Unsupported boundary_mode {boundary_mode}. Expected 'fixed' or 'learned'."
        )

    rnn_hidden_sizes = tuple(rnn_hidden_sizes)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_module = TemporalIntentionNetwork(
        output_size=parametric_action_distribution.param_size,
        encoder_layers=list(encoder_hidden_layer_sizes),
        latents=intention_latent_size,
        rnn_hidden_sizes=rnn_hidden_sizes,
        cell_type=rnn_type,
        boundary_mode=boundary_mode,
        macro_horizon=macro_horizon,
        min_macro_horizon=min_macro_horizon,
        max_macro_horizon=max_macro_horizon,
        proprioception_noise_std=proprioception_noise_std,
        horizon_ramp=horizon_ramp,
        horizon_ramp_steps=horizon_ramp_steps,
        eval_gate_threshold=eval_gate_threshold,
    )

    reset_segment_step = (
        macro_horizon if boundary_mode == "fixed" else max_macro_horizon
    )

    def policy_apply(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs: Mapping[str, Mapping[str, jnp.ndarray]],
        carry: TemporalPolicyCarry,
        key: jax.Array,
        deterministic: bool = False,
        gate_sample_override: jnp.ndarray | None = None,
        latent_override: jnp.ndarray | None = None,
        train_step: jnp.ndarray | None = None,
        get_activation: bool = False,
    ):
        policy_normalizer = normalizer_select(processor_params, policy_obs_key)
        normalized_obs = running_statistics.normalize(
            obs[policy_obs_key], policy_normalizer
        )
        return policy_module.apply(
            policy_params,
            obs=normalized_obs,
            carry=carry,
            key=key,
            deterministic=deterministic,
            gate_sample_override=gate_sample_override,
            latent_override=latent_override,
            train_step=train_step,
            get_activation=get_activation,
        )

    def policy_apply_sequence(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs_seq: Mapping[str, Mapping[str, jnp.ndarray]],
        initial_carry: TemporalPolicyCarry,
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        train_step: jnp.ndarray | None = None,
        stored_keys: jax.Array | None = None,
        stored_gate_samples: jax.Array | None = None,
        stored_latents: jnp.ndarray | None = None,
    ):
        policy_normalizer = normalizer_select(processor_params, policy_obs_key)
        obs_seq_normalized = running_statistics.normalize(
            obs_seq[policy_obs_key], policy_normalizer
        )

        _, _, eff_max = compute_effective_horizons(
            boundary_mode=boundary_mode,
            macro_horizon=macro_horizon,
            min_macro_horizon=min_macro_horizon,
            max_macro_horizon=max_macro_horizon,
            horizon_ramp=horizon_ramp,
            horizon_ramp_steps=horizon_ramp_steps,
            train_step=train_step,
        )

        if stored_keys is not None:
            ref_obs = obs_seq_normalized["task_obs"]
            expected_shape = (ref_obs.shape[0], ref_obs.shape[1], 2)
            if stored_keys.shape != expected_shape:
                raise ValueError(
                    f"stored_keys has shape {stored_keys.shape}, expected {expected_shape}."
                )

        if stored_gate_samples is not None:
            ref_obs = obs_seq_normalized["task_obs"]
            expected_gate_shape = ref_obs.shape[:2]
            if stored_gate_samples.shape != expected_gate_shape:
                raise ValueError(
                    f"stored_gate_samples has shape {stored_gate_samples.shape}, expected {expected_gate_shape}."
                )

        if stored_latents is not None:
            ref_obs = obs_seq_normalized["task_obs"]
            expected_latent_shape = ref_obs.shape[:2] + (
                initial_carry.current_latent.shape[-1],
            )
            if stored_latents.shape != expected_latent_shape:
                raise ValueError(
                    f"stored_latents has shape {stored_latents.shape}, expected {expected_latent_shape}."
                )

        if stored_keys is not None:
            if stored_gate_samples is None:
                stored_gate_samples = jnp.zeros_like(done_seq, dtype=jnp.float32)
            if stored_latents is not None:

                def scan_step(carry, inputs):
                    obs_t, done_t, keys_t, gate_t, latent_t = inputs
                    (
                        motor_logits,
                        latent_mean,
                        latent_logvar,
                        latent_z,
                        latent_log_prob,
                        latent_entropy,
                        gate_logits,
                        gate_probs,
                        gate_samples,
                        gate_valid,
                        refresh_mask,
                        new_carry,
                    ) = policy_module.apply(
                        policy_params,
                        obs=obs_t,
                        carry=carry,
                        key=keys_t,
                        deterministic=deterministic,
                        gate_sample_override=gate_t,
                        latent_override=latent_t,
                        train_step=train_step,
                    )
                    new_carry = reset_carry_on_done(
                        new_carry,
                        done_t,
                        cell_type=rnn_type,
                        reset_segment_step=eff_max,
                    )
                    return new_carry, (
                        motor_logits,
                        latent_mean,
                        latent_logvar,
                        latent_z,
                        latent_log_prob,
                        latent_entropy,
                        gate_logits,
                        gate_probs,
                        gate_samples,
                        gate_valid,
                        refresh_mask,
                    )

                final_carry, outputs = jax.lax.scan(
                    scan_step,
                    initial_carry,
                    (
                        obs_seq_normalized,
                        done_seq,
                        stored_keys,
                        stored_gate_samples,
                        stored_latents,
                    ),
                )
            else:

                def scan_step(carry, inputs):
                    obs_t, done_t, keys_t, gate_t = inputs
                    (
                        motor_logits,
                        latent_mean,
                        latent_logvar,
                        latent_z,
                        latent_log_prob,
                        latent_entropy,
                        gate_logits,
                        gate_probs,
                        gate_samples,
                        gate_valid,
                        refresh_mask,
                        new_carry,
                    ) = policy_module.apply(
                        policy_params,
                        obs=obs_t,
                        carry=carry,
                        key=keys_t,
                        deterministic=deterministic,
                        gate_sample_override=gate_t,
                        train_step=train_step,
                    )
                    new_carry = reset_carry_on_done(
                        new_carry,
                        done_t,
                        cell_type=rnn_type,
                        reset_segment_step=eff_max,
                    )
                    return new_carry, (
                        motor_logits,
                        latent_mean,
                        latent_logvar,
                        latent_z,
                        latent_log_prob,
                        latent_entropy,
                        gate_logits,
                        gate_probs,
                        gate_samples,
                        gate_valid,
                        refresh_mask,
                    )

                final_carry, outputs = jax.lax.scan(
                    scan_step,
                    initial_carry,
                    (obs_seq_normalized, done_seq, stored_keys, stored_gate_samples),
                )
        else:

            def scan_step(carry_key, inputs):
                carry, step_key = carry_key
                obs_t, done_t = inputs
                step_key, next_key = jax.random.split(step_key)

                (
                    motor_logits,
                    latent_mean,
                    latent_logvar,
                    latent_z,
                    latent_log_prob,
                    latent_entropy,
                    gate_logits,
                    gate_probs,
                    gate_samples,
                    gate_valid,
                    refresh_mask,
                    new_carry,
                ) = policy_module.apply(
                    policy_params,
                    obs=obs_t,
                    carry=carry,
                    key=step_key,
                    deterministic=deterministic,
                    latent_override=None,
                    train_step=train_step,
                )
                new_carry = reset_carry_on_done(
                    new_carry,
                    done_t,
                    cell_type=rnn_type,
                    reset_segment_step=eff_max,
                )

                return (new_carry, next_key), (
                    motor_logits,
                    latent_mean,
                    latent_logvar,
                    latent_z,
                    latent_log_prob,
                    latent_entropy,
                    gate_logits,
                    gate_probs,
                    gate_samples,
                    gate_valid,
                    refresh_mask,
                )

            (final_carry, _), outputs = jax.lax.scan(
                scan_step, (initial_carry, key), (obs_seq_normalized, done_seq)
            )

        (
            motor_logits,
            latent_means,
            latent_logvars,
            latent_z,
            latent_log_prob,
            latent_entropy,
            gate_logits,
            gate_probs,
            gate_samples,
            gate_valid,
            refresh_mask,
        ) = outputs

        return (
            motor_logits,
            latent_means,
            latent_logvars,
            latent_z,
            latent_log_prob,
            latent_entropy,
            gate_logits,
            gate_probs,
            gate_samples,
            gate_valid,
            refresh_mask,
            final_carry,
        )

    dummy_obs = {
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"]), dtype=jnp.float32),
        "proprioception": jnp.zeros(
            (1, obs_sizes["proprioception"]), dtype=jnp.float32
        ),
    }
    dummy_key = jax.random.PRNGKey(0)

    def policy_init(key):
        dummy_carry = init_temporal_carry(
            cell_type=rnn_type,
            rnn_hidden_sizes=rnn_hidden_sizes,
            latent_size=intention_latent_size,
            batch_size=1,
            reset_segment_step=reset_segment_step,
        )
        return policy_module.init(
            key,
            dummy_obs,
            dummy_carry,
            dummy_key,
            deterministic=False,
            train_step=None,
        )

    def policy_init_carry(batch_size: int) -> TemporalPolicyCarry:
        return init_temporal_carry(
            cell_type=rnn_type,
            rnn_hidden_sizes=rnn_hidden_sizes,
            latent_size=intention_latent_size,
            batch_size=batch_size,
            reset_segment_step=reset_segment_step,
        )

    policy_network = RecurrentNetwork(
        init=policy_init,
        apply=policy_apply,
        apply_sequence=policy_apply_sequence,
        init_carry=policy_init_carry,
    )

    value_network = make_temporal_value_network(
        obs_sizes=obs_sizes,
        latent_size=intention_latent_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
        value_obs_key=value_obs_key,
        condition_on_latent=condition_value_on_latent,
    )

    return TemporalPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        rnn_hidden_sizes=rnn_hidden_sizes,
        cell_type=rnn_type,
        boundary_mode=boundary_mode,
        macro_horizon=macro_horizon,
        min_macro_horizon=min_macro_horizon,
        max_macro_horizon=max_macro_horizon,
        horizon_ramp=horizon_ramp,
        horizon_ramp_steps=horizon_ramp_steps,
    )
