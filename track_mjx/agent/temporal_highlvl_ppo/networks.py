"""Temporal PPO networks for wrapped high-level decoder control."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training.acme import running_statistics
from flax import linen as nn

from track_mjx.agent.ff_ppo.intention_network import Encoder, reparameterize
from track_mjx.agent.observation_utils import normalizer_select
from track_mjx.agent.temporal_ppo import networks as shared_temporal_networks
from track_mjx.agent.temporal_ppo.types import (
    RNNCellType,
    TemporalBoundaryMode,
    TemporalPolicyCarry,
)


class TemporalHighLevelPolicyNetwork(nn.Module):
    """Temporal policy over wrapped high-level observations."""

    output_size: int
    encoder_layers: Sequence[int]
    latents: int = 60
    rnn_hidden_sizes: Sequence[int] = (256,)
    cell_type: RNNCellType = "gru"
    boundary_mode: TemporalBoundaryMode = "fixed"
    macro_horizon: int = 16
    min_macro_horizon: int = 4
    max_macro_horizon: int = 64
    horizon_ramp: bool = False
    horizon_ramp_steps: int = 0
    eval_gate_threshold: float = 0.5

    def setup(self):
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = shared_temporal_networks.TemporalDecoder(
            output_size=self.output_size,
            rnn_hidden_sizes=self.rnn_hidden_sizes,
            cell_type=self.cell_type,
        )
        if self.boundary_mode == "learned":
            self.gate_head = nn.Dense(1, name="gate_head")

    def _effective_horizons(
        self, train_step: jnp.ndarray | None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return shared_temporal_networks.compute_effective_horizons(
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
        obs: jnp.ndarray,
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
        jnp.ndarray,
        TemporalPolicyCarry,
    ]:
        del get_activation
        obs_is_batched = obs.ndim >= 2
        encoder_rng, _, gate_rng = (
            shared_temporal_networks._split_policy_rng(  # noqa: SLF001
                key, obs_is_batched
            )
        )
        eff_min, _, eff_max = self._effective_horizons(train_step)

        if self.boundary_mode == "learned":
            prev_top_hidden = jax.lax.stop_gradient(
                shared_temporal_networks._extract_top_hidden(  # noqa: SLF001
                    carry.decoder_hidden, self.cell_type
                )
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

        fresh_mean, fresh_logvar = self.encoder(obs, get_activation=False)
        fresh_z = (
            fresh_mean
            if deterministic
            else reparameterize(encoder_rng, fresh_mean, fresh_logvar)
        )

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

        fresh_latent_log_prob = shared_temporal_networks.gaussian_diag_log_prob(
            fresh_mean, fresh_logvar, selected_z
        )
        latent_log_prob = jnp.where(refresh_mask, fresh_latent_log_prob, 0.0)
        fresh_latent_entropy = shared_temporal_networks.gaussian_diag_entropy(
            fresh_logvar
        )
        latent_entropy = jnp.where(refresh_mask, fresh_latent_entropy, 0.0)

        decoder_input = jax.lax.stop_gradient(selected_z)
        motor_logits, new_hidden, _ = self.decoder(decoder_input, carry.decoder_hidden)

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


class LatentConditionedFlatValueMLP(nn.Module):
    """Value MLP over flat wrapped observations."""

    hidden_layer_sizes: Sequence[int]
    condition_on_latent: bool = True

    @nn.compact
    def __call__(self, obs: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, latent], axis=-1) if self.condition_on_latent else obs
        for i, hidden_size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(hidden_size, name=f"hidden_{i}")(x)
            x = nn.silu(x)
            x = nn.LayerNorm()(x)

        value = nn.Dense(1, name="value_head")(x)
        return jnp.squeeze(value, axis=-1)


def make_temporal_highlvl_value_network(
    obs_sizes: Mapping[str, int],
    latent_size: int,
    hidden_layer_sizes: Sequence[int],
    value_obs_key: str,
    condition_on_latent: bool,
) -> shared_temporal_networks.TemporalValueNetwork:
    """Creates a value network for flat wrapped observations."""
    if value_obs_key not in obs_sizes:
        raise KeyError(
            f"Missing value_obs_key '{value_obs_key}' in obs_sizes {sorted(obs_sizes)}."
        )

    value_module = LatentConditionedFlatValueMLP(
        hidden_layer_sizes=hidden_layer_sizes,
        condition_on_latent=condition_on_latent,
    )
    dummy_obs = jnp.zeros((1, obs_sizes[value_obs_key]), dtype=jnp.float32)
    dummy_latent = jnp.zeros((1, latent_size), dtype=jnp.float32)

    def init(key: jax.Array):
        return value_module.init(key, dummy_obs, dummy_latent)

    def apply(
        processor_params: running_statistics.RunningStatisticsState,
        value_params,
        obs: Mapping[str, jnp.ndarray],
        latent: jnp.ndarray,
    ) -> jnp.ndarray:
        value_normalizer = normalizer_select(processor_params, value_obs_key)
        normalized_obs = running_statistics.normalize(
            obs[value_obs_key], value_normalizer
        )
        return value_module.apply(value_params, normalized_obs, latent)

    return shared_temporal_networks.TemporalValueNetwork(init=init, apply=apply)


def make_inference_fn(temporal_ppo_networks):
    """Creates temporal policy factory for acting/evaluation."""
    return shared_temporal_networks.make_inference_fn(temporal_ppo_networks)


def make_logging_inference_fn(temporal_ppo_networks):
    """Creates temporal logging policy factory."""
    return shared_temporal_networks.make_logging_inference_fn(temporal_ppo_networks)


def make_temporal_highlvl_ppo_networks(
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
    value_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    condition_value_on_latent: bool = True,
    horizon_ramp: bool = False,
    horizon_ramp_steps: int = 0,
) -> shared_temporal_networks.TemporalPPONetworks:
    """Creates temporal PPO networks for wrapped high-level observations."""
    if boundary_mode not in ("fixed", "learned"):
        raise ValueError(
            f"Unsupported boundary_mode {boundary_mode}. Expected 'fixed' or 'learned'."
        )
    if policy_obs_key not in obs_sizes:
        raise KeyError(
            f"Missing policy_obs_key '{policy_obs_key}' in obs_sizes {sorted(obs_sizes)}."
        )

    rnn_hidden_sizes = tuple(rnn_hidden_sizes)
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_module = TemporalHighLevelPolicyNetwork(
        output_size=parametric_action_distribution.param_size,
        encoder_layers=tuple(encoder_hidden_layer_sizes),
        latents=intention_latent_size,
        rnn_hidden_sizes=rnn_hidden_sizes,
        cell_type=rnn_type,
        boundary_mode=boundary_mode,
        macro_horizon=macro_horizon,
        min_macro_horizon=min_macro_horizon,
        max_macro_horizon=max_macro_horizon,
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
        obs: Mapping[str, jnp.ndarray],
        carry: TemporalPolicyCarry,
        key: jax.Array,
        deterministic: bool = False,
        gate_sample_override: jnp.ndarray | None = None,
        latent_override: jnp.ndarray | None = None,
        train_step: jnp.ndarray | None = None,
        get_activation: bool = False,
    ):
        del get_activation
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
        )

    def policy_apply_sequence(
        processor_params: running_statistics.RunningStatisticsState,
        policy_params,
        obs_seq: Mapping[str, jnp.ndarray],
        initial_carry: TemporalPolicyCarry,
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        train_step: jnp.ndarray | None = None,
        stored_keys: jax.Array | None = None,
        stored_gate_samples: jnp.ndarray | None = None,
        stored_latents: jnp.ndarray | None = None,
    ):
        policy_normalizer = normalizer_select(processor_params, policy_obs_key)
        obs_seq_normalized = running_statistics.normalize(
            obs_seq[policy_obs_key], policy_normalizer
        )

        _, _, eff_max = shared_temporal_networks.compute_effective_horizons(
            boundary_mode=boundary_mode,
            macro_horizon=macro_horizon,
            min_macro_horizon=min_macro_horizon,
            max_macro_horizon=max_macro_horizon,
            horizon_ramp=horizon_ramp,
            horizon_ramp_steps=horizon_ramp_steps,
            train_step=train_step,
        )

        if stored_keys is not None:
            expected_shape = obs_seq_normalized.shape[:2] + (2,)
            if stored_keys.shape != expected_shape:
                raise ValueError(
                    f"stored_keys has shape {stored_keys.shape}, expected {expected_shape}."
                )
        if stored_gate_samples is not None:
            expected_gate_shape = obs_seq_normalized.shape[:2]
            if stored_gate_samples.shape != expected_gate_shape:
                raise ValueError(
                    "stored_gate_samples has shape "
                    f"{stored_gate_samples.shape}, expected {expected_gate_shape}."
                )
        if stored_latents is not None:
            expected_latent_shape = obs_seq_normalized.shape[:2] + (
                initial_carry.current_latent.shape[-1],
            )
            if stored_latents.shape != expected_latent_shape:
                raise ValueError(
                    f"stored_latents has shape {stored_latents.shape}, expected {expected_latent_shape}."
                )

        def step_outputs(new_carry, outputs, done_t):
            new_carry = shared_temporal_networks.reset_carry_on_done(
                new_carry,
                done_t,
                cell_type=rnn_type,
                reset_segment_step=eff_max,
            )
            return new_carry, outputs

        if stored_keys is not None:
            if stored_gate_samples is None:
                stored_gate_samples = jnp.zeros_like(done_seq, dtype=jnp.float32)

            if stored_latents is not None:

                def scan_step(carry, inputs):
                    obs_t, done_t, keys_t, gate_t, latent_t = inputs
                    outputs = policy_module.apply(
                        policy_params,
                        obs=obs_t,
                        carry=carry,
                        key=keys_t,
                        deterministic=deterministic,
                        gate_sample_override=gate_t,
                        latent_override=latent_t,
                        train_step=train_step,
                    )
                    new_carry = outputs[-1]
                    new_carry, _ = step_outputs(new_carry, outputs[:-1], done_t)
                    return new_carry, outputs[:-1]

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
                    outputs = policy_module.apply(
                        policy_params,
                        obs=obs_t,
                        carry=carry,
                        key=keys_t,
                        deterministic=deterministic,
                        gate_sample_override=gate_t,
                        train_step=train_step,
                    )
                    new_carry = outputs[-1]
                    new_carry, _ = step_outputs(new_carry, outputs[:-1], done_t)
                    return new_carry, outputs[:-1]

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
                outputs = policy_module.apply(
                    policy_params,
                    obs=obs_t,
                    carry=carry,
                    key=step_key,
                    deterministic=deterministic,
                    train_step=train_step,
                )
                new_carry = outputs[-1]
                new_carry, _ = step_outputs(new_carry, outputs[:-1], done_t)
                return (new_carry, next_key), outputs[:-1]

            (final_carry, _), outputs = jax.lax.scan(
                scan_step, (initial_carry, key), (obs_seq_normalized, done_seq)
            )

        return (*outputs, final_carry)

    dummy_obs = jnp.zeros((1, obs_sizes[policy_obs_key]), dtype=jnp.float32)
    dummy_key = jax.random.PRNGKey(0)

    def policy_init(key):
        dummy_carry = shared_temporal_networks.init_temporal_carry(
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
        return shared_temporal_networks.init_temporal_carry(
            cell_type=rnn_type,
            rnn_hidden_sizes=rnn_hidden_sizes,
            latent_size=intention_latent_size,
            batch_size=batch_size,
            reset_segment_step=reset_segment_step,
        )

    policy_network = shared_temporal_networks.RecurrentNetwork(
        init=policy_init,
        apply=policy_apply,
        apply_sequence=policy_apply_sequence,
        init_carry=policy_init_carry,
    )
    value_network = make_temporal_highlvl_value_network(
        obs_sizes=obs_sizes,
        latent_size=intention_latent_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
        value_obs_key=value_obs_key,
        condition_on_latent=condition_value_on_latent,
    )
    return shared_temporal_networks.TemporalPPONetworks(
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
