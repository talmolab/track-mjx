"""Encoder-decoder policy networks for KPMS-driven motor control.

The network receives a pre-computed syllable code (integer, passed as float in
``obs["kpms_code"]``), the agent's proprioception, and optionally the reference
trajectory (``obs["task_obs"]``). It embeds the code via a learned
embedding table, optionally encodes the reference trajectory into a continuous
latent, and feeds the concatenation through a decoder to produce action
parameters.

Two decoder variants are provided:

* ``MoSeqEncoderDecoderNetwork`` — feedforward MLP decoder (original).
* ``MoSeqRecurrentDecoderNetwork`` — GRU-based RNN decoder that maintains
  hidden state across timesteps, enabling closed-loop autonomous control
  within a syllable.

When ``use_continuous_encoder=False`` (default), the continuous encoder is
skipped and behavior is identical to the original decoder-only network.

When ``use_distillation_head=True`` (RNN decoder only), a distillation head
MLP reads the RNN hidden state ``h_t`` and predicts ``(mu_d, sigma_d)``.
A frozen pre-trained encoder provides target distributions ``(mu_e, sigma_e)``
for KL regularization.  In this mode, the encoder output z_e is completely
excluded from the action path — actions depend only on ``h_t``.
"""

from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MoSeqEncoderDecoderNetwork(nn.Module):
    """Encoder-decoder policy: code embedding + continuous latent + proprio -> action.

    Attributes:
        num_codes: Number of syllable codes (embedding table rows).
        code_embed_dim: Dimensionality of the code embedding.
        decoder_layer_sizes: Hidden layer sizes for the decoder MLP.
        action_param_size: Output dimension (2 * action_dim for NormalTanh).
        activation: Activation function.
        kernel_init: Initializer for Dense layers.
        use_continuous_encoder: Whether to encode imitation_target into z_e.
        encoder_layer_sizes: Hidden layer sizes for the encoder MLP.
        continuous_latent_dim: Dimensionality of the continuous latent (= code_embed_dim).
    """

    num_codes: int = 32
    code_embed_dim: int = 16
    decoder_layer_sizes: Sequence[int] = (512, 512, 256, 256)
    action_param_size: int = 1
    activation: Callable = nn.silu
    kernel_init: Callable = nn.initializers.lecun_uniform()
    use_continuous_encoder: bool = False
    encoder_layer_sizes: Sequence[int] = (256, 128)
    continuous_latent_dim: int = 16
    z_e_dropout_rate: float = 0.0

    def setup(self):
        self.code_embedding = nn.Embed(
            num_embeddings=self.num_codes,
            features=self.code_embed_dim,
        )

    @nn.compact
    def __call__(
        self,
        obs: dict[str, jnp.ndarray],
        key=None,
        deterministic: bool = False,
        z_e_scale: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Forward pass.

        Args:
            obs: Observation dict with ``kpms_code`` (raw int-as-float,
                shape ``[..., 1]``), ``proprioception`` (normalized,
                shape ``[..., proprio_dim]``), and optionally
                ``imitation_target`` (normalized, shape ``[..., target_dim]``).
            key: PRNG key for reparameterization (used when
                ``use_continuous_encoder=True`` and ``deterministic=False``).
            deterministic: If True, use mean (no sampling) for continuous latent.
            z_e_scale: Multiplier on z_e (1.0 = full, 0.0 = decoder-only).

        Returns:
            Tuple of ``(action_params, code_idx, mean, logvar)``.
            When ``use_continuous_encoder=False``, ``mean`` and ``logvar``
            are ``None``.
        """
        # Extract code indices — may be stacked [code_t, ..., code_{t+N-1}]
        kpms_code = obs["kpms_code"]  # [..., N] where N = code_stack_size
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)  # current code for metrics
        all_code_idx = jnp.round(kpms_code).astype(jnp.int32)
        all_emb = self.code_embedding(all_code_idx)  # [..., N, code_embed_dim]
        code_emb = all_emb.reshape(*all_emb.shape[:-2], -1)  # [..., N * code_embed_dim]

        # Proprioception (already normalized and flattened by the policy wrapper)
        proprio = obs["proprioception"]

        # Continuous encoder (optional)
        if self.use_continuous_encoder:
            imitation_target = obs["task_obs"]

            # Encoder MLP
            h = imitation_target
            for i, size in enumerate(self.encoder_layer_sizes):
                h = nn.Dense(size, kernel_init=self.kernel_init, name=f"enc_{i}")(h)
                h = self.activation(h)
                h = nn.LayerNorm(name=f"enc_ln_{i}")(h)

            mean = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="continuous_mean",
            )(h)
            logvar = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="continuous_logvar",
            )(h)

            # Reparameterization
            if deterministic:
                z_e = mean
            else:
                if key is None:
                    key = self.make_rng("params")
                eps = jax.random.normal(key, mean.shape)
                z_e = mean + jnp.exp(0.5 * logvar) * eps

            # Decoder input: code_emb + z_e_scaled + proprio
            z_e_scaled = z_e * z_e_scale

            # z_e dropout (training only)
            if not deterministic and self.z_e_dropout_rate > 0 and key is not None:
                _, _, dropout_key = jax.random.split(key, 3)
                keep = jax.random.bernoulli(
                    dropout_key, 1.0 - self.z_e_dropout_rate, shape=mean.shape[:1]
                ).astype(z_e_scaled.dtype)
                z_e_scaled = z_e_scaled * keep[..., None]

            x = jnp.concatenate([code_emb, z_e_scaled, proprio], axis=-1)
        else:
            mean = None
            logvar = None
            # Decoder input: code_emb + proprio (original behavior)
            x = jnp.concatenate([code_emb, proprio], axis=-1)

        # Decoder MLP
        for i, size in enumerate(self.decoder_layer_sizes):
            x = nn.Dense(size, kernel_init=self.kernel_init, name=f"dec_{i}")(x)
            x = self.activation(x)
            x = nn.LayerNorm(name=f"dec_ln_{i}")(x)
        action_params = nn.Dense(
            self.action_param_size,
            kernel_init=self.kernel_init,
            name="action_head",
        )(x)

        return action_params, code_idx, mean, logvar


class MoSeqRecurrentDecoderNetwork(nn.Module):
    """RNN decoder policy: code embedding + proprio -> GRU -> action (+z_e).

    The decoder maintains hidden state across timesteps so it can produce
    temporally coherent control autonomously given a syllable code. The
    optional continuous encoder (z_e) acts as a training scaffold.

    When ``z_e_at_action_head=True``, z_e does NOT enter the GRU recurrence.
    Instead it is concatenated with the GRU output at the action head. This
    forces the GRU to learn dynamics from code+proprio alone.

    When ``reinit_hidden_on_code=True``, the GRU hidden state is reinitialized
    at code transitions. If ``learned_hidden_init=True``, each code has a
    learned initial hidden state; otherwise hidden resets to zeros.

    Attributes:
        num_codes: Number of syllable codes (embedding table rows).
        code_embed_dim: Dimensionality of the code embedding.
        rnn_hidden_sizes: Hidden sizes for stacked GRU layers.
        action_param_size: Output dimension (2 * action_dim for NormalTanh).
        activation: Activation function for pre-RNN projection.
        kernel_init: Initializer for Dense layers.
        use_continuous_encoder: Whether to encode imitation_target into z_e.
        encoder_layer_sizes: Hidden layer sizes for the encoder MLP.
        continuous_latent_dim: Dimensionality of the continuous latent.
        z_e_dropout_rate: Probability of zeroing z_e per sample.
        z_e_at_action_head: If True, z_e enters at action head only (not GRU).
        reinit_hidden_on_code: If True, reinitialize hidden at code transitions.
        learned_hidden_init: If True, use learned per-code initial hidden states.
        use_distillation_head: If True, add a distill head MLP that predicts
            the encoder's latent distribution from h_t.  z_e is excluded
            from the action path entirely in this mode.
        distill_head_layer_sizes: Hidden layer sizes for the distill head MLP.
        distill_logvar_min: Optional min clamp for distill head log-variance.
        distill_logvar_max: Optional max clamp for distill head log-variance.
    """

    num_codes: int = 32
    code_embed_dim: int = 16
    rnn_hidden_sizes: Sequence[int] = (256,)
    action_param_size: int = 1
    activation: Callable = nn.silu
    kernel_init: Callable = nn.initializers.lecun_uniform()
    use_continuous_encoder: bool = False
    encoder_layer_sizes: Sequence[int] = (256, 128)
    continuous_latent_dim: int = 16
    z_e_dropout_rate: float = 0.0
    z_e_at_action_head: bool = False
    reinit_hidden_on_code: bool = False
    learned_hidden_init: bool = False
    use_distillation_head: bool = False
    distill_head_layer_sizes: Sequence[int] = (256, 128)
    distill_logvar_min: float | None = None
    distill_logvar_max: float | None = None

    def setup(self):
        self.code_embedding = nn.Embed(
            num_embeddings=self.num_codes,
            features=self.code_embed_dim,
        )

        # Pre-create encoder layers (must be in setup for apply_sequence/scan)
        if self.use_continuous_encoder:
            self.enc_layers = [
                nn.Dense(size, kernel_init=self.kernel_init, name=f"enc_{i}")
                for i, size in enumerate(self.encoder_layer_sizes)
            ]
            self.enc_lns = [
                nn.LayerNorm(name=f"enc_ln_{i}")
                for i in range(len(self.encoder_layer_sizes))
            ]
            self.mean_layer = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="continuous_mean",
            )
            self.logvar_layer = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="continuous_logvar",
            )

        self.rnn_cells = [
            nn.GRUCell(features=h, kernel_init=self.kernel_init)
            for h in self.rnn_hidden_sizes
        ]
        self.action_head = nn.Dense(
            self.action_param_size,
            kernel_init=self.kernel_init,
            name="action_head",
        )

        # Learned per-code initial hidden states
        if self.learned_hidden_init:
            self.hidden_init_params = [
                self.param(
                    f"hidden_init_{i}",
                    nn.initializers.zeros_init(),
                    (self.num_codes, h),
                )
                for i, h in enumerate(self.rnn_hidden_sizes)
            ]

        # Distillation head: h_t -> MLP -> (mu_d, logvar_d)
        if self.use_distillation_head:
            self.distill_layers = [
                nn.Dense(size, kernel_init=self.kernel_init, name=f"distill_{i}")
                for i, size in enumerate(self.distill_head_layer_sizes)
            ]
            self.distill_lns = [
                nn.LayerNorm(name=f"distill_ln_{i}")
                for i in range(len(self.distill_head_layer_sizes))
            ]
            self.distill_mean_layer = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="distill_mean",
            )
            self.distill_logvar_layer = nn.Dense(
                self.continuous_latent_dim,
                kernel_init=self.kernel_init,
                name="distill_logvar",
            )

    def _distill_head(
        self, h_t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Distillation head: predict encoder distribution from RNN hidden state.

        Args:
            h_t: Final GRU layer output, shape ``[..., H_last]``.

        Returns:
            ``(mu_d, logvar_d)`` each shape ``[..., continuous_latent_dim]``.
        """
        x = h_t
        for dense, ln in zip(self.distill_layers, self.distill_lns):
            x = dense(x)
            x = self.activation(x)
            x = ln(x)
        mu_d = self.distill_mean_layer(x)
        logvar_d = self.distill_logvar_layer(x)
        if self.distill_logvar_min is not None or self.distill_logvar_max is not None:
            logvar_d = jnp.clip(
                logvar_d,
                a_min=self.distill_logvar_min,
                a_max=self.distill_logvar_max,
            )
        return mu_d, logvar_d

    def _encode(
        self,
        obs: dict[str, jnp.ndarray],
        key=None,
        deterministic: bool = False,
        z_e_scale: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
        """Encode obs into (decoder_input, code_idx, mean, logvar, z_e_scaled)."""
        kpms_code = obs["kpms_code"]
        # Current code index (first in stack, used for metrics/logging)
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)
        # Embed all codes in the stack and concatenate
        all_code_idx = jnp.round(kpms_code).astype(jnp.int32)
        all_emb = self.code_embedding(all_code_idx)  # [..., N, code_embed_dim]
        # Flatten last two dims: [..., N * code_embed_dim]
        code_emb = all_emb.reshape(*all_emb.shape[:-2], -1)

        proprio = obs["proprioception"]

        if self.use_continuous_encoder:
            imitation_target = obs["task_obs"]

            h = imitation_target
            for dense, ln in zip(self.enc_layers, self.enc_lns):
                h = dense(h)
                h = self.activation(h)
                h = ln(h)

            mean = self.mean_layer(h)
            logvar = self.logvar_layer(h)

            # Freeze encoder when used as a distillation target (pre-trained weights)
            if self.use_distillation_head:
                mean = jax.lax.stop_gradient(mean)
                logvar = jax.lax.stop_gradient(logvar)

            if deterministic:
                z_e = mean
            else:
                if key is None:
                    key = self.make_rng("params")
                # Split key before reparameterization (matches reference
                # RecurrentIntentionNetwork pattern for PRNG hygiene)
                if key.ndim > 1:
                    # Per-sample keys [B, 2]: vmap split, use second half
                    _, encoder_rng = jax.vmap(jax.random.split)(key).swapaxes(0, 1)

                    def _reparam(k, m, lv):
                        return m + jnp.exp(0.5 * lv) * jax.random.normal(k, m.shape)

                    z_e = jax.vmap(_reparam)(encoder_rng, mean, logvar)
                elif mean.ndim == 1:
                    # Per-sample key but unbatched obs: use first key
                    _, encoder_rng = jax.random.split(key[0])
                    eps = jax.random.normal(encoder_rng, mean.shape)
                    z_e = mean + jnp.exp(0.5 * logvar) * eps
                else:
                    # Single key [2]: split for encoder
                    _, encoder_rng = jax.random.split(key)
                    eps = jax.random.normal(encoder_rng, mean.shape)
                    z_e = mean + jnp.exp(0.5 * logvar) * eps

            z_e_scaled = z_e * z_e_scale

            # z_e dropout: zero out entire z_e with prob p per sample (training only)
            if not deterministic and self.z_e_dropout_rate > 0 and key is not None:
                if key.ndim > 1:
                    # Per-sample keys [B, 2]: vmap bernoulli over batch
                    dropout_keys = jax.vmap(lambda k: jax.random.split(k, 3)[2])(key)
                    keep = jax.vmap(
                        lambda k: jax.random.bernoulli(k, 1.0 - self.z_e_dropout_rate)
                    )(dropout_keys).astype(z_e_scaled.dtype)
                else:
                    _, _, dropout_key = jax.random.split(key, 3)
                    keep = jax.random.bernoulli(
                        dropout_key,
                        1.0 - self.z_e_dropout_rate,
                        shape=mean.shape[:1],
                    ).astype(z_e_scaled.dtype)
                z_e_scaled = z_e_scaled * keep[..., None]

            if self.use_distillation_head:
                # Distill mode: encoder output is ONLY a distillation target.
                # z_e does NOT enter the action path in any way.
                z_e_scaled = None
                decoder_input = jnp.concatenate([code_emb, proprio], axis=-1)
            elif self.z_e_at_action_head:
                # z_e goes to action head, NOT into GRU input
                decoder_input = jnp.concatenate([code_emb, proprio], axis=-1)
            else:
                # z_e goes into GRU input (legacy behavior)
                decoder_input = jnp.concatenate([code_emb, z_e_scaled, proprio], axis=-1)
        else:
            mean = None
            logvar = None
            z_e_scaled = None
            decoder_input = jnp.concatenate([code_emb, proprio], axis=-1)

        return decoder_input, code_idx, mean, logvar, z_e_scaled

    def _decode_rnn(
        self,
        x: jnp.ndarray,
        hidden: list[jnp.ndarray],
        z_e_for_action: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray], jnp.ndarray]:
        """Run one timestep through the stacked GRU and action head.

        Args:
            x: GRU input (code_emb + proprio, or code_emb + z_e + proprio).
            hidden: List of GRU hidden states per layer.
            z_e_for_action: If provided, concatenated with GRU output before
                the action head (z_e at action head architecture).

        Returns:
            ``(action_params, new_hidden, h_t)`` where ``h_t`` is the final
            GRU layer output (used by the distillation head).
        """
        new_hidden = []
        rnn_input = x
        for cell, h in zip(self.rnn_cells, hidden):
            new_h, _ = cell(h, rnn_input)
            new_hidden.append(new_h)
            rnn_input = new_h
        h_t = rnn_input  # final GRU layer output
        # Optionally concat z_e at action head
        action_input = h_t
        if z_e_for_action is not None:
            action_input = jnp.concatenate([h_t, z_e_for_action], axis=-1)
        action_params = self.action_head(action_input)
        return action_params, new_hidden, h_t

    def __call__(
        self,
        obs: dict[str, jnp.ndarray],
        hidden: list[jnp.ndarray],
        key=None,
        deterministic: bool = False,
        z_e_scale: float = 1.0,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
        list[jnp.ndarray],
        jnp.ndarray | None,
        jnp.ndarray | None,
    ]:
        """Single-timestep forward pass (no per-code decay — used in rollout).

        Returns:
            ``(action_params, code_idx, mean, logvar, new_hidden,
            distill_mean, distill_logvar)``.
            When ``use_distillation_head=False``, distill outputs are ``None``.
        """
        decoder_input, code_idx, mean, logvar, z_e_scaled = self._encode(
            obs, key, deterministic, z_e_scale,
        )
        if self.use_distillation_head:
            z_e_arg = None  # z_e NEVER enters action path in distillation mode
        else:
            z_e_arg = z_e_scaled if self.z_e_at_action_head else None
        action_params, new_hidden, h_t = self._decode_rnn(
            decoder_input, hidden, z_e_for_action=z_e_arg,
        )
        if self.use_distillation_head:
            distill_mean, distill_logvar = self._distill_head(h_t)
        else:
            distill_mean = None
            distill_logvar = None
        return action_params, code_idx, mean, logvar, new_hidden, distill_mean, distill_logvar

    def apply_sequence(
        self,
        obs_seq: dict[str, jnp.ndarray],
        initial_hidden: list[jnp.ndarray],
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        stored_keys: jax.Array | None = None,
        z_e_scale: float = 1.0,
    ) -> (
        tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None,
              jnp.ndarray | None, jnp.ndarray | None, list[jnp.ndarray]]
    ):
        """Forward pass over a time sequence using jax.lax.scan.

        Args:
            obs_seq: Observations with shape ``[T, B, ...]`` per key.
            initial_hidden: Initial GRU hidden states (list of ``[B, H]``).
            done_seq: Episode-done flags ``[T, B]``.
            key: PRNG key (used only when ``stored_keys`` is None).
            deterministic: If True, use latent mean instead of sampling.
            stored_keys: Pre-stored PRNG keys ``[T, B, 2]`` for deterministic
                replay of z_e reparameterization. If None, fresh keys
                are generated at each timestep.
            z_e_scale: Multiplier on z_e.

        Returns:
            ``(logits, means, logvars, distill_means, distill_logvars,
            final_hidden)`` each ``[T, B, ...]``.  When
            ``use_distillation_head=False``, distill outputs are ``None``.
        """

        def _reset_hidden_on_done(hidden_list, done_t):
            """Reset hidden to zeros on episode done."""
            done_expanded = done_t[..., None]
            return [jnp.where(done_expanded, 0.0, h) for h in hidden_list]

        def _reinit_hidden_on_code(hidden_list, code_t, code_changed):
            """Reinitialize hidden at code transitions."""
            if not self.reinit_hidden_on_code:
                return hidden_list
            changed = code_changed[..., None]
            if self.learned_hidden_init:
                return [
                    jnp.where(changed, self.hidden_init_params[i][code_t], h)
                    for i, h in enumerate(hidden_list)
                ]
            else:
                return [jnp.where(changed, 0.0, h) for h in hidden_list]

        def _z_e_arg_for_action(z_e_scaled):
            """Determine z_e argument for the action head."""
            if self.use_distillation_head:
                return None  # z_e NEVER enters action path in distillation mode
            if self.z_e_at_action_head:
                return z_e_scaled
            return None

        # Validate stored_keys shape if provided
        if stored_keys is not None:
            ref_obs = jax.tree_util.tree_leaves(obs_seq)[0]
            expected_shape = (ref_obs.shape[0], ref_obs.shape[1], 2)
            if stored_keys.shape != expected_shape:
                raise ValueError(
                    f"stored_keys has shape {stored_keys.shape}, expected "
                    f"{expected_shape}. stored_keys must have shape [T, B, 2]."
                )

        # Initial prev_code for code-transition tracking (internal to scan)
        B = jax.tree_util.tree_leaves(obs_seq)[0].shape[1]
        init_prev_code = jnp.full((B,), -1, dtype=jnp.int32)

        if stored_keys is not None:

            if self.use_distillation_head:

                def step_stored(carry, inputs):
                    hidden_list, prev_code = carry
                    obs_t = {k: inputs[0][k] for k in inputs[0]}
                    done_t = inputs[1]
                    keys_t = inputs[2]

                    code_t = jnp.round(obs_t["kpms_code"][..., 0]).astype(jnp.int32)
                    code_changed = (code_t != prev_code) | (prev_code == -1)
                    hidden_list = _reinit_hidden_on_code(hidden_list, code_t, code_changed)

                    decoder_input, code_idx, mean, logvar, z_e_scaled = self._encode(
                        obs_t, keys_t, deterministic, z_e_scale
                    )
                    action_params, new_hidden, h_t = self._decode_rnn(
                        decoder_input, hidden_list, z_e_for_action=None,
                    )
                    d_mean, d_logvar = self._distill_head(h_t)
                    new_hidden = _reset_hidden_on_done(new_hidden, done_t)
                    return (new_hidden, code_t), (action_params, mean, logvar, d_mean, d_logvar)

                (final_hidden, _), (logits, means, logvars, d_means, d_logvars) = jax.lax.scan(
                    step_stored,
                    (initial_hidden, init_prev_code),
                    (obs_seq, done_seq, stored_keys),
                )

            else:

                def step_stored(carry, inputs):
                    hidden_list, prev_code = carry
                    obs_t = {k: inputs[0][k] for k in inputs[0]}
                    done_t = inputs[1]
                    keys_t = inputs[2]

                    code_t = jnp.round(obs_t["kpms_code"][..., 0]).astype(jnp.int32)
                    code_changed = (code_t != prev_code) | (prev_code == -1)
                    hidden_list = _reinit_hidden_on_code(hidden_list, code_t, code_changed)

                    decoder_input, code_idx, mean, logvar, z_e_scaled = self._encode(
                        obs_t, keys_t, deterministic, z_e_scale
                    )
                    z_e_arg = _z_e_arg_for_action(z_e_scaled)
                    action_params, new_hidden, _h_t = self._decode_rnn(
                        decoder_input, hidden_list, z_e_for_action=z_e_arg,
                    )
                    new_hidden = _reset_hidden_on_done(new_hidden, done_t)
                    return (new_hidden, code_t), (action_params, mean, logvar)

                (final_hidden, _), (logits, means, logvars) = jax.lax.scan(
                    step_stored,
                    (initial_hidden, init_prev_code),
                    (obs_seq, done_seq, stored_keys),
                )
                d_means = None
                d_logvars = None

        else:

            if self.use_distillation_head:

                def step_fresh(carry, inputs):
                    hidden_list, prev_code, step_key = carry
                    obs_t = {k: inputs[0][k] for k in inputs[0]}
                    done_t = inputs[1]
                    step_key, next_key = jax.random.split(step_key)

                    code_t = jnp.round(obs_t["kpms_code"][..., 0]).astype(jnp.int32)
                    code_changed = (code_t != prev_code) | (prev_code == -1)
                    hidden_list = _reinit_hidden_on_code(hidden_list, code_t, code_changed)

                    decoder_input, code_idx, mean, logvar, z_e_scaled = self._encode(
                        obs_t, step_key, deterministic, z_e_scale
                    )
                    action_params, new_hidden, h_t = self._decode_rnn(
                        decoder_input, hidden_list, z_e_for_action=None,
                    )
                    d_mean, d_logvar = self._distill_head(h_t)
                    new_hidden = _reset_hidden_on_done(new_hidden, done_t)
                    return (new_hidden, code_t, next_key), (action_params, mean, logvar, d_mean, d_logvar)

                (final_hidden, _, _), (logits, means, logvars, d_means, d_logvars) = jax.lax.scan(
                    step_fresh,
                    (initial_hidden, init_prev_code, key),
                    (obs_seq, done_seq),
                )

            else:

                def step_fresh(carry, inputs):
                    hidden_list, prev_code, step_key = carry
                    obs_t = {k: inputs[0][k] for k in inputs[0]}
                    done_t = inputs[1]
                    step_key, next_key = jax.random.split(step_key)

                    code_t = jnp.round(obs_t["kpms_code"][..., 0]).astype(jnp.int32)
                    code_changed = (code_t != prev_code) | (prev_code == -1)
                    hidden_list = _reinit_hidden_on_code(hidden_list, code_t, code_changed)

                    decoder_input, code_idx, mean, logvar, z_e_scaled = self._encode(
                        obs_t, step_key, deterministic, z_e_scale
                    )
                    z_e_arg = _z_e_arg_for_action(z_e_scaled)
                    action_params, new_hidden, _h_t = self._decode_rnn(
                        decoder_input, hidden_list, z_e_for_action=z_e_arg,
                    )
                    new_hidden = _reset_hidden_on_done(new_hidden, done_t)
                    return (new_hidden, code_t, next_key), (action_params, mean, logvar)

                (final_hidden, _, _), (logits, means, logvars) = jax.lax.scan(
                    step_fresh,
                    (initial_hidden, init_prev_code, key),
                    (obs_seq, done_seq),
                )
                d_means = None
                d_logvars = None

        return logits, means, logvars, d_means, d_logvars, final_hidden


# Backward-compatible alias
MoSeqDecoderNetwork = MoSeqEncoderDecoderNetwork
