"""Encoder-decoder policy networks for KPMS-driven motor control.

The network receives a pre-computed syllable code (integer, passed as float in
``obs["kpms_code"]``), the agent's proprioception, and optionally the reference
trajectory (``obs["imitation_target"]``). It embeds the code via a learned
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
            imitation_target = obs["imitation_target"]

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
    """RNN decoder policy: code embedding + z_e + proprio -> GRU -> action.

    The decoder maintains hidden state across timesteps so it can produce
    temporally coherent control autonomously given a syllable code. The
    optional continuous encoder (z_e) acts as a training scaffold that can
    be annealed to zero via ``z_e_scale``.

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

    def _encode(
        self,
        obs: dict[str, jnp.ndarray],
        key=None,
        deterministic: bool = False,
        z_e_scale: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Encode obs into (decoder_input, code_idx, mean, logvar)."""
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
            imitation_target = obs["imitation_target"]

            h = imitation_target
            for dense, ln in zip(self.enc_layers, self.enc_lns):
                h = dense(h)
                h = self.activation(h)
                h = ln(h)

            mean = self.mean_layer(h)
            logvar = self.logvar_layer(h)

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

            decoder_input = jnp.concatenate([code_emb, z_e_scaled, proprio], axis=-1)
        else:
            mean = None
            logvar = None
            decoder_input = jnp.concatenate([code_emb, proprio], axis=-1)

        return decoder_input, code_idx, mean, logvar

    def _decode_rnn(
        self,
        x: jnp.ndarray,
        hidden: list[jnp.ndarray],
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Run one timestep through the stacked GRU and action head."""
        new_hidden = []
        rnn_input = x
        for cell, h in zip(self.rnn_cells, hidden):
            new_h, _ = cell(h, rnn_input)
            new_hidden.append(new_h)
            rnn_input = new_h
        action_params = self.action_head(rnn_input)
        return action_params, new_hidden

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
    ]:
        """Single-timestep forward pass (no per-code decay — used in rollout).

        Returns:
            ``(action_params, code_idx, mean, logvar, new_hidden)``.
        """
        decoder_input, code_idx, mean, logvar = self._encode(
            obs, key, deterministic, z_e_scale,
        )
        action_params, new_hidden = self._decode_rnn(decoder_input, hidden)
        return action_params, code_idx, mean, logvar, new_hidden

    def apply_sequence(
        self,
        obs_seq: dict[str, jnp.ndarray],
        initial_hidden: list[jnp.ndarray],
        done_seq: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
        stored_keys: jax.Array | None = None,
        z_e_scale: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None, list[jnp.ndarray]]:
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
            ``(logits, means, logvars, final_hidden)`` each ``[T, B, ...]``.
        """

        def _reset_hidden(hidden_list, done_t):
            done_expanded = done_t[..., None]
            return [jnp.where(done_expanded, 0.0, h) for h in hidden_list]

        # Validate stored_keys shape if provided
        if stored_keys is not None:
            ref_obs = jax.tree_util.tree_leaves(obs_seq)[0]
            expected_shape = (ref_obs.shape[0], ref_obs.shape[1], 2)
            if stored_keys.shape != expected_shape:
                raise ValueError(
                    f"stored_keys has shape {stored_keys.shape}, expected "
                    f"{expected_shape}. stored_keys must have shape [T, B, 2]."
                )

        if stored_keys is not None:

            def step_stored(carry, inputs):
                hidden_list = carry
                obs_t = {k: inputs[0][k] for k in inputs[0]}
                done_t = inputs[1]
                keys_t = inputs[2]

                decoder_input, code_idx, mean, logvar = self._encode(
                    obs_t, keys_t, deterministic, z_e_scale
                )
                action_params, new_hidden = self._decode_rnn(decoder_input, hidden_list)
                new_hidden = _reset_hidden(new_hidden, done_t)
                return new_hidden, (action_params, mean, logvar)

            final_hidden, (logits, means, logvars) = jax.lax.scan(
                step_stored,
                initial_hidden,
                (obs_seq, done_seq, stored_keys),
            )
        else:

            def step_fresh(carry, inputs):
                hidden_list, step_key = carry
                obs_t = {k: inputs[0][k] for k in inputs[0]}
                done_t = inputs[1]
                step_key, next_key = jax.random.split(step_key)

                decoder_input, code_idx, mean, logvar = self._encode(
                    obs_t, step_key, deterministic, z_e_scale
                )
                action_params, new_hidden = self._decode_rnn(decoder_input, hidden_list)
                new_hidden = _reset_hidden(new_hidden, done_t)
                return (new_hidden, next_key), (action_params, mean, logvar)

            (final_hidden, _), (logits, means, logvars) = jax.lax.scan(
                step_fresh,
                (initial_hidden, key),
                (obs_seq, done_seq),
            )

        return logits, means, logvars, final_hidden


# Backward-compatible alias
MoSeqDecoderNetwork = MoSeqEncoderDecoderNetwork
