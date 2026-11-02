"""Slow-clock memory core variant of the B-aggressive kl-anchor DMPO networks.

The FF kl-anchor policy head (`_PolicyHeadModule`) plus a GRU "slow core"
that updates only every `tick_stride` env steps and holds its state in
between. The slow state is appended LAST to the FF head's MLP input, so the
head is the FF head with `hidden_0/kernel` widened by `slow_hidden_size`
trailing rows and one extra subtree, `slow_core`:

  * Graft: copy every leaf of an FF kl-anchor checkpoint and zero-fill the
    extra `hidden_0` rows, and the policy equals the FF policy for ANY
    hidden state -- slow_h only reaches the output through those rows.
    Param names therefore mirror `_PolicyHeadModule` exactly: auto-named
    `BinocularVisionEncoder_0`, `hidden_i` / `LayerNorm_i`, zero-init
    `residual`. The GRU is named explicitly ("slow_core") so it cannot shift
    an auto-name.
  * Clock: the hidden is the tuple `(slow_h, phase)`, phase a [1] float
    leaf. The core ticks when phase < 0.5 and phase advances mod
    tick_stride. The rollout zeroes every hidden leaf on done, so an episode
    reset restarts the clock with a tick on the new episode's first step.
    tick_stride=1 ticks every step (a plain per-step GRU feeding the FF MLP).

Prior and Decoder are the modules from networks_kl_anchor (warm-start splice
unchanged) and the head stays under `policy_head`, so
`optim_kl_anchor.label_param_tree` gives the slow core the policy_head LR.
"""

from __future__ import annotations

# Runtime import despite `from __future__ import annotations`: this module's
# Flax Module dataclass annotations are resolved at runtime by
# typing.get_type_hints() (e.g. the optional beartype runtime-checker
# extra), so a TYPE_CHECKING-only import raises NameError even though flax
# init/apply/clone never need it resolved.
from collections.abc import Sequence  # noqa: TC003
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

# Runtime import for the same reason as the Sequence import above.
from track_mjx.agent.dmpo.config import DMPOConfig  # noqa: TC001
from track_mjx.agent.dmpo.networks import DMPONetworks
from track_mjx.agent.dmpo.networks_kl_anchor import (
    _DecoderModule,
    _PriorModule,
    _splice_warm_start,
)
from track_mjx.agent.dmpo.networks_vision_bottleneck import (
    _ValueVisionCriticNet,
)
from track_mjx.agent.dmpo.vision_encoders import BinocularVisionEncoder

tfd = tfp.distributions


class SlowCoreMeta(NamedTuple):
    """Static description of the slow-core hidden state `(slow_h, phase)`.

    Drop-in for `RecurrentPolicyMeta` (rollout / eval / training loop only
    use `init_hidden` and `store_dtype`; `cell_type` / `hidden_sizes` are
    read for logging). `tick_stride` is the core's update period in env
    steps; `phase_leaf` is the index of the phase leaf in the hidden tuple,
    which the strided learner reads to find tick positions.
    """

    cell_type: str
    hidden_sizes: tuple
    store_dtype: Any
    tick_stride: int
    phase_leaf: int = 1

    def init_hidden(self, batch_size: int | None = None) -> tuple:
        """Zero `(slow_h, phase)`: `([B, H], [B, 1])`, or `([H], [1])` unbatched."""
        lead = () if batch_size is None else (batch_size,)
        return (
            jnp.zeros((*lead, self.hidden_sizes[0])),
            jnp.zeros((*lead, 1)),
        )


class _SlowCorePolicyHeadModule(nn.Module):
    """CNN + task_obs + proprio (+ held slow GRU state) -> residual.

    Creation order fixes the param names: encoder (auto
    `BinocularVisionEncoder_0`), GRU `slow_core`, `hidden_i` / `LayerNorm_i`,
    zero-init `residual`. Shape-agnostic over leading dims (unbatched is the
    production path: the rollout and learner vmap over envs).
    """

    layer_sizes: Sequence[int]
    slow_hidden_size: int
    tick_stride: int
    latent_size: int
    vision_shape: tuple
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool

    @nn.compact
    def __call__(self, vision, task_obs, proprio, hidden):
        slow_h, phase = hidden
        vis = BinocularVisionEncoder(
            feature_size=self.cnn_feature_size,
            channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
        )(vision)
        x = jnp.concatenate([vis, task_obs, proprio], axis=-1)

        # The core reads the tick frame only. It is evaluated every step and
        # selected by the tick mask (a cond would become a select under the
        # per-env vmap anyway). kernel_init matches the repo's GRU heads
        # (recurrent_ppo.get_rnn_cell), which cannot take an explicit name.
        tick = phase[..., :1] < 0.5  # [..., 1]; broadcasts over H
        g, _ = nn.GRUCell(
            features=self.slow_hidden_size,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            name="slow_core",
        )(slow_h, x)
        slow_h = jnp.where(tick, g, slow_h)

        h = jnp.concatenate([x, slow_h], axis=-1)  # slow_h LAST -> graftable
        for i, size in enumerate(self.layer_sizes):
            h = nn.Dense(size, name=f"hidden_{i}")(h)
            h = nn.LayerNorm(name=f"LayerNorm_{i}")(h)
            h = nn.silu(h)
        residual = nn.Dense(
            self.latent_size,
            name="residual",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)
        new_phase = jnp.mod(phase + 1.0, float(self.tick_stride))
        return residual, (slow_h, new_phase)


def _latent_residual(raw, z_logvar, mode: str, scale: float):
    """Head output -> latent residual. Same math as `_BAggressivePolicyNet` /
    `_BAggressiveRecurrentPolicyNet` (measured rationale for each mode lives
    in networks_kl_anchor.py); every mode maps raw == 0 to residual == 0."""
    if mode == "sigma_ball":
        # Norm-bounded eps in the prior's sigma units; sqrt(sum + 1e-12)
        # keeps the gradient finite at the zero-init raw == 0.
        sigma_prior = jnp.exp(0.5 * z_logvar)
        sq = jnp.sum(raw * raw, axis=-1, keepdims=True)
        norm = jnp.sqrt(sq + 1e-12)
        eps = raw * (scale * jnp.tanh(norm / scale) / norm)
        return sigma_prior * eps
    if mode == "sigma_tanh":
        sigma_prior = jnp.exp(0.5 * z_logvar)
        return scale * sigma_prior * jnp.tanh(raw)
    if mode == "tanh":
        return scale * jnp.tanh(raw)
    if mode == "none":
        return raw
    raise ValueError(
        f"unknown residual_mode {mode!r}; "
        "expected one of 'sigma_ball', 'sigma_tanh', 'tanh', 'none'"
    )


class _BAggressiveSlowCorePolicyNet(nn.Module):
    """Trainable Prior + slow-core PolicyHead + Decoder.

    Same apply paths as `_BAggressiveRecurrentPolicyNet`:
      - `__call__(obs, hidden) -> (tfd.MultivariateNormalDiag, new_hidden)`;
      - `raw(obs, hidden) -> (mu, scale, new_hidden)` (plain arrays, for the
        learner's `lax.scan`).
    """

    action_size: int
    latent_size: int
    vision_shape: tuple
    prior_layer_sizes: Sequence[int]
    decoder_layer_sizes: Sequence[int]
    policy_head_layer_sizes: Sequence[int]
    slow_hidden_size: int
    tick_stride: int
    cnn_feature_size: int
    cnn_channels: Sequence[int]
    mono_channels: int
    shared_weights: bool
    residual_mode: str = "sigma_ball"
    residual_scale: float = 5.0

    def setup(self):
        self.prior = _PriorModule(
            layer_sizes=tuple(self.prior_layer_sizes),
            latents=self.latent_size,
            name="prior",
        )
        self.decoder = _DecoderModule(
            layer_sizes=(*tuple(self.decoder_layer_sizes), 2 * self.action_size),
            name="decoder",
        )
        self.policy_head = _SlowCorePolicyHeadModule(
            layer_sizes=tuple(self.policy_head_layer_sizes),
            slow_hidden_size=self.slow_hidden_size,
            tick_stride=self.tick_stride,
            latent_size=self.latent_size,
            vision_shape=self.vision_shape,
            cnn_feature_size=self.cnn_feature_size,
            cnn_channels=tuple(self.cnn_channels),
            mono_channels=self.mono_channels,
            shared_weights=self.shared_weights,
            name="policy_head",
        )

    def raw(self, obs, hidden):
        proprio = obs["proprioception"]
        z_prior, z_logvar = self.prior(proprio)
        head_out, new_hidden = self.policy_head(
            obs["vision"], obs["imitation_target"], proprio, hidden
        )
        z = z_prior + _latent_residual(
            head_out, z_logvar, self.residual_mode, self.residual_scale
        )
        decoder_out = self.decoder(jnp.concatenate([z, proprio], axis=-1))
        mu = decoder_out[..., : self.action_size]
        # softplus + 1e-3 matches the imitation decoder's training transform
        # (brax NormalTanhDistribution); see networks_kl_anchor.py.
        scale = jax.nn.softplus(decoder_out[..., self.action_size :]) + 1e-3
        return mu, scale, new_hidden

    def __call__(self, obs, hidden):
        mu, scale, new_hidden = self.raw(obs, hidden)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=scale), new_hidden


def make_dmpo_kl_anchor_slow_networks(
    *,
    proprio_size: int,
    task_obs_size: int,
    action_size: int,
    latent_size: int,
    vision_shape: tuple,
    cfg: DMPOConfig,
    prior_layer_sizes: Sequence[int],
    decoder_layer_sizes: Sequence[int],
    policy_head_layer_sizes: Sequence[int] = (256, 256, 256),
    slow_hidden_size: int = 512,
    tick_stride: int = 10,
    rnn_cell: str = "gru",
    rnn_store_dtype: Any = "float16",
    cnn_feature_size: int = 32,
    cnn_channels: Sequence[int] = (4, 8, 16, 32),
    mono_channels: int = 1,
    shared_weights: bool = True,
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    warm_start_prior_params: dict | None = None,
    warm_start_decoder_params: dict | None = None,
    residual_mode: str = "sigma_ball",
    residual_scale: float = 5.0,
    critic_use_proprio: bool = False,
    critic_privileged_dim: int = 0,
) -> DMPONetworks:
    """Build the slow-core B-aggressive DMPO policy + FF critic (kl-anchor).

    Shares every kwarg of `make_dmpo_kl_anchor_rnn_networks` except the head
    block: `policy_head_layer_sizes` is the FF head's MLP (same meaning as in
    `make_dmpo_kl_anchor_networks`, so an FF checkpoint grafts in), plus
    `slow_hidden_size` (GRU width H) and `tick_stride` (core update period
    k in env steps). `rnn_cell` is "gru" only; `rnn_store_dtype` is the
    replay dtype of the stored hidden, both leaves -- it must hold every
    phase 0..k-1 exactly (float16: k <= 2049).

    Like the recurrent factory, `policy.init(rng, obs, hidden)` takes a
    hidden (use `recurrent_meta.init_hidden()`), and the warm-start
    prior/decoder are spliced after init.

    Returns:
        `DMPONetworks(policy, critic, recurrent_meta=SlowCoreMeta(...))`.
    """
    del proprio_size, task_obs_size  # unused directly; consumed via obs dict.

    if rnn_cell != "gru":
        raise ValueError(
            f"slow core cell {rnn_cell!r} is not supported; only 'gru' (the "
            "(slow_h, phase) hidden has no room for LSTM (c, h) pairs)."
        )
    tick_stride = int(tick_stride)
    if tick_stride < 1:
        raise ValueError(f"tick_stride must be >= 1, got {tick_stride}.")
    store_dtype = jnp.dtype(rnn_store_dtype)
    phases = np.arange(tick_stride, dtype=np.float64)
    if not np.array_equal(phases.astype(store_dtype).astype(np.float64), phases):
        raise ValueError(
            f"tick_stride={tick_stride}: rnn_store_dtype {store_dtype} cannot "
            f"hold every phase 0..{tick_stride - 1} exactly, so replay would "
            "store corrupted tick positions. Use a wider store dtype or a "
            "smaller tick_stride."
        )

    policy = _BAggressiveSlowCorePolicyNet(
        action_size=action_size,
        latent_size=latent_size,
        vision_shape=vision_shape,
        prior_layer_sizes=tuple(prior_layer_sizes),
        decoder_layer_sizes=tuple(decoder_layer_sizes),
        policy_head_layer_sizes=tuple(policy_head_layer_sizes),
        slow_hidden_size=int(slow_hidden_size),
        tick_stride=tick_stride,
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        residual_mode=residual_mode,
        residual_scale=residual_scale,
    )

    critic = _ValueVisionCriticNet(
        layer_sizes=tuple(value_hidden_layer_sizes),
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
        vision_shape=vision_shape,
        cnn_feature_size=cnn_feature_size,
        cnn_channels=tuple(cnn_channels),
        mono_channels=mono_channels,
        shared_weights=shared_weights,
        use_proprio=critic_use_proprio,
        privileged_dim=critic_privileged_dim,
    )

    meta = SlowCoreMeta(
        cell_type=rnn_cell,
        hidden_sizes=(int(slow_hidden_size),),
        store_dtype=store_dtype,
        tick_stride=tick_stride,
    )

    _orig_init = policy.init

    def policy_init(rng, obs, hidden):
        params = _orig_init(rng, obs, hidden)
        return _splice_warm_start(
            params, warm_start_prior_params, warm_start_decoder_params
        )

    policy.init = policy_init  # type: ignore[method-assign]

    return DMPONetworks(policy=policy, critic=critic, recurrent_meta=meta)
