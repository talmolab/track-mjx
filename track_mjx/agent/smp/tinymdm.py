"""Flax TinyMDM prior used by rodent SMP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from brax.training.types import Params


@dataclass(frozen=True)
class TinyMDMConfig:
    """Configuration for the unconditional TinyMDM DiT denoiser."""

    input_dim: int
    num_history_steps: int = 10
    num_train_timesteps: int = 50
    loss_type: str = "l1"
    prediction_type: str = "epsilon"
    beta_schedule: str = "squaredcos_cap_v2"
    num_layers: int = 2
    num_attention_heads: int = 4
    attention_head_dim: int = 64
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    normalizer_std_clip: float = 0.2

    @property
    def input_channel(self) -> int:
        if self.input_dim % self.num_history_steps != 0:
            raise ValueError(
                "input_dim must be divisible by num_history_steps: "
                f"{self.input_dim} vs {self.num_history_steps}"
            )
        return self.input_dim // self.num_history_steps

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TinyMDMConfig":
        normalized = dict(data)
        if "T" in normalized and "num_train_timesteps" not in normalized:
            normalized["num_train_timesteps"] = normalized.pop("T")
        if "estimate_mode" in normalized and "prediction_type" not in normalized:
            normalized["prediction_type"] = normalized.pop("estimate_mode")
        if "noise_schedule_mode" in normalized and "beta_schedule" not in normalized:
            normalized["beta_schedule"] = normalized.pop("noise_schedule_mode")
        return cls(**{k: v for k, v in normalized.items() if k in cls.__annotations__})


@flax.struct.dataclass
class SMPNormalizer:
    """Per-frame normalizer for flattened SMP windows."""

    mean: jnp.ndarray
    std: jnp.ndarray
    clip: float = 1.0e8

    def normalize(self, x: jnp.ndarray, num_history_steps: int) -> jnp.ndarray:
        frame_shape = x.shape[:-1] + (num_history_steps, -1)
        framed = x.reshape(frame_shape)
        norm = (framed - self.mean) / self.std
        norm = jnp.clip(norm, -self.clip, self.clip)
        return norm.reshape(x.shape)

    def unnormalize(self, x: jnp.ndarray, num_history_steps: int) -> jnp.ndarray:
        frame_shape = x.shape[:-1] + (num_history_steps, -1)
        framed = x.reshape(frame_shape)
        raw = framed * self.std + self.mean
        return raw.reshape(x.shape)


def normalizer_from_samples(
    samples: jnp.ndarray,
    num_history_steps: int,
    std_clip: float = 0.2,
    clip: float = 1.0e8,
) -> SMPNormalizer:
    """Computes a MimicKit-style per-frame normalizer from flat windows."""

    frames = samples.reshape((-1, samples.shape[-1] // num_history_steps))
    mean = jnp.mean(frames, axis=0)
    std = jnp.std(frames, axis=0)
    std = jnp.maximum(std, std_clip)
    return SMPNormalizer(mean=mean, std=std, clip=clip)


def betas_for_alpha_bar(
    num_train_timesteps: int,
    max_beta: float = 0.999,
) -> jnp.ndarray:
    """Diffusers ``squaredcos_cap_v2`` beta schedule."""

    def alpha_bar(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2

    steps = jnp.arange(num_train_timesteps, dtype=jnp.float32)
    t1 = steps / num_train_timesteps
    t2 = (steps + 1.0) / num_train_timesteps
    betas = 1.0 - alpha_bar(t2) / alpha_bar(t1)
    return jnp.clip(betas, 0.0, max_beta)


@flax.struct.dataclass
class DiffusionSchedule:
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray


def make_diffusion_schedule(config: TinyMDMConfig) -> DiffusionSchedule:
    if config.beta_schedule != "squaredcos_cap_v2":
        raise ValueError(f"Unsupported beta schedule: {config.beta_schedule}")
    betas = betas_for_alpha_bar(config.num_train_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def add_noise(
    x_start: jnp.ndarray,
    noise: jnp.ndarray,
    timesteps: jnp.ndarray,
    schedule: DiffusionSchedule,
) -> jnp.ndarray:
    alpha = schedule.alphas_cumprod[timesteps]
    while alpha.ndim < x_start.ndim:
        alpha = alpha[..., None]
    return jnp.sqrt(alpha) * x_start + jnp.sqrt(1.0 - alpha) * noise


def timestep_embedding(
    timesteps: jnp.ndarray,
    dim: int = 256,
    max_period: int = 10000,
    flip_sin_to_cos: bool = True,
) -> jnp.ndarray:
    """Sinusoidal embedding compatible with diffusers' timestep embedding."""

    half = dim // 2
    exponent = -jnp.log(float(max_period)) * jnp.arange(half, dtype=jnp.float32)
    exponent = exponent / jnp.maximum(half, 1)
    freqs = jnp.exp(exponent)
    args = timesteps.astype(jnp.float32)[:, None] * freqs[None]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if flip_sin_to_cos:
        emb = jnp.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def positional_embedding(length: int, dim: int) -> jnp.ndarray:
    pos = jnp.arange(length, dtype=jnp.float32)[:, None]
    half = dim // 2
    div = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / jnp.maximum(half, 1)
    )
    emb = jnp.concatenate([jnp.sin(pos * div), jnp.cos(pos * div)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb[None]


class TimestepMLP(nn.Module):
    inner_dim: int

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        emb = timestep_embedding(timesteps, 256)
        emb = nn.Dense(self.inner_dim)(emb)
        emb = nn.silu(emb)
        emb = nn.Dense(6 * self.inner_dim)(emb)
        return emb[:, None, :]


class SwiGLUFeedForward(nn.Module):
    dim: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_dim = self.dim * self.mlp_ratio
        x_proj = nn.Dense(2 * hidden_dim)(x)
        hidden, gate = jnp.split(x_proj, 2, axis=-1)
        x = hidden * nn.silu(gate)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        return nn.Dense(self.dim)(x)


class DiTBlock(nn.Module):
    dim: int
    num_heads: int
    head_dim: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        hidden: jnp.ndarray,
        time_hidden: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        table = self.param(
            "scale_shift_table",
            nn.initializers.normal(stddev=self.dim**-0.5),
            (1, 1, 6, self.dim),
        )
        shifts = table + time_hidden.reshape(time_hidden.shape[0], 1, 6, self.dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.moveaxis(
            shifts, -2, 0
        )

        h1 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5)(hidden)
        h1 = h1 * (1.0 + scale_msa) + shift_msa
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.head_dim,
            out_features=self.dim,
            use_bias=False,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(h1)
        hidden = hidden + gate_msa * attn

        h2 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-5)(hidden)
        h2 = h2 * (1.0 + scale_mlp) + shift_mlp
        ff = SwiGLUFeedForward(
            dim=self.dim,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
        )(h2, deterministic=deterministic)
        return hidden + gate_mlp * ff


class TinyDiTDenoiser(nn.Module):
    config: TinyMDMConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        timesteps: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        batch_size = x.shape[0]
        hidden = x.reshape((batch_size, cfg.num_history_steps, cfg.input_channel))

        residual = hidden
        hidden = nn.Dense(cfg.input_channel, use_bias=False, name="preprocess")(hidden)
        hidden = hidden + residual

        hidden = nn.Dense(cfg.inner_dim, use_bias=False, name="proj_in")(hidden)
        hidden = hidden + positional_embedding(cfg.num_history_steps, cfg.inner_dim)
        time_hidden = TimestepMLP(cfg.inner_dim)(timesteps)

        for i in range(cfg.num_layers):
            hidden = DiTBlock(
                dim=cfg.inner_dim,
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.attention_head_dim,
                mlp_ratio=cfg.mlp_ratio,
                dropout_rate=cfg.dropout_rate,
                name=f"block_{i}",
            )(hidden, time_hidden, deterministic=deterministic)

        hidden = nn.Dense(cfg.input_channel, use_bias=False, name="proj_out")(hidden)
        residual = hidden
        hidden = nn.Dense(cfg.input_channel, use_bias=False, name="postprocess")(hidden)
        hidden = hidden + residual
        return hidden.reshape((batch_size, cfg.input_dim))


def init_denoiser_params(
    rng: jax.Array,
    config: TinyMDMConfig,
) -> Params:
    model = TinyDiTDenoiser(config)
    dummy_x = jnp.zeros((1, config.input_dim), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    return model.init(rng, dummy_x, dummy_t)["params"]


def denoising_loss(
    params: Params,
    x0: jnp.ndarray,
    rng: jax.Array,
    config: TinyMDMConfig,
    model: TinyDiTDenoiser | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Computes TinyMDM denoising loss on normalized flat SMP windows."""

    if model is None:
        model = TinyDiTDenoiser(config)
    schedule = make_diffusion_schedule(config)
    rng_t, rng_noise = jax.random.split(rng)
    timesteps = jax.random.randint(
        rng_t, (x0.shape[0],), minval=0, maxval=config.num_train_timesteps
    )
    noise = jax.random.normal(rng_noise, x0.shape, dtype=x0.dtype)
    x_t = add_noise(x0, noise, timesteps, schedule)
    pred = model.apply({"params": params}, x_t, timesteps)

    if config.prediction_type != "epsilon":
        raise ValueError(f"Unsupported prediction type: {config.prediction_type}")
    target = noise
    err = pred - target
    if config.loss_type == "l1":
        loss = jnp.mean(jnp.abs(err))
    elif config.loss_type == "l2":
        loss = jnp.mean(jnp.square(err))
    else:
        raise ValueError(f"Unsupported loss type: {config.loss_type}")
    return loss, {
        "prior_loss": loss,
        "prior_pred_abs_mean": jnp.mean(jnp.abs(pred)),
        "prior_noise_abs_mean": jnp.mean(jnp.abs(noise)),
    }


@flax.struct.dataclass
class EMAState:
    params: Params
    step: jnp.ndarray
    decay: float = 0.995
    update_every: int = 10
    update_after_step: int = 5000


def update_ema(ema: EMAState, params: Params) -> EMAState:
    step = ema.step + 1
    should_update = (step % ema.update_every) == 0
    should_copy = step <= ema.update_after_step

    def copy_params(_: Params) -> Params:
        return params

    def blend_params(old_params: Params) -> Params:
        return jax.tree_util.tree_map(
            lambda old, new: ema.decay * old + (1.0 - ema.decay) * new,
            old_params,
            params,
        )

    new_params = jax.lax.cond(
        should_update,
        lambda old: jax.lax.cond(should_copy, copy_params, blend_params, old),
        lambda old: old,
        ema.params,
    )
    return replace(ema, params=new_params, step=step)


def make_optimizer(
    learning_rate: float, grad_clip_norm: float
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=0.0),
    )
