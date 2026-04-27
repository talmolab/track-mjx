"""SDS reward computation for SMP priors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import flax
import jax
import jax.numpy as jnp
from brax.training.types import Params

from track_mjx.agent.smp.tinymdm import (
    SMPNormalizer,
    TinyDiTDenoiser,
    TinyMDMConfig,
    add_noise,
    make_diffusion_schedule,
)


@dataclass(frozen=True)
class SMPRewardConfig:
    """SMP reward settings."""

    diffusion_steps: tuple[int, ...] = (22, 15, 8)
    sds_loss_scale: float = 6.0
    smp_reward_scale: float = 1.0
    task_reward_weight: float = 0.5
    smp_reward_weight: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["diffusion_steps"] = list(self.diffusion_steps)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMPRewardConfig":
        return cls(
            diffusion_steps=tuple(data.get("diffusion_steps", (22, 15, 8))),
            sds_loss_scale=float(data.get("sds_loss_scale", 6.0)),
            smp_reward_scale=float(data.get("smp_reward_scale", 1.0)),
            task_reward_weight=float(data.get("task_reward_weight", 0.5)),
            smp_reward_weight=float(data.get("smp_reward_weight", 0.5)),
        )


@flax.struct.dataclass
class DiffNormalizer:
    """Normalizer for raw SDS losses, one statistic per diffusion step."""

    mean_abs: jnp.ndarray
    min_diff: float = 1.0e-4

    @classmethod
    def identity(cls, num_steps: int) -> "DiffNormalizer":
        return cls(mean_abs=jnp.ones((num_steps,), dtype=jnp.float32))

    def normalize(self, losses: jnp.ndarray) -> jnp.ndarray:
        return losses / jnp.maximum(self.mean_abs, self.min_diff)


def diff_normalizer_from_losses(losses: jnp.ndarray) -> DiffNormalizer:
    return DiffNormalizer(mean_abs=jnp.maximum(jnp.mean(jnp.abs(losses), axis=0), 1e-4))


def esm_sds_losses(
    params: Params,
    normalizer: SMPNormalizer,
    x_obs: jnp.ndarray,
    rng: jax.Array,
    model_config: TinyMDMConfig,
    diffusion_steps: Sequence[int] = (22, 15, 8),
    model: TinyDiTDenoiser | None = None,
) -> jnp.ndarray:
    """Computes raw ESM/SDS losses for flat SMP observations.

    The current implementation supports epsilon prediction.  With unclipped DDIM
    reconstruction this is equivalent to the MimicKit ESM expression, because
    the implied ``eps_pred`` from ``pred_original_sample`` is the denoiser output.
    """

    if model is None:
        model = TinyDiTDenoiser(model_config)
    if model_config.prediction_type != "epsilon":
        raise ValueError(
            f"SMP reward currently supports epsilon prediction, got {model_config.prediction_type}."
        )

    x_norm = normalizer.normalize(x_obs, model_config.num_history_steps)
    schedule = make_diffusion_schedule(model_config)
    step_values = jnp.asarray(diffusion_steps, dtype=jnp.int32)
    keys = jax.random.split(rng, len(diffusion_steps))

    def loss_for_step(key: jax.Array, t_value: jnp.ndarray) -> jnp.ndarray:
        t = jnp.full((x_norm.shape[0],), t_value, dtype=jnp.int32)
        noise = jax.random.normal(key, x_norm.shape, dtype=x_norm.dtype)
        x_t = add_noise(x_norm, noise, t, schedule)
        pred_noise = model.apply({"params": params}, x_t, t)
        return jnp.mean(jnp.square(pred_noise - noise), axis=-1)

    losses = jax.vmap(loss_for_step)(keys, step_values)
    return jnp.swapaxes(losses, 0, 1)


def compute_smp_reward(
    params: Params,
    normalizer: SMPNormalizer,
    diff_normalizer: DiffNormalizer,
    x_obs: jnp.ndarray,
    rng: jax.Array,
    model_config: TinyMDMConfig,
    reward_config: SMPRewardConfig = SMPRewardConfig(),
    model: TinyDiTDenoiser | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Computes SMP prior reward for a batch of flat SMP observations."""

    raw_losses = esm_sds_losses(
        params=params,
        normalizer=normalizer,
        x_obs=x_obs,
        rng=rng,
        model_config=model_config,
        diffusion_steps=reward_config.diffusion_steps,
        model=model,
    )
    norm_losses = diff_normalizer.normalize(raw_losses)
    mean_norm_loss = jnp.mean(norm_losses, axis=-1)
    smp_reward = (
        jnp.exp(-mean_norm_loss * reward_config.sds_loss_scale)
        * reward_config.smp_reward_scale
    )
    raw_mean = jnp.mean(raw_losses, axis=-1)
    return smp_reward, {
        "smp_reward_mean": jnp.mean(smp_reward),
        "smp_reward_std": jnp.std(smp_reward),
        "sds_loss_mean": jnp.mean(raw_mean),
        "sds_loss_std": jnp.std(raw_mean),
    }


def blend_task_and_smp_rewards(
    task_reward: jnp.ndarray,
    smp_reward: jnp.ndarray,
    reward_config: SMPRewardConfig,
) -> jnp.ndarray:
    return (
        reward_config.task_reward_weight * task_reward
        + reward_config.smp_reward_weight * smp_reward
    )
