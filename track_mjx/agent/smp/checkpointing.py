"""Checkpoint helpers for SMP priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from brax.training.types import Params

from track_mjx.agent.smp.features import SMPFeatureSpec, numpy_metadata
from track_mjx.agent.smp.reward import DiffNormalizer, SMPRewardConfig
from track_mjx.agent.smp.tinymdm import (
    SMPNormalizer,
    TinyMDMConfig,
    init_denoiser_params,
)


def save_prior(
    output_dir: str | Path,
    params: Params,
    ema_params: Params,
    normalizer: SMPNormalizer,
    diff_normalizer: DiffNormalizer,
    model_config: TinyMDMConfig,
    feature_spec: SMPFeatureSpec,
    metadata: Mapping[str, Any],
    reward_config: SMPRewardConfig = SMPRewardConfig(),
) -> None:
    """Saves a portable SMP prior checkpoint."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    (output / "ema_params.msgpack").write_bytes(flax.serialization.to_bytes(ema_params))
    np.savez(
        output / "normalizer.npz",
        mean=np.asarray(normalizer.mean),
        std=np.asarray(normalizer.std),
        clip=np.asarray(normalizer.clip),
    )
    np.savez(
        output / "diff_normalizer.npz",
        mean_abs=np.asarray(diff_normalizer.mean_abs),
        min_diff=np.asarray(diff_normalizer.min_diff),
    )

    config = {
        "model": model_config.to_dict(),
        "feature_spec": feature_spec.to_dict(),
        "reward": reward_config.to_dict(),
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (output / "metadata.json").write_text(
        json.dumps(numpy_metadata(dict(metadata)), indent=2, sort_keys=True)
    )


def load_prior(
    checkpoint_dir: str | Path,
    use_ema: bool = True,
    init_rng: jax.Array | None = None,
) -> dict[str, Any]:
    """Loads an SMP prior checkpoint."""

    checkpoint = Path(checkpoint_dir)
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())
    model_config = TinyMDMConfig.from_dict(config["model"])
    feature_spec = SMPFeatureSpec.from_dict(config["feature_spec"])
    reward_config = SMPRewardConfig.from_dict(config.get("reward", {}))

    if init_rng is None:
        init_rng = jax.random.PRNGKey(0)
    init_params = init_denoiser_params(init_rng, model_config)

    params_name = "ema_params.msgpack" if use_ema else "params.msgpack"
    params_path = checkpoint / params_name
    if not params_path.exists():
        params_path = checkpoint / "params.msgpack"
    params = flax.serialization.from_bytes(init_params, params_path.read_bytes())

    norm_npz = np.load(checkpoint / "normalizer.npz")
    normalizer = SMPNormalizer(
        mean=jnp.asarray(norm_npz["mean"], dtype=jnp.float32),
        std=jnp.asarray(norm_npz["std"], dtype=jnp.float32),
        clip=float(norm_npz["clip"]),
    )
    diff_npz = np.load(checkpoint / "diff_normalizer.npz")
    diff_normalizer = DiffNormalizer(
        mean_abs=jnp.asarray(diff_npz["mean_abs"], dtype=jnp.float32),
        min_diff=float(diff_npz["min_diff"]),
    )
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "params": params,
        "model_config": model_config,
        "feature_spec": feature_spec,
        "reward_config": reward_config,
        "normalizer": normalizer,
        "diff_normalizer": diff_normalizer,
        "metadata": metadata,
    }
