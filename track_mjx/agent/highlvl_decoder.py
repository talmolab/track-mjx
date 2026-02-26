"""Helpers for loading high-level transfer decoders from mimic checkpoints."""

from __future__ import annotations

import os
from typing import Any

import hydra
from omegaconf import OmegaConf

from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks
from track_mjx.agent.recurrent_ppo import networks as recurrent_ppo_networks


def resolve_mimic_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve mimic checkpoint path to absolute path."""
    if os.path.isabs(checkpoint_path):
        return os.path.abspath(checkpoint_path)
    return hydra.utils.to_absolute_path(f"./model_checkpoints/{checkpoint_path}")


def load_mimic_checkpoint_and_decoder_fns(
    checkpoint_path: str,
) -> tuple[Any, dict[str, Any]]:
    """Load mimic config and decoder callable bundle for high-level transfer.

    Returns a dictionary with a ``mode`` key:
    - ``feedforward``: ``decoder_inference_fn``
    - ``recurrent``: ``decoder_step_fn``, ``init_decoder_hidden_fn``,
      ``reset_decoder_hidden_fn``
    """
    full_path = resolve_mimic_checkpoint_path(checkpoint_path)
    mimic_cfg = OmegaConf.create(checkpointing.load_config_from_checkpoint(full_path))
    arch_name = mimic_cfg.network_config.get("arch_name", "intention")

    if arch_name == "intention":
        return mimic_cfg, {
            "mode": "feedforward",
            "decoder_inference_fn": ff_ppo_networks.make_decoder_policy_fn(full_path),
        }

    if arch_name == "recurrent_intention":
        (
            decoder_step_fn,
            init_decoder_hidden_fn,
            reset_decoder_hidden_fn,
        ) = recurrent_ppo_networks.make_decoder_policy_fns(full_path)
        return mimic_cfg, {
            "mode": "recurrent",
            "decoder_step_fn": decoder_step_fn,
            "init_decoder_hidden_fn": init_decoder_hidden_fn,
            "reset_decoder_hidden_fn": reset_decoder_hidden_fn,
        }

    raise ValueError(
        "Unsupported mimic checkpoint architecture for high-level transfer: "
        f"{arch_name!r}. Supported architectures are 'intention' and "
        "'recurrent_intention'."
    )
