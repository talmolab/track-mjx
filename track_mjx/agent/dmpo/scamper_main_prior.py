"""Load SCAMPER-main stage-3 checkpoints (motion prior + behavior prior) for DMPO.

A SCAMPER-main stage-3 checkpoint (asirbu77/SCAMPER main, ``DistillPriorNetwork_<step>``) holds four
frozen networks under its ``policy`` item, plus the observation normalizer:

* ``encoder`` -- the imitation encoder q(z | x, g) (unused here);
* ``decoder`` -- p(a | x, z), the same module shape as the ``mlp_prior`` decoder;
* ``prior``   -- the **motion prior** p_psi(z | x), trained jointly with the decoder;
* ``rgm``     -- the **behavior prior** p_phi(z | x), distilled from the frozen encoder.

SCAMPER main's ``Prior`` and ``Decoder`` are the same classes the ``mlp_prior`` path builds, so the
existing inference makers in :mod:`prior_utils` and the kl-anchor network's ``_PriorModule`` /
``_DecoderModule`` apply unchanged. This module only restores the checkpoint, without importing
SCAMPER main (its venv pins different packages), and rebuilds the normalizer as the
``DictRunningStatisticsState`` the DMPO rollout and learner dispatch on.

Ported from ``scratch/scamper-dmpo-vnl`` (branch ``scott_claude/scamper-main-prior``, commits
``62ad4d9`` and ``5b6650e``), where it reproduced SCAMPER main's own inference functions to 2.6e-6.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training import types as brax_types
from brax.training.acme import running_statistics

from track_mjx.agent.dmpo.obs_utils import DictRunningStatisticsState

__all__ = [
    "PRIOR_FORMATS",
    "ScamperMainCheckpoint",
    "load_scamper_main_checkpoint",
]

# "mlp_prior": the SCAMPER-fork behavior-prior checkpoints every validated arm uses
# (prior_utils.load_prior_checkpoint). "scamper_main": a SCAMPER-main stage-3 checkpoint; the
# motion prior goes in the DMPO prior slot and the behavior prior is kept for exploration.
PRIOR_FORMATS = ("mlp_prior", "scamper_main")
_STEP_PREFIX = "DistillPriorNetwork"


class ScamperMainCheckpoint(NamedTuple):
    """The restored networks of a SCAMPER-main stage-3 checkpoint."""

    encoder_params: dict
    motion_prior_params: dict
    decoder_params: dict
    behavior_prior_params: dict
    normalizer_params: DictRunningStatisticsState
    config: dict
    step: int


def _running_stats_from_dict(d: dict) -> running_statistics.RunningStatisticsState:
    """Rebuild a brax running-statistics state from its restored dict."""
    count = d["count"]
    if isinstance(count, dict):  # brax UInt64 restored without a target
        count = brax_types.UInt64(
            hi=jnp.asarray(count["hi"]), lo=jnp.asarray(count["lo"])
        )
    return running_statistics.RunningStatisticsState(
        mean=jnp.asarray(d["mean"]),
        std=jnp.asarray(d["std"]),
        count=count,
        summed_variance=jnp.asarray(d["summed_variance"]),
        std_eps=jnp.asarray(d.get("std_eps", 0.0)),
    )


def load_scamper_main_checkpoint(
    checkpoint_path: str, step: int | None = None
) -> ScamperMainCheckpoint:
    """Restore a SCAMPER-main ``DistillPriorNetwork`` checkpoint.

    Args:
        checkpoint_path: The run directory holding ``DistillPriorNetwork_<step>`` folders.
        step: The step to restore; ``None`` restores the latest.

    Returns:
        The restored networks, normalizer, config and the step actually restored.

    Raises:
        FileNotFoundError: If the directory holds no ``DistillPriorNetwork_*`` step.
        KeyError: If the checkpoint lacks the behavior prior (``rgm``) or its layer sizes.
    """
    options = ocp.CheckpointManagerOptions(create=False, step_prefix=_STEP_PREFIX)
    with ocp.CheckpointManager(checkpoint_path, options=options) as mgr:
        if step is None:
            step = mgr.latest_step()
        if step is None:
            raise FileNotFoundError(f"no {_STEP_PREFIX}_* step under {checkpoint_path}")
        config = mgr.restore(
            step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )["config"]
        normalizer, network = mgr.restore(
            step, args=ocp.args.Composite(policy=ocp.args.StandardRestore())
        )["policy"]

    params = network["params"]
    for key in ("encoder", "prior", "decoder", "rgm"):
        if key not in params:
            raise KeyError(
                f"{checkpoint_path} step {step}: no '{key}' network; a SCAMPER-main stage-3 "
                f"checkpoint holds encoder/prior/decoder/rgm, found {sorted(params)}"
            )
    if "rgm_layer_sizes" not in config.get("network_config", {}):
        raise KeyError(
            f"{checkpoint_path} step {step}: config.network_config has no rgm_layer_sizes"
        )
    normalizer_params = DictRunningStatisticsState(
        imitation_target=_running_stats_from_dict(normalizer["imitation_target"]),
        proprioception=_running_stats_from_dict(normalizer["proprioception"]),
    )
    return ScamperMainCheckpoint(
        encoder_params=params["encoder"],
        motion_prior_params=params["prior"],
        decoder_params=params["decoder"],
        behavior_prior_params=params["rgm"],
        normalizer_params=normalizer_params,
        config=config,
        step=int(step),
    )
