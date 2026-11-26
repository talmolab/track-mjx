"""Frozen SCAMPER prior / decoder loading for DMPO task transfer.

The prior network (``Prior``), the checkpoint loader and the decoder / prior
inference factories come from SCAMPER (https://github.com/asirbu77/SCAMPER,
author Aidan Sirbu), installed as a pinned dependency rather than vendored:
``scamper.agent.mlp_prior.prior_networks`` and
``scamper.agent.task_transfer.walk_rear.checkpoint_utils`` at commit
``eb562779f0c79c7ddf03d9b3819295085265501e``. SCAMPER pins numpy, brax and
wandb versions that conflict with this package, so it is not listed in
``pyproject.toml``; install it with::

    pip install --no-deps -r requirements-dmpo-prior.txt

SCAMPER is imported lazily, so this module imports without it. Anything that
needs it (``load_prior_checkpoint``, ``Prior``, ``make_decoder_inference_fn``,
``make_prior_inference_fn``) raises an ``ImportError`` carrying the command
above when it is missing.

Only ``make_decoder_logits_fn`` is defined here, because SCAMPER has no
logits-only decoder factory. The other two factories delegate to SCAMPER; they
read only ``normalizer_params.proprioception``, so DMPO's normalizer type is
accepted as is.

The mlp_prior checkpoint layout is
``<ckpt>/PriorNetwork_<step>/{config, policy, train_state, _CHECKPOINT_METADATA}``
with ``policy = (normalizer_params, {"params": {"encoder", "prior", "decoder"}})``,
where ``normalizer_params`` is a ``DictRunningStatisticsState`` (dict-normalizer
format only; the legacy flat-normalizer format is not supported here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from brax.training import distribution
from brax.training.acme import running_statistics

from track_mjx.agent.dmpo.obs_utils import DictRunningStatisticsState
from track_mjx.agent.ff_ppo.intention_network import Decoder

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from scamper.agent.mlp_prior.prior_networks import Prior

__all__ = [
    "Prior",
    "load_prior_checkpoint",
    "make_decoder_inference_fn",
    "make_decoder_logits_fn",
    "make_prior_inference_fn",
]

_INSTALL_COMMAND = "pip install --no-deps -r requirements-dmpo-prior.txt"
_SCAMPER_PIN = (
    "scamper @ git+https://github.com/asirbu77/SCAMPER.git"
    "@eb562779f0c79c7ddf03d9b3819295085265501e"
)


def _import_scamper() -> tuple[ModuleType, ModuleType]:
    """Import the SCAMPER modules this module delegates to.

    Returns:
        ``(prior_networks, checkpoint_utils)`` modules from the pinned install.

    Raises:
        ImportError: If SCAMPER (or one of its own imports) is unavailable. The
            message names the install command and the pinned requirement.
    """
    try:
        from scamper.agent.mlp_prior import prior_networks
        from scamper.agent.task_transfer.walk_rear import checkpoint_utils
    except ImportError as err:
        raise ImportError(
            "Loading a frozen SCAMPER prior needs the pinned scamper package. "
            f"Install it with: {_INSTALL_COMMAND} "
            f"(or, without a source checkout: pip install --no-deps '{_SCAMPER_PIN}'). "
            f"Import failed: {err}"
        ) from err
    return prior_networks, checkpoint_utils


def __getattr__(name: str):
    """Re-export SCAMPER's ``Prior`` without importing SCAMPER at module load."""
    if name == "Prior":
        return _import_scamper()[0].Prior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_prior_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
) -> tuple[dict, dict, dict, DictRunningStatisticsState, dict]:
    """Load encoder, prior, and decoder parameters from an mlp_prior checkpoint.

    Delegates to SCAMPER's
    ``scamper.agent.task_transfer.walk_rear.checkpoint_utils.load_prior_checkpoint``
    and converts the normalizer from SCAMPER's own state class to
    ``track_mjx.agent.dmpo.obs_utils.DictRunningStatisticsState``, which the
    rollout and learner dispatch on with ``isinstance``. The restored values
    are passed through untouched.

    Only supports dict normalizer format.

    Args:
        checkpoint_path: Path to the mlp_prior checkpoint directory.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (encoder_params, prior_params, decoder_params,
                  normalizer_params, config_dict).
        normalizer_params is DictRunningStatisticsState.

    Raises:
        ImportError: If the pinned scamper package is not installed.
    """
    _, checkpoint_utils = _import_scamper()
    encoder_params, prior_params, decoder_params, normalizer_params, config = (
        checkpoint_utils.load_prior_checkpoint(checkpoint_path, step)
    )
    return (
        encoder_params,
        prior_params,
        decoder_params,
        _as_dmpo_normalizer(normalizer_params),
        config,
    )


def _as_dmpo_normalizer(normalizer_params) -> DictRunningStatisticsState:
    """Rebuild a dict normalizer state as DMPO's class, sharing the same arrays."""
    if isinstance(normalizer_params, DictRunningStatisticsState):
        return normalizer_params
    return DictRunningStatisticsState(
        imitation_target=normalizer_params.imitation_target,
        proprioception=normalizer_params.proprioception,
    )


def make_decoder_inference_fn(
    decoder_params: dict,
    normalizer_params: DictRunningStatisticsState,
    config: dict,
) -> Callable:
    """Create decoder inference function for use in high-level wrappers.

    Delegates to SCAMPER's
    ``scamper.agent.task_transfer.walk_rear.checkpoint_utils.make_decoder_inference_fn``.
    The decoder takes ``[latent, proprio]`` concatenated input and outputs an
    action (the mode of a ``NormalTanhDistribution``).

    Args:
        decoder_params: Frozen decoder parameters from the mlp_prior checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics. Only
            ``.proprioception`` is read, so DMPO's normalizer type works.
        config: Checkpoint config dict (as returned by ``load_prior_checkpoint``).

    Returns:
        Function ``(latent_proprio) -> (action, extras)`` where *latent_proprio*
        is ``[latent, proprio]`` concatenated along the last axis.

    Raises:
        ImportError: If the pinned scamper package is not installed.
    """
    _, checkpoint_utils = _import_scamper()
    return checkpoint_utils.make_decoder_inference_fn(
        decoder_params, normalizer_params, config
    )


def make_decoder_logits_fn(
    decoder_params: dict,
    normalizer_params: DictRunningStatisticsState,
    config: dict,
) -> Callable:
    """Create a decoder-logits inference function (raw pre-tanh, no mode).

    Returns the **raw** decoder logits (shape ``(..., 2*action_size)`` —
    concatenation of pre-tanh ``mu`` and ``log_std``). Used by the
    KL-anchor wrapper to expose `(mu_imit, log_std_imit)` to the policy
    loss for closed-form KL.

    Args:
        decoder_params: Frozen decoder parameters from the imit checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics.
        config: Checkpoint config dict (as returned by
            ``load_prior_checkpoint``).

    Returns:
        Function ``(latent_proprio) -> (logits, extras)`` where
        ``latent_proprio`` is ``[latent, proprio]`` concatenated and
        ``logits`` has shape ``(..., 2 * action_size)``.
    """
    action_size = config["network_config"]["action_size"]
    latent_size = config["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = tuple(config["network_config"]["decoder_layer_sizes"])

    # Use NormalTanhDistribution.param_size for the output layer to stay in
    # lockstep with make_decoder_inference_fn — both functions construct the
    # SAME decoder module shape; only the postprocessing differs (mode vs
    # raw logits). Hardcoding `2 * action_size` would silently desync if
    # brax ever ships a NormalTanh variant with a different param_size.
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    decoder_module = Decoder(
        layer_sizes=[
            *decoder_hidden_layer_sizes,
            parametric_action_distribution.param_size,
        ],
    )

    proprio_normalizer = normalizer_params.proprioception

    def decoder_logits_fn(latent_proprio: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        latent = latent_proprio[..., :latent_size]
        proprio = latent_proprio[..., latent_size:]
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)
        decoder_input = jnp.concatenate([latent, normalized_proprio], axis=-1)
        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)
        return logits, {"logits": logits}

    return decoder_logits_fn


def make_prior_inference_fn(
    prior_params: dict,
    normalizer_params: DictRunningStatisticsState,
    config: dict,
) -> Callable:
    """Create prior inference function for use in high-level wrappers.

    Delegates to SCAMPER's
    ``scamper.agent.task_transfer.walk_rear.checkpoint_utils.make_prior_inference_fn``.
    The prior takes proprioceptive observations and outputs the mean and
    log-variance of the latent distribution.

    Args:
        prior_params: Frozen prior parameters from the mlp_prior checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics. Only
            ``.proprioception`` is read, so DMPO's normalizer type works.
        config: Checkpoint config dict (as returned by ``load_prior_checkpoint``).

    Returns:
        Function ``(proprio) -> (mean, logvar)``.

    Raises:
        ImportError: If the pinned scamper package is not installed.
    """
    _, checkpoint_utils = _import_scamper()
    return checkpoint_utils.make_prior_inference_fn(
        prior_params, normalizer_params, config
    )
