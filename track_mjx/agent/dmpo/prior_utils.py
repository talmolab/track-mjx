"""Frozen SCAMPER prior / decoder loading for DMPO task transfer.

Provenance: ``Prior`` is vendored from SCAMPER
(https://github.com/asirbu77/SCAMPER, author Aidan Sirbu),
``scamper/agent/mlp_prior/prior_networks.py``; ``load_prior_checkpoint``,
``_create_abstract_prior_policy``, and ``_init_network_params`` are vendored
from the same repository's
``scamper/agent/task_transfer/walk_rear/checkpoint_utils.py``. Both files are
byte-identical to the public branch ``scott/dev`` at commit ``eb56277``.
SCAMPER's ``pyproject.toml`` declares the BSD-3-Clause license; no LICENSE
file is present at that commit. The three inference-fn factories
(``make_decoder_inference_fn``, ``make_decoder_logits_fn``,
``make_prior_inference_fn``) are from talmolab/vnl-playground's
``vnl_playground/tasks/prior_utils.py``, unchanged apart from import
retargets, so that DMPO no longer needs the `scamper` package importable at
all.

The mlp_prior checkpoint layout is
``<ckpt>/PriorNetwork_<step>/{config, policy, train_state, _CHECKPOINT_METADATA}``
with ``policy = (normalizer_params, {"params": {"encoder", "prior", "decoder"}})``,
where ``normalizer_params`` is a ``DictRunningStatisticsState`` (dict-normalizer
format only; the legacy flat-normalizer format is not supported here).
"""

from __future__ import annotations

# Runtime import despite `from __future__ import annotations`: Prior's Flax
# Module dataclass annotations are resolved at runtime by
# typing.get_type_hints() (e.g. the optional beartype runtime-checker extra),
# so a TYPE_CHECKING-only import raises NameError even though flax
# init/apply/clone never need it resolved. See networks_kl_anchor.py.
from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from brax.training import distribution, networks
from brax.training.acme import running_statistics, specs
from flax import linen as nn
from omegaconf import OmegaConf

from track_mjx.agent.dmpo.obs_utils import DictRunningStatisticsState
from track_mjx.agent.ff_ppo.intention_network import Decoder, Encoder

if TYPE_CHECKING:
    from collections.abc import Callable


class Prior(nn.Module):
    """Prior network that outputs distributions in latent space from proprioceptive observations.

    Takes only proprioceptive observations as input and outputs mean and log-variance
    of the latent distribution.

    Attributes:
        layer_sizes: Hidden layer dimensions for the MLP.
        latents: Dimension of the latent intention space.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
        bias: Whether to use bias terms in Dense layers.
    """

    layer_sizes: Sequence[int]
    latents: int
    activation: networks.ActivationFn = nn.silu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, get_activation: bool = False):
        activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            x = self.activation(x)
            x = nn.LayerNorm()(x)
            if get_activation:
                activations[f"layer_{i}"] = x

        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)

        if get_activation:
            activations["mean"] = mean_x
            activations["logvar"] = logvar_x
            return (mean_x, logvar_x), activations
        return mean_x, logvar_x


def load_prior_checkpoint(
    checkpoint_path: str,
    step: int | None = None,
) -> tuple[dict, dict, dict, DictRunningStatisticsState, dict]:
    """Load encoder, prior, and decoder parameters from an mlp_prior checkpoint.

    Ported from SCAMPER's
    ``scamper.agent.task_transfer.walk_rear.checkpoint_utils.load_prior_checkpoint``
    (commit 4380726): restores the config via ``JsonRestore``, builds an
    abstract policy pytree of the correct structure/dtype via
    ``_create_abstract_prior_policy``, then restores the real params/normalizer
    into that typed template with ``ocp.args.StandardRestore`` (not an untyped
    ``PyTreeRestore``, so orbax's own dtype/structure validation applies).

    Only supports dict normalizer format.

    Args:
        checkpoint_path: Path to the mlp_prior checkpoint directory.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (encoder_params, prior_params, decoder_params,
                  normalizer_params, config_dict).
        normalizer_params is DictRunningStatisticsState.
    """
    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix="PriorNetwork")

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        if step is None:
            step = ckpt_mgr.latest_step()

        cfg = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]

    cfg = OmegaConf.create(cfg)

    obs_sizes = cfg.network_config.get("obs_sizes", None)
    if obs_sizes is None:
        raise ValueError(
            "Checkpoint does not have obs_sizes in config. Only dict normalizer format is supported."
        )

    abstract_policy = _create_abstract_prior_policy(cfg)

    with ocp.CheckpointManager(checkpoint_path, options=mgr_options) as ckpt_mgr:
        policy_params = ckpt_mgr.restore(
            step,
            args=ocp.args.Composite(policy=ocp.args.StandardRestore(abstract_policy)),
        )["policy"]

    normalizer_params, network_params = policy_params

    encoder_params = network_params["params"]["encoder"]
    prior_params = network_params["params"]["prior"]
    decoder_params = network_params["params"]["decoder"]

    return (
        encoder_params,
        prior_params,
        decoder_params,
        normalizer_params,
        OmegaConf.to_container(cfg),
    )


def _create_abstract_prior_policy(cfg: OmegaConf) -> tuple:
    """Create abstract policy structure for checkpoint restoration.

    Args:
        cfg: Configuration from checkpoint (must have obs_sizes).

    Returns:
        Tuple of (normalizer_state, combined_policy_params) with correct structure.
    """
    latent_size = cfg.network_config.intention_size
    action_size = cfg.network_config.action_size
    obs_sizes = dict(cfg.network_config.obs_sizes)

    reference_obs_size = obs_sizes["imitation_target"]
    proprioceptive_obs_size = obs_sizes["proprioception"]

    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    prior_hidden_layer_sizes = tuple(
        cfg.network_config.get("prior_layer_sizes", [1024, 1024])
    )

    combined_params = _init_network_params(
        latent_size,
        action_size,
        reference_obs_size,
        proprioceptive_obs_size,
        encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes,
        prior_hidden_layer_sizes,
    )

    normalizer_state = DictRunningStatisticsState(
        imitation_target=running_statistics.init_state(
            specs.Array(reference_obs_size, jnp.dtype("float32"))
        ),
        proprioception=running_statistics.init_state(
            specs.Array(proprioceptive_obs_size, jnp.dtype("float32"))
        ),
    )

    return (normalizer_state, combined_params)


def _init_network_params(
    latent_size: int,
    action_size: int,
    reference_obs_size: int,
    proprioceptive_obs_size: int,
    encoder_hidden_layer_sizes: tuple,
    decoder_hidden_layer_sizes: tuple,
    prior_hidden_layer_sizes: tuple,
) -> dict:
    """Initialize network parameters for checkpoint structure matching."""
    encoder_module = Encoder(
        layer_sizes=list(encoder_hidden_layer_sizes),
        latents=latent_size,
    )

    action_param_size = action_size * 2
    decoder_module = Decoder(
        layer_sizes=[*decoder_hidden_layer_sizes, action_param_size],
    )

    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    key = jax.random.key(0)
    key_enc, key_dec, key_prior = jax.random.split(key, 3)

    dummy_traj_obs = jnp.zeros((1, reference_obs_size))
    dummy_proprio_obs = jnp.zeros((1, proprioceptive_obs_size))
    dummy_decoder_input = jnp.zeros((1, latent_size + proprioceptive_obs_size))

    encoder_init = encoder_module.init(key_enc, dummy_traj_obs)
    decoder_init = decoder_module.init(key_dec, dummy_decoder_input)
    prior_init = prior_module.init(key_prior, dummy_proprio_obs)

    return {
        "params": {
            "encoder": encoder_init["params"],
            "decoder": decoder_init["params"],
            "prior": prior_init["params"],
        }
    }


def make_decoder_inference_fn(
    decoder_params: dict,
    normalizer_params: DictRunningStatisticsState,
    config: dict,
) -> Callable:
    """Create decoder inference function for use in high-level wrappers.

    The decoder takes ``[latent, proprio]`` concatenated input and outputs an
    action (the mode of a ``NormalTanhDistribution``).

    Args:
        decoder_params: Frozen decoder parameters from the mlp_prior checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics.
        config: Checkpoint config dict (as returned by ``load_prior_checkpoint``).

    Returns:
        Function ``(latent_proprio) -> (action, extras)`` where *latent_proprio*
        is ``[latent, proprio]`` concatenated along the last axis.
    """
    action_size = config["network_config"]["action_size"]
    latent_size = config["network_config"]["intention_size"]
    decoder_hidden_layer_sizes = tuple(config["network_config"]["decoder_layer_sizes"])

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

    def decoder_fn(latent_proprio: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        """Apply decoder to latent + proprio input.

        Args:
            latent_proprio: Concatenated ``[latent, proprio]`` array.

        Returns:
            Tuple of ``(action, extras_dict)``.
        """
        latent = latent_proprio[..., :latent_size]
        proprio = latent_proprio[..., latent_size:]

        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        decoder_input = jnp.concatenate([latent, normalized_proprio], axis=-1)

        logits, _ = decoder_module.apply({"params": decoder_params}, decoder_input)

        action = parametric_action_distribution.mode(logits)

        return action, {"logits": logits}

    return decoder_fn


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

    The prior takes proprioceptive observations and outputs the mean and
    log-variance of the latent distribution.

    Args:
        prior_params: Frozen prior parameters from the mlp_prior checkpoint.
        normalizer_params: Dict normalizer with proprioception statistics.
        config: Checkpoint config dict (as returned by ``load_prior_checkpoint``).

    Returns:
        Function ``(proprio) -> (mean, logvar)``.
    """
    latent_size = config["network_config"]["intention_size"]
    prior_hidden_layer_sizes = tuple(
        config["network_config"].get("prior_layer_sizes", [1024, 1024])
    )

    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    proprio_normalizer = normalizer_params.proprioception

    def prior_fn(proprio: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply prior to proprioceptive observations.

        Args:
            proprio: Proprioceptive observations.

        Returns:
            Tuple of ``(mean, logvar)`` for the latent distribution.
        """
        normalized_proprio = running_statistics.normalize(proprio, proprio_normalizer)

        mean, logvar = prior_module.apply({"params": prior_params}, normalized_proprio)

        return mean, logvar

    return prior_fn
