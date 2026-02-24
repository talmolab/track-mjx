"""Shared types for temporal PPO policies."""

from typing import Literal, Union

import flax
import jax.numpy as jnp

RNNCellType = Literal["simple", "gru", "lstm"]
TemporalBoundaryMode = Literal["fixed", "learned"]
HiddenState = Union[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]


@flax.struct.dataclass
class TemporalPolicyCarry:
    """Recurrent carry for temporal encoder-decoder policies."""

    decoder_hidden: list[HiddenState]
    current_latent: jnp.ndarray
    current_latent_mean: jnp.ndarray
    current_latent_logvar: jnp.ndarray
    segment_step: jnp.ndarray

