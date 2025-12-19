"""Parameter masks for selective network freezing during training.

This module provides utilities for creating boolean masks over network parameters,
enabling selective freezing of network components (e.g., freeze encoder while
training decoder, or vice versa).
"""

import copy
from typing import Any


def create_decoder_mask(params: Any, decoder_name: str = "decoder") -> Any:
    """Create a parameter mask that freezes only the decoder.

    Creates a boolean mask with the same structure as the input parameters,
    where True indicates a frozen parameter and False indicates a trainable
    parameter. This is useful for two-phase training where you first train
    the full network, then freeze the decoder to fine-tune only the encoder.

    Args:
        params: PPO network parameters with .policy["params"] and .value
            attributes. Typically a PPONetworkParams namedtuple.
        decoder_name: Key name of the decoder in policy params. Defaults to
            "decoder".

    Returns:
        A deep copy of params with values replaced by boolean masks:
            - policy.params[decoder_name]: True (frozen)
            - policy.params[other keys]: False (trainable)
            - value[all keys]: False (trainable)

    Example:
        >>> mask = create_decoder_mask(ppo_params)
        >>> # Use with optax.multi_transform to apply different optimizers
        >>> optimizer = optax.multi_transform(
        ...     {"frozen": optax.set_to_zero(), "trainable": optax.adam(1e-4)},
        ...     param_labels=mask
        ... )
    """
    param_mask = copy.deepcopy(params)

    for key in param_mask.policy["params"]:
        param_mask.policy["params"][key] = (key == decoder_name)

    for key in param_mask.value:
        param_mask.value[key] = False

    return param_mask
