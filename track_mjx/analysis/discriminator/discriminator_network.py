"""Discriminator network definition.

A simple MLP that takes flattened motion clips as input and outputs
a binary classification logit (real=1, fake=0).
"""

from collections.abc import Callable, Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., jnp.ndarray]


class Discriminator(nn.Module):
    """MLP discriminator for motion clip classification.

    Takes flattened motion clip as input and outputs a single logit
    for binary classification.

    Attributes:
        layer_sizes: Sequence of hidden layer dimensions.
        dropout_rate: Dropout probability (0.0 to disable).
        use_layer_norm: Whether to apply layer normalization after each hidden layer.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
    """

    layer_sizes: Sequence[int]
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    activation: ActivationFn = nn.silu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass through the discriminator.

        Args:
            x: Input tensor of shape (batch, num_steps, qpos_dim) or
               (batch, num_steps * qpos_dim). Will be flattened if 3D.
            training: Whether in training mode (enables dropout).

        Returns:
            Logits of shape (batch, 1) for binary classification.
        """
        # Flatten input if needed: (batch, num_steps, qpos_dim) -> (batch, num_steps * qpos_dim)
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)

        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
            )(x)
            x = self.activation(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer: single logit for binary classification
        logits = nn.Dense(1, name="output", kernel_init=self.kernel_init)(x)
        return logits


def make_discriminator_network(
    input_size: int,
    hidden_layer_sizes: Sequence[int],
    dropout_rate: float = 0.1,
    use_layer_norm: bool = True,
) -> Tuple[Discriminator, Callable, Callable]:
    """Create discriminator network with init and apply functions.

    Args:
        input_size: Size of flattened input (num_steps * qpos_dim).
        hidden_layer_sizes: Sequence of hidden layer dimensions.
        dropout_rate: Dropout probability (0.0 to disable).
        use_layer_norm: Whether to use layer normalization.

    Returns:
        Tuple of (discriminator_module, init_fn, apply_fn) where:
            - discriminator_module: The Flax module instance
            - init_fn: Function (key) -> params to initialize parameters
            - apply_fn: Function (params, x, training, rngs) -> logits
    """
    discriminator = Discriminator(
        layer_sizes=list(hidden_layer_sizes),
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
    )

    def init_fn(key: jax.Array) -> dict:
        """Initialize network parameters."""
        dummy_input = jnp.zeros((1, input_size))
        return discriminator.init(key, dummy_input, training=False)

    def apply_fn(
        params: dict,
        x: jnp.ndarray,
        training: bool = True,
        rngs: dict | None = None,
    ) -> jnp.ndarray:
        """Apply network to input."""
        if rngs is None:
            rngs = {}
        return discriminator.apply(params, x, training=training, rngs=rngs)

    return discriminator, init_fn, apply_fn
