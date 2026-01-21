"""RNN Discriminator network with attention pooling.

A bidirectional GRU-based discriminator that processes sequential motion clips
and uses attention pooling to aggregate timesteps before classification.
"""

from collections.abc import Callable, Sequence
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., jnp.ndarray]


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence dimension.

    Learns a query vector that attends to all timesteps, producing a weighted
    sum as the pooled representation.

    Attributes:
        hidden_size: Size of the attention hidden layer.
    """

    hidden_size: int
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, return_weights: bool = False
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Apply attention pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            return_weights: If True, also return attention weights.

        Returns:
            If return_weights=False: pooled output of shape (batch, hidden_dim).
            If return_weights=True: tuple of (pooled, attention_weights)
                where attention_weights has shape (batch, seq_len).
        """
        # Learn attention scores through a small MLP
        scores = nn.Dense(
            self.hidden_size, name="attn_hidden", kernel_init=self.kernel_init
        )(x)
        scores = nn.tanh(scores)
        scores = nn.Dense(1, name="attn_score", kernel_init=self.kernel_init)(
            scores
        )  # (batch, seq_len, 1)

        # Softmax over sequence dimension
        weights = nn.softmax(scores, axis=1)  # (batch, seq_len, 1)

        # Weighted sum
        pooled = jnp.sum(x * weights, axis=1)  # (batch, hidden_dim)

        if return_weights:
            return pooled, weights.squeeze(-1)  # weights: (batch, seq_len)
        return pooled


class RNNDiscriminator(nn.Module):
    """Bidirectional GRU discriminator with attention pooling.

    Processes sequential motion clips through a bidirectional GRU, then uses
    attention pooling to aggregate timesteps before classification.

    Attributes:
        rnn_hidden_size: Hidden size for each GRU direction.
        num_layers: Number of stacked GRU layers.
        dropout_rate: Dropout probability (applied between layers and before output).
        bidirectional: Whether to use bidirectional GRU.
        attention_hidden_size: Hidden size for attention mechanism.
        return_attention: If True, return attention weights with logits.
        kernel_init: Weight initializer.
    """

    rnn_hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.1
    bidirectional: bool = True
    attention_hidden_size: int = 64
    return_attention: bool = False
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = True
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass through the RNN discriminator.

        Args:
            x: Input tensor of shape (batch, num_steps, qpos_dim).
            training: Whether in training mode (enables dropout).

        Returns:
            If return_attention=False: logits of shape (batch, 1).
            If return_attention=True: tuple of (logits, attention_weights)
                where attention_weights has shape (batch, num_steps).
        """
        # Process through stacked GRU layers
        for layer_idx in range(self.num_layers):
            # Forward GRU
            forward_cell = nn.GRUCell(
                features=self.rnn_hidden_size,
                kernel_init=self.kernel_init,
                name=f"gru_fwd_{layer_idx}",
            )
            forward_rnn = nn.RNN(forward_cell, return_carry=False)
            forward_out = forward_rnn(x)  # (batch, seq_len, hidden)

            if self.bidirectional:
                # Backward GRU (reverse sequence)
                backward_cell = nn.GRUCell(
                    features=self.rnn_hidden_size,
                    kernel_init=self.kernel_init,
                    name=f"gru_bwd_{layer_idx}",
                )
                backward_rnn = nn.RNN(backward_cell, return_carry=False, reverse=True)
                backward_out = backward_rnn(x)  # (batch, seq_len, hidden)

                # Concatenate forward and backward
                x = jnp.concatenate([forward_out, backward_out], axis=-1)
            else:
                x = forward_out

            # Apply dropout between layers (not after last layer)
            if layer_idx < self.num_layers - 1 and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Attention pooling over timesteps
        attention = AttentionPooling(
            hidden_size=self.attention_hidden_size,
            kernel_init=self.kernel_init,
        )
        pooled_output = attention(x, return_weights=self.return_attention)

        if self.return_attention:
            pooled, attention_weights = pooled_output
        else:
            pooled = pooled_output

        # Dropout before output layer
        if self.dropout_rate > 0:
            pooled = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(
                pooled
            )

        # Output layer: single logit for binary classification
        logits = nn.Dense(1, name="output", kernel_init=self.kernel_init)(pooled)

        if self.return_attention:
            return logits, attention_weights
        return logits


def make_rnn_discriminator_network(
    num_steps: int,
    qpos_dim: int,
    rnn_hidden_size: int = 128,
    num_layers: int = 2,
    dropout_rate: float = 0.1,
    bidirectional: bool = True,
    attention_hidden_size: int = 64,
    return_attention: bool = False,
) -> Tuple[RNNDiscriminator, Callable, Callable]:
    """Create RNN discriminator network with init and apply functions.

    Args:
        num_steps: Number of timesteps in input sequence.
        qpos_dim: Dimension of qpos at each timestep.
        rnn_hidden_size: Hidden size for each GRU direction.
        num_layers: Number of stacked GRU layers.
        dropout_rate: Dropout probability.
        bidirectional: Whether to use bidirectional GRU.
        attention_hidden_size: Hidden size for attention mechanism.
        return_attention: If True, apply_fn returns (logits, attention_weights).

    Returns:
        Tuple of (discriminator_module, init_fn, apply_fn) where:
            - discriminator_module: The Flax module instance
            - init_fn: Function (key) -> params to initialize parameters
            - apply_fn: Function (params, x, training, rngs) -> logits or (logits, attn)
    """
    discriminator = RNNDiscriminator(
        rnn_hidden_size=rnn_hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        attention_hidden_size=attention_hidden_size,
        return_attention=return_attention,
    )

    def init_fn(key: jax.Array) -> dict:
        """Initialize network parameters."""
        dummy_input = jnp.zeros((1, num_steps, qpos_dim))
        return discriminator.init(key, dummy_input, training=False)

    def apply_fn(
        params: dict,
        x: jnp.ndarray,
        training: bool = True,
        rngs: dict | None = None,
    ):
        """Apply network to input.

        Args:
            params: Network parameters.
            x: Input tensor of shape (batch, num_steps, qpos_dim).
            training: Whether in training mode (enables dropout).
            rngs: Optional dict with 'dropout' key for dropout randomness.

        Returns:
            If return_attention=False: logits of shape (batch, 1).
            If return_attention=True: tuple of (logits, attention_weights).
        """
        if rngs is None:
            rngs = {}
        return discriminator.apply(params, x, training=training, rngs=rngs)

    return discriminator, init_fn, apply_fn
