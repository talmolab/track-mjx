"""VAE network definition for feature extraction.

An MLP-based Variational Autoencoder that encodes flattened motion clips
into a latent space. The encoder's mu vectors are used as learned features
for computing distribution distances (FID, KID).
"""

from collections.abc import Callable, Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., jnp.ndarray]


class VAEEncoder(nn.Module):
    """MLP encoder that maps input to a latent distribution (mu, logvar).

    Attributes:
        layer_sizes: Sequence of hidden layer dimensions.
        latent_dim: Dimensionality of the latent space.
        dropout_rate: Dropout probability (0.0 to disable).
        use_layer_norm: Whether to apply layer normalization after each hidden layer.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
    """

    layer_sizes: Sequence[int]
    latent_dim: int
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    activation: ActivationFn = nn.silu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

        mu = nn.Dense(self.latent_dim, name="mu", kernel_init=self.kernel_init)(x)
        logvar = nn.Dense(self.latent_dim, name="logvar", kernel_init=self.kernel_init)(
            x
        )
        return mu, logvar


class VAEDecoder(nn.Module):
    """MLP decoder that reconstructs input from latent vectors.

    Attributes:
        layer_sizes: Sequence of hidden layer dimensions.
        output_dim: Dimensionality of the reconstructed output.
        dropout_rate: Dropout probability (0.0 to disable).
        use_layer_norm: Whether to apply layer normalization after each hidden layer.
        activation: Activation function (default: SiLU).
        kernel_init: Weight initializer (default: LeCun uniform).
    """

    layer_sizes: Sequence[int]
    output_dim: int
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    activation: ActivationFn = nn.silu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = z
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

        reconstruction = nn.Dense(
            self.output_dim, name="output", kernel_init=self.kernel_init
        )(x)
        return reconstruction


class VAE(nn.Module):
    """Variational Autoencoder composing encoder and decoder.

    Attributes:
        encoder_layer_sizes: Hidden layer sizes for the encoder.
        decoder_layer_sizes: Hidden layer sizes for the decoder.
        latent_dim: Dimensionality of the latent space.
        input_dim: Dimensionality of the flattened input.
        dropout_rate: Dropout probability.
        use_layer_norm: Whether to use layer normalization.
        activation: Activation function.
        kernel_init: Weight initializer.
    """

    encoder_layer_sizes: Sequence[int]
    decoder_layer_sizes: Sequence[int]
    latent_dim: int
    input_dim: int
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    activation: ActivationFn = nn.silu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    def setup(self):
        self.encoder = VAEEncoder(
            layer_sizes=self.encoder_layer_sizes,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            kernel_init=self.kernel_init,
        )
        self.decoder = VAEDecoder(
            layer_sizes=self.decoder_layer_sizes,
            output_dim=self.input_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
            activation=self.activation,
            kernel_init=self.kernel_init,
        )

    def __call__(
        self, x: jnp.ndarray, rng: jax.Array, training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)

        mu, logvar = self.encoder(x, training=training)

        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, shape=mu.shape)
        z = mu + std * eps

        reconstruction = self.decoder(z, training=training)
        return reconstruction, mu, logvar

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode input and return only mu (deterministic, no sampling)."""
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        mu, _ = self.encoder(x, training=False)
        return mu


def make_vae_network(
    input_size: int,
    latent_dim: int = 64,
    encoder_hidden_layer_sizes: Sequence[int] = (512, 256),
    decoder_hidden_layer_sizes: Sequence[int] | None = None,
    dropout_rate: float = 0.1,
    use_layer_norm: bool = True,
) -> Tuple[VAE, Callable, Callable, Callable]:
    """Create VAE network with init, apply, and encode functions.

    Args:
        input_size: Size of flattened input.
        latent_dim: Dimensionality of latent space.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
            If None, uses reversed encoder layers.
        dropout_rate: Dropout probability (0.0 to disable).
        use_layer_norm: Whether to use layer normalization.

    Returns:
        Tuple of (vae_module, init_fn, apply_fn, encode_fn) where:
            - vae_module: The Flax module instance
            - init_fn: Function (key) -> params to initialize parameters
            - apply_fn: Function (params, x, rng, training, rngs) -> (recon, mu, logvar)
            - encode_fn: Function (params, x) -> mu for feature extraction
    """
    if decoder_hidden_layer_sizes is None:
        decoder_hidden_layer_sizes = tuple(reversed(encoder_hidden_layer_sizes))

    vae = VAE(
        encoder_layer_sizes=list(encoder_hidden_layer_sizes),
        decoder_layer_sizes=list(decoder_hidden_layer_sizes),
        latent_dim=latent_dim,
        input_dim=input_size,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
    )

    def init_fn(key: jax.Array) -> dict:
        """Initialize network parameters."""
        dummy_input = jnp.zeros((1, input_size))
        init_key, rng_key = jax.random.split(key)
        return vae.init(init_key, dummy_input, rng=rng_key, training=False)

    def apply_fn(
        params: dict,
        x: jnp.ndarray,
        rng: jax.Array,
        training: bool = True,
        rngs: dict | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply VAE to input (forward pass with reparameterization)."""
        if rngs is None:
            rngs = {}
        return vae.apply(params, x, rng=rng, training=training, rngs=rngs)

    def encode_fn(params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """Encode input and return mu vectors (deterministic)."""
        return vae.apply(params, x, method=vae.encode)

    return vae, init_fn, apply_fn, encode_fn
