from typing import Callable, Sequence
from flax import nnx
from jax import numpy as jnp


class MLP(nnx.Module):
    """Simple MLP module with activation functions and layer normalization."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_fn: Callable = nnx.relu,
        activate_final: bool = False,
        layer_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize the MLP module.

        The layer size will be the input size of the first layer and the output size of the last layer, therefore, the length of the layer_sizes should be at least 2 and the layer size is len(layer_sizes) - 1.

        Args:
            layer_sizes (List[int]): _description_
            rngs (nnx.Rngs): _description_
            activation_fn (Callable, optional): _description_. Defaults to nnx.relu.
            activate_final (bool, optional): _description_. Defaults to False.
            layer_norm (bool, optional): _description_. Defaults to True.
        """
        self.layers = []
        num_layers = len(layer_sizes) - 1
        assert num_layers >= 1
        self.activation_fn = activation_fn
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        # create the layers with the given sizes and activation functions
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.append(nnx.Linear(in_size, out_size, use_bias=True, rngs=rngs))
            if i < num_layers - 1 or self.activate_final:
                self.layers.append(self.activation_fn)
                if self.layer_norm:
                    self.layers.append(nnx.LayerNorm(out_size, rngs=rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Call the MLP module, stepping through each layer.

        Args:
            x (jnp.ndarray): input tensor

        Returns:
            output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
