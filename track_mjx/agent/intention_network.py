import functools
import dataclasses
from functools import partial   # pylint: disable=g-importing-member
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP
import brax.training.agents.ppo.networks as ppo_networks
from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import lax, random

import flax
from flax import linen as nn
from flax.linen import initializers

from flax.linen.activation import relu
from flax.linen.activation import sigmoid
from flax.linen.activation import tanh
from flax.linen.activation import silu
from jax.nn import softplus

from flax.linen.linear import default_kernel_init
from flax.linen.linear import Dense


Array = Any
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?

class PosteriorSDECell(nn.Module):
  """A gated neural SDE cell that models the posterior process. The drift function is 
    parametrized by a gated feedforward neural network (FNN). 

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int] # is possible to default to []?
  alpha: float = 1.0 # float32?
  noise_level: float = 1.0
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, rnn_inputs, external_inputs, noise):
    """The gnSDE cell for the posterior process.
    Args:
      carry: the hidden state of the PosteriorSDE cell,
        initialized using `NODECell.initialize_carry`.
      rnn_inputs/external_inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      noise: an ndarray with the standard normal noise for the current time step.
    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    
    z = self.gate_fn(
      dense_h(name='hz')(h) + \
      dense_i(name='izr')(rnn_inputs) + \
      dense_i(name='izc')(external_inputs)
    )
    for i, feat in enumerate(self.features):
      if i == 0:
        h = dense_h(features=feat, name=f'layer_{i}_h')(h) + \
            dense_i(features=feat, name=f'layer_{i}_ir')(rnn_inputs) + \
            dense_i(features=feat, name=f'layer_{i}_ic')(external_inputs)
        h = silu(h)
      else:
        h = dense_h(features=feat, name=f'layer_{i}')(h)
        h = silu(h)
    if self.features:
      h = dense_h(features=hidden_features, name='layer_fin')(h)
    else:
      h = dense_h(features=hidden_features, name='layer_0_h')(h)
    h = tanh(h) # final nonlinearity
    logvar = self.param('logvar', lambda rng, shape: jnp.zeros(shape), (hidden_features,))
    std = self.noise_level * sigmoid(logvar) * jnp.ones_like(noise)
    mu = (1. - self.alpha * z) * carry + self.alpha * z * h
    new_h = mu + jnp.sqrt(self.alpha) * std * noise 
    mu_phi = z * (-carry + h)
    return new_h, ( new_h, mu_phi, mu, std )

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the PosteriorSDE cell carry.
    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given GNSDE cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)

class Flow(nn.Module):
  """The drift function of the prior process used in the training procedure.
    It is parametrized by a gated feedforward neural network (FNN). 

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int]
  alpha: float = 1.0
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, h, x):
    new_h = h
    hidden_features = h.shape[-1]
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    
    z = self.gate_fn(
      dense_h(name='hz')(new_h) + \
      dense_i(name='izx')(x)
    )
    for i, feat in enumerate(self.features):
      if i == 0:
        new_h = dense_h(features=feat, name=f'layer_{i}_h')(new_h) + \
                dense_i(features=feat, name=f'layer_{i}_i')(x)
        new_h = silu(new_h)
      else:
        new_h = dense_h(features=feat, name=f'layer_{i}')(new_h)
        new_h = silu(new_h)
    new_h = dense_h(features=hidden_features, name='layer_fin')(new_h)
    new_h = tanh(new_h) # final nonlinearity
    mu_theta = z * (-h + new_h)
    return mu_theta

class GNSDECell(nn.Module):
  """A gated neural SDE cell that models the prior process. This module is
  only used in the generative mode of FINDR which samples from prior.

  Attributes:
    features: the number of hidden units in the hidden layers of the FNN.
    alpha: the effective time step used for Euler-Maruyama integration of the SDE.
    gate_fn: activation function used for gates (default: sigmoid).
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: normalization proposed in Kim, Can, Krishnamurthy).
    bias_init: initializer for the bias parameters.
    dtype: the dtype of the computation (default: None).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  features: Sequence[int] # is possible to default to []?
  alpha: float = 1.0 # float32?
  gate_fn: Callable[..., Any] = sigmoid
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      default_kernel_init)
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.variance_scaling(scale=1.414, mode="fan_in", distribution="truncated_normal"))
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.normal(1e-3)
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, inputs, siglogvar_noise):
    """The gnSDE cell for the prior process.

    Args:
      carry: the hidden state of the GNSDE cell,
        initialized using `NSDECell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.
      siglogvar_noise: an ndarray with the noise for the current time step.

    Returns:
      A tuple with the new carry and the output.
    """
    h = carry
    hidden_features = h.shape[-1]
    # input and recurrent layers are summed so only one needs a bias.
    dense_h = partial(Dense,
                      features=hidden_features,
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.recurrent_kernel_init,
                      bias_init=self.bias_init)
    dense_i = partial(Dense,
                      features=hidden_features,
                      use_bias=False,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)
    
    z = self.gate_fn(
      dense_h(name='hz')(h) + \
      dense_i(name='izx')(inputs)
    )
    for i, feat in enumerate(self.features):
      if i == 0:
        h = dense_h(features=feat, name=f'layer_{i}_h')(h) + \
            dense_i(features=feat, name=f'layer_{i}_i')(inputs)
        h = silu(h)
      else:
        h = dense_h(features=feat, name=f'layer_{i}')(h)
        h = silu(h)
    h = dense_h(features=hidden_features, name='layer_fin')(h)
    h = tanh(h) # final nonlinearity
    mu = (1. - self.alpha * z) * carry + self.alpha * z * h
    new_h = mu + jnp.sqrt(self.alpha) * siglogvar_noise
    mu_theta = z * (-carry + h)
    return new_h, (new_h, mu_theta, mu)

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=initializers.zeros_init()):
    """Initialize the GNSDE cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given GNSDE cell.
    """
    mem_shape = batch_dims + (size,)
    return init_fn(rng, mem_shape)

class PosteriorSDE(nn.Module):
  """The gated neural SDE (gnSDE) model for the posterior process."""
  features: Sequence[int]
  alpha: float = 1.0
  noise_level: float = 1.0

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x, external_inputs, noise):
    gnsde_state, (y, mu_phi, mu, std) = PosteriorSDECell(self.features, self.alpha, self.noise_level)(
      carry, 
      x, 
      external_inputs,
      noise
    )
    return gnsde_state, (y, mu_phi, mu, std)
  
  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.zeros_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return PosteriorSDECell.initialize_carry(rng, (batch_size,), hidden_size, init_fn)

class GNSDE(nn.Module):
  """The gated neural SDE (gnSDE) model for the prior process."""
  features: Sequence[int]
  alpha: float = 1.0

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x, noise):
    gnsde_state, (y, mu_theta, mu) = GNSDECell(self.features, self.alpha)(
      carry,
      x,
      noise
    )
    return gnsde_state, (y, mu_theta, mu)
  
  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.zeros_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return GNSDECell.initialize_carry(rng, (batch_size,), hidden_size, init_fn)

class InitialState(nn.Module):
  num_latents: int

  @nn.compact
  def __call__(self):
    z0 = self.param(
      'init_state', 
      lambda rng, shape: jnp.zeros(shape), 
      (self.num_latents,)
    )
    return z0

class SimpleGRU(nn.Module):
  """A simple unidirectional GRU."""

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    new_carry, z = MyGRUCell()(carry, x)
    return new_carry, z

  @staticmethod
  def initialize_carry(
    rng: PRNGKey, 
    batch_size: int, 
    hidden_size: int, 
    init_fn=initializers.zeros_init()
  ):
    # Use a dummy key since the default state init fn is just zeros.
    return nn.GRUCell.initialize_carry(
        rng, (batch_size,), hidden_size, init_fn)

class SimpleBiGRU(nn.Module):
  """A simple bi-directional GRU."""
  hidden_size: int

  def setup(self):
    self.forward_gru = SimpleGRU()
    self.backward_gru = SimpleGRU()

  def __call__(self, data, rng):
    key_1, key_2 = random.split(rng, 2)
    batch_size = data.shape[0]

    # Forward GRU.
    initial_state = SimpleGRU.initialize_carry(
      key_1, 
      batch_size, 
      self.hidden_size
    )
    _, forward_outputs = self.forward_gru(
      initial_state, 
      data
    )

    # Backward GRU.
    reversed_data = jnp.flip(
      data, 
      axis=0
    )

    initial_state = SimpleGRU.initialize_carry(
      key_2, 
      batch_size, 
      self.hidden_size
    )
    _, backward_outputs = self.backward_gru(
      initial_state, 
      reversed_data
    )
    backward_outputs = jnp.flip(
      backward_outputs, 
      axis=0
    )

    # Concatenate the forward and backward representations.
    outputs = jnp.concatenate(
      [forward_outputs, backward_outputs], -1
    )
    return outputs

class Encoder(nn.Module):
    """outputs in the form of distributions in latent space"""

    layer_sizes: Sequence[int]
    latents: int  # intention size
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, get_activation: bool = False
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[Tuple[jnp.ndarray, jnp.ndarray], dict]]:
        activations = {}
        # For each layer in the sequence
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


class Decoder(nn.Module):
    """decode with action output"""

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, get_activation: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, dict]]:
        activations = {}
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
                x = nn.LayerNorm()(x)
                if get_activation:
                    activations[f"layer_{i}"] = x
        if get_activation:
            return x, activations
        return x

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

class IntentionNetwork(nn.Module): # is this taking in 5 reference steps??
    """Full VAE model, encode -> decode with sampled actions"""

    encoder_layers: Sequence[int]
    decoder_layers: Sequence[int]
    reference_obs_size: int
    latents: int = 60

    def setup(self):
        self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
        self.decoder = Decoder(layer_sizes=self.decoder_layers)

    def __call__(self, obs, key, get_activation: bool = False):
        _, encoder_rng = jax.random.split(key)
        traj = obs[..., : self.reference_obs_size] # obs
        # self.reference_obs_size = 470
        # obs.shape = (1, 617) then after that it is (128, 617) then (8192, 617), and now (20, 4096, 617) and then (617,)
        if get_activation:
            (latent_mean, latent_logvar), encoder_activations = self.encoder(traj, get_activation=True)
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            concatenated = jnp.concatenate([z, obs[..., self.reference_obs_size :]], axis=-1) #z
            action, decoder_activations = self.decoder(concatenated, get_activation=True)
            return action, latent_mean, latent_logvar, {"encoder": encoder_activations, "decoder": decoder_activations, "intention": z}
        else:
            latent_mean, latent_logvar = self.encoder(traj, get_activation=False) # latent_mean.shape = (1, 60) then (128, 60)
            z = reparameterize(encoder_rng, latent_mean, latent_logvar)
            action = self.decoder(
                jnp.concatenate([z, obs[..., self.reference_obs_size :]], axis=-1) #z
            )
            return action, latent_mean, latent_logvar

class RecurrentIntentionNetwork(nn.Module):
    """Full VAE model, encode -> decode with sampled actions"""

    prior_layers: Sequence[int]
    posterior_layers: Sequence[int]
    inference_network_size: int
    reference_obs_size: int
    latents: int = 60
    alpha: float = 1.0

    def setup(self):
        self.inference_network = SimpleBiGRU(
          hidden_size=self.inference_network_size
        )

        self.posterior_process = PosteriorSDE(
          features=self.posterior_layers,
          alpha=self.alpha
        )

        self.prior_process = Flow(
          features=self.prior_layers,
          alpha=self.alpha
        )

        self.gru_initial_state = InitialState(
           num_latents=self.inference_network_size
        )

        self.decoder = nn.Dense(
            self.num_neurons,
            use_bias=True
        )

    def __call__(self, obs, key, get_activation: bool = False):
        key_1, key_2, key_3 = jax.random.split(key, 3)

        hs = self.inference_network(obs, key_1)
        
        carry_dl = self.posterior_process.initialize_carry(
            key_2,
            obs.shape[0], 
            self.latents
        )
        
        noise_posterior = random.normal(
            key_3, 
            hs.shape[:-1] + (self.latents,)
        )

        _, (z, mu_phi, mu, std) = self.posterior_process(
        carry_dl, 
        hs,
        noise_posterior
        )
        mu_theta = self.prior_process(z)
        
        action = self.decoder(z)

        return action, mu, std

def make_intention_policy(
    param_size: int,
    latent_size: int,
    total_obs_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_hidden_layer_sizes: Sequence[int] = (1024, 1024),
) -> networks.FeedForwardNetwork:
    """Creates an intention policy network."""

    policy_module = IntentionNetwork(
        encoder_layers=list(encoder_hidden_layer_sizes),
        decoder_layers=list(decoder_hidden_layer_sizes) + [param_size], # action size
        reference_obs_size=reference_obs_size,
        latents=latent_size,
    )

    def apply(processor_params, policy_params, obs, key, get_activation: bool = False):
        """Applies the policy network with observation normalizer."""
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs=obs, key=key, get_activation=get_activation)

    dummy_total_obs = jnp.zeros((1, total_obs_size))
    dummy_key = jax.random.PRNGKey(0)

    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_total_obs, dummy_key),
        apply=apply,
    )
