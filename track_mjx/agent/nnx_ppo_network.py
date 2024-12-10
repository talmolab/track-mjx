"""
PPO network for imitation learning with intention, in Flax NNX API
"""

from typing import Any, Sequence, Tuple, Callable

from brax import base
from brax.training import networks
from brax.training import types
from brax.training.acme import running_statistics
from brax.training import distribution

# from brax.training.agents.ppo import networks as ppo_networks
from track_mjx.agent import custom_ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from brax import envs
from flax import nnx

from brax.training.types import PRNGKey
import jax
from flax import nnx
import dataclasses
from jax import numpy as jnp

from track_mjx.agent.nnx_intention_network import IntentionNetwork
from track_mjx.agent.nnx_util import MLP


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics


class PPOImitationNetworks(nnx.Module):
    policy_network: IntentionNetwork
    value_network: MLP
    parametric_action_distribution: distribution.ParametricDistribution

    def __init__(
        self,
        policy_network: IntentionNetwork,
        value_network: MLP,
        parametric_action_distribution: distribution.ParametricDistribution,
        *,
        rngs: nnx.Rngs,
    ):
        self.policy_network = policy_network
        self.value_network = value_network
        self.parametric_action_distribution = parametric_action_distribution
        self.rngs = rngs

    def policy(self, observations: types.Observation, key: PRNGKey | None = None) -> Tuple[types.Action, types.Extra]:
        """Policy function that returns actions and extra information."""
        if key is None:
            # if key is unspecified, then it will use nnx.Rngs stream
            # to generate the key, but it won't be a pure function
            key = self.rngs.param()
        key_sample, key_policy = jax.random.split(key)
        logits, _, _ = self.policy_network(observations, key=key_policy)
        # action sampling is happening here, according to distribution parameter logits
        raw_actions = self.parametric_action_distribution.sample_no_postprocessing(logits, key_sample)

        # probability of selection specific action, actions with higher reward should have higher probability
        log_prob = self.parametric_action_distribution.log_prob(logits, raw_actions)

        postprocessed_actions = self.parametric_action_distribution.postprocess(raw_actions)
        return postprocessed_actions, {
            # "latent_mean": latent_mean,
            # "latent_logvar": latent_logvar,
            "log_prob": log_prob,
            "raw_action": raw_actions,
            "logits": logits,
        }

    def value(self, observations: types.Observation) -> jnp.ndarray:
        """Value function that returns values."""
        return self.value_network(observations)

    def split_key(self) -> PRNGKey:
        """Splits the key and returns the first part."""
        return self.rngs.param()


def make_intention_policy(
    param_size: int,
    intention_latent_size: int,
    observation_size: int,
    reference_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    encoder_layers: Sequence[int] = (1024, 1024),
    decoder_layers: Sequence[int] = (1024, 1024),
) -> IntentionNetwork:
    """Creates an intention policy network, translate the policy network in nnx to Flax Linen module using the bridge"""

    policy_module = IntentionNetwork(
        action_size=param_size,
        latents=intention_latent_size,
        egocentric_obs_size=observation_size - reference_obs_size,
        reference_obs_size=reference_obs_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        rngs=nnx.Rngs(0),  # rngs stream
    )

    return policy_module


def make_value_network(
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (256,) * 2,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
):
    """Create a policy network with appropriate parameters. The output of the value network is the value of the state, which is a scalar.

    Args:
        obs_size (int): observation size
        hidden_layer_sizes (Sequence[int], optional): size of the neural network size. Defaults to (256,)*2.
        activation (Callable[[jnp.ndarray], jnp.ndarray], optional): activation function to. Defaults to nnx.relu.

    return:
        MLP: value network
    """
    value_module = MLP([obs_size] + list(hidden_layer_sizes) + [1], activation_fn=activation, rngs=nnx.Rngs(0))
    return value_module


def make_intention_ppo_networks(
    observation_size: int,
    reference_obs_size: int,
    action_size: int,
    preprocess_observations_fn: Callable = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_layers: Sequence[int] = (1024,) * 2,
    decoder_layers: Sequence[int] = (1024,) * 2,
    value_layers: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    # create policy network
    policy_network = make_intention_policy(
        param_size=parametric_action_distribution.param_size,
        intention_latent_size=intention_latent_size,
        observation_size=observation_size,
        reference_obs_size=reference_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
    )
    # create value network
    value_network = make_value_network(
        observation_size,
        hidden_layer_sizes=value_layers,
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        rngs=nnx.Rngs(44),
    )


@dataclasses.dataclass
class PPOTrainConfig:
    """Configuration for PPO training."""

    num_timesteps: int = 1000000
    episode_length: int = 1000
    action_repeat: int = 1
    num_envs: int = 16
    max_devices_per_host: int | None = None
    encoder_layers: Sequence[int] = (256,) * 2
    decoder_layers: Sequence[int] = (256,) * 2
    value_layer_sizes: Sequence[int] = (512,) * 3
    num_eval_envs: int = 128
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    kl_weight: float = 1e-3
    discounting: float = 0.9
    seed: int = 0
    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    num_evals: int = 1
    num_resets_per_eval: int = 0
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    clipping_epsilon: float = 0.3
    gae_lambda: float = 0.95
    deterministic_eval: bool = False
    network_factory: Callable[..., PPOImitationNetworks]  = make_intention_ppo_networks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None
    normalize_advantage: bool = True
    eval_env: envs.Env | None = None
    policy_params_fn: Callable[..., None] = lambda *args: None
    randomization_fn: Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]] | None = None
