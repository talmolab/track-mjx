from typing import Any, Sequence, Tuple, Callable

from brax.training import networks
from brax.training import types
from brax.training.acme import running_statistics
from brax.training import distribution
# from brax.training.agents.ppo import networks as ppo_networks
from track_mjx.agent import custom_ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
from flax import nnx

from brax.training.types import PRNGKey
import jax
from flax import nnx
import dataclasses
from jax import numpy as jnp

from track_mjx.agent.intention_network import IntentionNetwork
from track_mjx.agent.nnx_ppo_network import make_intention_ppo_networks
from track_mjx.agent.nnx_util import MLP

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics


@dataclasses.dataclass
class PPOTrainConfig:
    """Configuration for PPO training."""

    num_timesteps: int = 1000000
    episode_length: int = 1000
    action_repeat: int = 1
    num_envs: int = 1
    max_devices_per_host: int | None = None
    encoder_layers: Sequence[int] = (256,) * 2 
    decoder_layers: Sequence[int] = (256,) * 2
    value_hidden_layer_sizes: Sequence[int] = (512,) * 3
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
    network_factory: Callable = make_intention_ppo_networks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None
    normalize_advantage: bool = True
    eval_env: envs.Env | None = None
    policy_params_fn: Callable[..., None] = lambda *args: None
    randomization_fn: Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]] | None = None
    


@dataclasses.dataclass
class PPOImitationNetworks:
    policy_network: IntentionNetwork
    value_network: MLP
    parametric_action_distribution: distribution.ParametricDistribution


def make_intention_ppo_networks(
    observation_size: int,
    reference_obs_size: int,
    action_size: int,
    preprocess_observations_fn: Callable = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_layers: Sequence[int] = (1024,) * 2,
    decoder_layers: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_network = IntentionNetwork(
        action_size=parametric_action_distribution.param_size,
        latents=intention_latent_size,
        egocentric_obs_size=observation_size - reference_obs_size,
        reference_obs_size=reference_obs_size,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        rngs=nnx.Rngs(0),
    )
    value_network = MLP( # TODO(Scott): make a `make_value_network` function
        layer_sizes=value_hidden_layer_sizes, 
        # TODO(Scott): since we are using nnx, we need to explicitly pass in input and output sizes, which means we need to know the observation size here
        activation_fn=nnx.relu,
        rngs=nnx.Rngs(0),
    )

    return PPOImitationNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

# TODO(scott): is this the right way to do this? Can we use functools.partial instead?
def make_policy_fn(ppo_networks: PPOImitationNetworks):
    def policy(observations: jnp.ndarray, key: PRNGKey, ppo_networks: PPOImitationNetworks = ppo_networks, deterministic: bool = False):
        """Policy function that returns the action given an observation

        Args:
            params (Tuple[nnx.NestedMeanStd, Any]): parameters for the policy
            obs (jnp.ndarray): observation
            key (PRNGKey): random key

        Returns:
            Tuple[jnp.ndarray, Any]: action and policy info
        """
        key_sample, key_network = jax.random.split(key)
        # logits comes from policy directly, raw predictions that decoder generates (action, intention_mean, intention_logvar)
        logits, _, _ = ppo_networks.policy_network(observations, key_network)
        if deterministic:
            return ppo_networks.parametric_action_distribution.mode(logits), {}
        # action sampling is happening here, according to distribution parameter logits
        raw_actions = ppo_networks.parametric_action_distribution.sample_no_postprocessing(logits, key_sample)

        # probability of selection specific action, actions with higher reward should have higher probability
        log_prob = ppo_networks.parametric_action_distribution.log_prob(logits, raw_actions)

        postprocessed_actions = ppo_networks.parametric_action_distribution.postprocess(raw_actions)
        return postprocessed_actions, {
            "log_prob": log_prob,
            "raw_action": raw_actions,
            "logits": logits,
        }
    return policy
