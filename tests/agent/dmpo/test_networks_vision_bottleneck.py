import jax
import jax.numpy as jnp
import pytest

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.networks_vision_bottleneck import (
    make_dmpo_bottleneck_vision_networks,
)


@pytest.fixture
def cfg():
    return DMPOConfig(
        num_envs=4,
        unroll_length=4,
        batch_size=4,
        sequence_length=4,
        policy_layer_sizes=(64, 64),  # unused by bottleneck factory
        critic_layer_sizes=(64, 64),  # used as value head
    )


def test_policy_outputs_action_distribution_with_correct_shape(cfg):
    nets = make_dmpo_bottleneck_vision_networks(
        task_obs_size=12,
        action_size=18,
        vision_shape=(16, 16, 2),
        cfg=cfg,
        vision_latent_size=8,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        decoder_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32, 32),
        fusion_hidden_layer_sizes=(32,),
        mono_channels=1,
        shared_weights=True,
    )
    obs = {
        "vision": jnp.zeros((1, 16, 16, 2)),
        "imitation_target": jnp.zeros((1, 12)),
    }
    params = nets.policy.init(jax.random.PRNGKey(0), obs)
    dist = nets.policy.apply(params, obs)
    sample = dist.sample(seed=jax.random.PRNGKey(1))
    assert sample.shape == (1, 18)


def test_critic_outputs_categorical_distribution_with_correct_atoms(cfg):
    nets = make_dmpo_bottleneck_vision_networks(
        task_obs_size=12,
        action_size=18,
        vision_shape=(16, 16, 2),
        cfg=cfg,
        vision_latent_size=8,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        decoder_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32, 32),
        fusion_hidden_layer_sizes=(32,),
        mono_channels=1,
        shared_weights=True,
    )
    obs = {
        "vision": jnp.zeros((1, 16, 16, 2)),
        "imitation_target": jnp.zeros((1, 12)),
    }
    action = jnp.zeros((1, 18))
    params = nets.critic.init(jax.random.PRNGKey(0), obs, action)
    dist = nets.critic.apply(params, obs, action)
    # Categorical critic returns a tfp Distribution over num_atoms.
    assert dist.logits.shape == (1, cfg.num_atoms)


def test_policy_bottleneck_actually_squeezes(cfg):
    """The fusion stage must produce a `vision_latent_size`-D activation."""
    nets = make_dmpo_bottleneck_vision_networks(
        task_obs_size=12,
        action_size=18,
        vision_shape=(16, 16, 2),
        cfg=cfg,
        vision_latent_size=8,
        cnn_feature_size=8,
        cnn_channels=(2, 4, 8),
        decoder_hidden_layer_sizes=(32, 32),
        value_hidden_layer_sizes=(32, 32),
        fusion_hidden_layer_sizes=(32,),
        mono_channels=1,
        shared_weights=True,
    )
    obs = {
        "vision": jnp.zeros((1, 16, 16, 2)),
        "imitation_target": jnp.zeros((1, 12)),
    }
    params = nets.policy.init(jax.random.PRNGKey(0), obs)
    flat = jax.tree_util.tree_leaves(params)
    bottleneck_kernels = [k for k in flat if k.ndim == 2 and k.shape[-1] == 8]
    assert len(bottleneck_kernels) >= 1, (
        f"No Dense layer found projecting to vision_latent_size=8 in policy params. "
        f"Found shapes: {[k.shape for k in flat if k.ndim == 2]}"
    )
