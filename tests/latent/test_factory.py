def test_make_factory_returns_ppo_networks():
    from brax.training.acme import running_statistics
    from track_mjx.agent.latent_ppo.networks.factory import (
        make_latent_mimic_ppo_networks,
    )
    nets = make_latent_mimic_ppo_networks(
        observation_size={"proprioception": 30, "o_history": 60, "z_target": 16},
        action_size=12,
        preprocess_observations_fn=running_statistics.normalize,
        policy_layer_sizes=(64, 32),
        value_layer_sizes=(64, 32),
    )
    assert hasattr(nets, "policy_network")
    assert hasattr(nets, "value_network")
    assert hasattr(nets, "parametric_action_distribution")


def test_registry_includes_latent_mimic():
    from track_mjx.agent.network_registry import NETWORK_FACTORIES, PPO_MODULES
    assert "latent_mimic" in NETWORK_FACTORIES
    assert "latent_mimic" in PPO_MODULES
