def test_make_factory_returns_ppo_networks():
    from track_mjx.agent.latent_ppo.networks.factory import (
        make_latent_mimic_ppo_networks,
    )
    # Post-flatten obs schema: {imitation_target, proprioception} like the
    # existing intention factory's contract. The LatentMimic env wrapper packs
    # o_history into proprioception so this matches.
    nets = make_latent_mimic_ppo_networks(
        observation_size={"imitation_target": 16, "proprioception": 90},
        action_size=12,
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
