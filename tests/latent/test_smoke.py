def test_subpackage_imports():
    import track_mjx.agent.latent_ppo  # noqa: F401
    import track_mjx.agent.latent_ppo.networks  # noqa: F401
    import track_mjx.agent.latent_ppo.data  # noqa: F401
    import track_mjx.agent.latent_ppo.losses  # noqa: F401
