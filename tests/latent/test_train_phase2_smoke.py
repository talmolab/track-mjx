"""Smoke tests for the Phase 2 training entry point.

Trivial import-level + Hydra config-compose checks. Does NOT launch training.
"""


def test_train_phase2_imports():
    from track_mjx.agent.latent_ppo import train_phase2
    assert hasattr(train_phase2, "main")
    assert hasattr(train_phase2, "build_env")


def test_phase2_wandb_logging_imports():
    from track_mjx.agent.latent_ppo import wandb_logging
    assert hasattr(wandb_logging, "latent_mimic_rollout_logging_fn")


def test_phase2_config_loads():
    """Hydra config should compose without error."""
    from hydra import compose, initialize

    with initialize(config_path="../../track_mjx/config", version_base=None):
        cfg = compose(config_name="latent_mimic_phase2")
    # Should pick up env_config.mujoco_impl from v1_rodent_imitation_warp inheritance
    assert cfg.env_config.mujoco_impl == "warp"
    assert cfg.network_config.arch_name == "latent_mimic"
    assert cfg.latent_mimic.prior_dir is not None
