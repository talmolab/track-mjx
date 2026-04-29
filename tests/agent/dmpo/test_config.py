from track_mjx.agent.dmpo.config import DMPOConfig


def test_config_defaults_match_vnl_ray():
    cfg = DMPOConfig()
    # vnl-ray defaults from train_dmpo_ray.py:238-260
    assert cfg.epsilon == 0.1
    assert cfg.epsilon_mean == 0.0025
    assert cfg.epsilon_stddev == 1e-7
    assert cfg.epsilon_penalty == 0.1
    assert cfg.num_samples == 20
    assert cfg.target_policy_update_period == 101
    assert cfg.target_critic_update_period == 107
    assert cfg.per_dim_constraining is True
    assert cfg.action_penalization is True
    assert cfg.vmin == -150.0
    assert cfg.vmax == 150.0
    assert cfg.num_atoms == 51
    assert cfg.discount == 0.97
    assert cfg.n_step == 50
