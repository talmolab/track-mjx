import dataclasses
import math

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.train import compute_num_updates


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
    assert cfg.samples_per_insert == 2.0


def test_compute_num_updates_enforces_acme_samples_per_insert_ratio():
    """num_updates per rollout must produce the desired samples_per_insert
    ratio (Acme convention: samples_drawn / inserts).

    With:
      samples_drawn = num_updates * batch_size
      inserts = unroll_length * num_envs
    we require:
      samples_drawn / inserts ≈ samples_per_insert  (within int-truncation slop).

    The buggy formula produced 1/samples_per_insert instead of samples_per_insert
    — a 1024x error at the default samples_per_insert=32.
    """
    cfg = dataclasses.replace(
        DMPOConfig(),
        unroll_length=50,
        num_envs=2048,
        batch_size=256,
        samples_per_insert=32.0,
    )
    n = compute_num_updates(cfg)
    actual_ratio = (n * cfg.batch_size) / (cfg.unroll_length * cfg.num_envs)
    one_batch_slop = cfg.batch_size / (cfg.unroll_length * cfg.num_envs)
    assert math.isclose(actual_ratio, cfg.samples_per_insert, abs_tol=one_batch_slop), (
        f"samples_per_insert ratio mismatch: "
        f"got {actual_ratio} for n={n}, expected {cfg.samples_per_insert}"
    )


def test_compute_num_updates_lower_bounded_at_one():
    """num_updates must be at least 1 even when the ratio implies less than one
    SGD step per rollout (very small samples_per_insert)."""
    cfg = dataclasses.replace(
        DMPOConfig(),
        unroll_length=1,
        num_envs=1,
        batch_size=10_000,
        samples_per_insert=0.001,
    )
    n = compute_num_updates(cfg)
    assert n >= 1


def test_compute_num_updates_scales_linearly_with_samples_per_insert():
    """Doubling samples_per_insert should double num_updates (modulo int rounding)."""
    base = dataclasses.replace(
        DMPOConfig(),
        unroll_length=50, num_envs=2048, batch_size=256,
        samples_per_insert=4.0,
    )
    doubled = dataclasses.replace(base, samples_per_insert=8.0)
    n_base = compute_num_updates(base)
    n_doubled = compute_num_updates(doubled)
    assert n_doubled == 2 * n_base


def test_dmpo_config_has_kl_anchor_fields_with_zero_defaults():
    """kl_anchor_alpha and kl_anchor_w default to 0/0.5 so non-anchor entries
    see no behavioral change.
    """
    from track_mjx.agent.dmpo.config import DMPOConfig

    cfg = DMPOConfig()
    assert hasattr(cfg, "kl_anchor_alpha")
    assert hasattr(cfg, "kl_anchor_w")
    assert cfg.kl_anchor_alpha == 0.0
    assert cfg.kl_anchor_w == 0.5
    # Plain @dataclass does NOT enforce annotation types at runtime; assert
    # explicitly so a future edit that swaps the default to int(0) is caught
    # before it propagates into JAX (where mixed-dtype scalars cause silent
    # promotions).
    assert isinstance(cfg.kl_anchor_alpha, float)
    assert isinstance(cfg.kl_anchor_w, float)
    # Construct with overrides to confirm types.
    cfg2 = DMPOConfig(kl_anchor_alpha=1.5, kl_anchor_w=0.25)
    assert cfg2.kl_anchor_alpha == 1.5
    assert cfg2.kl_anchor_w == 0.25
    assert isinstance(cfg2.kl_anchor_alpha, float)
    assert isinstance(cfg2.kl_anchor_w, float)

    # Decay-schedule fields: defaults preserve static behavior (no decay).
    assert hasattr(cfg, "kl_anchor_w_floor")
    assert hasattr(cfg, "kl_anchor_decay_sgd_steps")
    assert cfg.kl_anchor_w_floor == 0.0
    assert cfg.kl_anchor_decay_sgd_steps == 0
    assert isinstance(cfg.kl_anchor_w_floor, float)
    assert isinstance(cfg.kl_anchor_decay_sgd_steps, int)
    cfg3 = DMPOConfig(
        kl_anchor_alpha=1.0,
        kl_anchor_w=1.0,
        kl_anchor_w_floor=0.05,
        kl_anchor_decay_sgd_steps=12_000,
    )
    assert cfg3.kl_anchor_w_floor == 0.05
    assert cfg3.kl_anchor_decay_sgd_steps == 12_000
    assert isinstance(cfg3.kl_anchor_w_floor, float)
    assert isinstance(cfg3.kl_anchor_decay_sgd_steps, int)
