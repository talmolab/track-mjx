"""`sgd_steps_per_rollout` must pin K exactly and must not disturb existing arms.

Background. Two formulas for K (SGD updates per rollout) coexist in the tree and
disagree:

    train_highlvl_dmpo_kl_anchor.py   K = unroll*num_envs / (batch*spi)   DIVIDES
    train.py:compute_num_updates      K = spi*unroll*num_envs / batch     MULTIPLIES

The live entry point uses the first, which inverts the Acme/Reverb meaning of
`samples_per_insert`: raising the knob *reduces* learner work. Rather than swap
the formula -- which would silently redefine `samples_per_insert` in every YAML
already run -- K became directly settable.

These tests lock in the two properties that make that safe:
  1. Unset (None) reproduces the legacy inverted formula exactly.
  2. Set pins K to that literal value, independent of samples_per_insert.

They also pin the arithmetic behind the Ray-parity target so a future edit to the
reasoning has to update a failing test rather than a comment.
"""

import pytest

from track_mjx.agent.dmpo.config import (
    DMPOConfig,
    realized_ratios,
    resolve_sgd_steps_per_rollout as resolve_K,
)
from track_mjx.agent.dmpo.train import compute_num_updates


def realized_spi(cfg, K):
    return realized_ratios(cfg, K)["realized_samples_per_insert"]


PROD = dict(num_envs=2048, unroll_length=50, batch_size=1024, samples_per_insert=2.0)


def test_default_reproduces_the_live_baseline():
    """Unset -> K=50, which is what dmpo_frozen_prior_vel08_sigmaball actually ran.

    Cross-check against its counters: 297,574,400 env steps / (2048*50) = 2906
    rollouts; wandb reports 145,250 learner updates; 145250/2906 = 50.
    """
    cfg = DMPOConfig(**PROD)
    assert cfg.sgd_steps_per_rollout is None
    K = resolve_K(cfg)
    assert K == 50
    # 2906 rollouts, of which EXACTLY ONE contributes no updates: flashbax's
    # min_length_time_axis is max(sequence_length+1, min_replay_size//num_envs)
    # = max(51, 24) = 51, so after the first 50-step add the buffer cannot be
    # sampled and train_dmpo_step.py gates the whole state pytree (including
    # `steps`) through a lax.select. After the second add the time index is 100
    # and the gate opens. Hence (2906 - 1) * 50 = 145,250 EXACTLY -- an exact fit,
    # not a rounded one. The MULTIPLY convention would predict 581,000.
    rollouts = 297_574_400 // (cfg.unroll_length * cfg.num_envs)
    assert rollouts == 2906
    assert (rollouts - 1) * K == 145_250
    assert (rollouts - 1) * compute_num_updates(cfg) == 581_000


def test_the_two_formulas_really_do_disagree():
    """Guards the premise. If someone unifies them, this test should fail loudly."""
    cfg = DMPOConfig(**PROD)
    assert resolve_K(cfg) == 50
    assert compute_num_updates(cfg) == 200
    assert resolve_K(cfg) != compute_num_updates(cfg)


def test_knob_is_inverted_in_the_legacy_path():
    """Raising samples_per_insert REDUCES learner work -- the defect being routed around."""
    lo = resolve_K(DMPOConfig(**{**PROD, "samples_per_insert": 1.0}))
    hi = resolve_K(DMPOConfig(**{**PROD, "samples_per_insert": 4.0}))
    assert lo > hi, "legacy formula should be inverted; if this fails it was silently fixed"


@pytest.mark.parametrize("k", [1, 50, 324, 1294])
def test_explicit_override_pins_K(k):
    """Set -> K is exactly k, whatever samples_per_insert says."""
    for spi in (0.5, 2.0, 32.0):
        cfg = DMPOConfig(**{**PROD, "samples_per_insert": spi, "sgd_steps_per_rollout": k})
        assert resolve_K(cfg) == k


def test_ray_parity_target_arithmetic():
    """The Ray run that solves the task draws 3.236 samples per actor step.

    Ray: 2,572,765 learner steps * batch 256 / 203,530,000 actor steps = 3.236.
    Pin the K needed to match that at both candidate batch sizes.
    """
    ray_spi = 2_572_765 * 256 / 203_530_000
    assert ray_spi == pytest.approx(3.236, abs=5e-3)

    # current port realizes 0.5 -- 6.5x less reuse
    cfg = DMPOConfig(**PROD)
    assert realized_spi(cfg, resolve_K(cfg)) == pytest.approx(0.5)
    assert ray_spi / 0.5 == pytest.approx(6.47, abs=0.02)

    for batch, expect in ((1024, 324), (256, 1294)):
        cfg = DMPOConfig(**{**PROD, "batch_size": batch})
        k = round(ray_spi * cfg.unroll_length * cfg.num_envs / batch)
        assert k == expect
        cfg = DMPOConfig(**{**PROD, "batch_size": batch, "sgd_steps_per_rollout": k})
        # K is an integer, so the realized ratio is quantised to one K-step,
        # batch/(unroll*num_envs). Match to that granularity, not tighter.
        quantum = batch / (cfg.unroll_length * cfg.num_envs)
        assert realized_spi(cfg, resolve_K(cfg)) == pytest.approx(ray_spi, abs=quantum)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_override_raises(bad):
    """0 must NOT silently mean 'never train'.

    0 is falsy, so a `if override:` guard would quietly fall back to the formula
    and train normally -- hiding a config typo that the user believes disabled
    SGD. Raise instead; None (omit the key) is the documented way to fall back.
    """
    cfg = DMPOConfig(**{**PROD, "sgd_steps_per_rollout": bad})
    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_K(cfg)


def test_realized_ratios_are_self_consistent():
    cfg = DMPOConfig(**PROD)
    K = resolve_K(cfg)
    r = realized_ratios(cfg, K)
    assert r["num_updates_per_rollout"] == K
    assert r["realized_samples_per_insert"] == pytest.approx(0.5)
    assert r["updates_per_actor_step"] == pytest.approx(K / (50 * 2048))
    # with sequence_length=50 stored per item, the learner now touches 50x more
    # timesteps per draw once use_n_step is on
    assert r["realized_uses_per_insert"] == pytest.approx(
        r["realized_samples_per_insert"] * cfg.sequence_length
    )
