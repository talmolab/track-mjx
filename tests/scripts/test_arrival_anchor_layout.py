"""No shipped DMPO arm asks the learner for a KL anchor its replay cannot hold.

On the feed-forward learner the policy loss is evaluated at the n-step arrival
state, so a loss-side KL anchor is read there too. The anchor is stored per
step for ``t < sequence_length`` only. ``sgd_step`` refuses a batch whose
arrival state lies past that at trace time; this test runs the same predicate
(``learner.arrival_anchor_missing``) on every composed arm, so a bad layout is
caught without building an env.
"""

import dataclasses

import pytest
from omegaconf import OmegaConf

from tests.scripts.test_configs_compose import (
    ARM_NAMES,
    ARMS,
    NOT_A_DMPO_ARM,
    _compose,
    _name,
)
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import arrival_anchor_missing
from track_mjx.agent.dmpo.train import _filter_dmpo_kwargs


@pytest.fixture(autouse=True)
def dummy_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/dummy/priors")
    monkeypatch.setenv("DMPO_CKPT_ROOT", "/dummy/checkpoints")


def _layout(name: str) -> tuple[DMPOConfig, int, bool, bool]:
    """The learner config and replay layout an entry builds for one arm.

    Mirrors the kl-anchor entry: the anchor arrays are stored whenever
    ``kl_anchor.alpha_anchor`` or ``kl_anchor.beta_linear`` is non-zero.
    """
    composed = OmegaConf.to_container(_compose(name), resolve=True)
    cfg = DMPOConfig(**_filter_dmpo_kwargs(composed["train_config"]))
    anchor = composed.get("kl_anchor") or {}
    alpha = float(anchor.get("alpha_anchor", 0.0))
    beta = float(anchor.get("beta_linear", 0.0))
    cfg = dataclasses.replace(cfg, kl_anchor_alpha=alpha, kl_anchor_beta_linear=beta)
    anchor_in_batch = alpha != 0.0 or beta != 0.0
    return cfg, int(cfg.sequence_length), cfg.store_next_observation, anchor_in_batch


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_no_arm_needs_an_anchor_past_its_sequence(arm):
    if _name(arm) in NOT_A_DMPO_ARM:
        pytest.skip(f"{_name(arm)} is a legacy PPO template")
    cfg, seq, has_next, anchor = _layout(_name(arm))
    assert not arrival_anchor_missing(cfg, seq, has_next, anchor), (
        f"{_name(arm)}: n_step={cfg.n_step}, sequence_length={seq}, "
        f"store_next_observation={has_next} -- the policy loss at the arrival "
        "state would read the KL anchor at t = sequence_length. Use "
        "sequence_length = n_step + 1 with store_next_observation: false."
    )


def _anchored_cfg(**overrides) -> DMPOConfig:
    return DMPOConfig(use_n_step=True, n_step=50, kl_anchor_alpha=1e-6, **overrides)


def test_predicate_flags_the_full_horizon_next_observation_layout():
    assert arrival_anchor_missing(_anchored_cfg(), 50, True, True)


@pytest.mark.parametrize(
    "cfg, seq, has_next, anchor",
    [
        (_anchored_cfg(), 51, False, True),  # compressed, n_step + 1
        (_anchored_cfg(), 51, True, True),  # next_observation, one spare step
        (_anchored_cfg(), 50, True, False),  # anchor not stored
        (_anchored_cfg(policy_loss_state="current"), 50, True, True),
        (_anchored_cfg(rnn_bptt_length=4), 54, False, True),  # recurrent
        (DMPOConfig(use_n_step=False, kl_anchor_alpha=1e-6), 2, True, True),
    ],
    ids=[
        "compressed",
        "next_obs_spare_step",
        "no_anchor",
        "current_state",
        "recurrent",
        "one_step",
    ],
)
def test_predicate_accepts_layouts_that_hold_the_anchor(cfg, seq, has_next, anchor):
    assert not arrival_anchor_missing(cfg, seq, has_next, anchor)
