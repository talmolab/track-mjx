"""Acme-alignment config fields: defaults, validation, "auto", legacy preset."""

import dataclasses
import pathlib

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from track_mjx.agent.dmpo.config import (
    ALIGNMENT_FIELDS,
    LEGACY_ALIGNMENT,
    DMPOConfig,
    validate_alignment,
)
from track_mjx.agent.dmpo.train import _filter_dmpo_kwargs

CONFIG_DIR = (
    pathlib.Path(__file__).resolve().parents[3] / "track_mjx" / "config" / "dmpo"
)

ALIGNED = {
    "critic_target_mode": "sample_mixture",
    "bootstrap_on_truncation": True,
    "critic_action_input": "executed",
    "action_penalty_formula": "acme",
    "policy_loss_state": "auto",
    "policy_grad_clip_scope": "global",
}


def test_defaults_are_aligned():
    cfg = DMPOConfig()
    assert set(ALIGNMENT_FIELDS) == set(ALIGNED)
    assert len(ALIGNMENT_FIELDS) == 6
    for name, value in ALIGNED.items():
        assert getattr(cfg, name) == value, name


def test_legacy_alignment_covers_exactly_the_alignment_fields():
    assert tuple(LEGACY_ALIGNMENT) == ALIGNMENT_FIELDS
    # Every legacy value differs from the aligned default except that the
    # policy-loss state "auto" is replaced by its concrete legacy value.
    for name in ALIGNMENT_FIELDS:
        assert LEGACY_ALIGNMENT[name] != ALIGNED[name], name
    # The legacy values must themselves validate.
    cfg = DMPOConfig(**LEGACY_ALIGNMENT)
    assert validate_alignment(cfg, recurrent=False) == cfg
    assert validate_alignment(cfg, recurrent=True) == cfg


@pytest.mark.parametrize(
    ("field", "bad", "expected_in_message"),
    [
        pytest.param(f, b, m, id=f"{f}={b}")
        for f, b, m in [
            ("critic_target_mode", "sample_mix", "sample_mixture"),
            ("critic_action_input", "bound", "executed"),
            ("action_penalty_formula", "l2", "acme"),
            ("policy_loss_state", "previous", "current"),
            ("policy_grad_clip_scope", "block", "per_block"),
            ("bootstrap_on_truncation", "yes", "True, False"),
        ]
    ],
)
def test_unknown_value_raises(field, bad, expected_in_message):
    cfg = dataclasses.replace(DMPOConfig(), **{field: bad})
    with pytest.raises(ValueError) as err:
        validate_alignment(cfg, recurrent=False)
    msg = str(err.value)
    assert field in msg
    assert expected_in_message in msg


def test_next_on_recurrent_raises():
    cfg = DMPOConfig(policy_loss_state="next")
    with pytest.raises(ValueError, match="recurrent"):
        validate_alignment(cfg, recurrent=True)
    # The same value is fine on the feed-forward learner.
    assert validate_alignment(cfg, recurrent=False).policy_loss_state == "next"


@pytest.mark.parametrize(
    ("recurrent", "expected"),
    [
        pytest.param(False, "next", id="ff->next"),
        pytest.param(True, "current", id="rnn->current"),
    ],
)
def test_auto_resolves(recurrent, expected):
    cfg = DMPOConfig()
    assert cfg.policy_loss_state == "auto"
    resolved = validate_alignment(cfg, recurrent=recurrent)
    assert resolved.policy_loss_state == expected
    # Returns a copy; the input is untouched, and nothing else changes.
    assert cfg.policy_loss_state == "auto"
    assert dataclasses.replace(resolved, policy_loss_state="auto") == cfg


def test_explicit_current_is_kept_on_both_learners():
    cfg = DMPOConfig(policy_loss_state="current")
    for recurrent in (False, True):
        assert validate_alignment(cfg, recurrent).policy_loss_state == "current"


def test_legacy_yaml_sets_all_six(tmp_path):
    """Compose an arm that includes the preset after /base and build the config."""
    arm = tmp_path / "arm_legacy_probe.yaml"
    arm.write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - /base\n"
        "  - /legacy_pre_alignment\n"
        "  - _self_\n"
    )
    assert (
        (CONFIG_DIR / "legacy_pre_alignment.yaml")
        .read_text()
        .startswith("# @package _global_\n")
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        composed = compose(
            config_name="arm_legacy_probe",
            overrides=[f"hydra.searchpath=[file://{CONFIG_DIR}]"],
        )
    GlobalHydra.instance().clear()
    train_cfg = dict(composed.train_config)
    for name, value in LEGACY_ALIGNMENT.items():
        assert train_cfg[name] == value, name
    cfg = DMPOConfig(**_filter_dmpo_kwargs(train_cfg))
    for name, value in LEGACY_ALIGNMENT.items():
        assert getattr(cfg, name) == value, name


def test_both_sgd_paths_and_fused_step_validate():
    """Every learner entry rejects a bad switch before touching state or nets."""
    from track_mjx.agent.dmpo.learner import _sgd_step_rnn, sgd_step
    from track_mjx.agent.dmpo.networks import DMPONetworks
    from track_mjx.agent.dmpo.train_dmpo_step import make_fused_train_step

    bad_ff = DMPOConfig(critic_target_mode="sample_mix")
    with pytest.raises(ValueError, match="sample_mixture"):
        sgd_step(None, {}, None, None, bad_ff)

    rnn_next = DMPOConfig(policy_loss_state="next", rnn_bptt_length=2)
    with pytest.raises(ValueError, match="recurrent"):
        sgd_step(None, {}, None, None, rnn_next)
    with pytest.raises(ValueError, match="recurrent"):
        _sgd_step_rnn(None, {}, None, None, rnn_next)

    recurrent_nets = DMPONetworks(policy=None, critic=None, recurrent_meta=object())
    with pytest.raises(ValueError, match="recurrent"):
        make_fused_train_step(
            None, recurrent_nets, None, None, DMPOConfig(policy_loss_state="next"), 1
        )
    with pytest.raises(ValueError, match="sample_mixture"):
        make_fused_train_step(None, None, None, None, bad_ff, 1)
