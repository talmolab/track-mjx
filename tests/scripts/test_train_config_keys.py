"""A ``train_config`` key that DMPOConfig does not take is refused, not dropped.

A misspelt switch (``critic_target_mod``) used to vanish in the filter and the
run trained on the default. The filter now raises, naming the key and the
closest field names, except for the few keys the entries read themselves.
"""

import pytest
from omegaconf import OmegaConf

from tests.scripts.test_configs_compose import (
    ARM_NAMES,
    ARMS,
    NOT_A_DMPO_ARM,
    TOP_LEVEL,
    _compose,
    _name,
)
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.train import (
    ENTRY_TRAIN_CONFIG_KEYS,
    INHERITED_PPO_TRAIN_KEYS,
    _filter_dmpo_kwargs,
)


@pytest.fixture(autouse=True)
def dummy_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/dummy/priors")
    monkeypatch.setenv("DMPO_CKPT_ROOT", "/dummy/checkpoints")


def test_unknown_train_config_key_raises():
    with pytest.raises(ValueError) as err:
        _filter_dmpo_kwargs({"n_step": 5, "critic_target_mod": "mean_action"})
    message = str(err.value)
    assert "critic_target_mod" in message
    assert "critic_target_mode" in message  # the close match


def test_every_unknown_key_is_named():
    with pytest.raises(ValueError) as err:
        _filter_dmpo_kwargs({"n_stpe": 5, "sequence_lenght": 9, "n_step": 5})
    message = str(err.value)
    for name in ("n_stpe", "sequence_lenght", "n_step", "sequence_length"):
        assert name in message


def test_known_fields_and_entry_keys_pass_through():
    raw = {"n_step": 5, "iters_per_chunk": 8, "episode_length": 200}
    assert _filter_dmpo_kwargs(raw) == {"n_step": 5}
    assert set(raw) - {"n_step"} <= ENTRY_TRAIN_CONFIG_KEYS


def test_entry_keys_are_not_dmpo_fields():
    assert not ENTRY_TRAIN_CONFIG_KEYS & set(DMPOConfig.__dataclass_fields__)


def test_ignore_argument_extends_the_allowlist():
    assert _filter_dmpo_kwargs(
        {"n_step": 5, "learning_rate": 1e-4}, ignore={"learning_rate"}
    ) == {"n_step": 5}


def _train_section(composed: dict) -> dict:
    """The block an entry builds its DMPOConfig from."""
    if composed.get("train_config") is not None:
        return composed["train_config"]
    return composed["train_setup"]["train_config"]


def _entry_ignore(name: str) -> frozenset:
    return (
        INHERITED_PPO_TRAIN_KEYS if name.startswith("rodent-imitation") else frozenset()
    )


@pytest.mark.parametrize("arm", ARMS, ids=ARM_NAMES)
def test_every_arm_builds_a_dmpo_config(arm):
    if _name(arm) in NOT_A_DMPO_ARM:
        pytest.skip(f"{_name(arm)} is a legacy PPO template")
    composed = OmegaConf.to_container(_compose(_name(arm)), resolve=True)
    DMPOConfig(**_filter_dmpo_kwargs(_train_section(composed)))


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_top_level_config_builds_a_dmpo_config(name):
    composed = OmegaConf.to_container(_compose(name), resolve=True)
    DMPOConfig(
        **_filter_dmpo_kwargs(_train_section(composed), ignore=_entry_ignore(name))
    )


def test_the_imitation_entry_ignores_only_inherited_ppo_keys():
    """A typo in the intention imitation config is still refused."""
    composed = OmegaConf.to_container(
        _compose("rodent-imitation-intention"), resolve=True
    )
    raw = dict(_train_section(composed))
    ignore = _entry_ignore("rodent-imitation-intention")
    raw["critic_target_mod"] = "mean_action"
    with pytest.raises(ValueError, match="critic_target_mod"):
        _filter_dmpo_kwargs(raw, ignore=ignore)
