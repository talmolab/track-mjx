import pytest

from track_mjx.agent.network_registry import (
    NETWORK_FACTORIES,
    PPO_MODULES,
    get_network_factory,
    get_ppo_module,
)


def test_registry_has_existing_archs():
    assert "intention" in NETWORK_FACTORIES
    assert "recurrent_intention" in NETWORK_FACTORIES


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="unknown arch_name"):
        get_network_factory("does_not_exist")


def test_intention_factory_is_ff_make_intention_ppo_networks():
    from track_mjx.agent.ff_ppo import ppo_networks as ff_networks
    assert get_network_factory("intention") is ff_networks.make_intention_ppo_networks


def test_intention_module_is_ff_ppo_train():
    from track_mjx.agent.ff_ppo import ppo as ff_ppo_train
    assert get_ppo_module("intention") is ff_ppo_train
