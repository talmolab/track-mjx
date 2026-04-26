"""Dispatch table mapping arch_name -> (network_factory, ppo_trainer_module).

Adding a new architecture = appending one entry to NETWORK_FACTORIES + PPO_MODULES.
No edit to train.py is required afterwards.
"""
from typing import Callable

from track_mjx.agent.ff_ppo import ppo_networks as ff_networks
from track_mjx.agent.ff_ppo import ppo as ff_ppo_train
from track_mjx.agent.recurrent_ppo import networks as recurrent_networks
from track_mjx.agent.recurrent_ppo import ppo as recurrent_ppo_train


NETWORK_FACTORIES: dict[str, Callable] = {
    "intention": ff_networks.make_intention_ppo_networks,
    "recurrent_intention": recurrent_networks.make_recurrent_intention_ppo_networks,
}

PPO_MODULES: dict[str, object] = {
    "intention": ff_ppo_train,
    "recurrent_intention": recurrent_ppo_train,
}


def get_network_factory(arch_name: str) -> Callable:
    if arch_name not in NETWORK_FACTORIES:
        raise ValueError(
            f"unknown arch_name {arch_name!r}; "
            f"registered: {sorted(NETWORK_FACTORIES.keys())}"
        )
    return NETWORK_FACTORIES[arch_name]


def get_ppo_module(arch_name: str):
    if arch_name not in PPO_MODULES:
        raise ValueError(
            f"unknown arch_name {arch_name!r}; "
            f"registered: {sorted(PPO_MODULES.keys())}"
        )
    return PPO_MODULES[arch_name]


# Register latent_mimic at module load (uses ff_ppo trainer).
# Imported here to avoid a circular import at the top of the file.
from track_mjx.agent.latent_ppo.networks.factory import (  # noqa: E402
    make_latent_mimic_ppo_networks,
)

NETWORK_FACTORIES["latent_mimic"] = make_latent_mimic_ppo_networks
PPO_MODULES["latent_mimic"] = ff_ppo_train
