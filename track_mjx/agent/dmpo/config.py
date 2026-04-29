"""DMPO configuration. Defaults mirror vnl-ray's train_dmpo_ray.py."""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class DMPOConfig:
    # MPO loss
    epsilon: float = 0.1
    epsilon_mean: float = 0.0025
    epsilon_stddev: float = 1e-7
    epsilon_penalty: float = 0.1
    init_log_temperature: float = 10.0
    init_log_alpha_mean: float = 10.0
    init_log_alpha_stddev: float = 1000.0
    per_dim_constraining: bool = True
    action_penalization: bool = True
    num_samples: int = 20

    # Distributional critic (C51)
    vmin: float = -150.0
    vmax: float = 150.0
    num_atoms: int = 51

    # RL
    discount: float = 0.97
    n_step: int = 50

    # Networks
    policy_layer_sizes: Sequence[int] = (256, 256, 256)
    critic_layer_sizes: Sequence[int] = (512, 512, 256)

    # Optim
    policy_lr: float = 1e-4
    critic_lr: float = 1e-4
    dual_lr: float = 1e-3
    grad_clip: float = 40.0

    # Targets
    target_policy_update_period: int = 101
    target_critic_update_period: int = 107

    # Replay
    min_replay_size: int = 50_000
    max_replay_size: int = 4_000_000
    batch_size: int = 256
    sequence_length: int = 50
    samples_per_insert: float = 32.0

    # Training loop
    num_envs: int = 2048
    num_timesteps: int = 1_000_000_000
    unroll_length: int = 50

    # Eval / logging
    eval_every_steps: int = 1_000_000
    log_every_steps: int = 10_000
