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
    # NOTE: previously 32.0 (vnl-ray distributed default). After fixing the
    # num_updates formula in train.py, 32.0 produced ~12 800 SGD updates per
    # rollout under the standard num_envs=2048 / unroll_length=50 / batch=256
    # config, which is impractical on a single device. Default lowered to 2.0
    # (2 samples drawn per insert = ~800 SGD updates per rollout under the
    # same config). Generating env experience is cheap on the MJX/JAX stack,
    # so the moderate ratio keeps wall-clock manageable while still providing
    # meaningful sample reuse. Override per-job in the YAML config if needed.
    samples_per_insert: float = 2.0

    # Training loop
    num_envs: int = 2048
    num_timesteps: int = 1_000_000_000
    unroll_length: int = 50

    # Eval / logging
    eval_every_steps: int = 1_000_000
    log_every_steps: int = 10_000

    # KL-anchor (loss-side) coefficients. When alpha > 0, the policy loss is
    # augmented with `-alpha * mean(exp(-w * KL(pi_theta || pi_imit)))` where
    # KL is closed-form Gaussian on pre-tanh logits and (mu_imit, log_std_imit)
    # are read from the batch (populated by the kl-anchor wrapper via state.info
    # → trajectory.extras). Defaults of 0 disable the term; non-kl-anchor
    # entries are unaffected.
    kl_anchor_alpha: float = 0.0
    kl_anchor_w: float = 0.5
    # Linear-decay schedule for `w`. When `kl_anchor_decay_sgd_steps > 0`,
    # `w_now` is linearly interpolated from `kl_anchor_w` (at SGD step 0) to
    # `kl_anchor_w_floor` (at step `kl_anchor_decay_sgd_steps`) and clamped at
    # the floor thereafter. Defaults preserve the static behavior: when
    # `decay_sgd_steps == 0`, `w` stays constant at `kl_anchor_w` for the
    # entire run.
    kl_anchor_w_floor: float = 0.0
    kl_anchor_decay_sgd_steps: int = 0
