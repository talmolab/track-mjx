"""DMPO configuration. Defaults mirror vnl-ray's train_dmpo_ray.py."""
from dataclasses import dataclass
from typing import Optional, Sequence


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
    # LINEAR KL penalty: `loss += kl_anchor_beta_linear * mean(kl)`.
    #
    # Distinct from `kl_anchor_alpha`, which adds `-alpha*mean(exp(-w*kl))` --
    # a SATURATING brake whose gradient is proportional to exp(-w*kl) and so
    # vanishes exactly when the policy has drifted far. Measured on
    # dmpo_frozen_prior_vel08_sigmaball: anchor_kl_mean = 15.34 with w = 0.5, so
    # exp(-w*kl) = 4.7e-4 -- the anchor had switched itself off. This linear form
    # does not, which is what makes it a usable brake on latent excursion.
    #
    # With prior+decoder frozen, pi_imit is the same network with residual = 0,
    # so kl is a pure function of the latent residual and this term is a direct
    # penalty on latent excursion -- the DMPO analogue of the L1-on-pre-tanh
    # brake brax PPO gets for free from entropy_cost (TanhBijector's
    # forward_log_det_jacobian -> -2|x|).
    kl_anchor_beta_linear: float = 0.0
    # EXPLICIT number of SGD updates per rollout. When None (default), K keeps
    # being derived from `samples_per_insert` by whatever formula the entry point
    # already used, so every previously-run arm stays bit-reproducible.
    #
    # WHY AN OVERRIDE RATHER THAN A FORMULA FIX. Two formulas disagree:
    #   train_highlvl_dmpo_kl_anchor.py:361
    #       K = unroll * num_envs / (batch * samples_per_insert)   <- DIVIDES
    #   train.py:compute_num_updates
    #       K = samples_per_insert * unroll * num_envs / batch     <- MULTIPLIES
    # The live entry point uses the first, which INVERTS the Acme/Reverb meaning
    # of samples_per_insert: raising it *reduces* learner work. Verified against
    # the counters of dmpo_frozen_prior_vel08_sigmaball -- 297,574,400 env steps
    # /(2048*50) = 2906 rollouts, 145,250 updates, 145250/2906 = exactly 50,
    # which is the inverted formula at samples_per_insert=2.0.
    #
    # Swapping the formula would silently change the meaning of `samples_per_insert`
    # in every existing YAML, so instead K becomes directly settable. The realized
    # ratio is logged as `replay/realized_samples_per_insert` either way.
    #
    # Ray/Acme reference for the run that SOLVES this task: 2,572,765 learner
    # steps over 203.53M actor steps at batch 256 = 3.236 samples drawn per actor
    # step. The MJX port's realized value is 50*1024/(50*2048) = 0.5, i.e. 6.5x
    # less reuse. To match at batch B: sgd_steps_per_rollout = 3.236*unroll*num_envs/B.
    sgd_steps_per_rollout: Optional[int] = None

    # Use an n-step return (n = min(n_step, sequence_length)) for the critic
    # target instead of single-step TD. `n_step` has been declared and unread
    # since the port; False keeps the historical single-step behaviour
    # bit-identical, so this is opt-in and does not disturb any existing arm.
    use_n_step: bool = False


def resolve_sgd_steps_per_rollout(cfg: DMPOConfig) -> int:
    """K = SGD updates per rollout. Explicit override wins, else the legacy formula.

    Lives here rather than in train.py because train.py's own
    ``compute_num_updates`` is dead code for every live entry point (it is reached
    only from that module's legacy ``@hydra.main`` and from test_config.py) AND it
    uses the opposite convention. Putting the shared resolver next to the field
    keeps the knob and its only consumer together.

    The ``None`` branch is a VERBATIM copy of the expression the five entry points
    each carried locally -- float division then ``int()`` truncation, not integer
    arithmetic and not ``round()`` -- so every completed arm stays bit-reproducible.
    """
    override = getattr(cfg, "sgd_steps_per_rollout", None)
    if override is not None:
        k = int(override)
        if k < 1:
            raise ValueError(
                f"sgd_steps_per_rollout must be >= 1 when set, got {k}. "
                "Use None (omit the key) to fall back to the samples_per_insert formula."
            )
        return k
    return max(
        1,
        int(cfg.unroll_length * cfg.num_envs / (cfg.batch_size * cfg.samples_per_insert)),
    )


def realized_ratios(cfg: DMPOConfig, K: int) -> dict:
    """Measured learner-throughput ratios, for logging.

    ``realized_samples_per_insert`` is the Acme/Reverb quantity: transitions the
    learner draws per transition the actors insert. The Ray run that solves this
    task realizes 3.236; this port realizes 0.5. It is reported as *measured* and
    not as *configured* because the configured knob is inverted in the live entry
    points -- see ``DMPOConfig.sgd_steps_per_rollout``.

    ``realized_uses_per_insert`` additionally counts every timestep of each sampled
    sequence, which is the honest figure once ``use_n_step`` is on and the learner
    actually consumes more than element ``[:, 0]``.
    """
    inserts = cfg.unroll_length * cfg.num_envs
    return {
        "num_updates_per_rollout": float(K),
        "realized_samples_per_insert": K * cfg.batch_size / inserts,
        "realized_uses_per_insert": K * cfg.batch_size * cfg.sequence_length / inserts,
        "updates_per_actor_step": K / inserts,
    }
