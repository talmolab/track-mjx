"""DMPO configuration. Defaults mirror vnl-ray's train_dmpo_ray.py."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass


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
    #   the original (pre-port) DMPO training entry point
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
    sgd_steps_per_rollout: int | None = None

    # Floor for the MPO temperature dual, in log space. The Acme/vnl-ray value
    # is -18 and is the default, so every completed arm is unchanged.
    #
    # WHY IT IS EXPOSED. Under a SPARSE reward Q can go exactly flat across the
    # N=20 sampled actions, and the temperature dual then has derivative
    # `epsilon` for EVERY temperature (see clip_mpo_params in losses.py), so it
    # decays without bound and pins at the floor. Measured on
    # a sparse gap-crossing run: log_temperature -4.06 @10.1M -> -9.31 @20.8M ->
    # -18.00 @31.4M, with the realized E-step KL (kl_q_rel) falling
    # 0.966 -> 0.441 -> 0.00002 and gap crossings/episode collapsing
    # 0.221 -> 0.260 -> 0.030. At the floor 1/t is ~7e7, so any residual critic
    # noise becomes a hard argmax over the sampled actions.
    #
    # A higher floor is the conservative setting: near-uniform E-step weights are
    # a no-op update, which is strictly better than chasing noise. A healthy
    # dense-reward reference run sits at -0.54 after ~300M steps.
    #
    # NOTE also that this floor is in ABSOLUTE temperature while Q scales with
    # the reward weight, so scaling the reward up is equivalent to lowering this
    # floor. That is why "make the sparse reward bigger" is not the fix it looks
    # like.
    min_log_temperature: float = -18.0

    # Use an n-step return for the critic target instead of single-step TD:
    # n = min(n_step, sequence_length) with next_observation stored,
    # min(n_step, sequence_length - 1) on the compressed replay layout. False
    # (the default) is single-step TD.
    use_n_step: bool = False

    # ------------------------------------------------------------------
    # Warm-start transition machinery. All defaults are "off"
    # so existing configs are bit-unchanged. Designed for dense->sparse
    # reward handover: warm-start the policy from a dense-reward checkpoint,
    # let a fresh critic fit the warm policy's data before the policy moves,
    # anneal the dense reward component to zero, and keep a decaying fraction
    # of the env batch driven by the frozen dense policy so dense-competent
    # behaviour stays in replay while the learner takes over.
    #
    # Schedules are functions of the env-step ESTIMATE derived inside jit
    # from the SGD counter: t = state.steps * (num_envs * unroll_length / K).
    # This follows the kl_anchor w-decay precedent (schedules from
    # state.steps, no signature changes to the fused-step contract). The
    # estimate ignores the replay warm-up offset (~2 rollouts, <0.5% of any
    # schedule used here).
    # ------------------------------------------------------------------

    # Critic-only warmup: while state.steps < N, policy and dual params (and
    # their optimizer states) are held fixed via lax.select in
    # learner.sgd_step; the critic and its target still train. The M-step and
    # dual gradients are computed and discarded (wasted FLOPs, numerically
    # inert), mirroring the existing can_sample gate.
    critic_warmup_sgd_steps: int = 0

    # Reward re-mix at rollout time:
    #   reward_stored = sparse + lambda(t) * (reward_total - sparse)
    # where sparse = nan_to_num(new_state.metrics[reward_anneal_sparse_key])
    # and lambda(t) falls linearly 1 -> 0 over [0, reward_anneal_env_steps],
    # clamped at 0 after. The env must be configured with BOTH the sparse and
    # the dense terms in reward_terms (reward_total is their sum); the mix is
    # applied to the value stored in replay, not to the env's own metrics.
    # None disables the remix entirely (bit-identical path).
    reward_anneal_sparse_key: str | None = None
    reward_anneal_env_steps: int = 0

    # Behavior mixing: the FIRST ceil(x(t) * num_envs) envs of the batch act
    # with a frozen copy of the warm-start policy instead of the learner.
    #   x(t) = behavior_mix_init                          t <  hold
    #        = init * (1 - (t-hold)/(end-hold))           hold <= t < end
    #        = 0                                          t >= end
    # Requires frozen_behavior_params to be supplied to the training loop
    # (the entry point passes the warm-start policy params). 0.0 disables.
    behavior_mix_init: float = 0.0
    behavior_mix_hold_env_steps: int = 0
    behavior_mix_end_env_steps: int = 0

    # ------------------------------------------------------------------
    # Replay-buffer compression. The buffer lives on-device
    # inside the fused JIT, so its depth is bounded by GPU memory; at the
    # uncompressed ~19.5 KB/transition (two f32 copies of the 32x32x2 vision
    # obs) 400k transitions ~= 7.8 GB ~= only 4 rollouts of history. Both
    # flags default to the historical schema.
    # ------------------------------------------------------------------

    # Drop the `next_observation` field from stored transitions. In flashbax
    # trajectory storage observation[t+1] IS next_observation[t] bit-for-bit
    # (the auto-reset wrapper swaps obs on the terminal step itself, and the
    # time axis is continuous across rollout adds because env_state persists),
    # so the field is pure duplication -- half the observation memory. The
    # learner then bootstraps from observation[:, n], which needs
    # sequence_length >= n_step + 1 (the learner caps n = min(n_step, T-1)).
    store_next_observation: bool = True

    # Store obs["vision"] as uint8 (round(v*255)) instead of float32. The
    # renderer unpacks 8-bit channels to f32 in [0,1], so quantization error
    # is <= 1/510 of range. normalize_dict_obs already dequantizes uint8
    # vision (/255) at sample time; the live rollout policy consumes the f32
    # render directly and is unaffected. 4x on the dominant obs component.
    vision_uint8_storage: bool = False

    # ------------------------------------------------------------------
    # Recurrent (GRU) policy head: short-BPTT learner.
    # ------------------------------------------------------------------

    # BPTT unroll depth L = the number of loss points per sampled sequence.
    # 0 (default) keeps the feed-forward learner: `sgd_step` branches on this
    # at Python trace time, so every completed FF arm stays bit-identical --
    # no retrace, no new ops, no schema change.
    #
    # When L > 0 the learner runs R2D2-style stored-state + short BPTT: the
    # online policy is unrolled L steps WITH gradients from the STORED
    # window-start hidden (batch["policy_hidden"][:, 0]), the MPO loss is
    # applied at each of the L unrolled points, and each point t bootstraps
    # with its own n-step return from observation[:, t + n] using the stored
    # hidden at t + n (single-step target apply). L is therefore the MEMORY
    # horizon -- how far back the GRU can learn what to remember (~100-400 ms
    # of context => L ~ 20-40 at 100 Hz) -- NOT the reward horizon, which
    # stays `n_step` (n = 100 for the gap task). The two meet only through
    # the sampled window: `sequence_length` MUST equal T = L*k + n_step, with
    # k = `rnn_tick_stride` (below; the default k = 1 gives
    # `rnn_bptt_length + n_step`), so that every loss point -- step t < L, or
    # tick t < L*k for a slow-core head -- has a full n-step reward window and
    # a bootstrap state at index t + n (checked fail-loud at trace time). The
    # recurrent learner also requires
    # `use_n_step=True` and the compressed replay schema
    # (`store_next_observation=False`); it raises otherwise.
    rnn_bptt_length: int = 0

    # Strided ("tick") BPTT for slow-core heads (networks_kl_anchor_slow).
    # k = the env steps between updates of a slow memory core whose hidden
    # carries a phase leaf (`recurrent_meta.phase_leaf`); the core ticks where
    # the stored phase is 0 and holds in between. With k > 1,
    # `rnn_bptt_length` counts TICKS (L), the sampled window is W = L * k
    # steps, and `sequence_length` MUST equal L * k + n_step (checked
    # fail-loud). The learner gathers frames, actions and loss points at the
    # first L ticks of the window (read from the stored phase), unrolls one
    # core update per tick, and still sums n-step rewards over every step. It
    # must equal the net's `tick_stride`. 1 (default) with a hidden that has
    # no phase leaf is the per-step learner above, bit-identical.
    rnn_tick_stride: int = 1

    # ------------------------------------------------------------------
    # Privileged critic input (asymmetric actor-critic).
    # ------------------------------------------------------------------

    # Batch key holding a per-step task-state vector that ONLY the critic sees
    # (e.g. which maze arm is baited). The env writes it to state.info[key];
    # the rollout stores it via `extra_state_extras` from the PRE-step state,
    # so batch[key][:, t] is aligned with observation[:, t]. The learner
    # attaches it as obs["privileged"] to critic applies only, gathered at the
    # index of the obs each apply consumes (t0 / bootstrap / loss points). The
    # critic must be built with a matching `critic_privileged_dim` (> 0).
    # "" (default) = off: no gather, no new ops -- every old path unchanged.
    critic_privileged_key: str = ""

    # ------------------------------------------------------------------
    # Acme-alignment switches. The defaults are the ALIGNED values (what
    # DeepMind acme's DMPO does); `LEGACY_ALIGNMENT` below lists the
    # historical value of each field, which reproduces the pre-alignment
    # numerics exactly (pinned by test_legacy_equivalence). Each field takes
    # effect behind its own switch; `validate_alignment` checks the values and
    # resolves "auto". The legacy preset is `config/dmpo/legacy_pre_alignment`.
    # ------------------------------------------------------------------

    # How the critic's bootstrap value is formed.
    #   "sample_mixture": average the target critic's distribution over
    #                     several actions sampled from the target policy
    #                     (acme).
    #   "mean_action":    one evaluation at the target policy's mean action.
    critic_target_mode: str = "sample_mixture"

    # Treat a time-limit cut as a bootstrap point, not as a terminal: the
    # n-step window is cut at the first time-out (read from the stored
    # truncation flag) and the target still bootstraps from there, while a
    # real termination keeps a zero bootstrap. False (the legacy preset)
    # treats every episode end as terminal, which biases values near the time
    # limit; True (the aligned default) bootstraps through time limits.
    bootstrap_on_truncation: bool = True

    # Which action the critic is evaluated on at the loss point.
    #   "executed": the action that was applied to the env (the tanh-bound
    #               policy sample).
    #   "raw":      the stored pre-bound policy sample.
    critic_action_input: str = "executed"

    # Form of the out-of-bound action penalty in the MPO loss.
    #   "acme":    -||a - clip(a, -1, 1)|| (only the out-of-bound part).
    #   "vnl_ray": -||a|| (every action magnitude).
    action_penalty_formula: str = "acme"

    # Which state the policy (E/M-step) loss is evaluated on.
    #   "auto":    "next" on a feed-forward learner, "current" on a recurrent
    #              one (resolved by `validate_alignment`).
    #   "next":    the state the n-step window arrives at (the bootstrap
    #              state), feed-forward only; shares its action samples with
    #              the critic target.
    #   "current": the state the sampled action was taken in.
    policy_loss_state: str = "auto"

    # Scope of the policy gradient-norm clip.
    #   "global":    one global-norm clip over all trainable policy parameters.
    #   "per_block": each trainable parameter block is clipped on its own.
    policy_grad_clip_scope: str = "global"


# The six acme-alignment fields, in a fixed order.
ALIGNMENT_FIELDS: tuple[str, ...] = (
    "critic_target_mode",
    "bootstrap_on_truncation",
    "critic_action_input",
    "action_penalty_formula",
    "policy_loss_state",
    "policy_grad_clip_scope",
)

# Historical value of every alignment field: the preset that reproduces the
# pre-alignment learner bit for bit.
LEGACY_ALIGNMENT: dict = {
    "critic_target_mode": "mean_action",
    "bootstrap_on_truncation": False,
    "critic_action_input": "raw",
    "action_penalty_formula": "vnl_ray",
    "policy_loss_state": "current",
    "policy_grad_clip_scope": "per_block",
}

# Accepted values of the string-valued fields (bools are checked by type).
_ALIGNMENT_CHOICES: dict[str, tuple[str, ...]] = {
    "critic_target_mode": ("sample_mixture", "mean_action"),
    "critic_action_input": ("executed", "raw"),
    "action_penalty_formula": ("acme", "vnl_ray"),
    "policy_loss_state": ("auto", "next", "current"),
    "policy_grad_clip_scope": ("global", "per_block"),
}


def validate_alignment(cfg: DMPOConfig, recurrent: bool) -> DMPOConfig:
    """Check the alignment fields and resolve ``policy_loss_state="auto"``.

    Returns a copy of ``cfg`` with ``policy_loss_state`` resolved to ``"next"``
    (feed-forward learner) or ``"current"`` (``recurrent=True``). Raises
    ``ValueError`` for an unknown value, naming the field and listing the valid
    values, and for ``policy_loss_state="next"`` on a recurrent learner (the
    recurrent learner has no next-state policy unroll). Runs at trace time, so
    a bad config fails before any device time is spent.
    """
    for name, choices in _ALIGNMENT_CHOICES.items():
        value = getattr(cfg, name)
        if value not in choices:
            raise ValueError(
                f"unknown {name}={value!r}; valid values: "
                + ", ".join(repr(c) for c in choices)
            )
    flag = cfg.bootstrap_on_truncation
    if not isinstance(flag, bool):
        raise ValueError(
            f"unknown bootstrap_on_truncation={flag!r}; valid values: True, False"
        )
    state = cfg.policy_loss_state
    if state == "next" and recurrent:
        raise ValueError(
            "policy_loss_state='next' is not supported by the recurrent "
            "learner; use 'current' or 'auto' (which resolves to 'current')."
        )
    if state == "auto":
        state = "current" if recurrent else "next"
    return dataclasses.replace(cfg, policy_loss_state=state)


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
        int(
            cfg.unroll_length * cfg.num_envs / (cfg.batch_size * cfg.samples_per_insert)
        ),
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
