# DMPO on the rodent gap-jump task: why the port underperformed, and what fixed it

Branch: `scott_claude/dmpo-gapjump-frozen-prior` (both `track-mjx` and `vnl-playground`)
Dates: 2026-08-11 → 2026-08-18

## The question

A Ray/Acme DMPO run solves the rodent corridor gap-jump task
(`yuy004/rodent-four-tasks/56cdO02Zd43jHUdCp23by`, episode_return 1156.7 at
episode_length 1500). The JAX/MJX port of the same algorithm, on the same task with the same
frozen prior, plateaued at roughly a quarter of the PPO reference. The question was whether DMPO
is unsuited to the task or whether the port was defective.

**Answer: the port was defective, in two independent ways, both in the learner.** Neither is a
property of DMPO. An earlier claim in `DIAGNOSIS_2.md` that DMPO's speed-vs-survival trade-off is
intrinsically capped is **wrong** and is superseded by this document.

## The defects

### 1. The n-step return was never ported (`learner.py`, commit `aab4f33`)

`cfg.n_step = 50` had been declared and never read. vnl-ray obtains its n-step return from
reverb's `NStepTransitionAdder`, which squashes n steps into each stored transition *before the
learner sees it*. The port copied the config value but not the mechanism: it sampled a length-50
sequence and used only `[:, 0]`, i.e. single-step TD.

Verified by the arithmetic of the counters, not by inspection alone — see defect 2.

### 2. `samples_per_insert` is inverted (`config.py`, commit `cd09c65`)

Two formulas coexisted and disagreed:

```
entry points          K = unroll * num_envs / (batch * samples_per_insert)   DIVIDES
train.py              K = samples_per_insert * unroll * num_envs / batch     MULTIPLIES
```

The live path used the first, so **raising `samples_per_insert` reduced learner work**.
`compute_num_updates` (the MULTIPLY form) is dead for every live entry point.

The cross-check is exact. `dmpo_frozen_prior_vel08_sigmaball` ran 297,574,400 env steps = 2,906
rollouts at 2048×50. flashbax's `min_length_time_axis = max(sequence_length+1,
min_replay_size//num_envs) = 51` gates precisely the first rollout's SGD (the whole state pytree,
including `steps`, goes through a `lax.select`). So (2906−1)×50 = **145,250**, the observed learner
count, to the step. The MULTIPLY convention predicts 581,000.

Realized reuse: **0.5** samples drawn per actor step, against vnl-ray's **3.236**
(2,572,765 learner steps × batch 256 / 203.53M actor steps).

Fixed by adding an explicit `sgd_steps_per_rollout` rather than swapping the formula, which would
have silently redefined `samples_per_insert` in every YAML already run.

### 3. The gap-crossing metric was unwired (`train_dmpo_eval.py`, commit `5e2b6b6`)

`batch/gap_crossings_per_env` read `rewards/gap_crossing_bonus`, a reward metric that `base.py`
only writes when that term is in `env_config.reward_terms`. Every frozen-prior arm deliberately
omits it. The `.get(key, zeros)` default therefore reported **0.000 crossings unconditionally** on
exactly the arms under study. This was reported as a behavioural finding for two sessions and
drove a false hypothesis about sub-pixel gaps at 32×32.

Reality: **every one of 2048 envs crosses gaps.** `batch/gap_measured` now distinguishes a real
zero from a blind spot.

## Results

Episode return, all-env estimator (`batch/mean_episode_reward`), each arm one factor off its
parent:

| arm | change | ep_ret | crossings/ep |
|---|---|---|---|
| baseline (rescored) | n=1 | 61.9 | unmeasured |
| `arm_h1_nstep50` | n=1 → 50 | ~145 | 0.60 |
| `arm_h3_nstep100` | n=50 → 100 | ~147 | 0.60 |
| `arm_j1_reuse324` | K=50 → 324 | 180 @95M | 0.98 |
| `arm_k1_raymdp` | vnl-ray reward + termination | n/c¹ | 0.23 |

¹ different reward function; returns are not comparable. Compare crossings and episode length.

In **PPO's own estimator** (brax `EvalWrapper`, first-episode censored), matched at 95M:

| | avg_episode_length | episode_reward |
|---|---|---|
| `arm_h1_nstep50` | 184.5 | 140.3 |
| **`arm_j1_reuse324`** | **278.4** | **228.8** |
| PPO reference @100M | 215.0 | 170.0 |
| PPO reference @150M | 277.0 | — |

**Do not compare `batch/mean_episode_reward` with the PPO reference.** PPO's numbers are
first-episode censored; the matching keys are `avg_episode_length` / `episode_reward`, emitted
specifically for this comparison.

### Findings that were negative, and matter

- **Horizon length saturates.** n=50 → n=100 (the exact vnl-ray 1.0 s horizon, since its ctrl_dt
  is 0.02 s not 0.025) is a **null**: +2.2 return against a 3.6 standard error, 0.6σ. Half a second
  already captures the available delayed credit.
- **The vnl-ray MDP is worse here.** `k1` crosses 0.23 gaps/episode against the baseline's 0.60.
  Making the MJX environment resemble Ray's *hurt*. Caveat: `k1` changed reward and termination
  together, so this does not attribute to either alone — the termination change on its own still
  looks promising and is untested (see Open items).

### Statistical caveat that repeatedly caused false readings

The eval SEM (~0.9 on ep_ret) measures precision **within** a checkpoint. Checkpoint-to-checkpoint
policy variance is ~10× larger. Comparing two runs requires the spread **across** checkpoints as
the denominator; using the printed SEM turns noise into a "significant" result. Over one 300M run,
reading three-to-four consecutive points produced, in order: "still climbing", "plateauing",
"degrading", "recovering", and "reward-per-step declining monotonically" — every one reversed
within two more evals.

## Reproducing

```bash
cd _implementation_log/DMPO/vnl-playground
../dmpo-env/bin/python -m vnl_playground.train_highlvl_dmpo_kl_anchor \
    --config-name=rodent_run_gap_dmpo/arm_j1_reuse324
```

Confirm at startup that the throughput knob took effect:

```
K=324 SGD updates/rollout via sgd_steps_per_rollout (explicit; samples_per_insert is unread)
  | realized_samples_per_insert=3.24 ...
```

Tests (**always** `JAX_PLATFORMS=cpu` — pytest initialises CUDA and will OOM-kill a running
trainer):

```bash
cd _implementation_log/DMPO/track-mjx
JAX_PLATFORMS=cpu ../dmpo-env/bin/python -m pytest tests/agent/dmpo/ -q
```

`test_train_dmpo_eval.py::test_render_eval_video_writes_file` is flaky in-suite only (ffmpeg
subprocess fork under multithreaded JAX); it passes in isolation.

## Operational notes

- **One GPU, strictly serial.** Anything that initialises JAX/CUDA while a trainer runs will
  OOM-kill it.
- **Resume works** (commit `cd09c65`): checkpoints are saved at `step=total_env_steps`, so
  `mgr.latest_step()` restores the counter. Before this, a resumed run trained `num_timesteps`
  *more* steps under colliding checkpoint names.
- **Replay is not checkpointed** but refills in ~4 rollouts (0.4% of a 300M run).
- `max_replay_size` counts **transitions**. The base default of 4,000,000 is ~78 GB and will not
  load; the arms override to 400,000.
- **`arm_l1_decoder_thaw` needs a fresh `checkpoint_dir`.** `decoder_lr_mult != 0` switches the
  optimizer block from `set_to_zero` to `chain(clip, adam)`, adding Adam moments for the decoder
  subtree and changing `policy_opt_state`'s shape.
- vnl-ray's `froze_decoder=True` set `trainable=False` only on the **target** decoder; the online
  decoder kept receiving gradients and was copied onto the target at each update. **The successful
  Ray run trained its decoder.** MJX's freeze is genuinely bit-exact (`optax.set_to_zero`), i.e. a
  strictly harder transfer problem.

## Analysis harness (NOT version controlled)

`analysis/2026-08-11-dmpo-gapjump-frozen-prior/scripts/` sits outside any git repo. Key entries:

| script | purpose |
|---|---|
| `arm_report.py` | matched-step comparison across arms; reads wandb (survives log truncation) |
| `make_arms.py` | generates arm configs from a parent, asserting each anchor is unique |
| `run_queue.sh` / `chain_arms.sh` / `run_arm.sh` | serial GPU queue with identity guards |
| `smoke_arm.sh` | ~2M-step validation into a throwaway dir, for never-executed code paths |
| `test_ray_mdp_parity.py` | verifies the ported reward/termination against dm_control itself |
| `rescore_eval.py` | re-scores a saved checkpoint under the current estimator |

Narrative record: `ClaudeCode_PromptHistory/2026-08-18-1-dmpo-ray-parity-fixes/SESSION_LOG.md`.

## Open items

1. **Extend `arm_j1_reuse324` to 300M.** It is a 100M screen. PPO keeps climbing to 803 by 1.95B,
   so the current comparison is same-step-count, not final-performance.
2. **Split `k1`** into termination-only and reward-only arms. The permissive termination lengthened
   untrained episodes 5.5× (997 steps vs ~180) and may help even though the package hurt.
3. **Combine `j1` with the proprioceptive critic**, and with n=100.
4. **Re-score the pre-fix arms for gap crossings.** Only h1 (≥199.9M), h3, j1, k1 have crossing
   data; the n=1 baseline, `arm_h2_nstep10` and `arm_a2_critic_proprio` predate the fix, so the
   n-step dose-response cannot yet be told in crossings.
5. **γ is not horizon-matched.** 0.97 at ctrl_dt 0.01 is a 0.33 s credit horizon; vnl-ray's 0.97 at
   0.02 s is 0.66 s. Matching needs γ = 0.98489. Untested, and must be its own arm.
6. `anchor/*` on runs before 2026-08-18 are env-0 (n=1) and are **not** comparable with the newer
   all-env `batch/anchor/*` keys.
