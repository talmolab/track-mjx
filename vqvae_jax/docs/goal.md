# Goal: Understanding and Composing Agent Decisions via Multi-Resolution Adjustment Patterns

## Core Idea

A VQ-VAE with Residual Vector Quantization (RVQ) decomposes the agent's internal
decision signal into adjustment patterns at multiple resolutions. During imitation
training, the encoder observes a reference trajectory and produces a continuous
embedding `z_e` that summarizes "what needs to happen." RVQ decomposes this into:

```
z_e  ≈  z_q_d0  +  z_q_d1
        ------     ------
        coarse     fine
```

Both `z_q_d0` and `z_q_d1` live in the same latent space (R^D). The decoder sees
their sum. It does not know which level contributed what -- it receives a single
D-dimensional adjustment pattern plus the agent's proprioceptive state and produces
actions. There is no qualitative distinction between levels, only a quantitative one:
D0 captures the largest-scale component, D1 captures the residual.

## The Agent's Decision as Multi-Scale Adjustment

The agent's full "intention" at any moment is the sum `z_q_d0 + z_q_d1`. Neither
level alone defines the behavior:

- **D0 alone** is a blurry version of the intention -- enough to identify the gross
  behavioral mode (turning, staying still, grooming) but not enough to execute it
  precisely.
- **D1 alone** is a fine correction that only makes sense relative to a D0 context.
  It sharpens the intention into something the decoder can execute accurately.

This is analogous to representing a number like 3.7 as integer part (3) plus
fractional part (0.7). Both are components of the same value at different precision.
The "meaning" of 3.7 requires both parts.

## Timescale Separation

The two levels naturally separate along temporal timescales:

- **D0 (coarse) changes slowly.** A behavioral mode like "turn right" persists for
  many consecutive timesteps. The stickiness bias in the quantizer reinforces this
  temporal coherence. D0 transitions correspond to meaningful behavioral switches.

- **D1 (fine) changes rapidly.** The correction needed at each timestep depends on
  the agent's exact physical state -- limb positions, velocities, contact forces.
  This changes at every control step.

This mirrors a natural structure in motor control: **intentions are slow, corrections
are fast.** You decide to turn right (slow timescale), and your motor system
continuously adjusts muscle activations to execute that turn given your current
balance and momentum (fast timescale).

## Training: Error-Driven Decomposition

During imitation training, the decomposition is learned end-to-end through error
signals. The reference trajectory provides a fixed goal at each moment, and the
system learns to encode that goal into discrete adjustment patterns that minimize
imitation error.

The training objective does not explicitly assign "intentions" to D0 and "corrections"
to D1. Instead, RVQ's structure -- D0 quantizes first, D1 quantizes the residual --
naturally allocates the largest variance component to D0 and successively finer
components to D1. The timescale separation emerges because large-variance behavioral
modes tend to be temporally persistent, while small-variance corrections tend to be
temporally variable.

## Control Theory Interpretation

The critical point: D1 isn't a second goal, it's the error signal. In control theory
terms, D0 is the reference command and D1 is the feedback correction. The system is:

```
D0 (feedforward)  ──→  (+)  ──→  decoder  ──→  actions  ──→  physics
                        ↑
D1 (feedback)     ──────┘
```

This is why D1 "adjusts" -- it's compensating for the gap between what D0 commands
and what the physical state actually requires.

## After Training: Composing Goals

After training, the learned codebooks form a vocabulary of adjustment patterns. The
key insight: **you can compose new goals by chaining D0 codes, and D1 provides the
real-time adaptation that makes each intention physically realizable.**

The intended deployment pipeline:

1. **Planner** selects a D0 code sequence at a coarse timescale (e.g., every 50
   steps): "turn right, then walk forward, then stay still."
2. **D1 prior** or **task-specific policy** selects D1 codes at a fine timescale
   (every step), conditioned on the current D0 code and proprioceptive state. This
   provides the state-dependent adjustment that bridges the gap between the coarse
   intention and the physical reality.
3. **Decoder** maps `(z_q_d0 + z_q_d1, proprioception) -> actions` using the
   patterns learned during imitation training.

This enables goal specification in a small discrete space (|D0| codes) rather than
a high-dimensional continuous action space, while retaining the expressiveness of
continuous control through the D1 correction channel.

## What Ablation Studies Reveal

Ablation experiments (forcing D0 to specific codes, zeroing D0, chaining code
sequences) test the structure of this decomposition:

- **D0 injection with compatible state**: D1 adapts and produces coherent behavior.
  This shows D1 functions as a correction channel within D0's operating regime.

- **D0 injection with incompatible state**: D1 cannot compensate, behavior collapses.
  This shows D0 carries real mode information -- the correction vocabulary (D1
  codebook) has finite range and cannot override a fundamentally wrong D0 pattern.

- **D0 zeroed out**: Some reference clips can still be tracked (their behavior is
  within D1's range alone), others cannot (they require D0's contribution). This
  maps the operating envelope of D1-only control.

- **Code sequence transitions**: When a forced D0 switch is compatible with the
  agent's physical state, behavior transitions smoothly. When incompatible, the
  system falls off the manifold. This tests whether D0 codes are robust enough to
  serve as composable building blocks.

These experiments characterize the **operating envelope** of each D0 code -- the set
of physical states from which that code produces coherent behavior -- and the **range**
of D1 correction -- how far D1 can compensate for state-intention mismatch.

## Open Questions

1. **D1 prior**: How to select D1 codes without the encoder (i.e., without a
   reference trajectory)? A learned prior `p(D1_t | D0_t, state_t)` trained from
   rollout data is the natural approach.

2. **Code robustness**: Current D0 codes are fragile -- they work from compatible
   states but break from incompatible ones. Can training be modified (data
   rebalancing, auxiliary losses, state-conditional codebook usage) to widen each
   code's operating envelope?

3. **Codebook allocation**: The training data is dominated by stationary behaviors,
   so the codebook over-allocates to stationary modes. How to ensure locomotion and
   other rare behaviors get adequate codebook representation?

4. **Timescale calibration**: What is the right temporal granularity for D0 codes?
   The current stickiness bias encourages persistence, but the optimal window size
   for code switches (and how it relates to the underlying behavioral timescales)
   is not yet characterized.
