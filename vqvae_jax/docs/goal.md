# Motor Decisions: Discrete Action Selection in a Virtual Rodent Predicts Neural Population Structure

## Project Goal

Understanding and composing agent decisions via multi-resolution adjustment patterns,
and testing whether the resulting discrete decision structure predicts the organization
of neural activity in motor cortex and striatum better than continuous alternatives.

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

---

## Neural Comparison: Discrete vs Continuous Motor Representations

### Motivation

Aldarondo et al. (2024), "A virtual rodent predicts the structure of neural activity
across behaviours" (Nature 632, 594-602), demonstrated that a virtual rodent with a
**continuous** Gaussian latent bottleneck (60-dim, KL-regularized) produces internal
representations that predict real motor cortex (MC) and dorsolateral striatum (DLS)
activity better than kinematic features alone. Their key claim: the virtual rodent
implements an inverse dynamics model, and this is what MC and DLS also implement.

Our framework uses the same inverse-dynamics structure -- encoder maps reference
trajectory to latent, decoder maps latent + proprioception to actions -- but forces
the latent through a **discrete** VQ-RVQ bottleneck. This creates a direct,
controlled comparison: does discretizing the motor plan make the internal
representations more or less brain-like?

### The Hypothesis

Real motor cortex activity shows a specific structure that is neither fully continuous
nor fully discrete:

- **Discrete preparatory states** before movement onset (Churchland & Shenoy):
  neural activity jumps to cluster-like preparatory configurations.
- **Smooth execution dynamics** during movement (Churchland et al. 2012):
  rotational trajectories within a low-dimensional manifold.
- **Categorical single-neuron tuning** (Aldarondo Fig 4): individual neurons
  in DLS and MC preferentially fire during specific behavioral categories.

A continuous latent model produces smooth representations everywhere -- no discrete
transitions. Our VQ model produces **piecewise-smooth representations**: smooth
trajectories within a code (proprioception varies continuously), sharp transitions
at code switches. This piecewise structure should better match the discrete-then-
smooth pattern observed in real neural data.

### Experimental Design

Four models isolate the effect of discretization from other architectural differences:

```
Model A: Continuous latent + MLP decoder   (ablation baseline)
Model B: VQ-RVQ latent + MLP decoder       (our current model)
Model C: Continuous latent + LSTM decoder   (Aldarondo's architecture)
Model D: VQ-RVQ latent + LSTM decoder       (novel: discrete plans + smooth execution)
```

The clean comparison is A vs B (same decoder, different latent). Models C and D
test whether adding temporal memory in the decoder interacts with discretization.
All models must achieve comparable imitation reward for the comparison to be valid.

### What to Measure

**Representation structure (no neural data needed):**

| Metric | What it tests | VQ prediction |
|--------|--------------|---------------|
| PCA dimensionality | Intrinsic complexity of hidden states | Lower (clustered) |
| Silhouette score | Discrete cluster quality in hidden space | Stronger clusters |
| HMM log-likelihood | Whether hidden dynamics have discrete states | Better fit |
| Tangling (Russo 2018) | State → future predictability | Lower (more predictable) |
| Transition sharpness | How abrupt are behavioral transitions | Sharper (step-like) |

**Neural comparison (with MC/DLS recordings):**

| Metric | What it tests |
|--------|--------------|
| GLM predictivity (CV-LLR) | Single-neuron prediction from network activations |
| RSA (whitened cosine) | Population-level representational geometry match |
| CKA | Layer-to-region alignment |
| Linear decodability | Can real neural activity be decoded into VQ codes? |

### Expected Findings

**Where VQ should win:**
- RSA structure: sharper block-diagonal RDMs matching categorical neural tuning.
- Transition dynamics: discrete code switches matching sharp neural transitions
  at movement onset (the "condition-independent signal," Kaufman et al. 2016).
- Temporal prediction: lower tangling because within-code dynamics are predictable.

**Where continuous should win:**
- Raw GLM predictivity: 60 continuous dims give the GLM more graded features to
  fit than 32 discrete codes. Continuous models likely score higher on single-neuron
  R-squared.
- Within-behavior variation: continuous latents capture graded vigor, speed, and
  postural nuance that VQ's finite codes cannot represent.

**The key finding** would be: despite lower raw predictivity, the VQ model's
representations better capture the **categorical and temporal structure** of MC/DLS
activity. This would provide computational evidence that the motor system organizes
around discrete primitives -- not just continuous dynamics.

### Relationship to Multi-Resolution Decisions

The D0/D1 decomposition described above maps directly onto a neural hierarchy:

```
Framework              Neural analog
---------              -------------
D0 (coarse code)   →   Premotor cortex / basal ganglia (action selection)
D1 (fine code)     →   Motor cortex / cerebellum (online correction)
Decoder + proprio  →   Motor cortex + spinal cord (state-dependent execution)
Stickiness bias    →   Commitment to motor plan (perseveration)
Code switch        →   Decision boundary crossing
```

The RVQ structure makes a specific prediction: **D0 transitions should align with
discrete state transitions in premotor/striatal recordings, while D1 variation should
align with continuous modulation in motor cortex.** This is testable if neural
recordings span both regions.

### Open Questions (Neural)

5. **Neural data access**: Aldarondo et al. recorded from DLS and MC in freely
   moving rats performing the same behavioral repertoire we train on. Can we
   access their data or collaborate for a direct comparison?

6. **Code-to-neuron mapping**: If VQ codes can be linearly decoded from neural
   population activity, that would be strong evidence that the brain uses a
   discrete motor vocabulary. What decoding accuracy is needed to make this claim?

7. **Timescale match**: Do D0 code dwell times (~200-800ms with stickiness)
   match the timescale of discrete neural state transitions in MC/DLS? This is
   a falsifiable prediction of the framework.

8. **LSTM + VQ (Model D)**: Does adding temporal memory to the decoder help or
   hurt neural predictivity? If it helps, the brain may use both discrete
   planning and recurrent execution -- consistent with separate premotor and
   motor cortex roles.
