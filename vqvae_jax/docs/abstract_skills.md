# Abstract Skill Representations in VQ-VAE

**Date**: 2026-01-24
**Branch**: kevin/vqvae-bias
**Status**: Design Philosophy Document

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem with Phase Decomposition](#2-the-problem-with-phase-decomposition)
3. [Abstract Symbols vs Semantic Segmentation](#3-abstract-symbols-vs-semantic-segmentation)
4. [Embracing Superposition](#4-embracing-superposition)
5. [Why Spanning the Codebook is Desirable](#5-why-spanning-the-codebook-is-desirable)
6. [Code Persistence via Bias](#6-code-persistence-via-bias)
7. [Examples of Abstract Concepts](#7-examples-of-abstract-concepts)
8. [Implications for Analysis](#8-implications-for-analysis)

---

## 1. Overview

This document describes the design philosophy for VQ-VAE codebook representations in motion imitation learning. The key insight is that codebook entries should represent **abstract motor primitives** rather than semantic behavior labels, and that **superposition** (the same code meaning different things in different contexts) is a feature to embrace, not a problem to solve.

### Core Principles

1. **Codes are abstract, not semantic**: Code 47 doesn't mean "walking"—it means something like "initiating forward momentum"
2. **Meaning emerges from context**: `code + proprioceptive_state = semantic_meaning`
3. **Superposition enables compression**: 512 codes can represent thousands of behaviors through combinatorial composition
4. **Temporal persistence matters**: Codes should represent extended phases, not instantaneous states

---

## 2. The Problem with Phase Decomposition

### What We Observe

Without intervention, VQ-VAE codes exhibit **rapid phase decomposition**:

```
Timestep:  1    2    3    4    5    6    7    8    9   10
Code:     [47] [23] [47] [112] [23] [47] [8] [112] [47] [23]
```

The codes switch rapidly, often oscillating between 2-3 codes within a single behavioral phase. This happens because:

1. **Small encoder fluctuations** cause boundary crossings in Voronoi space
2. **No temporal regularization** in standard VQ-VAE argmin
3. **Codes capture instantaneous state** rather than extended phases

### Why This is Problematic

- **Uninterpretable**: Rapid switching obscures what each code "means"
- **Noisy transitions**: Hard to distinguish meaningful state changes from noise
- **Lost temporal structure**: The duration of a phase carries information

### What We Want Instead

```
Timestep:  1    2    3    4    5    6    7    8    9   10
Code:     [47] [47] [47] [47] [23] [23] [23] [112][112][112]
           └── entering ──┘   └─ sustaining ─┘  └─ exiting ─┘
```

Codes should persist for meaningful durations, with transitions marking actual phase boundaries.

---

## 3. Abstract Symbols vs Semantic Segmentation

### The Semantic Segmentation Trap

A tempting but flawed goal is to have codes map directly to behaviors:

```
Code 1 = "walking"
Code 2 = "grooming"
Code 3 = "rearing"
...
```

This **semantic segmentation** approach fails because:

1. **Not enough codes**: 512 codes can't cover all behavior variations
2. **Arbitrary boundaries**: Where does "walking" end and "trotting" begin?
3. **Ignores compositionality**: Behaviors share sub-components
4. **Wastes capacity**: Similar phases in different behaviors get separate codes

### The Abstract Symbol Approach

Instead, codes represent **abstract motor primitives** that compose into behaviors:

```
Code 47 = "initiating/accelerating"
Code 23 = "sustaining steady-state"
Code 112 = "terminating/decelerating"
Code 8 = "weight-shifting"
...
```

These abstract symbols combine with proprioceptive context to produce meaning:

| Code | Proprioception | Semantic Meaning |
|------|----------------|------------------|
| 47 (initiating) | legs extending, body tilting forward | entering walk |
| 47 (initiating) | forelimbs raising, body tilting back | entering rear |
| 47 (initiating) | head lowering, forelimbs to face | entering groom |

The **same code** participates in **different behaviors** because the abstract concept (initiating) is shared.

### Benefits of Abstract Representations

1. **Efficient**: Shared primitives reduce redundancy
2. **Compositional**: New behaviors = new combinations of existing codes
3. **Generalizable**: Abstract concepts transfer across behavior types
4. **Interpretable**: Once decoded, primitives have consistent meaning

---

## 4. Embracing Superposition

### What is Superposition?

Superposition occurs when a single code represents multiple distinct concepts depending on context. In our VQ-VAE:

```
Code 47 in context A = "entering walk"
Code 47 in context B = "entering rear"
Code 47 in context C = "entering groom"
```

### Why Superposition is Necessary

With only 512 codes and thousands of possible motor states, superposition is mathematically inevitable. But more importantly, **superposition is why VQ-VAE works at all**:

1. **Compression**: 512 codes + continuous proprioception = unbounded expressiveness
2. **Abstraction**: Superposition forces codes to capture what's *common* across contexts
3. **Disentanglement**: The code captures the abstract component; proprioception captures the specific

### Superposition is Not a Bug

Early investigations might label superposition as a failure ("codes aren't clean!"). This is wrong. The goal is not:

```
code → behavior  (1:1 mapping, no superposition)
```

The goal is:

```
code + context → behavior  (many:many mapping, structured superposition)
```

### The Graph of Dependencies

Embracing superposition leads to a **dependency graph** where:

- **Nodes** = codes (abstract primitives)
- **Edges** = valid transitions
- **Paths** = behavior sequences

```
        ┌─────────────────────────────────────┐
        │                                     │
        v                                     │
    [initiating] ──────> [sustaining] ──────> [terminating]
        │                     │                    │
        │                     v                    │
        │              [weight-shifting]           │
        │                     │                    │
        └─────────────────────┴────────────────────┘
```

Different behaviors traverse different paths through this graph, but share nodes.

---

## 5. Why Spanning the Codebook is Desirable

### The "Problem" Restated

An early concern was: "Every clip uses all codes—shouldn't different clips use different code subsets?"

### Why Spanning is Actually Correct

If codes are abstract primitives, **all clips should use most codes** because:

1. **All clips have "initiating" phases** → Code for initiating appears everywhere
2. **All clips have "sustaining" phases** → Code for sustaining appears everywhere
3. **All clips have "terminating" phases** → Code for terminating appears everywhere

A walking clip and a grooming clip both contain initiating, sustaining, and terminating—just with different proprioceptive contexts.

### When Non-Spanning Would Occur

Codes would naturally separate across clips only if:

1. **Clips have no shared structure** (unlikely for motor behaviors)
2. **Codebook is massive** (enough codes for semantic segmentation)
3. **Explicit clip conditioning** (model knows clip identity)

### Spanning Enables Generalization

If a new behavior (never seen in training) shares abstract structure with training behaviors, the model can represent it using existing codes. This is only possible if codes are abstract and shared.

### What to Monitor Instead

Rather than worrying about spanning, monitor:

1. **Code consistency**: Does the same code have consistent abstract meaning?
2. **Transition structure**: Do transitions follow meaningful patterns?
3. **Context separation**: Does proprioception disambiguate superposed meanings?

---

## 6. Code Persistence via Bias

### The Mechanism

To achieve temporal persistence, we modify the VQ-VAE quantizer to **bias toward the previous code**:

```python
# Standard VQ-VAE
indices = argmin(||z_e - codebook||^2)

# With stickiness bias
indices = argmin(||z_e - codebook||^2 - bias * I(code == prev_code))
```

The bias makes the previous code appear closer than it really is, creating a "sticky" region that the encoder must overcome to trigger a transition.

### Why Bias > Cross-Entropy Loss

| Aspect | Cross-Entropy Loss | Bias Approach |
|--------|-------------------|---------------|
| Mechanism | Soft penalty via gradients | Direct modification of selection |
| Timing | Post-selection (loss term) | Pre-selection (inside argmin) |
| Effect | Pushes encoder toward prev | Expands prev code's selection region |
| Clarity | Indirect, interacts with other losses | Direct, single knob to tune |

### Hysteresis Dynamics

The bias creates **hysteresis**—the encoder must produce a significantly different output to trigger a code change:

```
                    Transition Threshold
                           │
     Code A    ════════════╪════════════>    Code B
               <═══════════╪═════════════
                           │
               z_e must cross this boundary
               to change codes
```

This naturally filters rapid fluctuations while preserving meaningful transitions.

### Calibrating the Bias

- **Too low** (< 0.1): Codes still switch rapidly
- **Just right** (0.5 - 2.0): Codes persist for meaningful durations
- **Too high** (> 5.0): Codebook collapse, everything uses one code

Monitor **perplexity** and **transition rate** to tune.

---

## 7. Examples of Abstract Concepts

### Temporal/Phase Dynamics

| Code Concept | Description | Appears In |
|--------------|-------------|------------|
| Initiating | Beginning movement, overcoming inertia | All behaviors |
| Sustaining | Maintaining steady-state | All behaviors |
| Terminating | Decelerating, coming to rest | All behaviors |
| Transitioning | Shifting between movement types | Behavior boundaries |

### Kinematic Primitives

| Code Concept | Description | Appears In |
|--------------|-------------|------------|
| Accelerating | Increasing velocity | Walk, run, lunge |
| Decelerating | Decreasing velocity | Stopping, landing |
| Oscillating | Rhythmic alternation | Walk, groom, scratch |
| Holding | Maintaining position | Rear, pause, balance |

### Balance & Stability

| Code Concept | Description | Appears In |
|--------------|-------------|------------|
| Weight-shifting | Transferring center of mass | All locomotion |
| Stabilizing | Correcting perturbation | Recovery, balance |
| Loading | Accepting weight onto limb | Stance phases |
| Unloading | Removing weight from limb | Swing phases |

### Example Decomposition

**Rearing behavior**:
```
[weight-shift-back] → [unload-front] → [elevating] → [stabilizing] → [holding]
```

**Walking behavior**:
```
[weight-shift-forward] → [unload-rear] → [accelerating] → [alternating] → [sustaining]
```

Notice shared codes: both behaviors include weight-shifting and unloading, just in different contexts.

---

## 8. Implications for Analysis

### How to Interpret Codes

1. **Don't expect semantic labels**: Code 47 ≠ "walking"
2. **Look for consistency**: Does code 47 always appear during initiating phases?
3. **Check context dependence**: What proprioceptive states accompany code 47?
4. **Map the graph**: What codes typically follow code 47?

### Metrics That Matter

| Metric | What It Tells You |
|--------|-------------------|
| Perplexity | How many codes are actively used (should be high) |
| Transition rate | How often codes change (should be moderate, not too high) |
| Code duration | How long codes persist (should match behavioral timescales) |
| Transition matrix | Which codes follow which (should show structure) |
| Context clustering | Do same-code instances cluster in proprioceptive space? |

### Visualization Approaches

1. **Transition graph**: Nodes = codes, edges = transition frequency
2. **Code timeline**: Color-coded sequence showing code persistence
3. **Context scatter**: PCA of proprioception, colored by code
4. **Per-code video**: Frames grouped by assigned code

### Questions to Ask

- "When code 47 is active, what is the body doing?" (look at video)
- "What proprioceptive features distinguish code 47 in walking vs rearing?" (context analysis)
- "What codes typically precede/follow code 47?" (transition structure)
- "Does code 47 appear at consistent phase positions?" (temporal alignment)

---

## Summary

| Principle | Implication |
|-----------|-------------|
| Codes are abstract | Don't expect semantic labels |
| Superposition is good | Same code, different contexts = different meanings |
| Spanning is expected | All clips use most codes because abstract concepts are shared |
| Persistence matters | Use bias to prevent rapid phase decomposition |
| Context completes meaning | Code + proprioception = semantic interpretation |

The goal is a **disentangled abstract representation** where:
- Codes capture **what's common** across behaviors (the abstract)
- Proprioception captures **what's specific** to each instance (the context)
- Composition of codes over time captures **behavioral structure** (the sequence)

This is fundamentally different from semantic segmentation, and that's intentional.
