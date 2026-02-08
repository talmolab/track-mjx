# Residual Vector Quantization for Motor Intention Learning

## Table of Contents

1. [Current System: VQ-VAE with State-Dependent Coding](#1-current-system-vq-vae-with-state-dependent-coding)
2. [Observed Phenomenon: Code Redundancy](#2-observed-phenomenon-code-redundancy)
3. [Why Superposition Is Not the Problem](#3-why-superposition-is-not-the-problem)
4. [Background: Standard VQ-VAE (Our Implementation)](#4-background-standard-vq-vae-our-implementation)
5. [STAR: Rotation-Augmented Residual Vector Quantization](#5-star-rotation-augmented-residual-vector-quantization)
6. [Side-by-Side Comparison: Our VQ-VAE vs STAR](#6-side-by-side-comparison-our-vq-vae-vs-star)
7. [How STAR Would Address Code Redundancy](#7-how-star-would-address-code-redundancy)
8. [References](#8-references)

---

## 1. Current System: VQ-VAE with State-Dependent Coding

Our motor intention policy uses a VQ-VAE encoder-decoder architecture
([van den Oord et al., 2017](#ref-vqvae)) to learn discrete skill codes for
rodent locomotion control. The architecture is:

```
                    ┌─────────────────────────────────┐
                    │        VQIntentionNetwork        │
                    │                                  │
  Reference        │  ┌─────────┐    ┌────────────┐   │
  Trajectory ──────┼─▶│ Encoder │───▶│ Quantizer  │   │
  (imitation       │  │ MLP     │ zₑ │ K=64, D=60 │   │
   target)         │  └─────────┘    └─────┬──────┘   │
                    │                       │ zq_st    │
                    │                       ▼          │
  Proprioceptive   │              ┌──────────────┐    │     Action
  State ───────────┼─────────────▶│   Decoder    │────┼───▶ Parameters
  (egocentric_obs) │              │ [zq, proprio] │    │     (μ, σ)
                    │              └──────────────┘    │
                    └─────────────────────────────────┘
```

**Key design feature:** The decoder receives `[z_q_st, egocentric_obs]` —
the quantized latent **concatenated with proprioceptive state**. This means
the same discrete code produces different motor outputs depending on the
animal's current body configuration. The code selects a **class of motor
strategy**, and the proprioceptive context resolves the specific output
within that class.

### Configuration

| Parameter | Value |
|-----------|-------|
| Codebook size ($K$) | 64 |
| Latent dimension ($D$) | 60 |
| Encoder architecture | MLP (1024, 1024) + LayerNorm |
| Decoder architecture | MLP (1024, 1024) + LayerNorm |
| Commitment cost ($\beta$) | 0.25 |
| Stickiness bias | Configurable (temporal persistence) |
| Codebook utilization | ~100% |

---

## 2. Observed Phenomenon: Code Redundancy

With $K = 64$ codes and ~100% utilization (no codebook collapse), we observe
that **multiple codes appear to encode similar motor intentions**. Concretely:

- Codes $i$ and $j$ may activate in similar proprioceptive contexts
- The transition distributions from codes $i$ and $j$ may overlap significantly
- When we replace code $i$ with code $j$ mid-trajectory, the resulting
  behavior is qualitatively similar

This is **code redundancy** — the codebook has allocated multiple embedding
vectors to represent what is functionally the same motor strategy, wasting
representational capacity.

```
     Latent Space (2D projection)
     ┌────────────────────────────────────┐
     │                                    │
     │    ●₁₂  ●₃₇                       │   Codes 12 and 37:
     │     ╲  ╱                           │   similar embeddings,
     │      ╲╱     Voronoi               │   similar decoder output,
     │      ╱╲     boundary               │   similar transitions
     │     ╱  ╲                           │
     │                                    │
     │                   ●₅              │   Code 5: clearly
     │                                    │   distinct function
     │         ●₂₁                       │
     │                                    │
     │              ●₄₈  ●₅₃            │   Codes 48 and 53:
     │               ╲  ╱                 │   another redundant
     │                ╲╱                  │   pair
     │                                    │
     └────────────────────────────────────┘
```

### Root Cause: The STE Gradient Problem

The straight-through estimator (STE) copies gradients from the decoder
output directly to the encoder output, **ignoring the geometry** of where
the encoder embedding $\mathbf{z}_e$ sits within its assigned Voronoi cell.

Every $\mathbf{z}_e$ that maps to the same code $k$ receives **identical
gradient information**, regardless of whether $\mathbf{z}_e$ is near the
center of the cell, near the boundary, or close to a neighboring code's
embedding. This creates two problems:

1. **Codes that start similar stay similar:** During training, if two codebook
   vectors happen to initialize near each other, they receive similar gradient
   updates (since their Voronoi cells contain similar encoder outputs) and
   fail to differentiate.

2. **No pressure to spread:** The STE provides no mechanism for codes to
   "repel" each other. The commitment loss pulls $\mathbf{z}_e$ toward the
   nearest code, and the codebook loss pulls codes toward their assigned
   $\mathbf{z}_e$ values, but neither creates explicit separation between
   code embeddings.

```
     STE Gradient Flow (identity)
     ┌──────────────────────────────────────────┐
     │                                          │
     │   Voronoi cell for code eₖ               │
     │   ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐            │
     │   │  ×₁ ──▶ ∇               │            │
     │   │      ×₂ ──▶ ∇           │  Same ∇    │
     │   │  ×₃ ──▶ ∇     ● eₖ     │  for all   │
     │   │            ×₄ ──▶ ∇     │  inputs    │
     │   │  ×₅ ──▶ ∇               │            │
     │   └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘            │
     │                                          │
     │   × = encoder outputs (zₑ)               │
     │   ∇ = gradient (all identical via STE)    │
     │                                          │
     └──────────────────────────────────────────┘
```

---

## 3. Why Superposition Is Not the Problem

In neuroscience, **superposition** refers to a single code representing
multiple unrelated functions depending on context. Our state-dependent
coding — where the same code produces different motor outputs depending on
proprioceptive state — might appear superficially like superposition, but
it is actually a desirable and efficient design.

### State-dependent coding is a feature

The decoder architecture `Decoder([z_q, proprioception])` is **designed** for
state-dependent coding. The code selects a motor strategy class, and the
proprioceptive context disambiguates:

```
   Same code k=7, different proprioceptive states:
   ┌──────────────────┐        ┌──────────────────┐
   │ Code 7           │        │ Code 7           │
   │ + standing pose  │        │ + crouching pose │
   │ ═══════════════  │        │ ═══════════════  │
   │ → extend legs    │        │ → coil to jump   │
   │   to walk        │        │                  │
   └──────────────────┘        └──────────────────┘
          Different outputs, same code — this is fine.
          The code means "locomote forward"; the body
          state determines what that looks like.
```

This is analogous to a **population code** in neuroscience: a compact
vocabulary of intentions that, combined with sensory state, generates the
full repertoire of motor behavior.

### The actual problem is redundancy, not superposition

The issue is not that codes are overloaded (superposition) but that
**multiple codes are underloaded** — they represent the same intention with
slightly different embeddings, reducing the effective vocabulary size below
the nominal $K = 64$.

| Property | Superposition | Code Redundancy (Our Issue) |
|----------|---------------|----------------------------|
| Utilization | Low (few codes, many meanings) | High (~100%) |
| Codes per function | 1 code → many functions | Many codes → 1 function |
| Cause | Insufficient capacity | STE gradient uniformity |
| Fix needed | More codes or higher dimension | Better gradient differentiation |

---

## 4. Background: Standard VQ-VAE (Our Implementation)

### 4.1 Encoder

The encoder maps reference trajectory observations to a continuous latent
embedding:

$$\mathbf{z}_e = f_\phi(\mathbf{x}_{\text{traj}}) \in \mathbb{R}^D$$

where $f_\phi$ is an MLP with layer normalization and SiLU activations. In
our implementation, $D = 60$ and the MLP has hidden layers of size
$(1024, 1024)$.

### 4.2 Codebook and Nearest-Neighbor Lookup

The codebook $\mathcal{C} = \{\mathbf{e}_1, \ldots, \mathbf{e}_K\} \subset \mathbb{R}^D$ contains $K = 64$ learned embedding vectors. Quantization selects
the nearest codebook entry:

$$k^* = \arg\min_{k \in \{1, \ldots, K\}} \|\mathbf{z}_e - \mathbf{e}_k\|_2^2$$

$$\mathbf{z}_q = \mathbf{e}_{k^*}$$

**Implementation** (`VectorQuantizer.__call__`, lines 130–188 of
`vq_intention_network.py`):

Distances are computed via the expansion
$\|\mathbf{z}_e - \mathbf{e}_k\|^2 = \|\mathbf{z}_e\|^2 + \|\mathbf{e}_k\|^2 - 2\,\mathbf{z}_e^\top \mathbf{e}_k$
for efficiency.

### 4.3 Straight-Through Estimator (STE)

The $\arg\min$ operation has zero gradients almost everywhere. The STE
bypasses this by copying decoder gradients directly to the encoder:

$$\mathbf{z}_{q}^{\text{st}} = \mathbf{z}_e - \texttt{sg}(\mathbf{z}_e) + \texttt{sg}(\mathbf{z}_q)$$

where $\texttt{sg}(\cdot)$ denotes `stop_gradient`. In the forward pass,
$\mathbf{z}_{q}^{\text{st}} = \mathbf{z}_q$. In the backward pass:

$$\frac{\partial \mathbf{z}_{q}^{\text{st}}}{\partial \mathbf{z}_e} = \mathbf{I}$$

The gradient is the **identity matrix** — every encoder output within the
same Voronoi cell receives the same gradient, regardless of its position.

### 4.4 Decoder

The decoder generates action distribution parameters conditioned on the
quantized latent and proprioceptive state:

$$(\boldsymbol{\mu}, \log \boldsymbol{\sigma}) = g_\psi([\mathbf{z}_{q}^{\text{st}}, \mathbf{o}_{\text{proprio}}])$$

where $[\cdot, \cdot]$ denotes concatenation. Actions are sampled from a
diagonal Gaussian $\mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$.

### 4.5 Training Loss

Our total loss combines PPO with VQ-VAE auxiliary terms:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{PPO}}}_{\text{policy}} + \underbrace{\mathcal{L}_{\text{value}}}_{\text{critic}} + \underbrace{\mathcal{L}_{\text{entropy}}}_{\text{exploration}} + \underbrace{\mathcal{L}_{\text{VQ}}}_{\text{quantization}}$$

The VQ loss consists of two terms:

$$\mathcal{L}_{\text{VQ}} = \underbrace{\|\texttt{sg}(\mathbf{z}_e) - \mathbf{z}_q\|_2^2}_{\text{codebook loss}} + \underbrace{\beta \|\mathbf{z}_e - \texttt{sg}(\mathbf{z}_q)\|_2^2}_{\text{commitment loss}}$$

- **Codebook loss:** Moves codebook vectors toward encoder outputs.
  Gradients flow to the codebook only.
- **Commitment loss:** Moves encoder outputs toward their assigned codebook
  vectors. Gradients flow to the encoder only. Weighted by $\beta = 0.25$.

**Implementation** (`compute_vq_loss`, lines 104–138 of `vq_losses.py`).

### 4.6 Stickiness Bias (Temporal Persistence)

To prevent rapid code switching, our system optionally applies a distance
bias that favors re-selecting the previous timestep's code:

$$d'_k = d_k - b \cdot \mathbb{1}[k = k_{t-1}]$$

where $b > 0$ is the stickiness bias and $d_k = \|\mathbf{z}_e - \mathbf{e}_k\|^2$. This creates hysteresis — the current code must be
significantly closer than the previous code to trigger a switch.

**Implementation** (`VectorQuantizer.__call__`, lines 162–171 and
`forward_temporal`, lines 361–470 of `vq_intention_network.py`).

### 4.7 Architecture Diagram

```
                    Our VQ-VAE Pipeline
    ════════════════════════════════════════════════════

    Trajectory Obs                 Proprioceptive Obs
         │                               │
         ▼                               │
    ┌─────────┐                          │
    │ Encoder │   MLP(1024,1024) + LN    │
    │  f_φ    │                          │
    └────┬────┘                          │
         │ zₑ ∈ ℝ⁶⁰                     │
         ▼                               │
    ┌──────────────────────┐             │
    │   Vector Quantizer   │             │
    │                      │             │
    │  k* = argmin ‖zₑ-eₖ‖²            │
    │  zq = e_{k*}         │             │
    │                      │             │
    │  STE: zq_st = zₑ     │             │
    │    - sg(zₑ) + sg(zq) │             │
    │                      │             │
    │  Codebook: 64 × 60   │             │
    └────────┬─────────────┘             │
             │ zq_st                     │
             ▼                           ▼
         ┌───────────────────────────────────┐
         │          Decoder  g_ψ             │
         │   Input: [zq_st, proprio]         │
         │   MLP(1024,1024) + LN             │
         └───────────────┬───────────────────┘
                         │
                         ▼
                  Action (μ, log σ)
```

---

## 5. STAR: Rotation-Augmented Residual Vector Quantization

STAR ([Li et al., 2025](#ref-star)) introduces two mechanisms to improve
VQ for skill learning:

1. **Residual Vector Quantization (RVQ):** A hierarchy of codebooks where
   each level quantizes the residual error from the previous level.
2. **Rotation-Augmented STE (RaRSQ):** Replaces the identity-gradient STE
   with a rotation matrix that preserves angular relationships between
   encoder outputs and their assigned codebook vectors.

### 5.1 Residual Vector Quantization (RVQ)

RVQ uses $D$ codebooks $\mathcal{C}_1, \ldots, \mathcal{C}_D$, each with
$K$ vectors. Starting from the encoder output $\mathbf{r}_0 = \mathbf{z}_e$,
each depth quantizes the **residual** left over from the previous depth.

**Depth $d = 1, \ldots, D$:**

$$k_d = \arg\min_{k \in \{1, \ldots, K\}} \|\mathbf{r}_{d-1} - \mathbf{e}_{d,k}\|_2^2 \tag{1}$$

$$\mathbf{r}_d = \mathbf{r}_{d-1} - \mathbf{e}_{d,k_d} \tag{2}$$

**Reconstructed quantized representation:**

$$\hat{\mathbf{z}} = \sum_{d=1}^{D} \mathbf{e}_{d,k_d} \tag{3}$$

The final code is a **tuple** $(k_1, k_2, \ldots, k_D)$, giving $K^D$
effective combinations from only $K \times D$ codebook vectors.

```
    Residual Vector Quantization (D=2)
    ══════════════════════════════════════════════════

    Encoder output: zₑ = r₀
         │
         ▼
    ┌──────────────────┐
    │  Codebook C₁     │   Depth 1 (coarse)
    │  k₁ = argmin     │
    │  ‖r₀ - e₁,ₖ‖²   │
    └────────┬─────────┘
             │  e_{1,k₁}       "trotting forward"
             ▼
         r₁ = r₀ - e_{1,k₁}    (residual error)
             │
             ▼
    ┌──────────────────┐
    │  Codebook C₂     │   Depth 2 (fine correction)
    │  k₂ = argmin     │
    │  ‖r₁ - e₂,ₖ‖²   │
    └────────┬─────────┘
             │  e_{2,k₂}       "left-leg-leads phase"
             ▼
         r₂ = r₁ - e_{2,k₂}    (remaining error)

    Final:  ẑ = e_{1,k₁} + e_{2,k₂}

    Effective codes: K² = 64² = 4096 combinations
    Stored vectors:  K×D = 64×2 = 128 vectors
```

**Analogy:** Think of decimal number representation. Instead of having one
codebook with 1000 entries (0.000 to 0.999), RVQ uses two codebooks of 10
entries each:
- Depth 1 captures the tenths digit: {0.0, 0.1, ..., 0.9}
- Depth 2 captures the hundredths digit: {0.00, 0.01, ..., 0.09}

The combination 0.3 + 0.07 = 0.37 uses only 20 entries to represent 100 values.

**Concrete example (rodent locomotion):**

- **Depth 1 (codebook 1):** Learns the broadest strokes. For our rodent, this might be "running forward" vs "rearing up" vs "turning left" vs "grooming". The $K$ codes tile the coarse behavior space.
- **Depth 2 (codebook 2):** Learns to correct the **error** of depth 1. It never sees the original $\mathbf{z}$ — it only sees $\mathbf{r}_1 = \mathbf{z} - \mathbf{e}_{1,k_1}$, which is "everything depth 1 got wrong." So it learns variations *within* a coarse category: "running forward **with slight left drift**" vs "running forward **with head down**".

The residual $\mathbf{r}_1$ is typically much smaller in magnitude than $\mathbf{z}$, so depth 2's codebook vectors are also smaller — they live in the space of corrections, not the space of full behaviors.

> *Note: when quantizing the mistakes, we are effectively doing a factorization, similar to the QueST paper.*

### 5.2 The Rotation Trick (Fifty et al., 2024)

Before describing STAR's full method, we review the rotation trick that it
builds upon. The core idea: instead of using the identity for the STE
backward pass, use a **rotation matrix** that maps $\mathbf{z}_e$ onto the
selected codebook vector $\mathbf{e}_k$.

For a single quantization level, the rotation-augmented quantization is:

$$\tilde{\mathbf{q}} = \frac{\|\mathbf{e}_k\|}{\|\mathbf{z}_e\|} \cdot \mathbf{R} \cdot \mathbf{z}_e \tag{4}$$

where $\mathbf{R}$ is an orthogonal matrix satisfying $\mathbf{R} \cdot \hat{\mathbf{z}}_e = \hat{\mathbf{e}}_k$ (hats denote unit vectors). In the
forward pass, $\tilde{\mathbf{q}} = \mathbf{e}_k$ exactly. In the backward
pass, the gradient becomes:

$$\frac{\partial \tilde{\mathbf{q}}}{\partial \mathbf{z}_e} = \frac{\|\mathbf{e}_k\|}{\|\mathbf{z}_e\|} \cdot \mathbf{R} \neq \mathbf{I}$$

This is a **scaled rotation** — not the identity. Encoder outputs at
different angles relative to their codebook vector receive **different
gradient directions**, even within the same Voronoi cell.

```
    Gradient Comparison: STE vs Rotation Trick
    ══════════════════════════════════════════════════

         Standard STE                  Rotation-Augmented STE
    ┌─────────────────────┐      ┌─────────────────────┐
    │                     │      │                     │
    │  ×₁ ──▶ ∇           │      │  ×₁ ──▶ ∇₁  ╲      │
    │      ×₂ ──▶ ∇       │      │      ×₂ ──▶ ∇₂  ╲  │
    │           ● eₖ      │      │           ● eₖ     │
    │  ×₃ ──▶ ∇           │      │  ×₃ ──▶ ∇₃  ╱      │
    │      ×₄ ──▶ ∇       │      │      ×₄ ──▶ ∇₄  ╱  │
    │                     │      │                     │
    │  All gradients ∇    │      │  ∇₁ ≠ ∇₂ ≠ ∇₃ ≠ ∇₄ │
    │  are IDENTICAL      │      │  Direction-dependent │
    └─────────────────────┘      └─────────────────────┘

    × = encoder outputs (zₑ)
    ∇ = gradient received during backpropagation
```

### 5.3 Rotation Matrix Construction

#### Why do we need a rotation at all?

Recall the STE problem: the encoder produces $\mathbf{z}_e$, but the
decoder receives $\mathbf{e}_k$ (the nearest codebook vector). These two
vectors point in **different directions** — $\mathbf{z}_e$ is wherever the
encoder put it, while $\mathbf{e}_k$ is the closest codebook entry. The STE
ignores this mismatch and passes gradients through as if
$\mathbf{z}_e = \mathbf{e}_k$.

The rotation trick says: instead of pretending they're the same, let's
**acknowledge the angular mismatch** and build it into the gradient. We
construct a rotation matrix $\mathbf{R}$ that maps the direction of
$\mathbf{z}_e$ onto the direction of $\mathbf{e}_k$:

$$\mathbf{R} \cdot \hat{\mathbf{r}} = \hat{\mathbf{q}}$$

where $\hat{\mathbf{r}} = \mathbf{z}_e / \|\mathbf{z}_e\|$ and
$\hat{\mathbf{q}} = \mathbf{e}_k / \|\mathbf{e}_k\|$.

```
    The mismatch that R corrects
    ══════════════════════════════════════════════

                 q̂ (codebook direction)
                ╱
               ╱  ← R rotates r̂ to here
              ╱ θ
             ╱
            ○──────── r̂ (encoder output direction)

    θ = angle between encoder output and its
        assigned codebook vector.

    STE pretends θ = 0 (ignores the mismatch).
    Rotation trick encodes θ into the gradient.
```

**Why this matters:** Two encoder outputs $\mathbf{z}_e^{(a)}$ and
$\mathbf{z}_e^{(b)}$ that map to the same code $k$ but approach it from
different angles have different $\theta$ values, so they get **different**
rotation matrices and therefore **different gradients**. Under STE, they
would get the identical gradient $\mathbf{I}$.

#### What does $\mathbf{R}$ look like?

In RVQ, the rotation at depth $d$ maps the residual direction onto the
codebook direction. The inputs are:

$$\hat{\mathbf{q}}_d = \frac{\mathbf{e}_{d,k_d}}{\|\mathbf{e}_{d,k_d}\|}, \qquad \hat{\mathbf{r}}_{d-1} = \frac{\mathbf{r}_{d-1}}{\|\mathbf{r}_{d-1}\|} \tag{6}$$

The rotation matrix is constructed via a Householder-like composition:

$$\hat{\mathbf{m}}_d = \frac{\hat{\mathbf{r}}_{d-1} + \hat{\mathbf{q}}_d}{\|\hat{\mathbf{r}}_{d-1} + \hat{\mathbf{q}}_d\|} \quad \text{(bisector direction)} \tag{7}$$

$$\mathbf{R}_d = \mathbf{I} - 2\,\hat{\mathbf{m}}_d\,\hat{\mathbf{m}}_d^\top + 2\,\hat{\mathbf{q}}_d\,\hat{\mathbf{r}}_{d-1}^\top \tag{5}$$

The construction details (Householder reflection + rank-1 sign correction)
are standard linear algebra — what matters for intuition is:

- $\mathbf{R}_d$ is an **orthogonal matrix** (preserves lengths and angles)
- It satisfies $\mathbf{R}_d \hat{\mathbf{r}}_{d-1} = \hat{\mathbf{q}}_d$
  (rotates residual direction onto codebook direction)
- It is **input-dependent**: different $\hat{\mathbf{r}}_{d-1}$ values
  produce different $\mathbf{R}_d$ matrices, which is the whole point —
  different encoder outputs get different gradients

#### Why "residual direction" and not "encoder direction"?

In standard (single-level) VQ, the rotation maps the encoder output
direction $\hat{\mathbf{z}}_e$ onto the codebook direction
$\hat{\mathbf{e}}_k$. In RVQ, depth $d$ doesn't see the original encoder
output — it sees the **residual** $\mathbf{r}_{d-1}$ (what previous depths
got wrong). So the rotation at depth $d$ must align the residual direction
with the depth-$d$ codebook vector direction.

This is natural: each depth's "input" is the residual, so the angular
mismatch that matters is between the residual and the codebook entry chosen
to approximate it.

```
    Rotation at each depth of RVQ
    ══════════════════════════════════════════════

    Depth 1:  R₁ rotates  r̂₀ = ẑₑ       →  q̂₁ = ê₁,ₖ₁
              (encoder direction → coarse codebook direction)

    Depth 2:  R₂ rotates  r̂₁ = r̂esidual  →  q̂₂ = ê₂,ₖ₂
              (error direction → fine codebook direction)

    Each depth has its own angular mismatch,
    its own rotation matrix, and therefore its
    own direction-dependent gradient.
```

### 5.4 STAR's RaRSQ: Full Forward Pass

STAR combines RVQ with the rotation trick at each depth. The complete
forward pass for $D$ depths:

**Initialize:** $\mathbf{r}_0 = \mathbf{z}_e = f_\phi(\mathbf{x})$

**For $d = 1, \ldots, D$:**

$$k_d = \arg\min_{k \in \{1, \ldots, K\}} \|\mathbf{r}_{d-1} - \mathbf{e}_{d,k}\|_2^2 \tag{8}$$

$$\mathbf{R}_d = \text{ComputeRotation}(\mathbf{r}_{d-1}, \mathbf{e}_{d,k_d}) \tag{9}$$

$$\tilde{\mathbf{q}}_d = \texttt{sg}\!\left[\frac{\|\mathbf{e}_{d,k_d}\|}{\|\mathbf{r}_{d-1}\|} \cdot \mathbf{R}_d\right] \cdot \mathbf{r}_{d-1} \tag{10}$$

$$\mathbf{r}_d = \mathbf{r}_{d-1} - \tilde{\mathbf{q}}_d \tag{11}$$

**Reconstructed skill representation:**

$$\hat{\mathbf{z}} = \sum_{d=1}^{D} \tilde{\mathbf{q}}_d \tag{12}$$

**Key detail in Eq. 10:** The `stop_gradient` wraps the scaling factor and
rotation matrix, so in the backward pass, gradients flow through
$\mathbf{r}_{d-1}$ but the rotation/scaling are treated as constants. The
resulting gradient at depth $d$ is:

$$\frac{\partial \hat{\mathbf{z}}}{\partial \mathbf{r}_{d-1}} = \frac{\|\mathbf{e}_{d,k_d}\|}{\|\mathbf{r}_{d-1}\|} \cdot \mathbf{R}_d \tag{13}$$

This is a **scaled rotation** — not the identity. Encoder outputs that differ
in angle receive different gradient updates.

### 5.5 STAR's Training Loss

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{commit}} \tag{14}$$

**Reconstruction loss:**

$$\mathcal{L}_{\text{recon}} = \|\mathbf{a}_{t:t+T} - \psi(\hat{\mathbf{z}})\|_2^2 \tag{15}$$

where $\mathbf{a}_{t:t+T}$ is a ground-truth action sequence of length $T = 8$ and $\psi$ is the decoder.

**Rotation-augmented commitment loss:**

$$\mathcal{L}_{\text{commit}} = \beta \sum_{d=1}^{D} \left\| \texttt{sg}(\mathbf{r}_{d-1}) - \frac{\|\mathbf{e}_{d,k_d}\|}{\|\mathbf{r}_{d-1}\|} \cdot \mathbf{R}_d \cdot \mathbf{r}_{d-1} \right\|_2^2 \tag{16}$$

Note: STAR does **not** use a separate codebook loss term or EMA updates.
The codebooks are updated through gradient descent on the commitment loss.

### 5.6 STAR Architecture Diagram

```
    STAR: RaRSQ Forward Pass (D=2, K=16)
    ══════════════════════════════════════════════════════

    Action sequence a_{t:t+8}
         │
         ▼
    ┌─────────┐
    │ Encoder │   Single-layer MLP, dim=128
    │  f_φ    │
    └────┬────┘
         │ zₑ = r₀
         │
    ═════╪═══════════════════════════════════════ Depth 1
         │
         ▼
    ┌──────────────────────────────┐
    │  Codebook C₁  (K=16)        │
    │  k₁ = argmin ‖r₀ - e₁,ₖ‖²  │
    │                              │
    │  R₁ = Rotation(r₀, e₁,ₖ₁)  │
    │  q̃₁ = sg[‖e₁,ₖ₁‖/‖r₀‖·R₁]·r₀ │
    └────────┬─────────────────────┘
             │ q̃₁         (coarse skill)
             │
         r₁ = r₀ - q̃₁     (residual error)
             │
    ═════════╪═══════════════════════════════════ Depth 2
             │
             ▼
    ┌──────────────────────────────┐
    │  Codebook C₂  (K=16)        │
    │  k₂ = argmin ‖r₁ - e₂,ₖ‖²  │
    │                              │
    │  R₂ = Rotation(r₁, e₂,ₖ₂)  │
    │  q̃₂ = sg[‖e₂,ₖ₂‖/‖r₁‖·R₂]·r₁ │
    └────────┬─────────────────────┘
             │ q̃₂         (fine correction)
             │
    ═════════╪═══════════════════════════════════ Combine
             │
         ẑ = q̃₁ + q̃₂     (full skill code)
             │
             ▼
    ┌──────────────────┐
    │  Decoder  ψ      │   Transformer: 4 heads, 4 layers
    │  â = ψ(ẑ)       │   dim=128
    └────────┬─────────┘
             │
             ▼
       Reconstructed actions â_{t:t+8}
```

### 5.7 STAR Hyperparameters

| Parameter | STAR Value | Our Value |
|-----------|-----------|-----------|
| Codebook size ($K$) | 16 | 64 |
| Depth ($D$) | 2 | 1 |
| Effective combinations | $16^2 = 256$ | 64 |
| Stored vectors | $16 \times 2 = 32$ | 64 |
| Latent dimension | 128 | 60 |
| Encoder | 1-layer MLP | 2-layer MLP + LN |
| Decoder | Transformer (4h, 4L) | MLP + LN |
| STE type | Rotation-augmented | Standard identity |
| EMA updates | No | No |

---

## 6. Side-by-Side Comparison: Our VQ-VAE vs STAR

### 6.1 Quantization

| Aspect | Our VQ-VAE | STAR (RaRSQ) |
|--------|-----------|--------------|
| Codebooks | 1 | $D$ (typically 2) |
| Code representation | Single index $k \in \{1,\ldots,K\}$ | Tuple $(k_1, \ldots, k_D)$ |
| Quantized output | $\mathbf{z}_q = \mathbf{e}_{k^*}$ | $\hat{\mathbf{z}} = \sum_d \tilde{\mathbf{q}}_d$ |
| Backward gradient | $\mathbf{I}$ (identity) | $\frac{\|\mathbf{e}_{d,k_d}\|}{\|\mathbf{r}_{d-1}\|} \cdot \mathbf{R}_d$ (scaled rotation) |
| Effective capacity | $K$ | $K^D$ |

### 6.2 Loss Functions

**Our system:**

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{entropy}} + \underbrace{\beta\|\mathbf{z}_e - \texttt{sg}(\mathbf{z}_q)\|^2 + \|\texttt{sg}(\mathbf{z}_e) - \mathbf{z}_q\|^2}_{\mathcal{L}_{\text{VQ}}}$$

**STAR (RVQ encoder-decoder, not RL):**

$$\mathcal{L} = \|\mathbf{a} - \psi(\hat{\mathbf{z}})\|^2 + \beta \sum_{d=1}^{D} \|\texttt{sg}(\mathbf{r}_{d-1}) - \tilde{\mathbf{q}}_d\|^2$$

**Key difference:** Our system trains end-to-end with RL (PPO), where the
policy loss provides the reconstruction signal. STAR uses direct behavior
cloning with an explicit reconstruction loss on action sequences.

### 6.3 Decoder Conditioning

| | Our VQ-VAE | STAR |
|--|-----------|------|
| **Input** | $[\mathbf{z}_q, \mathbf{o}_{\text{proprio}}]$ | $\hat{\mathbf{z}}$ only |
| **State-dependence** | Yes — same code, different behavior per state | No — code fully determines action |
| **Implication** | Codes are motor strategy *classes* | Codes are motor strategy *instances* |

This is a fundamental architectural difference. Our decoder's state
conditioning allows 64 codes to cover a large behavioral space by leveraging
proprioceptive context. STAR's decoder produces a fixed action sequence per
code tuple, requiring $K^D$ capacity to cover the same space.

### 6.4 Training Paradigm

```
    Our System                         STAR
    ══════════                         ════

    ┌──────────────────┐               ┌──────────────────┐
    │  Environment     │               │  Expert Demos    │
    │  (MuJoCo MJX)    │               │  (offline data)  │
    └────────┬─────────┘               └────────┬─────────┘
             │ rewards                          │ action sequences
             ▼                                  ▼
    ┌──────────────────┐               ┌──────────────────┐
    │  PPO + VQ Loss   │               │ Reconstruction + │
    │  (online RL)     │               │ Commitment Loss   │
    │                  │               │ (supervised)       │
    └──────────────────┘               └──────────────────┘
```

---

## 7. How STAR Would Address Code Redundancy

### 7.1 Mechanism 1: Rotation-Augmented Gradients

The rotation trick directly attacks the gradient uniformity problem that
causes code redundancy.

**Current problem:** When codes $\mathbf{e}_i$ and $\mathbf{e}_j$ have
similar embeddings, their Voronoi cells contain encoder outputs at similar
positions. Under STE, these outputs receive identical gradients $\mathbf{I}$,
so the codebook updates for $\mathbf{e}_i$ and $\mathbf{e}_j$ are also
similar. The codes drift together during training rather than
differentiating.

**With rotation:** Two encoder outputs $\mathbf{z}_e^{(a)}$ and $\mathbf{z}_e^{(b)}$ that map to the same code $k$ but sit at different angles
relative to $\mathbf{e}_k$ receive **different** gradient directions:

$$\nabla_a = \frac{\|\mathbf{e}_k\|}{\|\mathbf{z}_e^{(a)}\|} \cdot \mathbf{R}^{(a)}, \qquad \nabla_b = \frac{\|\mathbf{e}_k\|}{\|\mathbf{z}_e^{(b)}\|} \cdot \mathbf{R}^{(b)}$$

Since $\mathbf{R}^{(a)} \neq \mathbf{R}^{(b)}$ (different rotation angles),
the encoder receives position-aware feedback. This creates pressure for
encoder outputs near Voronoi boundaries to either commit more strongly to
their current code or shift toward a neighboring code — **actively
differentiating** codes that would otherwise remain redundant.

```
    Effect on Redundant Codes Over Training
    ══════════════════════════════════════════════════

    Before (STE):                 After (Rotation):
    ┌──────────────────────┐      ┌──────────────────────┐
    │                      │      │                      │
    │   ●ᵢ    ●ⱼ          │      │   ●ᵢ                │
    │   similar embeddings │      │         ●ⱼ          │
    │   similar gradients  │      │                      │
    │   similar updates    │      │   differentiated!    │
    │   → stay similar     │      │   → capture different│
    │                      │      │     sub-behaviors    │
    └──────────────────────┘      └──────────────────────┘
```

### 7.2 Mechanism 2: Residual Hierarchy

The residual decomposition structurally **eliminates the need** for
redundant codes at any single level.

**Current problem:** With a flat codebook of $K = 64$, codes must jointly
capture both coarse distinctions (trotting vs rearing vs grooming) and fine
distinctions (left-leg-leads trot vs right-leg-leads trot). When fine
distinctions exist within a coarse category, multiple codes end up allocated
to the same coarse behavior — they are "redundant" because they differ only
in the fine detail.

**With RVQ ($D = 2$, $K = 64$):**

- **Depth 1 (coarse):** Quantizes $\mathbf{r}_0 = \mathbf{z}_e$. These 64
  codes capture broad motor strategy distinctions. "Left-leg trot" and
  "right-leg trot" both map to the **same** depth-1 code (e.g.,
  "trotting forward"). No redundancy at this level.

- **Depth 2 (fine correction):** Quantizes
  $\mathbf{r}_1 = \mathbf{z}_e - \mathbf{e}_{1,k_1}$, the residual error.
  These codes tile a **much smaller** region of latent space (the correction
  space) and can resolve fine distinctions efficiently.

```
    Flat Codebook (current):       Residual Codebook (RVQ):
    ══════════════════════════     ════════════════════════════

    Code 12: trot, left-lead       Depth 1, Code 8: "trotting"
    Code 37: trot, right-lead      Depth 2, Code 3: "left-lead"
    Code 41: trot, slight-left     Depth 2, Code 19: "right-lead"
    Code 55: trot, symmetric       Depth 2, Code 7: "slight-left"
                                   Depth 2, Code 12: "symmetric"
    4 codes for one coarse
    behavior (redundancy!)         1 coarse code + 4 fine codes
                                   (no redundancy at either level)
```

**Capacity comparison:**

| | Flat ($K = 64$) | RVQ ($D = 2$, $K = 64$) |
|--|-----------------|------------------------|
| Stored vectors | 64 | 128 |
| Effective combinations | 64 | 4,096 |
| Coarse categories | ~16 (with 4x avg redundancy) | 64 (no redundancy) |
| Fine distinctions per category | 0 (absorbed into coarse) | 64 |

### 7.3 How the Two Mechanisms Work Together

The rotation trick and residual hierarchy are **complementary**:

| Mechanism | What it solves | How |
|-----------|---------------|-----|
| **Rotation** (optimization fix) | Codes that *are* similar get pushed apart during training | Direction-dependent gradients break the STE uniformity that lets similar codes persist |
| **Residual hierarchy** (architectural fix) | Eliminates the *need* for similar codes at any single level | Coarse/fine factorization means each level has a distinct, well-scoped role |

Without rotation, even RVQ can suffer from within-level redundancy (the STE
still gives identical gradients at each depth). Without the hierarchy, even
rotation-augmented VQ may waste capacity on fine distinctions that could be
captured more efficiently by a correction codebook.

### 7.4 Considerations for Our System

Adopting STAR's mechanisms would require changes at several levels:

**1. VectorQuantizer → ResidualVectorQuantizer:**
Replace the single codebook with $D$ codebooks and implement the iterative
quantize-subtract loop. The `__call__` method would iterate over depths,
computing residuals at each level.

**2. STE → Rotation-Augmented STE:**
Replace `z_q_st = z_e - sg(z_e) + sg(z_q)` with the rotation construction
(Eqs. 5–7) and the scaled rotation quantization (Eq. 10).

**3. Loss Function:**
Replace `compute_vq_loss` with the rotation-augmented commitment loss
(Eq. 16), summed over depths. The codebook loss term may be dropped (STAR
does not use it).

**4. Code Representation:**
Analysis pipelines would need to handle code tuples $(k_1, \ldots, k_D)$
instead of single indices. Transition matrices become higher-dimensional,
and community detection operates on composite code sequences.

**5. Stickiness Bias:**
The temporal persistence mechanism would need to operate on the composite
code tuple. Options include: bias at depth 1 only (coarse persistence),
independent bias per depth, or bias on the concatenated code tuple.

**6. Decoder Conditioning:**
Our state-dependent decoder `[z_q, proprio]` remains unchanged — the
quantized output $\hat{\mathbf{z}} = \sum_d \tilde{\mathbf{q}}_d$ replaces
the single-level $\mathbf{z}_q$ as input.

---

## 8. References

<a id="ref-vqvae"></a>
**[1]** van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
*Neural Discrete Representation Learning.* NeurIPS 2017.
[arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

<a id="ref-vqvae2"></a>
**[2]** Razavi, A., van den Oord, A., & Vinyals, O. (2019).
*Generating Diverse High-Fidelity Images with VQ-VAE-2.* NeurIPS 2019.
[arXiv:1906.00446](https://arxiv.org/abs/1906.00446)

<a id="ref-soundstream"></a>
**[3]** Zeghidour, N., et al. (2021).
*SoundStream: An End-to-End Neural Audio Codec.* IEEE/ACM TASLP.
[arXiv:2107.03312](https://arxiv.org/abs/2107.03312)

<a id="ref-encodec"></a>
**[4]** D&eacute;fossez, A., et al. (2022).
*High Fidelity Neural Audio Compression.* TMLR 2023.
[arXiv:2210.13438](https://arxiv.org/abs/2210.13438)

<a id="ref-rqvae"></a>
**[5]** Lee, D., et al. (2022).
*Autoregressive Image Generation using Residual Quantization.* CVPR 2022.
[arXiv:2203.01941](https://arxiv.org/abs/2203.01941)

<a id="ref-vqbet"></a>
**[6]** Lee, S., et al. (2024).
*Behavior Generation with Latent Actions.* ICML 2024.
[arXiv:2403.03181](https://arxiv.org/abs/2403.03181)

<a id="ref-rotation"></a>
**[7]** Fifty, C., et al. (2024).
*Restructuring Vector Quantization with the Rotation Trick.* ICML 2024.
[arXiv:2410.06424](https://arxiv.org/abs/2410.06424)

<a id="ref-star"></a>
**[8]** Li, H., et al. (2025).
*STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented
Vector Quantization.* ICML 2025.
[arXiv:2506.03863](https://arxiv.org/abs/2506.03863)
