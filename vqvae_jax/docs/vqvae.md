# VQ-VAE Integration Plan for track-mjx

**Date**: 2026-01-09
**Author**: Investigation for minimal VQ-VAE integration
**Status**: Planning Phase - No Code Yet

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Understanding VQ-VAE: Component-by-Component](#2-understanding-vq-vae-component-by-component)
3. [Current VAE Architecture Analysis](#3-current-vae-architecture-analysis)
4. [Integration Strategy: Minimal Changes Philosophy](#4-integration-strategy-minimal-changes-philosophy)
5. [JAX-Specific Challenges and Solutions](#5-jax-specific-challenges-and-solutions)
6. [Detailed Integration Plan](#6-detailed-integration-plan)
7. [Data Flow Analysis](#7-data-flow-analysis)
8. [Shape Management Strategy](#8-shape-management-strategy)
9. [Loss Function Design](#9-loss-function-design)
10. [Decision Justifications](#10-decision-justifications)
11. [Risk Assessment](#11-risk-assessment)
12. [Implementation Checklist](#12-implementation-checklist)
13. [References](#13-references)

---

## 1. Executive Summary

### Goal

Integrate Vector Quantized Variational Autoencoder (VQ-VAE) into the track-mjx motion imitation pipeline with **minimal structural changes** to the existing codebase, while respecting JAX's functional programming paradigm.

### Key Insight

**The codebook can be treated as a standard trainable parameter updated via gradients, eliminating the need for complex EMA state management.**

This insight allows us to:
- Keep the training loop nearly identical to the existing VAE implementation
- Avoid complex mutable state handling that caused issues in the previous pilot
- Maintain full compatibility with the existing `pmap`-based multi-device training
- Preserve the clean separation between network, loss, and training logic

### Scope of Changes

| Component | Change Level | Description |
|-----------|--------------|-------------|
| `intention_network.py` | **New file** | VQ encoder + quantizer + decoder |
| `losses.py` | **Minor modification** | Add VQ loss computation alongside existing |
| `ppo_networks.py` | **Minor modification** | Add VQ network factory |
| `ppo.py` | **No changes needed** | Training loop unchanged |
| `train.py` | **Config-driven** | Select VAE vs VQ-VAE via config |

---

## 2. Understanding VQ-VAE: Component-by-Component

### 2.1 What is VQ-VAE?

VQ-VAE (Vector Quantized Variational Autoencoder) replaces the continuous latent space of a standard VAE with a **discrete latent space** defined by a learned codebook. Instead of sampling from a Gaussian distribution, VQ-VAE finds the nearest vector in a finite dictionary.

**Original paper**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (van den Oord et al., 2017)

### 2.2 Component Breakdown

#### 2.2.1 Encoder

**Purpose**: Map input observations to a continuous embedding vector.

```
Input: x ∈ ℝ^d_input
Output: z_e ∈ ℝ^d_latent (continuous embedding)
```

**Key difference from VAE encoder**:
- VAE encoder outputs `(mean, logvar)` - two vectors parameterizing a Gaussian
- VQ encoder outputs `z_e` - a single continuous embedding vector

**Why this matters**: VQ-VAE doesn't need the reparameterization trick because there's no sampling from a distribution. The quantization itself provides the discrete bottleneck.

#### 2.2.2 Codebook (Embedding Dictionary)

**Purpose**: A learnable lookup table of K prototype vectors.

```
Codebook: E = {e_1, e_2, ..., e_K} where e_k ∈ ℝ^d_latent
Shape: [num_codes, latent_dim]
```

**Key properties**:
- Each entry represents a "motor primitive" or "intention prototype"
- Finite vocabulary enables discrete, interpretable representations
- K typically ranges from 128 to 2048 (we use 512 as default)

**Why this matters for motor control**: Each codebook entry can be interpreted as a learned motor pattern. The policy learns to map reference trajectories to one of K discrete motor intentions.

#### 2.2.3 Quantization Operation

**Purpose**: Map continuous encoder output to nearest codebook entry.

```
k* = argmin_k ||z_e - e_k||²    (find nearest neighbor)
z_q = e_{k*}                      (quantized output)
```

**The non-differentiability problem**: `argmin` has zero gradients almost everywhere. We cannot backpropagate through this discrete selection.

**Solution**: Straight-Through Estimator (see 2.2.4)

#### 2.2.4 Straight-Through Estimator (STE)

**Purpose**: Enable gradient flow through the non-differentiable quantization.

**Mechanism**:
- **Forward pass**: Use quantized value `z_q`
- **Backward pass**: Copy gradients from `z_q` directly to `z_e`

**JAX implementation** (numerically stable Sterbenz lemma pattern):
```python
def straight_through(z_e, z_q):
    # Forward: returns z_q
    # Backward: gradients flow to z_e
    zero = z_e - jax.lax.stop_gradient(z_e)  # Exactly 0.0 with grad = 1
    return zero + jax.lax.stop_gradient(z_q)  # Value = z_q, grad = 1
```

**Why this pattern works**:
1. `z_e - stop_gradient(z_e)` = 0.0 exactly (Sterbenz lemma guarantees)
2. Gradient of step 1 w.r.t. z_e = 1.0
3. `stop_gradient(z_q)` contributes the value but no gradient
4. Result: value = z_q, gradient flows to z_e

**Why not the simpler pattern** `z_e + stop_gradient(z_q - z_e)`?
- Mathematically equivalent but can accumulate floating-point errors
- The Sterbenz pattern is numerically exact

#### 2.2.5 Decoder

**Purpose**: Map quantized latent (+ proprioceptive state) to action distribution.

```
Input: [z_q, proprio_obs]
Output: action_params (mean, std for Gaussian policy)
```

**Identical to VAE decoder**: The decoder architecture remains unchanged. It receives a latent vector and proprioceptive observations, outputting action distribution parameters.

#### 2.2.6 Loss Functions

VQ-VAE has three loss components (in addition to the task loss):

**1. Reconstruction Loss (Task Loss)**
```
L_task = PPO_loss + Value_loss + Entropy_loss
```
This is unchanged from the VAE - it's our RL objective.

**2. Commitment Loss**
```
L_commit = β × ||z_e - sg[z_q]||²
```
- `sg[]` = stop_gradient
- Forces encoder to "commit" to codebook entries
- Prevents encoder outputs from drifting arbitrarily
- β typically 0.25 (from original paper)

**3. Codebook Loss (Gradient Mode)**
```
L_codebook = ||sg[z_e] - z_q||²
```
- Moves codebook entries toward encoder outputs
- Only used when codebook is updated via gradients (not EMA)

**Total VQ Loss**:
```
L_vq = β × ||z_e - sg[z_q]||² + ||sg[z_e] - z_q||²
```

**Why two separate losses?**
- Commitment loss: Gradients to encoder only (codebook is stopped)
- Codebook loss: Gradients to codebook only (encoder is stopped)
- This bidirectional pressure brings encoder and codebook together

### 2.3 VQ-VAE vs VAE: Side-by-Side Comparison

| Aspect | VAE | VQ-VAE |
|--------|-----|--------|
| Latent type | Continuous (Gaussian) | Discrete (codebook index) |
| Encoder output | (mean, logvar) | z_e |
| Sampling | Reparameterization trick | Nearest neighbor lookup |
| Regularization | KL divergence | Commitment + codebook loss |
| Interpretability | Latent directions | Discrete motor primitives |
| Codebook | None | K learned embeddings |

---

## 3. Current VAE Architecture Analysis

### 3.1 Network Flow (Current VAE)

```
                    CURRENT VAE PIPELINE
                    ====================

Observation [T, B, obs_dim]
         │
         ▼
    ┌─────────────────────────────────────────┐
    │         Split by reference_obs_size      │
    └─────────────────────────────────────────┘
         │                              │
         ▼                              ▼
    traj_obs                       proprio_obs
    [T, B, ref_size]               [T, B, proprio_size]
         │                              │
         ▼                              │
    ┌──────────────┐                    │
    │   ENCODER    │                    │
    │  MLP+LN+SiLU │                    │
    └──────────────┘                    │
         │                              │
         ▼                              │
    ┌──────────────────┐                │
    │ mean, logvar     │                │
    │ [T, B, latent]   │ ◄──── KL Loss  │
    └──────────────────┘                │
         │                              │
         ▼                              │
    ┌──────────────────┐                │
    │ REPARAMETERIZE   │                │
    │ z = μ + σ×ε      │                │
    └──────────────────┘                │
         │                              │
         ▼                              ▼
    ┌─────────────────────────────────────────┐
    │           CONCATENATE [z, proprio]       │
    └─────────────────────────────────────────┘
                        │
                        ▼
                  ┌──────────────┐
                  │   DECODER    │
                  │  MLP+LN+SiLU │
                  └──────────────┘
                        │
                        ▼
                  action_params
                  [T, B, action*2]
```

### 3.2 Key Interfaces

**IntentionNetwork.__call__ signature**:
```python
def __call__(self, obs, key, deterministic=False, get_activation=False):
    # Returns: (action_params, latent_mean, latent_logvar)
```

**Loss function interface**:
```python
def compute_ppo_loss(params, normalizer_params, data, rng, step, ...):
    # Network forward pass
    policy_logits, latent_mean, latent_logvar = policy_apply(...)

    # KL divergence computed from mean, logvar
    kl_loss = compute_kl_divergence(latent_mean, latent_logvar, ...)

    # Returns: (total_loss, metrics_dict)
```

**Training state structure**:
```python
@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: PPONetworkParams  # Contains policy + value
    normalizer_params: RunningStatisticsState
    env_steps: jnp.ndarray
```

### 3.3 What We Must Preserve

1. **PPONetworkParams structure**: Policy and value params in a dataclass
2. **Training loop flow**: generate_unroll → compute_loss → gradient_update
3. **pmap compatibility**: All operations must be pmappable
4. **Observation normalization**: Running statistics integration
5. **Checkpoint format**: Compatible save/restore

---

## 4. Integration Strategy: Minimal Changes Philosophy

### 4.1 Core Principle: Codebook as Standard Parameter

**The key realization**: If we use gradient-based codebook updates (not EMA), the codebook is simply another trainable parameter. The optimizer handles it automatically.

**Why this works**:
- JAX/Flax treat all parameters uniformly in the pytree
- Optax applies gradients to the entire param tree
- No special state management required

**Trade-off acknowledged**:
- EMA updates are theoretically more stable for VQ-VAE
- But gradient updates work well in practice (see MaskGIT, VQGAN implementations)
- Dramatically simpler integration outweighs marginal EMA benefits

### 4.2 Parameter Structure Strategy

**Current VAE**:
```python
params.policy = {
    'encoder': {...},   # Weights for encoder MLP
    'decoder': {...},   # Weights for decoder MLP
}
```

**Proposed VQ-VAE**:
```python
params.policy = {
    'encoder': {...},       # Weights for encoder MLP
    'codebook': {...},      # Codebook embeddings [K, latent_dim]
    'decoder': {...},       # Weights for decoder MLP
}
```

**Why this structure**:
- Codebook lives alongside encoder/decoder as sibling in param tree
- Optimizer sees all three and updates them via gradients
- No changes to PPONetworkParams dataclass needed
- Checkpoint loading/saving works automatically

### 4.3 Network Output Strategy

**Current VAE returns**: `(action_params, latent_mean, latent_logvar)`
- `latent_mean`: Used for KL computation and logging
- `latent_logvar`: Used for KL computation and logging

**Proposed VQ-VAE returns**: `(action_params, z_e, indices)`
- `z_e`: Continuous encoder output, used for commitment loss
- `indices`: Codebook indices, used for logging/analysis

**Why NOT return z_q**:
- z_q can be reconstructed from codebook[indices]
- Returning indices is more memory efficient
- Avoids redundant data in transition buffer

**Alternative considered**: `(action_params, commitment_loss, indices)`
- Pre-compute commitment loss in forward pass
- Simpler loss function
- **Rejected**: Better to compute losses centrally for consistency

### 4.4 Loss Function Strategy

**Current VAE loss structure**:
```python
total_loss = policy_loss + v_loss + entropy_loss + kl_latent_loss
```

**Proposed VQ-VAE loss structure**:
```python
total_loss = policy_loss + v_loss + entropy_loss + vq_loss
# where vq_loss = commitment_loss + codebook_loss
```

**Key insight**: The regularization term changes, but the structure is identical.

---

## 5. JAX-Specific Challenges and Solutions

### 5.1 Challenge: Immutability

**Problem**: JAX arrays are immutable. Cannot do `codebook[idx] = new_value`.

**Solution for VQ-VAE**: This is a non-issue with gradient-based updates!
- We don't need to update individual codebook entries
- Optimizer updates the entire codebook via gradients
- Standard JAX autodiff handles everything

**If we needed EMA** (we don't, but for reference):
```python
# Wrong: codebook[idx] = new_value
# Right: Reconstruct entire array
new_codebook = codebook.at[idx].set(new_value)  # Returns new array
```

### 5.2 Challenge: JIT Shape Requirements

**Problem**: JIT-compiled functions require static shapes. Dynamic shapes cause recompilation.

**Shapes in our VQ-VAE**:
| Tensor | Shape | Static? |
|--------|-------|---------|
| observations | [T, B, obs_dim] | ✓ All dims from config |
| z_e | [T, B, latent_dim] | ✓ latent_dim from config |
| codebook | [num_codes, latent_dim] | ✓ Both from config |
| indices | [T, B] | ✓ Inherited from z_e batch dims |
| z_q | [T, B, latent_dim] | ✓ Same as z_e |

**All shapes are statically determined from config**: No dynamic shape issues.

**Potential gotcha**: Indices tensor has shape [T, B] (no latent_dim), different from z_e [T, B, latent_dim]. This affects data processing (see Section 8).

### 5.3 Challenge: Gradient Flow Through Discrete Operations

**Problem**: `argmin` for nearest neighbor has zero gradient.

**Solution**: Straight-through estimator with `jax.lax.stop_gradient`.

```python
def quantize(z_e, codebook):
    """Quantize encoder output to nearest codebook entry.

    JAX-compatible implementation with proper gradient flow.
    """
    # Compute squared distances: ||z_e - e_k||²
    # Using expansion: ||a-b||² = ||a||² + ||b||² - 2<a,b>
    # This is numerically stable and avoids broadcasting issues

    z_e_sq = jnp.sum(z_e ** 2, axis=-1, keepdims=True)      # [*, 1]
    codebook_sq = jnp.sum(codebook ** 2, axis=-1)           # [K]
    cross = jnp.matmul(z_e, codebook.T)                      # [*, K]

    distances = z_e_sq + codebook_sq - 2 * cross            # [*, K]

    # Find nearest (non-differentiable)
    indices = jnp.argmin(distances, axis=-1)                 # [*]

    # Look up quantized vectors
    z_q = codebook[indices]                                  # [*, latent_dim]

    # Straight-through estimator (Sterbenz pattern)
    z_q_st = z_e - jax.lax.stop_gradient(z_e) + jax.lax.stop_gradient(z_q)

    return z_q_st, indices, z_q  # z_q for loss computation
```

**Why this works**:
1. Forward pass: z_q_st = z_q (the quantized value)
2. Backward pass: d(z_q_st)/d(z_e) = 1 (gradients flow through)

### 5.4 Challenge: Loss Function Gradient Routing

**Problem**: Commitment loss should only update encoder. Codebook loss should only update codebook.

**Solution**: Strategic use of `stop_gradient`.

```python
def compute_vq_loss(z_e, z_q, commitment_cost=0.25):
    """Compute VQ-VAE auxiliary losses with proper gradient routing.

    Args:
        z_e: Encoder output [*, latent_dim]
        z_q: Quantized vectors (NOT straight-through) [*, latent_dim]
        commitment_cost: Beta coefficient

    Returns:
        vq_loss: Combined commitment + codebook loss
    """
    # Commitment loss: encoder learns to commit to codebook entries
    # Gradient flows to z_e only (codebook stopped)
    commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

    # Codebook loss: codebook moves toward encoder outputs
    # Gradient flows to z_q only (encoder stopped)
    codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)

    return commitment_cost * commitment_loss + codebook_loss
```

**Gradient flow verification**:
- `d(commitment_loss)/d(z_e)` = gradient exists (trains encoder)
- `d(commitment_loss)/d(codebook)` = 0 (codebook frozen for this term)
- `d(codebook_loss)/d(z_e)` = 0 (encoder frozen for this term)
- `d(codebook_loss)/d(codebook)` = gradient exists (trains codebook via z_q)

### 5.5 Challenge: Control Flow in JIT

**Problem**: Python `if/else` with traced values causes recompilation.

**Our situation**: We have a `deterministic` flag for eval vs train.

**Analysis of our forward pass**:
```python
def __call__(self, obs, key, deterministic=False, ...):
    z_e = self.encoder(traj_obs)
    z_q_st, indices, z_q = self.quantize(z_e)  # No conditional needed!
    ...
```

**Key insight**: VQ-VAE quantization is the same for train and eval!
- No reparameterization sampling that depends on `deterministic`
- Quantization is deterministic (nearest neighbor)
- No `if deterministic:` branch in the critical path

**This eliminates a major source of complexity from the VAE**.

### 5.6 Challenge: Batched Codebook Indexing

**Problem**: Looking up multiple indices from codebook.

```python
# indices: [T, B] - batch of indices
# codebook: [K, latent_dim] - the codebook
# Want: z_q [T, B, latent_dim]
```

**JAX solution**: Advanced indexing works naturally.
```python
z_q = codebook[indices]  # Shape: [T, B, latent_dim]
```

**Why this works in JAX**:
- JAX supports NumPy-style advanced indexing
- Integer array indexing broadcasts correctly
- Result shape = index_shape + indexed_array_trailing_shape
- [T, B] indexing into [K, D] gives [T, B, D]

### 5.7 Challenge: pmap Compatibility

**Problem**: Codebook is shared across devices. How do we handle this?

**Solution**: Standard pmap parameter replication handles this automatically.

```python
# In training loop (existing pattern)
training_state = jax.device_put_replicated(
    training_state, jax.local_devices()[:local_devices_to_use]
)
```

**Gradient synchronization**:
```python
# Existing pattern in gradient_update_fn
grads = jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)
```

**No changes needed**: The codebook is part of params, so it's:
1. Replicated across devices at initialization
2. Gradients averaged via pmean
3. Updated identically on all devices

---

## 6. Detailed Integration Plan

### 6.1 File Structure

```
track_mjx/agent/ff_ppo/
├── intention_network.py          # EXISTING: VAE implementation
├── vq_intention_network.py       # NEW: VQ-VAE implementation
├── losses.py                     # MODIFY: Add compute_vq_ppo_loss
├── ppo_networks.py               # MODIFY: Add make_vq_ppo_networks
└── ppo.py                        # NO CHANGES (training loop unchanged)
```

### 6.2 New File: vq_intention_network.py

**Purpose**: VQ-VAE encoder, quantizer, decoder, and network assembly.

**Classes to implement**:

1. **VQEncoder(nn.Module)**
   - Input: traj_obs [*, ref_size]
   - Output: z_e [*, latent_dim]
   - Architecture: MLP + LayerNorm + SiLU (same as VAE encoder body)
   - Single output (not mean/logvar)

2. **VectorQuantizer(nn.Module)**
   - Codebook as `self.param('codebook', initializer, shape)`
   - Pure quantization function
   - Straight-through estimator
   - Returns: (z_q_st, indices, z_q)

3. **Decoder(nn.Module)**
   - Reuse existing decoder or copy
   - Input: [z_q_st, proprio_obs]
   - Output: action_params

4. **VQIntentionNetwork(nn.Module)**
   - Combines encoder + quantizer + decoder
   - Handles observation splitting
   - Forward signature: `__call__(obs, key, deterministic=False)`
   - Returns: `(action_params, z_e, indices)`

5. **make_vq_intention_policy()**
   - Factory function matching existing pattern
   - Returns: FeedForwardNetwork with init/apply

### 6.3 Modifications to losses.py

**Add new function**: `compute_vq_ppo_loss`

**Structure**:
```python
def compute_vq_ppo_loss(
    params: PPONetworkParams,
    normalizer_params,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network,
    # Existing PPO params...
    entropy_cost: float = 1e-4,
    clipping_epsilon: float = 0.3,
    # ... etc
    # VQ-specific params
    commitment_cost: float = 0.25,
    codebook_loss_weight: float = 1.0,
) -> tuple[jnp.ndarray, dict]:
    """PPO loss with VQ-VAE regularization instead of KL divergence."""

    # Standard PPO computation (copy from compute_ppo_loss)
    # ...

    # VQ-VAE specific: compute commitment and codebook losses
    # Need z_e and indices from forward pass
    # Reconstruct z_q from codebook[indices] for loss computation

    # Return (total_loss, metrics) - same signature as existing
```

**Key decisions**:
- Same return signature as `compute_ppo_loss`
- Metrics dict includes VQ-specific metrics (commitment_loss, codebook_loss, perplexity)
- No state_updates return - codebook is updated via gradients

### 6.4 Modifications to ppo_networks.py

**Add new factory function**: `make_vq_intention_ppo_networks`

```python
def make_vq_intention_ppo_networks(
    observation_size: int,
    reference_obs_size: int,
    action_size: int,
    preprocess_observations_fn,
    intention_latent_size: int = 60,
    num_codes: int = 512,
    commitment_cost: float = 0.25,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPOImitationNetworks:
    """Create VQ-VAE intention-based PPO networks."""
    # ...
```

**Add new dataclass** (optional, may reuse existing):
```python
@flax.struct.dataclass
class VQPPOImitationNetworks:
    policy_network: vq_intention_network.VQIntentionNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    # Additional for convenience:
    num_codes: int
    latent_dim: int
```

### 6.5 Configuration Additions

**In config YAML**:
```yaml
network_config:
  arch_name: vqvae_intention  # or "intention" for VAE

  # VQ-VAE specific (only used if arch_name == vqvae_intention)
  num_codes: 512
  commitment_cost: 0.25
  codebook_loss_weight: 1.0
  codebook_init_scale: 0.01  # Small initialization
```

### 6.6 No Changes to ppo.py

**Why no changes needed**:
1. Training loop operates on generic `params` pytree
2. Loss function is passed as argument (can swap VAE for VQ)
3. Network factory is passed as argument (can swap VAE for VQ)
4. All VQ-specific logic contained in network and loss

**Verification checklist**:
- [ ] `training_step` doesn't assume param structure
- [ ] `gradient_update_fn` uses generic gradients.loss_and_pgrad
- [ ] Checkpoint save/restore uses generic pytree serialization
- [ ] Evaluation uses generic make_policy interface

---

## 7. Data Flow Analysis

### 7.1 Training Data Flow

```
                        VQ-VAE TRAINING DATA FLOW
                        =========================

┌─────────────────────────────────────────────────────────────────────┐
│                        ROLLOUT GENERATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  env.reset() → state                                                 │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    POLICY FORWARD                            │    │
│  │                                                              │    │
│  │  obs [B, obs_dim]                                            │    │
│  │       │                                                      │    │
│  │       ▼                                                      │    │
│  │  VQIntentionNetwork:                                         │    │
│  │    traj_obs → Encoder → z_e [B, latent]                     │    │
│  │                           │                                  │    │
│  │                           ▼                                  │    │
│  │                    VectorQuantizer:                          │    │
│  │                      argmin distances → indices [B]          │    │
│  │                      codebook[indices] → z_q [B, latent]     │    │
│  │                      straight_through → z_q_st               │    │
│  │                           │                                  │    │
│  │    proprio_obs ───────────┤                                  │    │
│  │                           ▼                                  │    │
│  │                    [z_q_st, proprio] → Decoder → action      │    │
│  │                                                              │    │
│  │  Returns: (action_params, z_e, indices)                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  action_dist.sample(action_params, rng) → action                    │
│       │                                                              │
│       ▼                                                              │
│  extras = {                                                          │
│      'log_prob': log_prob,     # [B]                                │
│      'raw_action': raw_action, # [B, action_dim]                    │
│      # Note: z_e and indices NOT stored (recomputed in loss)        │
│  }                                                                   │
│       │                                                              │
│       ▼                                                              │
│  env.step(action) → next_state, reward, done                        │
│       │                                                              │
│       ▼                                                              │
│  Transition(obs, action, reward, next_obs, done, extras)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LOSS COMPUTATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  data: Transition batch [B, T, ...]                                 │
│       │                                                              │
│       ▼                                                              │
│  Reshape: [B, T] → [T, B] (time-major for GAE)                      │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  POLICY FORWARD (again)                      │    │
│  │                                                              │    │
│  │  data.observation [T, B, obs_dim]                            │    │
│  │       │                                                      │    │
│  │       ▼                                                      │    │
│  │  VQIntentionNetwork:                                         │    │
│  │    → (action_params, z_e, indices) [T, B, ...]              │    │
│  │                                                              │    │
│  │  # Reconstruct z_q for loss (not stored in transition)      │    │
│  │  z_q = params.policy['codebook'][indices]                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    LOSS COMPONENTS                           │    │
│  │                                                              │    │
│  │  PPO Loss: clipped surrogate from log_prob ratio            │    │
│  │  Value Loss: MSE(V_pred, V_target)                          │    │
│  │  Entropy Loss: -entropy_cost × entropy                       │    │
│  │                                                              │    │
│  │  Commitment Loss: β × ||z_e - sg[z_q]||²                    │    │
│  │  Codebook Loss: ||sg[z_e] - z_q||²                          │    │
│  │                                                              │    │
│  │  Total = PPO + Value + Entropy + Commitment + Codebook      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│       │                                                              │
│       ▼                                                              │
│  (total_loss, metrics_dict)                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GRADIENT UPDATE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  jax.grad(loss_fn)(params) → grads                                  │
│       │                                                              │
│       │  grads.policy contains:                                     │
│       │    - encoder gradients (from PPO + commitment)              │
│       │    - codebook gradients (from codebook loss via z_q)        │
│       │    - decoder gradients (from PPO)                           │
│       │                                                              │
│       ▼                                                              │
│  pmean(grads) → synchronized grads                                  │
│       │                                                              │
│       ▼                                                              │
│  optimizer.update(grads, params) → new_params                       │
│       │                                                              │
│       │  All params updated uniformly:                              │
│       │    - encoder weights                                        │
│       │    - codebook embeddings ← KEY: standard gradient update    │
│       │    - decoder weights                                        │
│       │    - value network weights                                  │
│       │                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Key Data Flow Decisions

**Decision 1**: Don't store z_e/indices in transition extras.
- **Why**: Recompute in loss function for fresh gradients
- **Trade-off**: Extra compute vs memory/complexity
- **Justification**: Matches VAE pattern (recomputes mean/logvar in loss)

**Decision 2**: Reconstruct z_q from codebook[indices] in loss.
- **Why**: z_q must use current codebook params for correct gradients
- **Note**: indices are discrete, so they're valid across gradient steps

**Decision 3**: Keep transition buffer structure unchanged.
- **Why**: Minimal changes to data pipeline
- **Benefit**: Compatible with existing checkpoint format, replay

---

## 8. Shape Management Strategy

### 8.1 Shape Inventory

| Tensor | Shape | Notes |
|--------|-------|-------|
| observation | [T, B, obs_dim] | obs_dim = ref_size + proprio_size |
| traj_obs | [T, B, ref_size] | Reference trajectory |
| proprio_obs | [T, B, proprio_size] | Proprioceptive state |
| z_e | [T, B, latent_dim] | Encoder output |
| codebook | [num_codes, latent_dim] | Shared across batch |
| indices | [T, B] | **Note: No latent_dim!** |
| z_q | [T, B, latent_dim] | Quantized vectors |
| action_params | [T, B, action_dim * 2] | Mean + std |

### 8.2 The Indices Shape Challenge

**Problem**: indices has shape [T, B] while other tensors have [T, B, D].

**Where this matters**:

1. **Transition extras**: If we store indices in extras
   ```python
   extras = {
       'log_prob': [T, B],        # 2D - OK
       'raw_action': [T, B, D],   # 3D - OK
       'indices': [T, B],         # 2D - matches log_prob
   }
   ```
   This is actually fine! Same shape as log_prob.

2. **Minibatch reshaping**: The existing code does:
   ```python
   data = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), data)
   ```
   For [T, B, D] → [T*B, D]
   For [T, B] → [T*B] (still works!)

3. **Data swapping**: The existing code does:
   ```python
   data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
   ```
   For [T, B, D] → [B, T, D]
   For [T, B] → [B, T] (still works!)

**Conclusion**: No special handling needed for indices shape.

### 8.3 Codebook Indexing Shape Analysis

```python
# Setup
codebook = jnp.zeros((512, 60))  # [K, D]
indices = jnp.zeros((20, 4096), dtype=jnp.int32)  # [T, B]

# Indexing
z_q = codebook[indices]
# Result shape: [20, 4096, 60] = [T, B, D]
```

**Why this works**:
- JAX follows NumPy advanced indexing rules
- Integer array indexing: result_shape = index_shape + trailing_dims
- [T, B] indexing [K, D] → [T, B] + [D] = [T, B, D]

### 8.4 Distance Computation Shape Analysis

```python
# Setup
z_e = jnp.zeros((20, 4096, 60))  # [T, B, D]
codebook = jnp.zeros((512, 60))   # [K, D]

# Need to compute distances [T, B, K]

# Method: Reshape z_e for matmul, then reshape back
z_e_flat = z_e.reshape(-1, 60)    # [T*B, D]

z_e_sq = jnp.sum(z_e_flat ** 2, axis=-1, keepdims=True)  # [T*B, 1]
codebook_sq = jnp.sum(codebook ** 2, axis=-1)            # [K]
cross = jnp.matmul(z_e_flat, codebook.T)                  # [T*B, K]

distances = z_e_sq + codebook_sq - 2 * cross             # [T*B, K]

indices_flat = jnp.argmin(distances, axis=-1)            # [T*B]
indices = indices_flat.reshape(z_e.shape[:-1])           # [T, B]
```

**Why flatten-unflatten pattern**:
- Matmul operates on last two dimensions
- z_e has 3 dims, codebook has 2 dims
- Flattening [T, B, D] to [T*B, D] makes matmul clean
- Unflatten indices back to [T, B]

---

## 9. Loss Function Design

### 9.1 Complete Loss Structure

```python
def compute_vq_ppo_loss(
    params,
    normalizer_params,
    data,
    rng,
    step,
    ppo_network,
    # PPO hyperparameters (existing)
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    # VQ-VAE hyperparameters (new)
    commitment_cost: float = 0.25,
    codebook_loss_weight: float = 1.0,
):
    """PPO loss with VQ-VAE commitment and codebook regularization."""

    # ========== Standard PPO Setup (unchanged) ==========

    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    parametric_action_distribution = ppo_network.parametric_action_distribution

    # Time-major format
    data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # ========== Policy Forward Pass ==========

    # Returns (action_params, z_e, indices)
    policy_logits, z_e, indices = policy_apply(
        normalizer_params, params.policy, data.observation, rng
    )

    # ========== VQ-VAE Loss Components ==========

    # Reconstruct z_q from current codebook (important for correct gradients)
    codebook = params.policy['codebook']['embeddings']  # Access codebook
    z_q = codebook[indices]  # [T, B, latent_dim]

    # Commitment loss: encoder commits to codebook
    commitment_loss = jnp.mean(
        (z_e - jax.lax.stop_gradient(z_q)) ** 2
    )

    # Codebook loss: codebook moves toward encoder
    codebook_loss = jnp.mean(
        (jax.lax.stop_gradient(z_e) - z_q) ** 2
    )

    vq_loss = commitment_cost * commitment_loss + codebook_loss_weight * codebook_loss

    # ========== Standard PPO Losses (unchanged) ==========

    # Value function
    baseline = value_apply(normalizer_params, params.value, data.observation)
    bootstrap_value = value_apply(
        normalizer_params, params.value, data.next_observation[-1]
    )

    # GAE
    rewards = data.reward * reward_scaling
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    vs, advantages = compute_gae(
        truncation, termination, rewards, baseline, bootstrap_value,
        gae_lambda, discounting
    )

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    target_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action']
    )
    behaviour_log_probs = data.extras['policy_extras']['log_prob']

    rho = jnp.exp(target_log_probs - behaviour_log_probs)
    surrogate1 = rho * advantages
    surrogate2 = jnp.clip(rho, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    # Value loss
    v_loss = 0.5 * jnp.mean((vs - baseline) ** 2) * 0.5

    # Entropy loss
    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, rng)
    )
    entropy_loss = entropy_cost * -entropy

    # ========== Total Loss ==========

    total_loss = policy_loss + v_loss + entropy_loss + vq_loss

    # ========== Metrics ==========

    # Perplexity: exp(entropy of code usage)
    # Higher = more codes used, lower = codebook collapse
    code_one_hot = jax.nn.one_hot(indices.reshape(-1), codebook.shape[0])
    code_probs = jnp.mean(code_one_hot, axis=0)
    code_entropy = -jnp.sum(
        jnp.where(code_probs > 0, code_probs * jnp.log(code_probs + 1e-10), 0)
    )
    perplexity = jnp.exp(code_entropy)

    # Codebook utilization
    codes_used = jnp.sum(code_probs > 0)
    utilization = codes_used / codebook.shape[0]

    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'entropy_loss': entropy_loss,
        'commitment_loss': commitment_loss,
        'codebook_loss': codebook_loss,
        'vq_loss': vq_loss,
        'perplexity': perplexity,
        'codebook_utilization': utilization,
        'codes_used': codes_used,
    }

    return total_loss, metrics
```

### 9.2 Loss Gradient Flow Diagram

```
                    GRADIENT FLOW IN VQ-VAE LOSS
                    ============================

                         total_loss
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
      policy_loss        v_loss            vq_loss
           │                  │                  │
           │                  │        ┌─────────┴─────────┐
           │                  │        │                   │
           │                  │        ▼                   ▼
           │                  │   commitment          codebook
           │                  │      loss               loss
           │                  │        │                   │
           │                  │        │                   │
    ┌──────┴──────┐    ┌─────┴─────┐  │                   │
    │             │    │           │  │                   │
    ▼             ▼    ▼           │  │                   │
 decoder      action  value        │  │                   │
 weights      dist   network       │  │                   │
                                   │  │                   │
                                   │  ▼                   ▼
                                   │  β×||z_e-sg[z_q]||²  ||sg[z_e]-z_q||²
                                   │       │                   │
                                   │       │                   │
                                   │       ▼                   ▼
                                   │    encoder            codebook
                                   │    weights           embeddings
                                   │       │                   │
                                   │       │                   │
                                   ▼       ▼                   │
                                straight-through ◄─────────────┘
                                estimator        (z_q = codebook[indices])
                                   │
                                   ▼
                                encoder
                                weights
```

---

## 10. Decision Justifications

### 10.1 Gradient-Based Codebook Updates (vs EMA)

**Decision**: Use gradient descent for codebook updates, not EMA.

**Rationale**:

| Factor | Gradient Updates | EMA Updates |
|--------|------------------|-------------|
| Training loop changes | None | Significant |
| State management | Standard params | Custom mutable state |
| pmap compatibility | Automatic | Requires custom handling |
| Theoretical quality | Good (proven in VQGAN, MaskGIT) | Better (original VQ-VAE) |
| Implementation risk | Low | High (as seen in pilot) |
| Debugging | Easy (standard autodiff) | Hard (EMA state issues) |

**Supporting evidence**:
- [MaskGIT](https://arxiv.org/abs/2202.04200) uses gradient-based VQ successfully
- [VQGAN](https://arxiv.org/abs/2012.09841) uses gradient-based VQ successfully
- [jax-vqvae-vqgan](https://github.com/kvfrans/jax-vqvae-vqgan) reproduces paper results with gradients

**Risk mitigation**: If gradient updates cause codebook collapse, we can add EMA later. Start simple.

### 10.2 Codebook as Nested Param (vs Separate State)

**Decision**: Codebook is part of `params.policy['codebook']`, not separate state.

**Rationale**:
- Existing training loop operates on generic params pytree
- Optimizer handles all params uniformly
- Checkpoint save/restore works automatically
- No changes to TrainingState dataclass

**Implementation detail**:
```python
# Codebook is initialized as a parameter in VectorQuantizer
class VectorQuantizer(nn.Module):
    def setup(self):
        self.codebook = self.param(
            'embeddings',
            nn.initializers.uniform(scale=init_scale),
            (self.num_codes, self.latent_dim)
        )
```

### 10.3 Recompute z_e/indices in Loss (vs Store in Transition)

**Decision**: Recompute encoder output in loss function, don't store in extras.

**Rationale**:
- Matches VAE pattern (recomputes mean/logvar in loss)
- Ensures gradients flow through current params
- Avoids transition buffer shape changes
- Simpler data pipeline

**Trade-off**:
- Extra forward pass through encoder
- But: encoder is small relative to environment step cost
- Memory savings from not storing [T, B, latent_dim] tensor

### 10.4 Single Output Encoder (vs Reusing VAE Encoder)

**Decision**: Create new VQEncoder that outputs single z_e, not (mean, logvar).

**Rationale**:
- VAE encoder computes unnecessary logvar
- Cleaner architecture
- Removes confusion about what's used
- Minor efficiency gain

**Note**: Encoder body (MLP + LayerNorm + SiLU) is identical, just final layer differs.

### 10.5 Straight-Through via Sterbenz Pattern

**Decision**: Use `zero = x - stop_grad(x); return zero + stop_grad(y)` pattern.

**Rationale**:
- Numerically exact (no floating point accumulation)
- Recommended by JAX team (Issue #9032)
- Used in [dm-haiku VQ-VAE](https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/nets/vqvae.py)

**Alternative rejected**: `x + stop_grad(y - x)`
- Mathematically equivalent
- Can accumulate floating-point errors in deep networks

---

## 11. Risk Assessment

### 11.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Codebook collapse (few codes used) | Medium | High | Monitor perplexity, add entropy bonus if needed |
| Gradient instability | Low | Medium | Use standard initialization, gradient clipping exists |
| Shape mismatches | Low | High | Thorough testing at each integration point |
| Training slower than VAE | Medium | Low | Expected due to quantization overhead |
| Worse motion quality than VAE | Medium | Medium | Hyperparameter tuning, ablation studies |

### 11.2 Codebook Collapse Prevention

**Indicators**:
- Perplexity << num_codes (e.g., < 100 for 512 codes)
- utilization < 0.5
- Same indices repeated across batch

**Mitigation strategies** (implement if needed):
1. **Entropy regularization**: Add `entropy_weight * code_entropy` to loss
2. **Codebook reset**: Reinitialize dead codes from batch samples
3. **Larger initialization**: Spread codebook wider initially
4. **Lower commitment cost**: Let encoder explore more

### 11.3 Integration Risks

| Risk | Mitigation |
|------|------------|
| Checkpoint incompatibility | Test save/load explicitly |
| pmap gradient sync issues | Verify pmean on all grads |
| Evaluation mode differences | VQ is deterministic, simpler than VAE |
| Logging compatibility | Add VQ-specific metrics to wandb |

---

## 12. Implementation Checklist

### Phase 1: Core Network Implementation

- [ ] Create `vq_intention_network.py`
  - [ ] VQEncoder class (MLP → z_e)
  - [ ] VectorQuantizer class (codebook as param, quantize function)
  - [ ] Decoder class (reuse or copy from intention_network.py)
  - [ ] VQIntentionNetwork class (combines all)
  - [ ] make_vq_intention_policy factory function
  - [ ] Unit tests for shapes and gradient flow

### Phase 2: Loss Function Implementation

- [ ] Add `compute_vq_ppo_loss` to `losses.py`
  - [ ] Copy PPO loss structure from compute_ppo_loss
  - [ ] Add commitment loss computation
  - [ ] Add codebook loss computation
  - [ ] Add perplexity/utilization metrics
  - [ ] Unit test: verify gradient routing with stop_gradient

### Phase 3: Network Factory Integration

- [ ] Add `make_vq_intention_ppo_networks` to `ppo_networks.py`
  - [ ] VQPPOImitationNetworks dataclass (if needed)
  - [ ] make_vq_inference_fn (for evaluation)
  - [ ] Integration test with dummy data

### Phase 4: Configuration Integration

- [ ] Add VQ config options to YAML schema
- [ ] Add `arch_name` switch in `train.py`
- [ ] Test config loading

### Phase 5: Logging Integration

- [ ] Add VQ metrics to wandb_logging.py
  - [ ] Codebook health (perplexity, utilization)
  - [ ] Code usage histogram
  - [ ] Code trajectory visualization (optional)

### Phase 6: End-to-End Testing

- [ ] Run short training (few iterations) on rodent
- [ ] Verify all metrics logged
- [ ] Verify checkpoint save/load
- [ ] Verify multi-GPU training works

### Phase 7: Validation

- [ ] Compare reward curves: VAE vs VQ-VAE
- [ ] Analyze learned codebook
- [ ] Motion quality assessment

---

## 13. References

### VQ-VAE Papers

1. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) - Original VQ-VAE (van den Oord et al., 2017)
2. [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) - Hierarchical VQ-VAE
3. [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) - VQGAN

### JAX VQ-VAE Implementations

1. [dm-haiku VQ-VAE](https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/nets/vqvae.py) - DeepMind's official implementation
2. [jax-vqvae-vqgan](https://github.com/kvfrans/jax-vqvae-vqgan) - Clean JAX implementation with FSQ support
3. [VQVAE_Flax](https://github.com/aillaud/VQVAE_Flax) - Flax-based implementation

### JAX Documentation

1. [jax.lax.stop_gradient](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html) - Official stop_gradient docs
2. [JAX Issue #9032](https://github.com/jax-ml/jax/discussions/23700) - Straight-through estimator discussion
3. [Flax Module documentation](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html) - self.param usage

### Related VQ Research

1. [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505) - FSQ as simpler alternative
2. [Understanding VQ in VQ-VAE](https://huggingface.co/blog/ariG23498/understand-vq) - Educational walkthrough
3. [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) - Uses gradient-based VQ

---

## Appendix A: Comparison with Previous Pilot

The previous pilot in `vqvae_jax/` encountered 9 bugs and significant complexity due to EMA-based codebook updates. This plan avoids those issues:

| Previous Pilot Issue | This Plan's Solution |
|---------------------|---------------------|
| EMA updates never executed | No EMA - gradient updates |
| Codebook as trainable param with EMA | Codebook IS trainable - gradient only |
| Loss mixes EMA and gradient | Only gradient updates |
| Missing Laplace smoothing | Not needed without EMA |
| JAX mutable state handling | No mutable state |
| State updates in loss return | Standard (loss, metrics) return |
| Complex parameter restructuring | Standard params pytree |
| Custom gradient_update_fn | Use existing gradient utility |

---

## Appendix B: Future EMA Extension

If gradient-based updates prove insufficient (codebook collapse despite mitigations), here's the path to add EMA:

1. Change codebook from `self.param()` to `self.variable('codebook', ...)`
2. Add EMA state variables: cluster_size, ema_dw
3. Modify forward pass to return state updates
4. Modify loss function return: `(loss, (metrics, state_updates))`
5. Modify training loop to apply state updates after gradient step
6. Add Laplace smoothing for numerical stability

This is a well-understood path (the pilot implemented it), but should only be pursued if gradient updates fail.

---

**End of Planning Document**

*Ready for implementation upon approval.*
