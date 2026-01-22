# Temporal Downsampling for Semantic VQ-VAE Codes

## The Problem: Codes Represent Gait Phases, Not Behaviors

### Current Architecture

The current VQ-VAE encoder processes each frame **independently**:

```
                    Frame 0    Frame 1    Frame 2    ...    Frame T
                       │          │          │                 │
                       ▼          ▼          ▼                 ▼
                    ┌─────┐   ┌─────┐   ┌─────┐           ┌─────┐
Encoder (MLP):      │ MLP │   │ MLP │   │ MLP │    ...    │ MLP │
                    └──┬──┘   └──┬──┘   └──┬──┘           └──┬──┘
                       │          │          │                 │
                       ▼          ▼          ▼                 ▼
                      z_e₀       z_e₁       z_e₂      ...     z_eₜ
                       │          │          │                 │
                       ▼          ▼          ▼                 ▼
Quantizer:          Code 3     Code 5     Code 5    ...     Code 2
```

Mathematically, for observation $x_t$ at time $t$:

$$z_{e,t} = \text{MLP}(x_t)$$
$$c_t = \arg\min_k \| z_{e,t} - e_k \|^2$$

where $e_k$ is the $k$-th codebook entry.

**Key Issue**: Each $z_{e,t}$ is computed from $x_t$ alone. The encoder has **no temporal context**.

### What the Codes Learn

Since each code is assigned based on a single frame's pose, codes end up representing **phases of the gait cycle**:

```
Walking Gait Cycle (simplified):

    ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐      ┌──────┐
    │      │      │  /\  │      │      │      │  /\  │      │      │
    │  /\  │      │ /  \ │      │  ──  │      │ /  \ │      │  /\  │
    │ /  \ │      │/    \│      │ /  \ │      │/    \│      │ /  \ │
    └──────┘      └──────┘      └──────┘      └──────┘      └──────┘
     Code 3        Code 5        Code 7        Code 5        Code 3
   "left foot    "mid-stride"   "right foot  "mid-stride"  "left foot
    forward"                     forward"                   forward"
```

The 8 codes divide the **pose space**, not the **behavior space**:

| Code | What It Represents | What We Wanted |
|------|-------------------|----------------|
| 0-7  | Different poses in gait cycle | Different behaviors (walk, rear, groom) |

### Evidence from Training

From our training run (iteration 20):
- **Perplexity**: 6.39 (using nearly all 8 codes)
- **Transition rate**: 37.4% (code changes every ~3 frames)
- **All codes used in every clip** (because every clip has all gait phases)

```
Code Sequence for a Walking Clip:
Time:  0   10   20   30   40   50   60   70   80   90  100  ...
Code:  3 5 7 5 3 5 7 5 3 5 7 5 3 5 7 5 3 5 7 5 3 5 7 5 ...
       └─────────────┘ └─────────────┘ └─────────────┘
         gait cycle      gait cycle      gait cycle
```

**Problem**: A walking-only clip uses all codes. A rearing clip also uses all codes. We can't distinguish behaviors by which codes are active.

---

## The Solution: Temporal Downsampling

### Core Idea

Instead of assigning one code per frame, assign **one code per temporal chunk**:

```
Before:  20 frames  →  20 codes  (one per frame)
After:   20 frames  →   5 codes  (one per 4-frame chunk)
```

**Multiple frames must share the same code**, forcing the code to represent something that's **true for the entire chunk**.

```
┌────────────────────────────────────────────────────────────────────┐
│  Before (per-frame):                                               │
│  Code answers: "What POSE is the rodent in right now?"             │
│  → Codes = gait phases (left foot forward, mid-stride, etc.)       │
├────────────────────────────────────────────────────────────────────┤
│  After (per-chunk):                                                │
│  Code answers: "What BEHAVIOR happens in this 80ms?"               │
│  → Codes = behavioral primitives (walking, rearing, grooming)      │
└────────────────────────────────────────────────────────────────────┘
```

### Two Implementation Approaches

We present two methods to achieve temporal downsampling:

| Approach | Method | Complexity | Flexibility |
|----------|--------|------------|-------------|
| **Approach A: Pooling** | MLP per-frame, then pool | Simple | Fixed chunks |
| **Approach B: Temporal Conv** | Conv1D with stride | Moderate | Learned aggregation |

---

## Approach A: MLP + Pooling

### Overview

Keep the existing MLP encoder, but add a **pooling step** that aggregates multiple frame embeddings into one:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLP + POOLING APPROACH                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: MLP encodes each frame independently (same as before)              │
│                                                                             │
│  Frame:   x₀      x₁      x₂      x₃      x₄      x₅      x₆      x₇       │
│           │       │       │       │       │       │       │       │        │
│           ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼        │
│        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │
│  MLP:  │ MLP │ │ MLP │ │ MLP │ │ MLP │ │ MLP │ │ MLP │ │ MLP │ │ MLP │    │
│        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘    │
│           │       │       │       │       │       │       │       │        │
│           ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼        │
│  z:      z₀      z₁      z₂      z₃      z₄      z₅      z₆      z₇       │
│                                                                             │
│  Step 2: Group into chunks of size 4                                        │
│                                                                             │
│          ┌───────────────────┐       ┌───────────────────┐                 │
│          │  z₀   z₁   z₂   z₃│       │  z₄   z₅   z₆   z₇│                 │
│          │     Chunk 0       │       │     Chunk 1       │                 │
│          └─────────┬─────────┘       └─────────┬─────────┘                 │
│                    │                           │                            │
│  Step 3: Pool      ▼ mean/max                  ▼ mean/max                   │
│                                                                             │
│                 ┌─────┐                     ┌─────┐                         │
│                 │z_e⁰ │                     │z_e¹ │                         │
│                 └──┬──┘                     └──┬──┘                         │
│                    │                           │                            │
│  Step 4:           ▼                           ▼                            │
│  Quantize       Code 3                      Code 3                          │
│                "walking"                   "walking"                        │
│                                                                             │
│  Result: 8 frames → 2 codes                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematics

**Step 1**: Encode each frame with existing MLP:
$$z_t = \text{MLP}(x_t) \in \mathbb{R}^d \quad \forall t \in [0, T)$$

**Step 2**: Group into chunks of size $s$:
$$Z^{(i)} = [z_{is}, z_{is+1}, ..., z_{is+s-1}] \in \mathbb{R}^{s \times d}$$

**Step 3**: Pool each chunk:

*Mean pooling:*
$$z_e^{(i)} = \frac{1}{s} \sum_{j=0}^{s-1} z_{is+j}$$

*Max pooling:*
$$z_e^{(i)} = \max_{j \in [0, s)} z_{is+j}$$
(element-wise max across the $s$ vectors)

**Step 4**: Quantize:
$$c^{(i)} = \arg\min_{k \in [K]} \| z_e^{(i)} - e_k \|^2$$

**Output**: $\lfloor T/s \rfloor$ codes instead of $T$ codes.

### Detailed Example (stride=4)

```
Input: 12 frames of walking

Frame:     0    1    2    3  │  4    5    6    7  │  8    9   10   11
Pose:      🦵→  🦶   🦵←  🦶  │  🦵→  🦶   🦵←  🦶  │  🦵→  🦶   🦵←  🦶
           │    │    │    │  │  │    │    │    │  │  │    │    │    │
           ▼    ▼    ▼    ▼  │  ▼    ▼    ▼    ▼  │  ▼    ▼    ▼    ▼
MLP out:  [.2] [.5] [.8] [.5]│ [.2] [.5] [.8] [.5]│ [.2] [.5] [.8] [.5]
           └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                    │                    │                    │
Mean Pool:         [.5]                 [.5]                 [.5]
                    │                    │                    │
Quantize:        Code 3              Code 3              Code 3
                "walking"           "walking"           "walking"

Result: 12 frames → 3 codes, all "walking"
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Minimal code change | Fixed chunk boundaries |
| Reuses existing MLP | No learned temporal features |
| Easy to understand | Pooling may lose information |
| Fast to implement | Mean pooling = "average pose" |

---

## Approach B: Temporal Convolution

### Overview

Replace the MLP encoder with **1D convolutions** that have **stride > 1**. Each conv layer both extracts features AND reduces temporal resolution:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPORAL CONVOLUTION APPROACH                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: 8 frames                                                            │
│                                                                             │
│  Frame:   x₀      x₁      x₂      x₃      x₄      x₅      x₆      x₇       │
│           │       │       │       │       │       │       │       │        │
│           └───┬───┘       └───┬───┘       └───┬───┘       └───┬───┘        │
│               │               │               │               │            │
│  Conv1D       ▼               ▼               ▼               ▼            │
│  stride=2   ┌───┐           ┌───┐           ┌───┐           ┌───┐         │
│  kernel=3   │ * │           │ * │           │ * │           │ * │         │
│             └─┬─┘           └─┬─┘           └─┬─┘           └─┬─┘         │
│               │               │               │               │            │
│               ▼               ▼               ▼               ▼            │
│  Layer 1:    h₀              h₁              h₂              h₃           │
│  (4 outputs)  │               │               │               │            │
│               └───────┬───────┘               └───────┬───────┘            │
│                       │                               │                    │
│  Conv1D               ▼                               ▼                    │
│  stride=2           ┌───┐                           ┌───┐                  │
│  kernel=3           │ * │                           │ * │                  │
│                     └─┬─┘                           └─┬─┘                  │
│                       │                               │                    │
│                       ▼                               ▼                    │
│  Layer 2:           z_e⁰                            z_e¹                   │
│  (2 outputs)          │                               │                    │
│                       ▼                               ▼                    │
│  Quantize:         Code 3                          Code 3                  │
│                   "walking"                       "walking"                │
│                                                                            │
│  Result: 8 frames → 4 hidden → 2 codes                                     │
│          (total stride = 2 × 2 = 4)                                        │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Strided Convolution Works

A 1D convolution with **stride=2** and **kernel_size=3** processes the input like this:

```
Input sequence (T=8):
Position:    0     1     2     3     4     5     6     7
Value:      [a]   [b]   [c]   [d]   [e]   [f]   [g]   [h]


Kernel (size=3) slides with stride=2:

Output 0:   [a]   [b]   [c]                              → weighted sum → h₀
             └─────┼─────┘
                kernel

Output 1:               [c]   [d]   [e]                  → weighted sum → h₁
                         └─────┼─────┘
                            kernel

Output 2:                           [e]   [f]   [g]      → weighted sum → h₂
                                     └─────┼─────┘
                                        kernel

Output 3:                                       [g]   [h]  (+ padding) → h₃
                                                 └─────┼─────┘
                                                    kernel

Result: 8 inputs → 4 outputs (T/stride = 8/2 = 4)
```

### Mathematics

**Layer 1** (Conv1D, stride $s_1$, kernel size $k$, input channels $C_{in}$, output channels $C_1$):

$$h_i^{(1)} = \sigma\left( \sum_{j=0}^{k-1} W^{(1)}_j \cdot x_{i \cdot s_1 + j} + b^{(1)} \right)$$

Output length: $T_1 = \lfloor T / s_1 \rfloor$

**Layer 2** (Conv1D, stride $s_2$, input channels $C_1$, output channels $C_2$):

$$h_i^{(2)} = \sigma\left( \sum_{j=0}^{k-1} W^{(2)}_j \cdot h_{i \cdot s_2 + j}^{(1)} + b^{(2)} \right)$$

Output length: $T_2 = \lfloor T_1 / s_2 \rfloor = \lfloor T / (s_1 \cdot s_2) \rfloor$

**Projection to latent**:
$$z_e^{(i)} = W^{proj} \cdot h_i^{(2)} + b^{proj}$$

**Total temporal stride**: $s_{total} = s_1 \cdot s_2$

With two layers of stride=2: $s_{total} = 4$

### Receptive Field

Each output code "sees" multiple input frames through the conv layers:

```
Two Conv layers (stride=2, kernel=3 each):

Layer 2 output z_e⁰ sees:
    └── Layer 1 outputs h₀, h₁, h₂ (kernel=3)
            │      │      │
            ▼      ▼      ▼
        ┌───┴───┐ ┌┴┐ ┌───┴───┐
        │       │ │ │ │       │
Layer 1: h₀ sees   h₁ sees   h₂ sees
        x₀,x₁,x₂  x₂,x₃,x₄  x₄,x₅,x₆

Effective receptive field of z_e⁰: frames x₀ through x₆ (7 frames!)
```

The convolution **learns** which temporal patterns matter, unlike pooling which just averages.

### Detailed Example

```
Input: 8 frames, Conv with 2 layers (stride=2 each)

Frame:        x₀      x₁      x₂      x₃      x₄      x₅      x₆      x₇
              │       │       │       │       │       │       │       │
              └───────┼───────┘       └───────┼───────┘       │       │
                      │                       │               │       │
Layer 1               ▼                       ▼               ▼       ▼
(stride=2):          h₀                      h₁              h₂      h₃
                      │                       │               │       │
                      └───────────┬───────────┘               └───┬───┘
                                  │                               │
Layer 2                           ▼                               ▼
(stride=2):                     z_e⁰                            z_e¹
                                  │                               │
                                  ▼                               ▼
Quantize:                      Code 3                          Code 3
                              "walking"                       "walking"

8 frames → 4 hidden → 2 codes
Total stride = 2 × 2 = 4 frames per code
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Learned temporal features | More parameters |
| Overlapping receptive fields | Requires new encoder architecture |
| Can capture motion patterns | Slightly more complex |
| Flexible (adjust layers/strides) | Need to tune kernel size |

---

## Comparison: Pooling vs Convolution

### Visual Comparison

```
                        POOLING                          CONVOLUTION

Input:    [x₀][x₁][x₂][x₃][x₄][x₅][x₆][x₇]    [x₀][x₁][x₂][x₃][x₄][x₅][x₆][x₇]
              │   │   │   │   │   │   │   │          \   |   /       \   |   /
              │   │   │   │   │   │   │   │           \  |  /         \  |  /
              ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼            \ | /           \ | /
MLP:        [z₀][z₁][z₂][z₃][z₄][z₅][z₆][z₇]           \|/             \|/
              │   │   │   │   │   │   │   │              │               │
              └───┴───┴───┘   └───┴───┴───┘         [h₀][h₁][h₂][h₃]   (stride=2)
                    │               │                 \   |   /   |
                    │               │                  \  |  /    |
                    ▼ pool          ▼ pool              \ | /     |
                                                        \|/      |
                  [z_e⁰]          [z_e¹]              [z_e⁰]  [z_e¹] (stride=2)
                    │               │                    │       │
                    ▼               ▼                    ▼       ▼
                  Code            Code                 Code    Code

Each code sees:   4 frames        4 frames           ~7 frames  ~7 frames
                 (disjoint)      (disjoint)        (overlapping)(overlapping)
```

### Feature Comparison

| Aspect | Pooling | Convolution |
|--------|---------|-------------|
| **Implementation** | Add pooling after MLP | New encoder architecture |
| **Parameters** | None added | Conv kernels learned |
| **Chunk boundaries** | Fixed, non-overlapping | Overlapping receptive fields |
| **Temporal features** | Only through averaging | Learned motion patterns |
| **Information loss** | Mean = average pose | Selective feature extraction |
| **Complexity** | Very simple | Moderate |

### When to Use Which

**Use Pooling when:**
- You want a quick experiment
- The MLP already extracts good per-frame features
- You want minimal code changes

**Use Convolution when:**
- You want the encoder to learn temporal patterns
- Motion dynamics matter (e.g., velocity, acceleration)
- You have enough data to train conv layers

---

## Upsampling for Action Prediction

Both approaches need to **upsample** back to per-frame for action prediction:

```
After quantization (stride=4):

Codes:            [Code 3]              [Code 3]              [Code 3]
                      │                     │                     │
                      ▼                     ▼                     ▼
Upsample         ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
(repeat):        │ │ │ │ │ │          │ │ │ │ │ │          │ │ │ │ │ │
                 ▼ ▼ ▼ ▼ ▼ ▼          ▼ ▼ ▼ ▼ ▼ ▼          ▼ ▼ ▼ ▼ ▼ ▼
z_q repeated:   [3][3][3][3]         [3][3][3][3]         [3][3][3][3]
                 │  │  │  │           │  │  │  │           │  │  │  │
                 +  +  +  +           +  +  +  +           +  +  +  +
proprio:        [p₀][p₁][p₂][p₃]     [p₄][p₅][p₆][p₇]     [p₈][p₉][p₁₀][p₁₁]
                 │  │  │  │           │  │  │  │           │  │  │  │
                 ▼  ▼  ▼  ▼           ▼  ▼  ▼  ▼           ▼  ▼  ▼  ▼
Decoder:       [a₀][a₁][a₂][a₃]     [a₄][a₅][a₆][a₇]     [a₈][a₉][a₁₀][a₁₁]
```

Mathematically:
$$\tilde{z}_q^{(t)} = z_q^{(\lfloor t/s \rfloor)} \quad \forall t \in [0, T)$$

Then decode with per-frame proprioception:
$$a_t = \text{Decoder}([\tilde{z}_q^{(t)}; p_t])$$

---

## Choosing the Stride

### Trade-offs

| Stride | Frames/Code | Time/Code (50Hz) | Codes for 20 frames | Semantic Level |
|--------|-------------|------------------|---------------------|----------------|
| 1 | 1 | 20ms | 20 | Pose (too fine) |
| 2 | 2 | 40ms | 10 | Sub-motif |
| **4** | **4** | **80ms** | **5** | **Motif (recommended)** |
| 8 | 8 | 160ms | 2 | Coarse behavior |
| 16 | 16 | 320ms | 1 | Very coarse |

### Recommended: Stride = 4

- **80ms** ≈ duration of a single behavioral motif
- Enough codes per unroll (5) for PPO credit assignment
- Matches biological timescales for motor primitives

### Behavior Timescale Hierarchy

```
├── 20ms   - Single physics step / muscle twitch
├── 80ms   - Motor primitive / motif      ← TARGET (stride=4)
├── 320ms  - Behavioral bout
├── 1-10s  - Behavioral episode
└── minutes - Session-level patterns
```

---

## Expected Outcome

### Before: All Clips Use All Codes

```
Walking Clip:     3 5 7 5 3 5 7 5 3 5 7 5 ...   Uses: {3, 5, 7}
Rearing Clip:     3 5 7 5 3 5 7 5 3 5 7 5 ...   Uses: {3, 5, 7}  (same!)
Grooming Clip:    3 5 7 5 3 5 7 5 3 5 7 5 ...   Uses: {3, 5, 7}  (same!)
```

### After: Different Clips Use Different Code Subsets

```
Walking Clip:     [3, 3, 3, 3, 3, 3, 3, ...]    Uses: {3}
Rearing Clip:     [3, 3, 5, 5, 5, 3, 3, ...]    Uses: {3, 5}
Grooming Clip:    [7, 7, 7, 7, 7, 7, 7, ...]    Uses: {7}
Complex Clip:     [3, 3, 5, 5, 7, 7, 3, ...]    Uses: {3, 5, 7}
```

---

## Summary

| Aspect | Current | Pooling Approach | Conv Approach |
|--------|---------|------------------|---------------|
| Codes per 20 frames | 20 | 5 | 5 |
| Code meaning | Pose phase | Behavioral chunk | Behavioral chunk |
| Implementation | MLP | MLP + pool | Conv encoder |
| Learned aggregation | No | No | Yes |
| Complexity | - | Low | Medium |

**Key insight**: By forcing multiple frames to share a code, we shift from encoding "what pose" to encoding "what behavior".

---

## References

- VQ-MAP: Behavioral Representation Learning (2025) - Temporal conv blocks for semantic codes
- VQ-VAE: Neural Discrete Representation Learning (2017) - Original VQ-VAE paper
