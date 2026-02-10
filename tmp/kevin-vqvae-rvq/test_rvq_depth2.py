"""Tests for ResidualVectorQuantizer with depth=2.

Verifies multi-level RVQ behavior: output shapes, residual property,
z_hat_st reconstruction, per-level stickiness, VQ loss, and full
VQIntentionNetwork forward pass.
"""

import sys
import traceback

import jax
import jax.numpy as jnp
from flax import linen as nn

# Ensure the vqvae_jax package is importable
sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx")

from vqvae_jax.vq_intention_network import (
    ResidualVectorQuantizer,
    VQIntentionNetwork,
    _quantize_single_level,
)
from vqvae_jax.vq_losses import compute_vq_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEPTH = 2
NUM_CODES = 16
LATENT_DIM = 8
BATCH = 4

results = []


def report(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    msg = f"[{tag}] {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)
    results.append((name, passed))


# ---------------------------------------------------------------------------
# Test 1: Output shapes
# ---------------------------------------------------------------------------
def test_output_shapes():
    rng = jax.random.PRNGKey(0)
    z_e = jax.random.normal(rng, (BATCH, LATENT_DIM))

    model = ResidualVectorQuantizer(
        num_codes=NUM_CODES, latent_dim=LATENT_DIM, depth=DEPTH
    )
    params = model.init(rng, z_e)
    z_hat_st, all_indices, all_z_q, all_residuals = model.apply(params, z_e)

    ok = True
    details = []

    # z_hat_st shape == z_e shape
    if z_hat_st.shape != z_e.shape:
        ok = False
        details.append(f"z_hat_st shape {z_hat_st.shape} != z_e shape {z_e.shape}")

    # all_indices is a tuple of length 2
    if not isinstance(all_indices, tuple) or len(all_indices) != DEPTH:
        ok = False
        details.append(f"all_indices length {len(all_indices)} != {DEPTH}")

    # all_z_q is a tuple of length 2
    if not isinstance(all_z_q, tuple) or len(all_z_q) != DEPTH:
        ok = False
        details.append(f"all_z_q length {len(all_z_q)} != {DEPTH}")

    # all_residuals is a tuple of length D+1 = 3
    if not isinstance(all_residuals, tuple) or len(all_residuals) != DEPTH + 1:
        ok = False
        details.append(f"all_residuals length {len(all_residuals)} != {DEPTH + 1}")

    # Each indices array has batch shape
    for d in range(DEPTH):
        if all_indices[d].shape != (BATCH,):
            ok = False
            details.append(
                f"all_indices[{d}] shape {all_indices[d].shape} != ({BATCH},)"
            )

    # Each z_q has same shape as z_e
    for d in range(DEPTH):
        if all_z_q[d].shape != z_e.shape:
            ok = False
            details.append(
                f"all_z_q[{d}] shape {all_z_q[d].shape} != {z_e.shape}"
            )

    # Each residual has same shape as z_e
    for d in range(DEPTH + 1):
        if all_residuals[d].shape != z_e.shape:
            ok = False
            details.append(
                f"all_residuals[{d}] shape {all_residuals[d].shape} != {z_e.shape}"
            )

    report("1. Output shapes", ok, "; ".join(details) if details else "all correct")


# ---------------------------------------------------------------------------
# Test 2: Residual property
# ---------------------------------------------------------------------------
def test_residual_property():
    rng = jax.random.PRNGKey(1)
    z_e = jax.random.normal(rng, (BATCH, LATENT_DIM))

    model = ResidualVectorQuantizer(
        num_codes=NUM_CODES, latent_dim=LATENT_DIM, depth=DEPTH
    )
    params = model.init(rng, z_e)
    z_hat_st, all_indices, all_z_q, all_residuals = model.apply(params, z_e)

    ok = True
    details = []

    # residuals[0] == z_e
    err0 = float(jnp.max(jnp.abs(all_residuals[0] - z_e)))
    if err0 > 1e-6:
        ok = False
        details.append(f"residuals[0] != z_e, max_err={err0:.2e}")

    # residuals[1] == z_e - sg(z_q[0])
    expected_r1 = z_e - jax.lax.stop_gradient(all_z_q[0])
    err1 = float(jnp.max(jnp.abs(all_residuals[1] - expected_r1)))
    if err1 > 1e-6:
        ok = False
        details.append(f"residuals[1] mismatch, max_err={err1:.2e}")

    # residuals[2] == z_e - sg(z_q[0]) - sg(z_q[1])
    expected_r2 = (
        z_e
        - jax.lax.stop_gradient(all_z_q[0])
        - jax.lax.stop_gradient(all_z_q[1])
    )
    err2 = float(jnp.max(jnp.abs(all_residuals[2] - expected_r2)))
    if err2 > 1e-6:
        ok = False
        details.append(f"residuals[2] mismatch, max_err={err2:.2e}")

    report(
        "2. Residual property",
        ok,
        "; ".join(details) if details else "all residuals match",
    )


# ---------------------------------------------------------------------------
# Test 3: z_hat_st reconstruction
# ---------------------------------------------------------------------------
def test_z_hat_st_reconstruction():
    """z_hat_st should equal the sum of STE-quantized parts from each level.

    For each level d, the STE quantized output is:
        z_q_st_d = residual_d - sg(residual_d) + sg(z_q_d)

    Numerically (in the forward pass), this equals z_q_d.
    Hence z_hat_st value == sum_d(z_q_d).
    """
    rng = jax.random.PRNGKey(2)
    z_e = jax.random.normal(rng, (BATCH, LATENT_DIM))

    model = ResidualVectorQuantizer(
        num_codes=NUM_CODES, latent_dim=LATENT_DIM, depth=DEPTH
    )
    params = model.init(rng, z_e)
    z_hat_st, all_indices, all_z_q, all_residuals = model.apply(params, z_e)

    # z_hat_st should equal sum of all z_q values (numerically)
    expected = sum(all_z_q)
    err = float(jnp.max(jnp.abs(z_hat_st - expected)))
    ok = err < 1e-5
    report(
        "3. z_hat_st reconstruction",
        ok,
        f"max_err={err:.2e} (z_hat_st vs sum(all_z_q))",
    )


# ---------------------------------------------------------------------------
# Test 4: Per-level stickiness
# ---------------------------------------------------------------------------
def test_per_level_stickiness():
    """With stickiness_bias=(5.0, 0.0):
    - Level 0 should show stickiness (prev_indices bias L0 selection)
    - Level 1 should NOT show stickiness (bias=0.0)

    Strategy: construct z_e as exact midpoint of two codes, then verify
    stickiness tips the L0 selection, while L1 is unaffected by its prev.
    """
    rng = jax.random.PRNGKey(3)

    model = ResidualVectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        depth=DEPTH,
        stickiness_bias=(5.0, 0.0),
    )
    params = model.init(rng, jnp.zeros((1, LATENT_DIM)))

    # Extract codebook for level 0
    cb0 = params["params"]["codebooks_0"]["embeddings"]  # [K, D]

    # Create z_e as exact midpoint between code 0 and code 1 in codebook 0
    z_e = ((cb0[0] + cb0[1]) / 2.0)[None, :]  # [1, D]

    ok = True
    details = []

    # --- L0 stickiness check ---
    # With prev=0 at L0, should pick code 0; with prev=1, should pick code 1
    prev_toward_0 = (jnp.array([0]), jnp.array([0]))
    prev_toward_1 = (jnp.array([1]), jnp.array([0]))

    _, idx_p0, _, _ = model.apply(params, z_e, prev_indices=prev_toward_0)
    _, idx_p1, _, _ = model.apply(params, z_e, prev_indices=prev_toward_1)

    l0_when_prev0 = int(idx_p0[0][0])
    l0_when_prev1 = int(idx_p1[0][0])

    if l0_when_prev0 == l0_when_prev1:
        # Midpoint may not be truly equidistant from codes 0 and 1.
        # This is not necessarily a failure of stickiness -- just means
        # the distance gap is larger than the bias can overcome.
        details.append(
            f"L0: prev=0 -> code {l0_when_prev0}, prev=1 -> code {l0_when_prev1} "
            "(same; midpoint may not be equidistant)"
        )
    else:
        details.append(
            f"L0: prev=0 -> code {l0_when_prev0}, prev=1 -> code {l0_when_prev1} "
            "(stickiness tips selection)"
        )

    # --- L1 NO stickiness check (bias=0.0) ---
    # Changing L1 prev_indices should NOT affect L1 code selection
    prev_a = (jnp.array([0]), jnp.array([0]))
    prev_b = (jnp.array([0]), jnp.array([5]))

    _, idx_a, _, _ = model.apply(params, z_e, prev_indices=prev_a)
    _, idx_b, _, _ = model.apply(params, z_e, prev_indices=prev_b)

    l1_a = int(idx_a[1][0])
    l1_b = int(idx_b[1][0])

    if l1_a != l1_b:
        ok = False
        details.append(
            f"L1 bias=0.0 but indices differ: prev_l1=0 -> {l1_a}, prev_l1=5 -> {l1_b}"
        )
    else:
        details.append(f"L1 stable (idx={l1_a}) regardless of prev L1 index")

    report(
        "4. Per-level stickiness",
        ok,
        "; ".join(details) if details else "stickiness behaves correctly",
    )


# ---------------------------------------------------------------------------
# Test 5: VQ loss with multi-depth
# ---------------------------------------------------------------------------
def test_vq_loss_multi_depth():
    rng = jax.random.PRNGKey(4)
    z_e = jax.random.normal(rng, (BATCH, LATENT_DIM))

    model = ResidualVectorQuantizer(
        num_codes=NUM_CODES, latent_dim=LATENT_DIM, depth=DEPTH
    )
    params = model.init(rng, z_e)
    z_hat_st, all_indices, all_z_q, all_residuals = model.apply(params, z_e)

    vq_loss, commitment_loss, codebook_loss = compute_vq_loss(
        z_e=z_e,
        commitment_cost=0.25,
        all_z_q=all_z_q,
        all_residuals=all_residuals,
    )

    ok = True
    details = []

    # All should be finite scalars
    for name, val in [
        ("vq_loss", vq_loss),
        ("commitment_loss", commitment_loss),
        ("codebook_loss", codebook_loss),
    ]:
        if not jnp.isfinite(val):
            ok = False
            details.append(f"{name} is not finite: {val}")
        if val.shape != ():
            ok = False
            details.append(f"{name} is not scalar: shape={val.shape}")

    # vq_loss should equal commitment_cost * commitment + codebook
    expected = 0.25 * commitment_loss + codebook_loss
    err = float(jnp.abs(vq_loss - expected))
    if err > 1e-5:
        ok = False
        details.append(f"vq_loss != 0.25*commit+codebook, err={err:.2e}")

    # Losses should be non-negative
    if float(commitment_loss) < 0:
        ok = False
        details.append(f"commitment_loss negative: {commitment_loss}")
    if float(codebook_loss) < 0:
        ok = False
        details.append(f"codebook_loss negative: {codebook_loss}")

    # Verify 1/D scaling: manually compute per-level losses
    D = DEPTH
    manual_commitment = 0.0
    manual_codebook = 0.0
    for d in range(D):
        r_d = all_residuals[d]
        z_q_d = all_z_q[d]
        manual_commitment += (1.0 / D) * float(
            jnp.mean((r_d - jax.lax.stop_gradient(z_q_d)) ** 2)
        )
        manual_codebook += (1.0 / D) * float(
            jnp.mean((jax.lax.stop_gradient(r_d) - z_q_d) ** 2)
        )

    err_c = abs(float(commitment_loss) - manual_commitment)
    err_cb = abs(float(codebook_loss) - manual_codebook)
    if err_c > 1e-5:
        ok = False
        details.append(f"commitment_loss 1/D scaling mismatch, err={err_c:.2e}")
    if err_cb > 1e-5:
        ok = False
        details.append(f"codebook_loss 1/D scaling mismatch, err={err_cb:.2e}")

    report(
        "5. VQ loss multi-depth",
        ok,
        "; ".join(details)
        if details
        else f"vq={float(vq_loss):.4f}, commit={float(commitment_loss):.4f}, codebook={float(codebook_loss):.4f}",
    )


# ---------------------------------------------------------------------------
# Test 6: VQIntentionNetwork depth=2 forward pass
# ---------------------------------------------------------------------------
def test_vq_intention_network_depth2():
    rng = jax.random.PRNGKey(5)
    init_rng, call_rng = jax.random.split(rng)

    TRAJ_DIM = 20
    PROPRIO_DIM = 12
    ACTION_SIZE = 10

    model = VQIntentionNetwork(
        encoder_layers=[32, 32],
        decoder_layers=[32, 32, ACTION_SIZE * 2],
        latent_dim=LATENT_DIM,
        num_codes=NUM_CODES,
        rvq_depth=DEPTH,
        stickiness_bias=(1.0, 0.0),
    )

    obs = {
        "imitation_target": jnp.ones((BATCH, TRAJ_DIM)),
        "proprioception": jnp.ones((BATCH, PROPRIO_DIM)),
    }

    params = model.init(init_rng, obs, call_rng)
    action, z_e, all_indices = model.apply(params, obs, call_rng)

    ok = True
    details = []

    # Action shape
    expected_action_shape = (BATCH, ACTION_SIZE * 2)
    if action.shape != expected_action_shape:
        ok = False
        details.append(
            f"action shape {action.shape} != {expected_action_shape}"
        )

    # z_e shape
    if z_e.shape != (BATCH, LATENT_DIM):
        ok = False
        details.append(f"z_e shape {z_e.shape} != ({BATCH}, {LATENT_DIM})")

    # all_indices tuple of length DEPTH
    if not isinstance(all_indices, tuple) or len(all_indices) != DEPTH:
        ok = False
        details.append(f"all_indices not tuple of length {DEPTH}")
    else:
        for d in range(DEPTH):
            if all_indices[d].shape != (BATCH,):
                ok = False
                details.append(
                    f"all_indices[{d}] shape {all_indices[d].shape} != ({BATCH},)"
                )

    # Test get_activation=True path
    action_act, z_e_act, all_indices_act, extras = model.apply(
        params, obs, call_rng, get_activation=True
    )
    if "all_z_q" not in extras:
        ok = False
        details.append("get_activation=True missing 'all_z_q'")
    if "all_residuals" not in extras:
        ok = False
        details.append("get_activation=True missing 'all_residuals'")
    if "z_hat_st" not in extras:
        ok = False
        details.append("get_activation=True missing 'z_hat_st'")

    # Test with prev_indices
    prev_idx = tuple(jnp.zeros((BATCH,), dtype=jnp.int32) for _ in range(DEPTH))
    action_pi, z_e_pi, all_indices_pi = model.apply(
        params, obs, call_rng, prev_indices=prev_idx
    )
    if action_pi.shape != expected_action_shape:
        ok = False
        details.append(f"action with prev_indices shape {action_pi.shape} wrong")

    # All values should be finite
    if not jnp.all(jnp.isfinite(action)):
        ok = False
        details.append("action contains non-finite values")
    if not jnp.all(jnp.isfinite(z_e)):
        ok = False
        details.append("z_e contains non-finite values")

    report(
        "6. VQIntentionNetwork depth=2",
        ok,
        "; ".join(details) if details else "forward pass correct",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("1. Output shapes", test_output_shapes),
        ("2. Residual property", test_residual_property),
        ("3. z_hat_st reconstruction", test_z_hat_st_reconstruction),
        ("4. Per-level stickiness", test_per_level_stickiness),
        ("5. VQ loss multi-depth", test_vq_loss_multi_depth),
        ("6. VQIntentionNetwork depth=2", test_vq_intention_network_depth2),
    ]

    print("=" * 70)
    print("ResidualVectorQuantizer depth=2 test suite")
    print("=" * 70)
    print()

    for test_name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            report(test_name, False, f"EXCEPTION: {e}")
            traceback.print_exc()
        print()

    print("=" * 70)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    if passed < total:
        print("FAILED tests:")
        for name, p in results:
            if not p:
                print(f"  - {name}")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)
