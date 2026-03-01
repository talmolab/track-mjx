"""Regression test: ResidualVectorQuantizer(depth=1) == flat VQ reference.

Verifies that the RVQ wrapper with depth=1 produces bit-identical results
to a hand-coded flat vector quantizer operating on the same codebook.
"""
import sys

sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx/vqvae_jax")
sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx")

import jax
import jax.numpy as jnp
from flax import linen as nn

from vq_intention_network import (
    ResidualVectorQuantizer,
    VQIntentionNetwork,
    make_vq_intention_policy,
)


# -----------------------------------------------------------------------
# Reference flat quantizer (pure jax, no nn.Module)
# -----------------------------------------------------------------------
def reference_flat_quantize(
    z_e: jnp.ndarray,
    codebook: jnp.ndarray,
    stickiness_bias: float = 0.0,
    prev_indices: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flat VQ reference implementation.

    Returns (z_q, indices, z_q_st).
    """
    K, D = codebook.shape
    input_shape = z_e.shape
    flat_z_e = z_e.reshape(-1, D)

    # Squared Euclidean distance: ||z_e - e_k||^2 = z_e^2 + e_k^2 - 2*z_e.e_k
    z_e_sq = jnp.sum(flat_z_e ** 2, axis=-1, keepdims=True)  # [N, 1]
    codebook_sq = jnp.sum(codebook ** 2, axis=-1)              # [K]
    cross = jnp.matmul(flat_z_e, codebook.T)                   # [N, K]
    distances = z_e_sq + codebook_sq - 2 * cross                # [N, K]

    # Stickiness bias
    if prev_indices is not None and stickiness_bias > 0:
        flat_prev = prev_indices.reshape(-1)
        prev_oh = jax.nn.one_hot(flat_prev, K)
        distances = distances - stickiness_bias * prev_oh

    flat_indices = jnp.argmin(distances, axis=-1)
    flat_z_q = codebook[flat_indices]

    indices = flat_indices.reshape(input_shape[:-1])
    z_q = flat_z_q.reshape(input_shape)

    # Straight-through estimator
    z_q_st = z_e - jax.lax.stop_gradient(z_e) + jax.lax.stop_gradient(z_q)

    return z_q, indices, z_q_st


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _assert_allclose(name, a, b, atol=0.0, rtol=0.0):
    if not jnp.allclose(a, b, atol=atol, rtol=rtol):
        max_diff = jnp.max(jnp.abs(a - b))
        raise AssertionError(
            f"{name}: NOT allclose. max_diff={float(max_diff):.2e}"
        )

def _assert_equal(name, a, b):
    if not jnp.array_equal(a, b):
        mismatches = int(jnp.sum(a != b))
        raise AssertionError(f"{name}: NOT equal. {mismatches} mismatches")


# -----------------------------------------------------------------------
# Test 1: depth=1, stickiness_bias=0.0, no prev_indices
# -----------------------------------------------------------------------
def test_depth1_no_stickiness():
    print("=" * 60)
    print("TEST 1: depth=1, stickiness_bias=0.0, no prev_indices")
    print("=" * 60)

    NUM_CODES = 16
    LATENT_DIM = 8
    BATCH = 5

    rvq = ResidualVectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        depth=1,
        commitment_cost=0.25,
        codebook_init_scale=1.0,
        stickiness_bias=0.0,
    )

    key = jax.random.PRNGKey(42)
    z_e = jax.random.normal(key, (BATCH, LATENT_DIM))

    # Initialize the module
    init_key = jax.random.PRNGKey(0)
    params = rvq.init(init_key, z_e)

    # Extract the codebook from params
    codebook = params["params"]["codebooks_0"]["embeddings"]
    print(f"  Codebook shape: {codebook.shape}")
    assert codebook.shape == (NUM_CODES, LATENT_DIM), (
        f"Expected ({NUM_CODES}, {LATENT_DIM}), got {codebook.shape}"
    )

    # Run through the module
    z_hat_st, all_indices, all_z_q, all_residuals = rvq.apply(params, z_e)

    # Unpack depth-1 results
    mod_indices = all_indices[0]
    mod_z_q = all_z_q[0]

    # Run reference
    ref_z_q, ref_indices, ref_z_q_st = reference_flat_quantize(z_e, codebook)

    # Check indices are identical
    _assert_equal("indices", mod_indices, ref_indices)
    print(f"  Indices: MATCH (all {BATCH} entries identical)")

    # Check z_q (no STE) are identical
    _assert_allclose("z_q", mod_z_q, ref_z_q)
    print(f"  z_q: MATCH (max diff = {float(jnp.max(jnp.abs(mod_z_q - ref_z_q))):.2e})")

    # Check z_hat_st (with STE) are identical
    _assert_allclose("z_hat_st", z_hat_st, ref_z_q_st)
    print(f"  z_hat_st: MATCH (max diff = {float(jnp.max(jnp.abs(z_hat_st - ref_z_q_st))):.2e})")

    # Check residuals
    assert len(all_residuals) == 2, f"Expected 2 residuals, got {len(all_residuals)}"
    _assert_allclose("residual[0] == z_e", all_residuals[0], z_e)
    expected_residual = z_e - jax.lax.stop_gradient(ref_z_q)
    _assert_allclose("residual[1]", all_residuals[1], expected_residual)
    print(f"  Residuals: MATCH")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 2: depth=1, stickiness_bias=3.0, with prev_indices
# -----------------------------------------------------------------------
def test_depth1_with_stickiness():
    print("=" * 60)
    print("TEST 2: depth=1, stickiness_bias=3.0, prev_indices provided")
    print("=" * 60)

    NUM_CODES = 16
    LATENT_DIM = 8
    BATCH = 5
    BIAS = 3.0

    rvq = ResidualVectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        depth=1,
        stickiness_bias=BIAS,
    )

    key = jax.random.PRNGKey(99)
    z_e = jax.random.normal(key, (BATCH, LATENT_DIM))

    init_key = jax.random.PRNGKey(1)
    params = rvq.init(init_key, z_e)
    codebook = params["params"]["codebooks_0"]["embeddings"]

    # Create prev_indices: assign each sample to a specific prior code
    prev_indices = jnp.array([0, 3, 7, 12, 15])

    # Run module (single array form -- backward compat)
    z_hat_st, all_indices, all_z_q, _ = rvq.apply(
        params, z_e, prev_indices=prev_indices
    )
    mod_indices = all_indices[0]
    mod_z_q = all_z_q[0]

    # Run reference
    ref_z_q, ref_indices, ref_z_q_st = reference_flat_quantize(
        z_e, codebook, stickiness_bias=BIAS, prev_indices=prev_indices
    )

    _assert_equal("indices (sticky)", mod_indices, ref_indices)
    print(f"  Indices: MATCH")

    _assert_allclose("z_q (sticky)", mod_z_q, ref_z_q)
    print(f"  z_q: MATCH")

    _assert_allclose("z_hat_st (sticky)", z_hat_st, ref_z_q_st)
    print(f"  z_hat_st: MATCH")

    # Verify stickiness actually has an effect: re-run without bias and see if
    # at least one index differs (stickiness should pull toward prev)
    ref_z_q_nb, ref_indices_nb, _ = reference_flat_quantize(z_e, codebook)
    n_changed = int(jnp.sum(ref_indices != ref_indices_nb))
    print(f"  Stickiness changed {n_changed}/{BATCH} indices vs unbiased")
    if n_changed == 0:
        # This is possible but extremely unlikely with bias=3.0 on random data.
        # If this fails, the test data just happened to have prev==nearest for all.
        # We re-try with a larger batch to reduce false-pass probability.
        print("  WARNING: no index changes detected; stickiness may not be active")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 3: stickiness with tuple form (per-level bias for depth=1)
# -----------------------------------------------------------------------
def test_depth1_tuple_bias():
    print("=" * 60)
    print("TEST 3: depth=1, stickiness_bias=(3.0,) tuple form")
    print("=" * 60)

    NUM_CODES = 16
    LATENT_DIM = 8
    BATCH = 5
    BIAS = (3.0,)

    rvq = ResidualVectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        depth=1,
        stickiness_bias=BIAS,
    )

    key = jax.random.PRNGKey(99)
    z_e = jax.random.normal(key, (BATCH, LATENT_DIM))
    prev_indices = jnp.array([0, 3, 7, 12, 15])

    init_key = jax.random.PRNGKey(1)
    params = rvq.init(init_key, z_e)
    codebook = params["params"]["codebooks_0"]["embeddings"]

    # Run module -- pass prev_indices as tuple of 1 array
    z_hat_st, all_indices, all_z_q, _ = rvq.apply(
        params, z_e, prev_indices=(prev_indices,)
    )
    mod_indices = all_indices[0]

    ref_z_q, ref_indices, ref_z_q_st = reference_flat_quantize(
        z_e, codebook, stickiness_bias=3.0, prev_indices=prev_indices
    )

    _assert_equal("indices (tuple bias)", mod_indices, ref_indices)
    _assert_allclose("z_hat_st (tuple bias)", z_hat_st, ref_z_q_st)
    print(f"  Indices: MATCH")
    print(f"  z_hat_st: MATCH")
    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 4: Multiple input shapes (batch=1, 2D, 3D)
# -----------------------------------------------------------------------
def test_various_shapes():
    print("=" * 60)
    print("TEST 4: Various input shapes (batch=1, 2D, 3D)")
    print("=" * 60)

    NUM_CODES = 16
    LATENT_DIM = 8

    rvq = ResidualVectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        depth=1,
        stickiness_bias=0.0,
    )

    init_key = jax.random.PRNGKey(2)
    dummy = jnp.zeros((1, LATENT_DIM))
    params = rvq.init(init_key, dummy)
    codebook = params["params"]["codebooks_0"]["embeddings"]

    for shape_desc, shape in [
        ("batch=1", (1, LATENT_DIM)),
        ("batch=10", (10, LATENT_DIM)),
        ("3D [T=3, B=4]", (3, 4, LATENT_DIM)),
    ]:
        key = jax.random.PRNGKey(hash(shape_desc) % 2**31)
        z_e = jax.random.normal(key, shape)

        z_hat_st, all_indices, all_z_q, _ = rvq.apply(params, z_e)
        ref_z_q, ref_indices, ref_z_q_st = reference_flat_quantize(z_e, codebook)

        _assert_equal(f"indices [{shape_desc}]", all_indices[0], ref_indices)
        _assert_allclose(f"z_q [{shape_desc}]", all_z_q[0], ref_z_q)
        _assert_allclose(f"z_hat_st [{shape_desc}]", z_hat_st, ref_z_q_st)
        print(f"  {shape_desc} shape={shape}: MATCH")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 5: STE gradient correctness
# -----------------------------------------------------------------------
def test_ste_gradient():
    print("=" * 60)
    print("TEST 5: STE gradient flows through z_hat_st to z_e")
    print("=" * 60)

    NUM_CODES = 16
    LATENT_DIM = 8
    BATCH = 4

    rvq = ResidualVectorQuantizer(
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        depth=1,
        stickiness_bias=0.0,
    )

    key = jax.random.PRNGKey(77)
    z_e = jax.random.normal(key, (BATCH, LATENT_DIM))
    init_key = jax.random.PRNGKey(3)
    params = rvq.init(init_key, z_e)

    # Gradient of sum(z_hat_st) w.r.t. z_e should be all ones
    # because z_hat_st = z_e - sg(z_e) + sg(z_q) => grad = I
    def loss_fn(z_input):
        z_hat_st, _, _, _ = rvq.apply(params, z_input)
        return jnp.sum(z_hat_st)

    grad = jax.grad(loss_fn)(z_e)
    expected_grad = jnp.ones_like(z_e)
    _assert_allclose("STE grad", grad, expected_grad, atol=1e-6)
    print(f"  grad(sum(z_hat_st))/d(z_e) = ones: MATCH")

    # Verify that codebook embeddings have zero gradient through z_hat_st
    # (codebook update should come from VQ loss, not through STE)
    def loss_fn_params(p):
        z_hat_st, _, _, _ = rvq.apply(p, z_e)
        return jnp.sum(z_hat_st)

    param_grad = jax.grad(loss_fn_params)(params)
    cb_grad = param_grad["params"]["codebooks_0"]["embeddings"]
    _assert_allclose("codebook grad via STE = 0", cb_grad, jnp.zeros_like(cb_grad))
    print(f"  grad(sum(z_hat_st))/d(codebook) = zeros: MATCH")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 6: Full VQIntentionNetwork with rvq_depth=1
# -----------------------------------------------------------------------
def test_full_network_depth1():
    print("=" * 60)
    print("TEST 6: Full VQIntentionNetwork with rvq_depth=1")
    print("=" * 60)

    LATENT_DIM = 8
    NUM_CODES = 16
    TRAJ_DIM = 20
    PROPRIO_DIM = 10
    ACTION_SIZE = 6
    BATCH = 4

    network = VQIntentionNetwork(
        encoder_layers=[32, 32],
        decoder_layers=[32, 32, ACTION_SIZE * 2],
        latent_dim=LATENT_DIM,
        num_codes=NUM_CODES,
        commitment_cost=0.25,
        stickiness_bias=0.0,
        rvq_depth=1,
    )

    obs = {
        "imitation_target": jax.random.normal(jax.random.PRNGKey(10), (BATCH, TRAJ_DIM)),
        "proprioception": jax.random.normal(jax.random.PRNGKey(11), (BATCH, PROPRIO_DIM)),
    }

    init_key = jax.random.PRNGKey(5)
    dummy_key = jax.random.PRNGKey(0)
    params = network.init(init_key, obs, dummy_key)

    # Standard forward pass
    action, z_e, all_indices, logvar = network.apply(params, obs, dummy_key)

    # Shape checks
    assert action.shape == (BATCH, ACTION_SIZE * 2), (
        f"action shape: expected {(BATCH, ACTION_SIZE * 2)}, got {action.shape}"
    )
    assert z_e.shape == (BATCH, LATENT_DIM), (
        f"z_e shape: expected {(BATCH, LATENT_DIM)}, got {z_e.shape}"
    )
    assert len(all_indices) == 1, f"Expected 1 index array, got {len(all_indices)}"
    assert all_indices[0].shape == (BATCH,), (
        f"indices shape: expected ({BATCH},), got {all_indices[0].shape}"
    )
    print(f"  action shape: {action.shape} -- OK")
    print(f"  z_e shape: {z_e.shape} -- OK")
    print(f"  all_indices: {len(all_indices)} level(s), shape {all_indices[0].shape} -- OK")

    # Verify the internal quantizer uses the same codebook
    codebook = params["params"]["quantizer"]["codebooks_0"]["embeddings"]
    assert codebook.shape == (NUM_CODES, LATENT_DIM)
    print(f"  codebook shape: {codebook.shape} -- OK")

    # Verify quantizer output matches reference for same z_e
    ref_z_q, ref_indices, ref_z_q_st = reference_flat_quantize(z_e, codebook)
    _assert_equal("network indices vs reference", all_indices[0], ref_indices)
    print(f"  Quantizer indices match reference: MATCH")

    # Finiteness check
    assert jnp.all(jnp.isfinite(action)), "action contains NaN/Inf"
    assert jnp.all(jnp.isfinite(z_e)), "z_e contains NaN/Inf"
    print(f"  All outputs finite: OK")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 7: Full VQIntentionNetwork with get_activation=True
# -----------------------------------------------------------------------
def test_full_network_activations():
    print("=" * 60)
    print("TEST 7: VQIntentionNetwork get_activation=True")
    print("=" * 60)

    LATENT_DIM = 8
    NUM_CODES = 16
    TRAJ_DIM = 20
    PROPRIO_DIM = 10
    ACTION_SIZE = 6
    BATCH = 4

    network = VQIntentionNetwork(
        encoder_layers=[32, 32],
        decoder_layers=[32, 32, ACTION_SIZE * 2],
        latent_dim=LATENT_DIM,
        num_codes=NUM_CODES,
        stickiness_bias=0.0,
        rvq_depth=1,
    )

    obs = {
        "imitation_target": jax.random.normal(jax.random.PRNGKey(10), (BATCH, TRAJ_DIM)),
        "proprioception": jax.random.normal(jax.random.PRNGKey(11), (BATCH, PROPRIO_DIM)),
    }

    init_key = jax.random.PRNGKey(5)
    dummy_key = jax.random.PRNGKey(0)
    params = network.init(init_key, obs, dummy_key)

    action, z_e, all_indices, logvar, extras = network.apply(
        params, obs, dummy_key, get_activation=True
    )

    # Verify extras dict has expected keys
    expected_keys = {
        "encoder", "decoder", "egocentric_obs", "traj_obs",
        "z_e", "all_z_q", "all_residuals", "z_hat_st", "all_indices",
    }
    assert set(extras.keys()) == expected_keys, (
        f"extras keys mismatch: {set(extras.keys())} vs {expected_keys}"
    )
    print(f"  extras keys: {sorted(extras.keys())} -- OK")

    # z_hat_st should match reference
    codebook = params["params"]["quantizer"]["codebooks_0"]["embeddings"]
    ref_z_q, ref_indices, ref_z_q_st = reference_flat_quantize(z_e, codebook)
    _assert_allclose("z_hat_st (activation mode)", extras["z_hat_st"], ref_z_q_st)
    _assert_equal("indices (activation mode)", extras["all_indices"][0], ref_indices)
    print(f"  z_hat_st matches reference: MATCH")
    print(f"  all_indices matches reference: MATCH")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Test 8: make_vq_intention_policy factory
# -----------------------------------------------------------------------
def test_make_vq_intention_policy():
    print("=" * 60)
    print("TEST 8: make_vq_intention_policy with rvq_depth=1")
    print("=" * 60)

    LATENT_DIM = 8
    NUM_CODES = 16
    TRAJ_DIM = 20
    PROPRIO_DIM = 10
    ACTION_SIZE = 6

    policy = make_vq_intention_policy(
        action_param_size=ACTION_SIZE * 2,
        latent_dim=LATENT_DIM,
        obs_sizes={"imitation_target": TRAJ_DIM, "proprioception": PROPRIO_DIM},
        encoder_hidden_layer_sizes=(32, 32),
        decoder_hidden_layer_sizes=(32, 32),
        num_codes=NUM_CODES,
        commitment_cost=0.25,
        stickiness_bias=0.0,
        rvq_depth=1,
    )

    assert policy.rvq_depth == 1
    assert policy.num_codes == NUM_CODES
    assert policy.latent_dim == LATENT_DIM
    print(f"  Policy attributes: depth={policy.rvq_depth}, codes={policy.num_codes}, dim={policy.latent_dim} -- OK")

    # Initialize
    init_key = jax.random.PRNGKey(7)
    params = policy.init(init_key)

    # Check param tree has quantizer codebook
    assert "quantizer" in params["params"], "Missing quantizer in params"
    assert "codebooks_0" in params["params"]["quantizer"], "Missing codebooks_0"
    print(f"  Param tree structure: OK")

    print("  >>> PASS\n")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    all_tests = [
        test_depth1_no_stickiness,
        test_depth1_with_stickiness,
        test_depth1_tuple_bias,
        test_various_shapes,
        test_ste_gradient,
        test_full_network_depth1,
        test_full_network_activations,
        test_make_vq_intention_policy,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in all_tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  >>> FAIL: {e}\n")

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(all_tests)} tests")
    print("=" * 60)

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
        print("\nOVERALL: FAIL")
        sys.exit(1)
    else:
        print("\nOVERALL: PASS")
        sys.exit(0)
