"""Tests for Householder rotation-augmented STE in _rotation_quantize.

Verifies forward correctness, gradient structure (scale * R Jacobian),
degenerate/anti-parallel fallback, codebook selection invariance, backward
regression for vanilla STE, batched shape handling, and stop-gradient safety.

References:
- STAR: arXiv:2506.03863, Fifty et al. 2024 (Eq. 7-9)
"""

import sys

sys.path.insert(0, "/home/jovyan/vast/kaiwen/track-mjx/vqvae_jax")

import jax
import jax.numpy as jnp
import pytest

from vq_intention_network import _quantize_single_level, _rotation_quantize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LATENT_DIM = 8
NUM_CODES = 16
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Test 1: Forward correctness -- _rotation_quantize returns z_q in forward
# ---------------------------------------------------------------------------
def test_rotation_quantize_forward_returns_z_q():
    """_rotation_quantize(r, z_q) should return z_q in the forward pass,
    not r or any rotation of r."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    r = jax.random.normal(k1, (5, LATENT_DIM))
    z_q = jax.random.normal(k2, (5, LATENT_DIM))
    # z_q should be stop-gradiented as the real call site does
    z_q_sg = jax.lax.stop_gradient(z_q)

    result = _rotation_quantize(r, z_q_sg)

    # Forward value must equal z_q, not r
    assert jnp.allclose(result, z_q_sg, atol=ATOL), (
        f"Forward value != z_q. max diff = {float(jnp.max(jnp.abs(result - z_q_sg))):.2e}"
    )
    assert not jnp.allclose(result, r, atol=ATOL), (
        "Forward value should NOT equal r (unless r == z_q by coincidence)"
    )


# ---------------------------------------------------------------------------
# Test 2: Gradient is NOT identity (unlike vanilla STE)
# ---------------------------------------------------------------------------
def test_rotation_gradient_is_not_identity():
    """The gradient of sum(_rotation_quantize(r, z_q)) w.r.t. r should NOT
    be all-ones. Vanilla STE gives all-ones; rotation STE gives scale * R."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    r = jax.random.normal(k1, (LATENT_DIM,))
    z_q = jax.random.normal(k2, (LATENT_DIM,))
    z_q_sg = jax.lax.stop_gradient(z_q)

    def loss_fn(r_input):
        return jnp.sum(_rotation_quantize(r_input, z_q_sg))

    grad = jax.grad(loss_fn)(r)
    ones = jnp.ones_like(r)

    # Gradient should NOT be all-ones
    assert not jnp.allclose(grad, ones, atol=1e-3), (
        "Rotation STE gradient should differ from vanilla STE (all-ones)"
    )
    # Gradient should be finite
    assert jnp.all(jnp.isfinite(grad)), "Rotation STE gradient contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 3: Gradient is scale * R (Jacobian verification)
# ---------------------------------------------------------------------------
def test_rotation_jacobian_matches_analytical():
    """For a specific r and z_q, verify the Jacobian of _rotation_quantize
    matches the analytical formula: scale * R where
        scale = ||z_q|| / ||r||
        R = I - 2*m_hat*m_hat^T + 2*q_hat*r_hat^T
        m_hat = normalize(r_hat + q_hat)
    """
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    r = jax.random.normal(k1, (LATENT_DIM,))
    z_q = jax.random.normal(k2, (LATENT_DIM,))
    z_q_sg = jax.lax.stop_gradient(z_q)

    # Compute analytical Jacobian
    eps = 1e-8
    r_norm = jnp.linalg.norm(r) + eps
    z_q_norm = jnp.linalg.norm(z_q_sg) + eps
    r_hat = r / r_norm
    q_hat = z_q_sg / z_q_norm
    scale = z_q_norm / r_norm

    m = r_hat + q_hat
    m_norm = jnp.linalg.norm(m) + eps
    m_hat = m / m_norm

    D = LATENT_DIM
    I = jnp.eye(D)
    # R = I - 2*m_hat*m_hat^T + 2*q_hat*r_hat^T
    R = I - 2.0 * jnp.outer(m_hat, m_hat) + 2.0 * jnp.outer(q_hat, r_hat)
    expected_jacobian = scale * R

    # Compute autodiff Jacobian
    def rotation_fn(r_input):
        return _rotation_quantize(r_input, z_q_sg)

    autodiff_jacobian = jax.jacobian(rotation_fn)(r)

    assert jnp.allclose(autodiff_jacobian, expected_jacobian, atol=1e-4), (
        f"Jacobian mismatch. max diff = "
        f"{float(jnp.max(jnp.abs(autodiff_jacobian - expected_jacobian))):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: Degenerate fallback (anti-parallel: r ~ -z_q)
# ---------------------------------------------------------------------------
def test_degenerate_anti_parallel_no_nan():
    """When r and z_q are anti-parallel, m = r_hat + q_hat ~ 0, which is
    degenerate. The function should fall back to vanilla STE without NaN."""
    r = jnp.array([1.0, 0.0, 0.0])
    z_q = jnp.array([-1.0, 0.0, 0.0])

    result = _rotation_quantize(r, z_q)

    # Should be finite (no NaN)
    assert jnp.all(jnp.isfinite(result)), (
        f"Anti-parallel case produced NaN/Inf: {result}"
    )
    # Forward value should still be z_q (vanilla STE fallback)
    assert jnp.allclose(result, z_q, atol=ATOL), (
        f"Anti-parallel forward != z_q. Got {result}, expected {z_q}"
    )

    # Gradient should also be finite (vanilla STE fallback => grad = ones)
    def loss_fn(r_input):
        return jnp.sum(_rotation_quantize(r_input, z_q))

    grad = jax.grad(loss_fn)(r)
    assert jnp.all(jnp.isfinite(grad)), (
        f"Anti-parallel gradient contains NaN/Inf: {grad}"
    )
    # Fallback is vanilla STE, so gradient should be identity (all-ones)
    expected_grad = jnp.ones_like(r)
    assert jnp.allclose(grad, expected_grad, atol=1e-3), (
        f"Anti-parallel fallback gradient != ones. Got {grad}"
    )


# ---------------------------------------------------------------------------
# Test 5: Near-parallel (NOT degenerate) -- uses rotation path
# ---------------------------------------------------------------------------
def test_near_parallel_uses_rotation_path():
    """When r ~ z_q (same direction, possibly different magnitude),
    m = r_hat + q_hat is NOT near zero, so the rotation path should be used,
    not the fallback. Forward value should still be z_q."""
    # Same direction, different magnitudes
    r = jnp.array([3.0, 0.0, 0.0, 0.0])
    z_q = jnp.array([1.0, 0.0, 0.0, 0.0])

    result = _rotation_quantize(r, z_q)

    # Forward should be z_q
    assert jnp.allclose(result, z_q, atol=ATOL), (
        f"Near-parallel forward != z_q. Got {result}"
    )

    # Gradient should NOT be all-ones (rotation path, not fallback)
    def loss_fn(r_input):
        return jnp.sum(_rotation_quantize(r_input, z_q))

    grad = jax.grad(loss_fn)(r)
    assert jnp.all(jnp.isfinite(grad)), f"Near-parallel gradient has NaN/Inf: {grad}"

    # For parallel vectors, the rotation matrix R maps r_hat to q_hat (= r_hat),
    # so R = I. The gradient should be scale * I applied to dr, giving scale * ones.
    # scale = ||z_q|| / ||r|| = 1/3
    scale = jnp.linalg.norm(z_q) / jnp.linalg.norm(r)
    expected_grad = scale * jnp.ones_like(r)
    assert jnp.allclose(grad, expected_grad, atol=1e-4), (
        f"Near-parallel gradient != scale*ones. Got {grad}, expected {expected_grad}"
    )

    ones = jnp.ones_like(r)
    assert not jnp.allclose(grad, ones, atol=1e-3), (
        "Near-parallel gradient should NOT equal all-ones (rotation path, not vanilla STE)"
    )


# ---------------------------------------------------------------------------
# Test 6: Codebook selection unchanged with use_rotation=True
# ---------------------------------------------------------------------------
def test_codebook_selection_unchanged_by_rotation():
    """_quantize_single_level with use_rotation=True should select the SAME
    codebook entry as with use_rotation=False. Rotation only affects
    gradients, not the argmin."""
    key = jax.random.PRNGKey(10)
    k1, k2 = jax.random.split(key)
    z_e = jax.random.normal(k1, (8, LATENT_DIM))
    codebook = jax.random.normal(k2, (NUM_CODES, LATENT_DIM))

    z_q_vanilla, indices_vanilla, z_q_st_vanilla = _quantize_single_level(
        z_e=z_e,
        codebook=codebook,
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        stickiness_bias=0.0,
        prev_indices=None,
        use_rotation=False,
    )

    z_q_rotation, indices_rotation, z_q_st_rotation = _quantize_single_level(
        z_e=z_e,
        codebook=codebook,
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        stickiness_bias=0.0,
        prev_indices=None,
        use_rotation=True,
    )

    # Indices must be identical
    assert jnp.array_equal(indices_vanilla, indices_rotation), (
        f"Indices differ: vanilla={indices_vanilla}, rotation={indices_rotation}"
    )

    # z_q (no STE) must be identical
    assert jnp.allclose(z_q_vanilla, z_q_rotation, atol=1e-7), (
        f"z_q differs. max diff = "
        f"{float(jnp.max(jnp.abs(z_q_vanilla - z_q_rotation))):.2e}"
    )

    # Forward values of z_q_st must also be identical (both equal z_q in forward)
    assert jnp.allclose(z_q_st_vanilla, z_q_st_rotation, atol=ATOL), (
        f"z_q_st forward values differ. max diff = "
        f"{float(jnp.max(jnp.abs(z_q_st_vanilla - z_q_st_rotation))):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 7: Depth=1 backward regression -- vanilla STE unchanged
# ---------------------------------------------------------------------------
def test_vanilla_ste_backward_regression():
    """With use_rotation=False, _quantize_single_level should produce vanilla
    STE: z_q_st = z_e - sg(z_e) + sg(z_q), so grad(sum(z_q_st))/dz_e = ones."""
    key = jax.random.PRNGKey(20)
    k1, k2 = jax.random.split(key)
    z_e = jax.random.normal(k1, (4, LATENT_DIM))
    codebook = jax.random.normal(k2, (NUM_CODES, LATENT_DIM))

    def loss_fn(z_input):
        _, _, z_q_st = _quantize_single_level(
            z_e=z_input,
            codebook=codebook,
            num_codes=NUM_CODES,
            latent_dim=LATENT_DIM,
            stickiness_bias=0.0,
            prev_indices=None,
            use_rotation=False,
        )
        return jnp.sum(z_q_st)

    grad = jax.grad(loss_fn)(z_e)
    expected = jnp.ones_like(z_e)

    assert jnp.allclose(grad, expected, atol=1e-6), (
        f"Vanilla STE gradient != ones. max diff = "
        f"{float(jnp.max(jnp.abs(grad - expected))):.2e}"
    )

    # Also verify the z_q_st forward value equals z_q
    z_q, _, z_q_st = _quantize_single_level(
        z_e=z_e,
        codebook=codebook,
        num_codes=NUM_CODES,
        latent_dim=LATENT_DIM,
        stickiness_bias=0.0,
        prev_indices=None,
        use_rotation=False,
    )
    assert jnp.allclose(z_q_st, z_q, atol=1e-6), (
        "Vanilla STE z_q_st != z_q in forward"
    )


# ---------------------------------------------------------------------------
# Test 8: Batched shapes -- [D], [B, D], [T, B, D]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (LATENT_DIM,),
        (4, LATENT_DIM),
        (3, 4, LATENT_DIM),
    ],
    ids=["1D", "2D_batch", "3D_time_batch"],
)
def test_batched_shapes(shape):
    """Verify _rotation_quantize handles various batch dimensions correctly."""
    key = jax.random.PRNGKey(30)
    k1, k2 = jax.random.split(key)
    r = jax.random.normal(k1, shape)
    z_q = jax.random.normal(k2, shape)

    result = _rotation_quantize(r, z_q)

    # Output shape must match input shape
    assert result.shape == shape, f"Expected shape {shape}, got {result.shape}"

    # Forward value must equal z_q
    assert jnp.allclose(result, z_q, atol=ATOL), (
        f"Forward != z_q for shape {shape}. "
        f"max diff = {float(jnp.max(jnp.abs(result - z_q))):.2e}"
    )

    # All values finite
    assert jnp.all(jnp.isfinite(result)), (
        f"Non-finite output for shape {shape}"
    )

    # Gradient should also have correct shape and be finite
    def loss_fn(r_input):
        return jnp.sum(_rotation_quantize(r_input, z_q))

    grad = jax.grad(loss_fn)(r)
    assert grad.shape == shape, f"Gradient shape {grad.shape} != {shape}"
    assert jnp.all(jnp.isfinite(grad)), (
        f"Non-finite gradient for shape {shape}"
    )


# ---------------------------------------------------------------------------
# Test 9: z_q already stop-gradiented
# ---------------------------------------------------------------------------
def test_z_q_already_stop_gradiented():
    """Verify that passing jax.lax.stop_gradient(z_q) works correctly.
    The function expects z_q to be already stop-gradiented (as done in
    _quantize_single_level). Double stop-gradient should be harmless."""
    key = jax.random.PRNGKey(50)
    k1, k2 = jax.random.split(key)
    r = jax.random.normal(k1, (4, LATENT_DIM))
    z_q_raw = jax.random.normal(k2, (4, LATENT_DIM))

    # Single stop_gradient (as the call site does)
    z_q_sg1 = jax.lax.stop_gradient(z_q_raw)
    result_1 = _rotation_quantize(r, z_q_sg1)

    # Double stop_gradient (should be identical)
    z_q_sg2 = jax.lax.stop_gradient(jax.lax.stop_gradient(z_q_raw))
    result_2 = _rotation_quantize(r, z_q_sg2)

    # Forward values should be identical
    assert jnp.allclose(result_1, result_2, atol=1e-7), (
        f"Double stop_gradient changes result. "
        f"max diff = {float(jnp.max(jnp.abs(result_1 - result_2))):.2e}"
    )

    # Forward should equal z_q
    assert jnp.allclose(result_1, z_q_raw, atol=ATOL), (
        f"Forward != z_q. max diff = "
        f"{float(jnp.max(jnp.abs(result_1 - z_q_raw))):.2e}"
    )

    # Verify that gradients w.r.t. r are finite and identical
    def loss_fn_1(r_input):
        return jnp.sum(_rotation_quantize(r_input, z_q_sg1))

    def loss_fn_2(r_input):
        return jnp.sum(_rotation_quantize(r_input, z_q_sg2))

    grad_1 = jax.grad(loss_fn_1)(r)
    grad_2 = jax.grad(loss_fn_2)(r)

    assert jnp.all(jnp.isfinite(grad_1)), "Gradient with single sg has NaN/Inf"
    assert jnp.allclose(grad_1, grad_2, atol=1e-7), (
        f"Gradients differ between single/double stop_gradient. "
        f"max diff = {float(jnp.max(jnp.abs(grad_1 - grad_2))):.2e}"
    )

    # Verify no gradient flows to z_q_raw through z_q_sg
    def loss_fn_wrt_z_q(z_q_input):
        z_q_stopped = jax.lax.stop_gradient(z_q_input)
        return jnp.sum(_rotation_quantize(r, z_q_stopped))

    grad_z_q = jax.grad(loss_fn_wrt_z_q)(z_q_raw)
    assert jnp.allclose(grad_z_q, jnp.zeros_like(z_q_raw), atol=1e-7), (
        f"Gradient leaked through stop_gradient to z_q. "
        f"max abs grad = {float(jnp.max(jnp.abs(grad_z_q))):.2e}"
    )
