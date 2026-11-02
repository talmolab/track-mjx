"""Tests for the vendored C51 support projection (`categorical.py`).

`categorical_l2_project` was ported verbatim from rlax 0.1.8
(`rlax._src.value_learning.categorical_l2_project`) so the DMPO package no
longer depends on rlax. These tests do NOT import rlax: rlax is not a
dependency of the package, and an `importorskip` guard would
silently skip the golden-value comparison for anyone who doesn't happen to
have rlax installed, defeating the point of pinning exact equivalence.
"""

import jax.numpy as jnp
import numpy as np

from track_mjx.agent.dmpo.categorical import categorical_l2_project


def test_identity_support_is_a_no_op():
    z = jnp.linspace(-1.0, 1.0, 5)
    p = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    np.testing.assert_allclose(categorical_l2_project(z, p, z), p, atol=1e-6)


def test_mass_is_split_between_neighbouring_atoms():
    z_q = jnp.array([0.0, 1.0, 2.0])
    out = categorical_l2_project(jnp.array([0.25]), jnp.array([1.0]), z_q)
    np.testing.assert_allclose(out, [0.75, 0.25, 0.0], atol=1e-6)


def test_out_of_range_mass_clips_to_the_edges():
    z_q = jnp.array([-1.0, 0.0, 1.0])
    out = categorical_l2_project(jnp.array([-5.0, 5.0]), jnp.array([0.5, 0.5]), z_q)
    np.testing.assert_allclose(out, [0.5, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)


def test_golden_value_matches_rlax_0_1_8():
    """Pin exact numerical equivalence with the rlax 0.1.8 implementation.

    Non-trivial fixed input: K=11 target atoms with one duplicate value
    (-0.5 appears twice, at indices 3 and 4), 7 source atoms several of
    which lie outside the target range ([-2.0, 3.0]), and non-uniform
    source probabilities.

    The expected literal below was generated ONCE, offline, by running:

        import jax.numpy as jnp, rlax
        z_q = jnp.array([-2.0, -1.5, -1.0, -0.5, -0.5, 0.0, 0.5, 1.0, 1.5,
                          2.0, 2.5], dtype=jnp.float32)
        z_p = jnp.array([-5.0, -2.0, -1.2, 0.3, 1.1, 2.5, 6.0],
                         dtype=jnp.float32)
        probs = jnp.array([0.05, 0.10, 0.25, 0.30, 0.15, 0.10, 0.05],
                           dtype=jnp.float32)
        rlax.categorical_l2_project(z_p, probs, z_q)  # rlax==0.1.8

    Note the duplicate atom makes the zero-gap normalization degenerate
    (see `test_mass_conservation_requires_distinct_atoms` below): the
    output here sums to ~2.0, not 1.0. That is expected rlax 0.1.8
    behavior for a degenerate target grid, reproduced verbatim by the
    vendored port, not a bug in this test.
    """
    z_q = jnp.array(
        [-2.0, -1.5, -1.0, -0.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        dtype=jnp.float32,
    )
    z_p = jnp.array([-5.0, -2.0, -1.2, 0.3, 1.1, 2.5, 6.0], dtype=jnp.float32)
    probs = jnp.array([0.05, 0.10, 0.25, 0.30, 0.15, 0.10, 0.05], dtype=jnp.float32)

    expected = np.array(
        [
            0.15000000596046448,
            0.10000002384185791,
            0.1499999761581421,
            0.6000000238418579,
            0.4000000059604645,
            0.11999999731779099,
            0.18000000715255737,
            0.11999999731779099,
            0.030000008642673492,
            0.0,
            0.15000000596046448,
        ]
    )

    out = categorical_l2_project(z_p, probs, z_q)
    assert out.ndim == 1
    assert out.shape == (11,)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_mass_conservation_requires_distinct_atoms():
    """For a non-degenerate (strictly increasing) target grid, projecting a
    normalized distribution conserves total probability mass.

    Same source distribution as the golden-value test above, but with the
    duplicate target atom (-0.5, -0.5) replaced by a distinct value (3.0
    appended) so the projection is a proper partition of unity. Expected
    literal generated the same way, offline, against rlax 0.1.8:

        z_q = jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0,
                          2.5, 3.0], dtype=jnp.float32)
        rlax.categorical_l2_project(z_p, probs, z_q)  # rlax==0.1.8
    """
    z_q = jnp.array(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        dtype=jnp.float32,
    )
    z_p = jnp.array([-5.0, -2.0, -1.2, 0.3, 1.1, 2.5, 6.0], dtype=jnp.float32)
    probs = jnp.array([0.05, 0.10, 0.25, 0.30, 0.15, 0.10, 0.05], dtype=jnp.float32)

    out = categorical_l2_project(z_p, probs, z_q)
    np.testing.assert_allclose(float(out.sum()), 1.0, atol=1e-6)
