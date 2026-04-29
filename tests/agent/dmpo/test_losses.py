"""Numerical parity: JAX MPO loss must match vnl-ray TF MPO loss on a fixed batch.

This is the firewall for the DMPO port. Tasks 8-17 all depend on the MPO loss
being numerically correct. A bug in the vendoring or a subtle difference
between Acme's MPO and vnl-ray's MPO would silently break every downstream
component, so we pin the JAX implementation to a TF reference here.

The reference JSON ships in-tree (``vnl_ray_reference.json``); the next dev
does NOT need the TF venv to run these tests. To regenerate the JSON (for
example, after intentionally changing the loss math), run::

    source /tmp/vnl_ray_tf_env/bin/activate
    python tests/agent/dmpo/vnl_ray_reference.py
    deactivate

NOTE on action penalization:
    vnl-ray's MPO defaults to ``cost_out_of_bound = -tf.norm(actions, axis=-1)``
    (penalises ALL action magnitudes), while Acme JAX uses
    ``-norm(actions - clip(actions, -1, 1))`` (only out-of-bound actions). The
    reference dump passes a custom ``penalization_cost`` to vnl-ray that
    replicates Acme's formulation, so this parity test confirms the JAX port
    matches Acme's *original* behaviour. If we ever decide to track vnl-ray's
    modified penalisation in the JAX port, the change goes in losses.py and
    the reference JSON is regenerated *without* the custom callable.
"""

import json
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp

from track_mjx.agent.dmpo.losses import MPO, MPOParams, clip_mpo_params

tfd = tfp.distributions

REF_OUTPUT = pathlib.Path(__file__).parent / "vnl_ray_reference.json"

# Matched stats keys (present in both Acme MPOStats and vnl-ray's stats dict).
# vnl-ray adds a few extra diagnostics (pi_mean_abs_mean, pi_stddev_abs_mean,
# pi_mean_stddev, pi_stddev_stddev) that Acme does not log; we skip those.
SCALAR_STATS_TO_COMPARE = (
    "dual_alpha_mean",
    "dual_alpha_stddev",
    "dual_temperature",
    "loss_policy",
    "loss_alpha",
    "loss_temperature",
    "kl_q_rel",
    "penalty_kl_q_rel",
    "q_min",
    "q_max",
    "pi_stddev_min",
    "pi_stddev_max",
    "pi_stddev_cond",
)
ARRAY_STATS_TO_COMPARE = (
    "kl_mean_rel",
    "kl_stddev_rel",
)
TOL = 1e-4


@pytest.fixture(scope="module")
def reference():
    if not REF_OUTPUT.exists():
        pytest.skip(
            f"Missing {REF_OUTPUT}. Generate with: "
            "source /tmp/vnl_ray_tf_env/bin/activate && "
            "python tests/agent/dmpo/vnl_ray_reference.py"
        )
    return json.loads(REF_OUTPUT.read_text())


def _run_jax_mpo(reference):
    inp = reference["inputs"]
    online = tfd.MultivariateNormalDiag(
        loc=jnp.array(inp["online_loc"]),
        scale_diag=jnp.array(inp["online_scale"]),
    )
    target = tfd.MultivariateNormalDiag(
        loc=jnp.array(inp["target_loc"]),
        scale_diag=jnp.array(inp["target_scale"]),
    )
    sampled_actions = jnp.array(inp["sampled_actions"])
    q_values = jnp.array(inp["q_values"])

    loss_fn = MPO(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=10.0,
        init_log_alpha_mean=10.0,
        init_log_alpha_stddev=1000.0,
        per_dim_constraining=True,
        action_penalization=True,
    )
    A = jnp.array(inp["online_loc"]).shape[-1]
    params = loss_fn.init_params(action_dim=A, dtype=jnp.float32)

    loss, stats = loss_fn(
        params=params,
        online_action_distribution=online,
        target_action_distribution=target,
        actions=sampled_actions,
        q_values=q_values,
    )
    return loss, stats


def test_mpo_loss_parity_with_vnl_ray(reference):
    """Top-level loss value must match vnl-ray within TOL."""
    loss, _ = _run_jax_mpo(reference)
    # Both vnl-ray and our JAX impl return shape (1,) loss; squeeze to scalar.
    loss_scalar = float(np.squeeze(np.asarray(loss)))
    ref_loss = float(reference["outputs"]["loss"])
    diff = abs(loss_scalar - ref_loss)
    assert diff < TOL, (
        f"MPO loss disagrees with vnl-ray TF reference: "
        f"jax={loss_scalar} ref={ref_loss} |diff|={diff} (tol={TOL})"
    )


def test_mpo_scalar_stats_parity_with_vnl_ray(reference):
    """Scalar stats present in both Acme MPOStats and vnl-ray stats must match."""
    _, stats = _run_jax_mpo(reference)
    jax_stats = stats._asdict()
    failures = []
    for k in SCALAR_STATS_TO_COMPARE:
        v_ref = reference["stats"][k]
        v_jax = jax_stats[k]
        if v_jax is None:
            failures.append(f"{k}: jax stat is None (ref={v_ref})")
            continue
        v_jax_f = float(np.squeeze(np.asarray(v_jax)))
        v_ref_f = float(v_ref)
        if abs(v_jax_f - v_ref_f) >= TOL:
            failures.append(
                f"{k}: jax={v_jax_f} ref={v_ref_f} |diff|={abs(v_jax_f - v_ref_f)}"
            )
    assert not failures, "Stat disagreement(s):\n  " + "\n  ".join(failures)


def test_mpo_array_stats_parity_with_vnl_ray(reference):
    """Per-dimension stats (kl_mean_rel, kl_stddev_rel) must match elementwise.

    These quantities are KL divergences divided by very small ``epsilon`` (down
    to 1e-7 for ``epsilon_stddev``), so the magnitudes can be ~1e5. Pure
    ``atol=1e-4`` is too tight given float32 reduction-order noise — we use
    ``rtol`` instead. A relative agreement at 1e-5 is the realistic float32
    bound for sums-of-products of this magnitude across two different XLA/TF
    reduction orders.
    """
    _, stats = _run_jax_mpo(reference)
    jax_stats = stats._asdict()
    for k in ARRAY_STATS_TO_COMPARE:
        v_ref = np.asarray(reference["stats"][k], dtype=np.float32)
        v_jax = np.asarray(jax_stats[k], dtype=np.float32)
        np.testing.assert_allclose(
            v_jax,
            v_ref,
            rtol=1e-5,
            atol=TOL,
            err_msg=f"per-dim stat {k} disagrees",
        )


def test_mpo_init_params():
    loss_fn = MPO(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=10.0,
        init_log_alpha_mean=10.0,
        init_log_alpha_stddev=1000.0,
        per_dim_constraining=True,
        action_penalization=True,
    )
    params = loss_fn.init_params(action_dim=6, dtype=jnp.float32)
    assert isinstance(params, MPOParams)
    # Per-dim constraining -> alphas have shape (action_dim,).
    assert params.log_alpha_mean.shape == (6,)
    assert params.log_alpha_stddev.shape == (6,)
    assert params.log_temperature.shape == (1,)
    assert params.log_penalty_temperature.shape == (1,)


def test_mpo_clip_params_floors_at_min():
    loss_fn = MPO(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=-100.0,
        init_log_alpha_mean=-100.0,
        init_log_alpha_stddev=-100.0,
        per_dim_constraining=True,
        action_penalization=True,
    )
    params = loss_fn.init_params(action_dim=6, dtype=jnp.float32)
    clipped = clip_mpo_params(params, per_dim_constraining=True)
    # Both _MIN_LOG_TEMPERATURE and _MIN_LOG_ALPHA are -18.0 in the vendored
    # losses.py.
    assert float(clipped.log_temperature[0]) == -18.0
    assert float(clipped.log_penalty_temperature[0]) == -18.0
    assert float(clipped.log_alpha_mean[0]) == -18.0
    assert float(clipped.log_alpha_stddev[0]) == -18.0
