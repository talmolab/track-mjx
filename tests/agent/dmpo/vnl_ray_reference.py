"""Regenerate the vnl-ray TF MPO reference used by test_losses.py.

This is a regeneration SCRIPT, not a collected pytest test: it is not
imported/executed by the test suite. The committed ``vnl_ray_reference.json``
next to this file is what test_losses.py actually reads; this script exists
only to reproduce that JSON if the reference ever needs to be regenerated
(e.g. to check parity against a newer vnl-ray).

Regenerating requires two things this repo does not vendor:
  1. The archived vnl-ray checkout (a sibling package, not on PyPI). Point
     the ``VNL_RAY_ROOT`` env var at its root (the directory that itself
     contains the ``vnl_ray/`` package, e.g.
     ``/path/to/vnl-ray``). The script
     raises a clear error at import time if this is unset.
  2. A TensorFlow venv (vnl-ray's MPO loss is a TF1-style
     tensorflow_probability implementation, incompatible with this repo's
     JAX env) — e.g. ``/tmp/vnl_ray_tf_env`` on the machine this was
     authored on.

To regenerate (from the repo root, with VNL_RAY_ROOT set and the TF venv
active):
    source /tmp/vnl_ray_tf_env/bin/activate
    export VNL_RAY_ROOT=/path/to/vnl-ray
    python tests/agent/dmpo/vnl_ray_reference.py
    deactivate

NOTE on action penalization parity:
    This reference uses vnl-ray's untouched default action-penalization
    formula: ``cost_out_of_bound = -tf.norm(actions, axis=-1)``, which
    penalizes ALL action magnitudes (see
    vnl-ray's vnl_ray/agents/losses_mpo.py:265-270 for the active
    line and the `# OLD` Acme reference). Our vendored JAX loss has been
    aligned to this same formula (see
    track_mjx/agent/dmpo/losses.py module docstring), so no callable
    override is needed here.
"""

import json
import os
import pathlib
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp_tf

_VNL_RAY_ROOT = os.environ.get("VNL_RAY_ROOT")
if not _VNL_RAY_ROOT:
    raise RuntimeError(
        "vnl_ray_reference.py needs the archived vnl-ray checkout to regenerate "
        "vnl_ray_reference.json. Set VNL_RAY_ROOT to its root directory (the "
        "directory containing the vnl_ray/ package), e.g.\n"
        "  export VNL_RAY_ROOT=/path/to/vnl-ray\n"
        "This is a regeneration script, not a collected test -- the committed "
        "vnl_ray_reference.json is what test_losses.py actually reads."
    )
sys.path.insert(0, _VNL_RAY_ROOT)
from vnl_ray.agents.losses_mpo import MPO as VnlMPO  # type: ignore  # noqa: E402

tfd = tfp_tf.distributions

OUTPUT = pathlib.Path(__file__).parent / "vnl_ray_reference.json"


def main():
    np.random.seed(0)
    B, A, N = 4, 6, 20
    online_loc = np.random.randn(B, A).astype(np.float32) * 0.1
    online_scale = np.exp(np.random.randn(B, A).astype(np.float32) * 0.1) + 0.5
    target_loc = np.random.randn(B, A).astype(np.float32) * 0.1
    target_scale = np.exp(np.random.randn(B, A).astype(np.float32) * 0.1) + 0.5
    sampled_actions = np.random.randn(N, B, A).astype(np.float32)
    q_values = np.random.randn(N, B).astype(np.float32)

    online = tfd.MultivariateNormalDiag(
        loc=tf.constant(online_loc), scale_diag=tf.constant(online_scale)
    )
    target = tfd.MultivariateNormalDiag(
        loc=tf.constant(target_loc), scale_diag=tf.constant(target_scale)
    )
    mpo = VnlMPO(
        epsilon=0.1,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-7,
        epsilon_penalty=0.1,
        init_log_temperature=10.0,
        init_log_alpha_mean=10.0,
        init_log_alpha_stddev=1000.0,
        per_dim_constraining=True,
        action_penalization=True,
        # Use vnl-ray's default action-penalization formula
        # (-norm(actions, axis=-1)) — no callable override.
    )
    loss, stats = mpo(
        online_action_distribution=online,
        target_action_distribution=target,
        actions=tf.constant(sampled_actions),
        q_values=tf.constant(q_values),
    )

    # vnl-ray returns stats as a Dict[str, tf.Tensor]; dump only scalars/arrays
    # we'll compare.
    stat_scalars = {}
    for k, v in stats.items():
        v_np = v.numpy()
        if v_np.ndim == 0:
            stat_scalars[k] = float(v_np)
        else:
            stat_scalars[k] = v_np.tolist()

    loss_np = loss.numpy()
    # vnl-ray's loss has shape (1,) (it inherits the leading dim from the
    # log_temperature variable, which is shape (1,)). Squeeze to scalar.
    loss_scalar = float(np.squeeze(loss_np))

    out = {
        "inputs": {
            "online_loc": online_loc.tolist(),
            "online_scale": online_scale.tolist(),
            "target_loc": target_loc.tolist(),
            "target_scale": target_scale.tolist(),
            "sampled_actions": sampled_actions.tolist(),
            "q_values": q_values.tolist(),
        },
        "outputs": {
            "loss": loss_scalar,
            "loss_shape": list(loss_np.shape),
        },
        "stats": stat_scalars,
    }
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
