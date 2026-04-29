"""Run vnl-ray's TF MPO loss on a fixed batch and dump outputs to JSON.

Executed once (and cached) by test_losses.py. Requires the TF venv at
/tmp/vnl_ray_tf_env.

To regenerate:
    source /tmp/vnl_ray_tf_env/bin/activate
    cd /home/talmolab/Desktop/SalkResearch/track-mjx
    python tests/agent/dmpo/vnl_ray_reference.py
    deactivate

NOTE on action penalization parity:
    vnl-ray's MPO defaults to ``cost_out_of_bound = -tf.norm(actions, axis=-1)``
    which penalizes ALL action magnitudes (see
    archive/vnl-ray/vnl_ray/agents/losses_mpo.py:265-270). Acme JAX uses the
    original ``-norm(actions - clip(actions, -1, 1))`` formulation that only
    penalizes out-of-bound actions. To get a numerical apples-to-apples
    comparison against Acme's vendored loss, we pass a custom
    ``penalization_cost`` callable to vnl-ray's MPO that replicates Acme's
    formulation. The behavioural divergence is documented in the test file.
"""

import json
import pathlib
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp_tf

sys.path.insert(0, "/home/talmolab/Desktop/SalkResearch/archive/vnl-ray")
from vnl_ray.agents.losses_mpo import MPO as VnlMPO  # type: ignore  # noqa: E402

tfd = tfp_tf.distributions

OUTPUT = pathlib.Path(__file__).parent / "vnl_ray_reference.json"


def acme_style_penalization_cost(actions: tf.Tensor) -> tf.Tensor:
    """Replicates Acme JAX MPO's out-of-bound-only penalization for parity."""
    diff_out_of_bound = actions - tf.clip_by_value(actions, -1.0, 1.0)
    return -tf.norm(diff_out_of_bound, axis=-1)


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
        # Match Acme's original out-of-bound-only formulation (Acme JAX MPO
        # uses -norm(actions - clip(actions, -1, 1))).
        penalization_cost=acme_style_penalization_cost,
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
