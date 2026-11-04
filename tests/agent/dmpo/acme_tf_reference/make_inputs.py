"""Inputs of the acme parity fixture: config, initial weights, batches, noise.

Runs in the track-mjx (JAX) environment. ``generate.py`` reads what this
writes and runs acme's TF learner on it; ``test_acme_parity.py`` rebuilds the
same initial weights with ``build_inputs`` and checks them against the
fixture before replaying the steps through ``sgd_step``.

Usage::

    python -m tests.agent.dmpo.acme_tf_reference.make_inputs inputs.json
"""

import json
import sys

import jax
import numpy as np

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks import make_dmpo_networks

OBS, ACT = 5, 2
BATCH, N_STEP = 6, 3
SEQ = N_STEP + 1  # compressed layout: the bootstrap state is observation[:, n]
NUM_STEPS = 4
INIT_SEED = 0
PERTURB_SEED, BATCH_SEED, EPS_SEED = 1, 2, 3
PERTURB_STD = 0.1

# Everything but the six alignment fields, which stay at their (aligned)
# defaults. The MPO, discount and learning-rate values are the learner's
# defaults where they keep the comparison well conditioned in float32 (see
# the README).
CFG = dict(
    policy_layer_sizes=(16, 16),
    critic_layer_sizes=(16, 16),
    num_atoms=11,
    vmin=-5.0,
    vmax=5.0,
    num_samples=4,
    discount=0.9,
    use_n_step=True,
    n_step=N_STEP,
    sequence_length=SEQ,
    store_next_observation=False,
    batch_size=BATCH,
    target_policy_update_period=2,
    target_critic_update_period=3,
    policy_lr=1e-3,
    critic_lr=1e-3,
    dual_lr=1e-2,
    grad_clip=40.0,
    epsilon=0.1,
    epsilon_mean=2.5e-3,
    epsilon_stddev=1e-3,
    epsilon_penalty=1e-3,
    init_log_temperature=10.0,
    init_log_alpha_mean=10.0,
    init_log_alpha_stddev=1000.0,
    per_dim_constraining=True,
    action_penalization=True,
)

# Where each row's episode ends inside the n-step window: None = mid-episode,
# k = the step k terminates (discount[k] = 0). Steps after k belong to the next
# episode and carry arbitrary values the learner must ignore.
ROW_TERMINATION = (None, None, 0, 1, 2, None)


def make_cfg() -> DMPOConfig:
    return DMPOConfig(**CFG)


def flatten(tree) -> dict[str, np.ndarray]:
    """``{"a/b/c": leaf}`` for a nested dict of arrays."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        out["/".join(str(p.key) for p in path)] = np.asarray(leaf)
    return out


def unflatten(flat: dict[str, np.ndarray]) -> dict:
    tree: dict = {}
    for key, value in flat.items():
        node = tree
        *parents, leaf = key.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return tree


def build_inputs() -> dict:
    """The fixture's inputs, as float32 numpy arrays.

    The initial weights are the JAX init (seed 0) plus N(0, PERTURB_STD^2)
    noise on every leaf, so the policy head (initialized near zero) and the
    LayerNorm scales and offsets are not trivially constant.
    """
    cfg = make_cfg()
    nets = make_dmpo_networks(OBS, ACT, cfg)
    state = init_training_state(
        jax.random.key(INIT_SEED), nets, {"obs_size": OBS, "action_size": ACT}, cfg
    )
    rs = np.random.RandomState(PERTURB_SEED)

    def _perturb(tree):
        flat = flatten(tree)
        return {
            k: (v + PERTURB_STD * rs.standard_normal(v.shape)).astype(np.float32)
            for k, v in sorted(flat.items())
        }

    policy = _perturb(state.policy_params)
    critic = _perturb(state.critic_params)

    rb = np.random.RandomState(BATCH_SEED)
    batches = []
    for _ in range(NUM_STEPS):
        discount = np.ones((BATCH, SEQ), np.float32)
        for row, k in enumerate(ROW_TERMINATION):
            if k is not None:
                discount[row, k] = 0.0
        batches.append(
            {
                "observation": rb.standard_normal((BATCH, SEQ, OBS)).astype(np.float32),
                "action": (1.2 * rb.standard_normal((BATCH, SEQ, ACT))).astype(
                    np.float32
                ),
                "reward": rb.standard_normal((BATCH, SEQ)).astype(np.float32),
                "discount": discount,
                "truncation": np.zeros((BATCH, SEQ), np.float32),
            }
        )
    re = np.random.RandomState(EPS_SEED)
    eps = [
        re.standard_normal((cfg.num_samples, BATCH, ACT)).astype(np.float32)
        for _ in range(NUM_STEPS)
    ]
    return {
        "cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in CFG.items()},
        "obs_size": OBS,
        "action_size": ACT,
        "row_termination": list(ROW_TERMINATION),
        "policy": policy,
        "critic": critic,
        "batches": batches,
        "eps": eps,
    }


def to_json(x):
    """Arrays to nested lists of float32 values (shortest round-trip repr)."""
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [to_json(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.dtype == np.float32:
            return np.vectorize(lambda v: float(str(v)), otypes=[object])(x).tolist()
        return x.tolist()
    return x


def main(path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_json(build_inputs()), f)


if __name__ == "__main__":
    main(sys.argv[1])
