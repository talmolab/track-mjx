"""Run acme's TF DMPO learner on the parity inputs and write ``fixture.json``.

Runs ONLY in the TF reference environment (see README.md), never on CI; the
JAX side (``test_acme_parity.py``) reads the fixture this writes. acme itself
is used unmodified: its ``DistributionalMPOLearner``, ``losses.MPO`` and the
n-step accumulation of ``NStepTransitionAdder``. What this file adds is only
what the comparison needs:

* sonnet networks that compute what track-mjx's ``_PolicyNet`` and
  ``_CriticNet`` compute (Linear -> LayerNorm(eps 1e-6) -> SiLU blocks, the
  ``MultivariateNormalDiagHead`` scale formula, a categorical critic head),
  loaded with the JAX initial weights from the inputs;
* the critic's action branch applies ``clip(tanh(a), -(1 - 1e-6), 1 - 1e-6)``
  to its action input, the analogue of acme's ``ClipToSpec`` for a policy whose
  actions are tanh-bound at the env;
* the target policy's ``sample(n)`` returns ``mu + sigma * eps`` with the
  stored ``eps``, so both learners see the same samples;
* each transition is built from a per-step window with acme's own n-step
  accumulation. A window whose episode terminates at step k is the shorter
  transition acme writes at an episode end, with ``next_observation`` set to
  s_k, the last state the JAX replay stores for that episode.

Usage (TF environment)::

    LD_LIBRARY_PATH=<python 3.10 lib dir> CUDA_VISIBLE_DEVICES="" \\
        python generate.py --acme <acme checkout> --inputs inputs.json \\
        --out fixture.json
"""

import argparse
import collections
import json
import os
import pathlib
import sys
import types as pytypes

HERE = pathlib.Path(__file__).resolve().parent


def _import_acme(acme_root: str):
    """Import the pieces of acme the reference needs.

    ``_stubs`` stands in for jax, which acme's TF code references only in type
    hints and one adders helper import; it is appended to ``sys.path`` (so a
    real jax would win) and only here.
    """
    sys.path.insert(0, acme_root)
    sys.path.append(str(HERE / "_stubs"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "acme_dmpo_learning",
        os.path.join(acme_root, "acme", "agents", "tf", "dmpo", "learning.py"),
    )
    learning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learning)
    from acme import types as acme_types
    from acme.adders.reverb.transition import NStepTransitionAdder
    from acme.tf import losses
    from acme.tf import networks as acme_networks

    return learning, acme_types, NStepTransitionAdder, losses, acme_networks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acme", required=True, help="acme checkout (read only)")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", default=str(HERE / "fixture.json"))
    args = parser.parse_args()

    learning, acme_types, NStepTransitionAdder, losses, acme_networks = _import_acme(
        args.acme
    )
    import numpy as np
    import sonnet as snt
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    inputs = json.loads(pathlib.Path(args.inputs).read_text())
    cfg = inputs["cfg"]
    obs_size, act_size = inputs["obs_size"], inputs["action_size"]
    f32 = np.float32

    def arr(x):
        return np.asarray(x, dtype=f32)

    # ---- networks ----------------------------------------------------------

    class Torso(snt.Module):
        """Linear -> LayerNorm -> SiLU per block, as flax Dense/LayerNorm."""

        def __init__(self, sizes, name=None):
            super().__init__(name=name)
            self.linears = [snt.Linear(s) for s in sizes]
            self.norms = [
                snt.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6)
                for _ in sizes
            ]

        def __call__(self, x):
            for lin, norm in zip(self.linears, self.norms):
                x = tf.nn.silu(norm(lin(x)))
            return x

        def pairs(self):
            """(flax path prefix, sonnet variable) for every variable."""
            for i, (lin, norm) in enumerate(zip(self.linears, self.norms)):
                yield f"params/Dense_{i}/kernel", lin.w
                yield f"params/Dense_{i}/bias", lin.b
                yield f"params/LayerNorm_{i}/scale", norm.scale
                yield f"params/LayerNorm_{i}/bias", norm.offset

    class NoiseSource:
        """Plain holder (not a tf.Module, so it adds no network variable)."""

        def __init__(self):
            self.eps = None

    class PolicyNet(snt.Module):
        """Torso -> MultivariateNormalDiagHead (acme's scale formula)."""

        init_scale, min_scale = 0.7, 1e-6

        def __init__(self, sizes, noise=None, name=None):
            super().__init__(name=name)
            self.torso = Torso(sizes)
            self.loc = snt.Linear(act_size)
            self.scale = snt.Linear(act_size)
            self._noise = noise

        def __call__(self, obs):
            h = self.torso(obs)
            loc = self.loc(h)
            scale = tf.nn.softplus(self.scale(h))
            scale = scale * (self.init_scale / tf.nn.softplus(0.0)) + self.min_scale
            dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            if self._noise is not None:
                eps = self._noise.eps

                def _sample(n):
                    del n  # eps fixes the number of samples
                    return loc[None] + scale[None] * eps

                dist.sample = _sample
            return dist

        def pairs(self):
            yield from self.torso.pairs()
            head = "params/GaussianPolicyHead_0"
            yield f"{head}/loc/kernel", self.loc.w
            yield f"{head}/loc/bias", self.loc.b
            yield f"{head}/scale/kernel", self.scale.w
            yield f"{head}/scale/bias", self.scale.b

    support = tf.constant(
        np.linspace(cfg["vmin"], cfg["vmax"], num=cfg["num_atoms"]).astype(f32)
    )

    class CriticNet(snt.Module):
        """concat([obs, bound action]) -> Torso -> categorical head."""

        def __init__(self, sizes, name=None):
            super().__init__(name=name)
            self.torso = Torso(sizes)
            self.logits = snt.Linear(cfg["num_atoms"])

        def __call__(self, obs, action):
            bound = tf.clip_by_value(tf.tanh(action), -1.0 + 1e-6, 1.0 - 1e-6)
            h = self.torso(tf.concat([obs, bound], axis=-1))
            return acme_networks.DiscreteValuedDistribution(
                values=support, logits=self.logits(h)
            )

        def pairs(self):
            yield from self.torso.pairs()
            head = "params/CategoricalCriticHead_0/logits"
            yield f"{head}/kernel", self.logits.w
            yield f"{head}/bias", self.logits.b

    noise = NoiseSource()
    policy = PolicyNet(cfg["policy_layer_sizes"], name="policy")
    target_policy = PolicyNet(cfg["policy_layer_sizes"], noise, name="target_policy")
    critic = CriticNet(cfg["critic_layer_sizes"], name="critic")
    target_critic = CriticNet(cfg["critic_layer_sizes"], name="target_critic")

    batch_size = len(inputs["batches"][0]["reward"])
    noise.eps = tf.Variable(np.zeros((cfg["num_samples"], batch_size, act_size), f32))
    dummy_obs = tf.zeros((1, obs_size))
    for net in (policy, target_policy):
        net(dummy_obs)
    for net in (critic, target_critic):
        net(dummy_obs, tf.zeros((1, act_size)))

    def load(net, flat):
        pairs = dict(net.pairs())
        assert set(pairs) == set(flat), (sorted(pairs), sorted(flat))
        for key, var in pairs.items():
            value = arr(flat[key])
            assert tuple(var.shape) == value.shape, (key, var.shape, value.shape)
            var.assign(value)

    def dump(net):
        return {key: var.numpy() for key, var in net.pairs()}

    load(policy, inputs["policy"])
    load(target_policy, inputs["policy"])
    load(critic, inputs["critic"])
    load(target_critic, inputs["critic"])

    # ---- transitions ---------------------------------------------------------
    n_step, gamma = cfg["n_step"], f32(cfg["discount"])

    def transition_fields(batch):
        obs, act = arr(batch["observation"]), arr(batch["action"])
        rew, disc = arr(batch["reward"]), arr(batch["discount"])
        out = collections.defaultdict(list)
        for b in range(obs.shape[0]):
            term = [i for i in range(n_step) if disc[b, i] == 0.0]
            length = term[0] + 1 if term else n_step
            adder = pytypes.SimpleNamespace(_n_step=length, _discount=gamma)
            ret, tot = NStepTransitionAdder._compute_cumulative_quantities(
                adder, rew[b, :length], disc[b, :length]
            )
            arrival = term[0] if term else n_step  # s_k at an episode end
            out["observation"].append(obs[b, 0])
            out["action"].append(act[b, 0])
            out["reward"].append(f32(ret))
            out["discount"].append(f32(tot))
            out["next_observation"].append(obs[b, arrival])
        return {k: np.stack(v).astype(f32) for k, v in out.items()}

    per_step = [transition_fields(b) for b in inputs["batches"]]
    stacked = {k: np.stack([s[k] for s in per_step]) for k in per_step[0]}
    Sample = collections.namedtuple("Sample", ["data"])

    def _to_sample(fields):
        return Sample(
            data=acme_types.Transition(
                observation=fields["observation"],
                action=fields["action"],
                reward=fields["reward"],
                discount=fields["discount"],
                next_observation=fields["next_observation"],
            )
        )

    dataset = tf.data.Dataset.from_tensor_slices(stacked).map(_to_sample)

    # ---- the learner, unmodified -------------------------------------------
    class _NullLogger:
        def write(self, data):
            del data

    mpo = losses.MPO(
        epsilon=cfg["epsilon"],
        epsilon_mean=cfg["epsilon_mean"],
        epsilon_stddev=cfg["epsilon_stddev"],
        init_log_temperature=cfg["init_log_temperature"],
        init_log_alpha_mean=cfg["init_log_alpha_mean"],
        init_log_alpha_stddev=cfg["init_log_alpha_stddev"],
        per_dim_constraining=cfg["per_dim_constraining"],
        action_penalization=True,
        epsilon_penalty=cfg["epsilon_penalty"],
    )
    from acme.utils import counting

    learner = learning.DistributionalMPOLearner(
        policy_network=policy,
        critic_network=critic,
        target_policy_network=target_policy,
        target_critic_network=target_critic,
        discount=cfg["discount"],
        num_samples=cfg["num_samples"],
        target_policy_update_period=cfg["target_policy_update_period"],
        target_critic_update_period=cfg["target_critic_update_period"],
        dataset=dataset,
        policy_loss_module=mpo,
        policy_optimizer=snt.optimizers.Adam(cfg["policy_lr"]),
        critic_optimizer=snt.optimizers.Adam(cfg["critic_lr"]),
        dual_optimizer=snt.optimizers.Adam(cfg["dual_lr"]),
        clipping=True,
        counter=counting.Counter(),
        logger=_NullLogger(),
        checkpoint=False,
    )

    steps = []
    for j, eps in enumerate(inputs["eps"]):
        noise.eps.assign(arr(eps))
        fetches = learner._step()
        steps.append(
            {
                "step": j + 1,
                "fetches": {
                    k: float(np.asarray(v).reshape(())) for k, v in fetches.items()
                },
                "policy": dump(policy),
                "target_policy": dump(target_policy),
                "critic": dump(critic),
                "target_critic": dump(target_critic),
                "dual": {
                    "log_temperature": mpo._log_temperature.numpy(),
                    "log_alpha_mean": mpo._log_alpha_mean.numpy(),
                    "log_alpha_stddev": mpo._log_alpha_stddev.numpy(),
                    "log_penalty_temperature": mpo._log_penalty_temperature.numpy(),
                },
                "transitions": per_step[j],
            }
        )

    def to_json(x):
        if isinstance(x, dict):
            return {k: to_json(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_json(v) for v in x]
        if isinstance(x, np.ndarray):
            flat = [float(str(f32(v))) for v in x.reshape(-1)]
            return np.asarray(flat, dtype=object).reshape(x.shape).tolist()
        if isinstance(x, float):
            return float(str(f32(x)))
        return x

    import importlib.metadata as md

    versions = {
        name: md.version(name)
        for name in (
            "tensorflow",
            "tensorflow-probability",
            "dm-sonnet",
            "trfl",
            "dm-launchpad",
            "dm-reverb",
            "dm-tree",
            "dm-env",
            "numpy",
            "protobuf",
        )
    }
    versions["python"] = sys.version.split()[0]
    fixture = {
        "description": (
            "acme TF DistributionalMPOLearner reference for test_acme_parity.py; "
            "regenerate with generate.py (see README.md)."
        ),
        "versions": versions,
        "inputs": inputs,
        "reference": to_json(steps),
    }
    pathlib.Path(args.out).write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
