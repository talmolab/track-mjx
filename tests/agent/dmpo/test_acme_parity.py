"""Learner-step parity with acme's TF DMPO learner.

``acme_tf_reference/fixture.json`` holds four ``_step`` calls of acme's
``DistributionalMPOLearner`` (TF, unmodified) on inputs built by
``acme_tf_reference/make_inputs.py``: the initial weights, one batch and one
set of target-policy noise per step. Here the same inputs go through
``sgd_step`` on the aligned defaults, and every value acme reports or holds
after each step is compared: the critic and policy losses, every MPO
statistic, and every online, target and dual variable.

Two bookkeeping differences are applied to the reference before comparing,
both of timing only:

* acme copies online to target weights at the START of a step
  (``num_steps % period == 0`` before its increment), the JAX learner at the
  END of the previous one (``(steps + 1) % period == 0``). The copy is the same
  one; after step j the JAX target already holds the weights acme's step j + 1
  will copy in.
* acme floors the log dual variables at -18 at the start of a step, the JAX
  learner after its update. The floor is applied to the reference likewise.

The fixture is produced by ``generate.py`` in a TF environment (see the
README); this test needs only the JSON.
"""

import functools
import json
import pathlib
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo import learner as learner_mod
from track_mjx.agent.dmpo.config import ALIGNMENT_FIELDS, DMPOConfig
from track_mjx.agent.dmpo.learner import (
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.losses import MPO
from track_mjx.agent.dmpo.networks import make_dmpo_networks

REF_DIR = pathlib.Path(__file__).parent / "acme_tf_reference"
FIXTURE = REF_DIR / "fixture.json"
RTOL, ATOL = 1e-4, 1e-5
NUM_STEPS = 4
MIN_LOG_DUAL = -18.0

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"{FIXTURE.name} is missing; regenerate it as {REF_DIR / 'README.md'} "
    "describes",
)

# acme stat -> the sgd_step metric that reports it.
METRIC_OF = {
    "critic_loss": "critic_loss",
    "policy_loss": "policy_loss",
    "dual_alpha_mean": "dual_alpha_mean",
    "dual_alpha_stddev": "dual_alpha_stddev",
    "dual_temperature": "dual_temperature",
    "loss_policy": "loss_policy",
    "loss_alpha": "loss_alpha",
    "loss_temperature": "loss_temperature",
    "kl_q_rel": "kl_q_rel",
    "kl_mean_rel": "kl_mean_rel",
    "kl_stddev_rel": "kl_stddev_rel",
    "q_min": "q_min",
    "q_max": "q_max",
    "pi_stddev_min": "pi_stddev_min",
    "pi_stddev_max": "pi_stddev_max",
    "pi_stddev_cond": "pi_stddev_cond",
}
# acme stats sgd_step does not report; read from the MPO loss itself.
SPIED = (
    "penalty_kl_q_rel",
    "kl_mean_rel_min",
    "kl_mean_rel_max",
    "kl_stddev_rel_min",
    "kl_stddev_rel_max",
)
DUALS = (
    "log_temperature",
    "log_alpha_mean",
    "log_alpha_stddev",
    "log_penalty_temperature",
)


def test_fixture_present():
    assert FIXTURE.exists()
    assert FIXTURE.stat().st_size <= 300 * 1024


@functools.lru_cache(maxsize=1)
def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _cfg(inputs: dict) -> DMPOConfig:
    values = {
        k: tuple(v) if isinstance(v, list) else v for k, v in inputs["cfg"].items()
    }
    assert not set(values) & set(ALIGNMENT_FIELDS), "alignment fields stay default"
    return DMPOConfig(**values)


def _unflatten(flat: dict) -> dict:
    tree: dict = {}
    for key, value in flat.items():
        node = tree
        *parents, leaf = key.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = jnp.asarray(np.asarray(value, np.float32))
    return tree


def _flatten(tree) -> dict[str, np.ndarray]:
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        out["/".join(str(p.key) for p in path)] = np.asarray(leaf)
    return out


class _SpyMPO(MPO):
    """The learner's MPO loss, reporting the stats sgd_step does not log."""

    records: ClassVar[list] = []

    def __call__(self, *args, **kwargs):
        loss, stats = super().__call__(*args, **kwargs)
        jax.debug.callback(
            lambda *v: _SpyMPO.records.append([np.asarray(x) for x in v]),
            stats.penalty_kl_q_rel,
            stats.kl_mean_rel,
            stats.kl_stddev_rel,
        )
        return loss, stats


@functools.lru_cache(maxsize=1)
def _run() -> list[dict]:
    """The four JAX steps: per step the metrics, spied stats and state."""
    inputs = _fixture()["inputs"]
    cfg = _cfg(inputs)
    obs, act = inputs["obs_size"], inputs["action_size"]
    nets = make_dmpo_networks(obs, act, cfg)
    state = init_training_state(
        jax.random.key(0), nets, {"obs_size": obs, "action_size": act}, cfg
    )
    policy, critic = _unflatten(inputs["policy"]), _unflatten(inputs["critic"])
    optimizers = make_optimizers(cfg)
    pol_opt, crit_opt, _ = optimizers
    state = state._replace(
        policy_params=policy,
        target_policy_params=policy,
        critic_params=critic,
        target_critic_params=critic,
        policy_opt_state=pol_opt.init(policy),
        critic_opt_state=crit_opt.init(critic),
    )

    original = learner_mod.MPO
    learner_mod.MPO = _SpyMPO
    _SpyMPO.records.clear()
    try:
        step = jax.jit(
            lambda s, b, e: sgd_step(s, b, nets, optimizers, cfg, sample_eps=e)
        )
        out = []
        for batch, eps in zip(inputs["batches"], inputs["eps"]):
            batch = {
                k: jnp.asarray(np.asarray(v, np.float32)) for k, v in batch.items()
            }
            state, metrics = step(
                state, batch, jnp.asarray(np.asarray(eps, np.float32))
            )
            jax.effects_barrier()
            penalty_kl, kl_mean, kl_stddev = _SpyMPO.records[-1]
            spied = {
                "penalty_kl_q_rel": penalty_kl,
                "kl_mean_rel_min": kl_mean.min(),
                "kl_mean_rel_max": kl_mean.max(),
                "kl_stddev_rel_min": kl_stddev.min(),
                "kl_stddev_rel_max": kl_stddev.max(),
            }
            out.append(
                {
                    "metrics": {k: np.asarray(v) for k, v in metrics.items()},
                    "spied": spied,
                    "state": state,
                }
            )
        assert len(_SpyMPO.records) == NUM_STEPS
        return out
    finally:
        learner_mod.MPO = original


def _expected_targets(ref: dict, cfg: DMPOConfig, j: int) -> tuple[dict, dict]:
    """The reference targets the JAX learner holds after step j (see top)."""
    pol = ref["policy" if j % cfg.target_policy_update_period == 0 else "target_policy"]
    crit = ref[
        "critic" if j % cfg.target_critic_update_period == 0 else "target_critic"
    ]
    return pol, crit


def comparisons(j: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(quantity, JAX value, acme value) for every compared quantity at step j."""
    fixture = _fixture()
    cfg = _cfg(fixture["inputs"])
    ref = fixture["reference"][j - 1]
    assert ref["step"] == j
    got = _run()[j - 1]
    pairs = []
    fetches = ref["fetches"]
    assert set(fetches) == set(METRIC_OF) | set(SPIED), sorted(fetches)
    for name, metric in METRIC_OF.items():
        pairs.append((f"stat/{name}", got["metrics"][metric], fetches[name]))
    for name in SPIED:
        pairs.append((f"stat/{name}", got["spied"][name], fetches[name]))

    state = got["state"]
    target_pol, target_crit = _expected_targets(ref, cfg, j)
    trees = {
        "policy": (state.policy_params, ref["policy"]),
        "target_policy": (state.target_policy_params, target_pol),
        "critic": (state.critic_params, ref["critic"]),
        "target_critic": (state.target_critic_params, target_crit),
    }
    for group, (tree, want) in trees.items():
        flat = _flatten(tree)
        assert set(flat) == set(want), (group, sorted(flat), sorted(want))
        for key in sorted(want):
            pairs.append((f"{group}/{key}", flat[key], want[key]))
    for name in DUALS:
        want = np.maximum(np.asarray(ref["dual"][name], np.float32), MIN_LOG_DUAL)
        pairs.append((f"dual/{name}", getattr(state.dual_params, name), want))
    return [
        (q, np.asarray(a, np.float32), np.asarray(b, np.float32)) for q, a, b in pairs
    ]


def test_inputs_are_the_documented_jax_init():
    """The fixture's inputs are what make_inputs.py builds from the JAX init."""
    from tests.agent.dmpo.acme_tf_reference.make_inputs import build_inputs

    built = build_inputs()
    stored = _fixture()["inputs"]
    for group in ("policy", "critic"):
        assert set(built[group]) == set(stored[group])
        for key, value in built[group].items():
            np.testing.assert_array_equal(
                value, np.asarray(stored[group][key], np.float32), err_msg=key
            )
    for got, want in zip(built["batches"], stored["batches"]):
        for key, value in got.items():
            np.testing.assert_array_equal(value, np.asarray(want[key], np.float32))
    for got, want in zip(built["eps"], stored["eps"]):
        np.testing.assert_array_equal(got, np.asarray(want, np.float32))


@pytest.mark.parametrize("j", range(1, NUM_STEPS + 1), ids=lambda j: f"step{j}")
def test_learner_step_parity(j):
    failures = []
    for quantity, got, want in comparisons(j):
        if got.shape != want.shape:
            failures.append(f"{quantity}: shape {got.shape} != {want.shape}")
        elif not np.allclose(got, want, rtol=RTOL, atol=ATOL):
            err = np.abs(got - want)
            failures.append(
                f"{quantity}: max abs err {err.max():.3g} "
                f"(jax {got.reshape(-1)[err.argmax()]:.8g}, "
                f"acme {want.reshape(-1)[err.argmax()]:.8g})"
            )
    assert not failures, f"step {j}: " + "; ".join(failures)
