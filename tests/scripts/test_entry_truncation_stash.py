"""The DMPO entries compose their training env with the truncation stash.

An off-policy learner that bootstraps through time-outs needs the post-step
truncation flag after every step. brax's ``BraxAutoResetWrapper(full_reset=True)``
wipes ``info["truncation"]`` on exactly that step, so the entries build the
training env with ``wrap_for_dmpo_training``, which keeps the flag under
``info["AutoResetWrapper_preserve_info"]``. The eval env is not touched.

The kl-anchor entry's ``_wrap_for_brax_training`` is exercised on CPU with a
stub env, for a maze arm (``wrappers.full_reset: true``) and a run-gap arm
(``wrappers.info_reset_on_done: true``, ``full_reset`` off). The high-level
and imitation entries hard-code their stacks inline, so they are checked from
their source, as is each entry's replay template.
"""

import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from mujoco_playground._src import mjx_env

from tests.scripts._entries import ENTRIES, SCRIPTS_DIR
from tests.scripts.test_configs_compose import CONFIG_DIR

pytest.importorskip(
    "vnl_playground.tasks.wrappers_truncation",
    reason="needs the truncation wrapper, which the pinned vnl-playground lacks",
)
DEFAULT_RUN_GAP_KEYS = pytest.importorskip(
    "vnl_playground.tasks.wrappers_info_reset",
    reason="needs the info-reset wrapper, which the pinned vnl-playground lacks",
).DEFAULT_RUN_GAP_KEYS

PRESERVE_KEY = "AutoResetWrapper_preserve_info"
EPISODE_LENGTH = 3
BATCH = 2


def _first_run_gap_arm_with_info_reset():
    """A shipped run-gap config whose ``wrappers:`` block resets info on done."""
    for path in sorted((CONFIG_DIR / "run_gap").glob("*.yaml")):
        if "info_reset_on_done: true" in path.read_text():
            return f"run_gap/{path.stem}"
    raise AssertionError("no run_gap config sets wrappers.info_reset_on_done")


# arm id -> (shipped config, whether it enables full_reset)
ARMS = {
    "maze": ("maze/maze_gen1_prior_mlp", True),
    "run_gap": (_first_run_gap_arm_with_info_reset(), False),
}


class _StubEnv:
    """Stands in for a registered env: no physics, every info key a run-gap
    env resets (so the run-gap arm's info-reset wrapper has something to
    restore), obs = steps taken."""

    action_size = 1
    observation_size = 1

    def reset(self, rng):
        del rng
        return mjx_env.State(
            data=None,
            obs=jnp.zeros(1),
            reward=jnp.float32(0),
            done=jnp.float32(0),
            metrics={},
            info={k: jnp.float32(0) for k in DEFAULT_RUN_GAP_KEYS},
        )

    def step(self, state, action):
        del action
        steps = state.info["steps"] + 1
        return state.replace(obs=jnp.reshape(steps, (1,)), reward=jnp.float32(1))


def _compose(name):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name)


def _chain(env):
    out = []
    while True:
        out.append(type(env).__name__)
        if not hasattr(env, "env"):
            return out
        env = env.env


@pytest.fixture(autouse=True)
def dummy_roots(monkeypatch):
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/dummy/priors")
    monkeypatch.setenv("DMPO_CKPT_ROOT", "/dummy/checkpoints")


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_entry_builds_env_with_truncation_stash(arm, kl_anchor_entry):
    name, expect_full_reset = ARMS[arm]
    cfg = _compose(name)
    full_reset, _, _ = kl_anchor_entry["_wrapper_settings"](cfg)
    assert full_reset is expect_full_reset

    env = kl_anchor_entry["_wrap_for_brax_training"](
        _StubEnv(), cfg, episode_length=EPISODE_LENGTH, action_repeat=1
    )
    chain = _chain(env)
    assert "TruncationStashWrapper" in chain, chain
    # Between the episode counter and the auto-reset, as the stash requires.
    stash = chain.index("TruncationStashWrapper")
    assert chain.index("BraxAutoResetWrapper") < stash < chain.index("EpisodeWrapper")

    # And it does its job through the entry's whole stack, info reset included:
    # the flag is 1 on the step that hits the limit, 0 elsewhere.
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), BATCH))
    step = jax.jit(env.step)
    flags = []
    for _ in range(2 * EPISODE_LENGTH + 1):
        state = step(state, jnp.zeros((BATCH, 1)))
        flags.append(np.asarray(state.info[PRESERVE_KEY]["truncation"]))
    expected = [0, 0, 1, 0, 0, 1, 0]
    assert [float(f[0]) for f in flags] == expected
    assert all(np.all(f == f[0]) for f in flags)


def _called_names(path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            names.add(
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
    return names


@pytest.mark.parametrize("entry", ["kl_anchor", "highlvl", "imitation"])
def test_entry_does_not_build_training_env_without_stash(entry):
    """No entry calls the plain brax training wrap any more."""
    called = _called_names(SCRIPTS_DIR / ENTRIES[entry][0])
    assert "wrap_for_dmpo_training" in called
    assert "wrap_for_brax_training" not in called


@pytest.mark.parametrize("entry", ["kl_anchor", "highlvl", "imitation"])
def test_entry_stores_truncation_under_the_switch(entry):
    """The replay template gains the time-out flag when the learner reads it."""
    source = (SCRIPTS_DIR / ENTRIES[entry][0]).read_text()
    assert 'transition_template["truncation"]' in source
    assert ".bootstrap_on_truncation:" in source


@pytest.mark.parametrize("entry", ["highlvl", "imitation"])
def test_entry_requests_the_stash_with_full_reset_off(entry):
    tree = ast.parse((SCRIPTS_DIR / ENTRIES[entry][0]).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "wrap_for_dmpo_training"
    ]
    assert len(calls) == 1
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert kwargs["full_reset"] == "False"
