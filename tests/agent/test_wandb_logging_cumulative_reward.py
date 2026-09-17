"""Coverage for the cumulative episode-reward scalar logger.

Trainers that do not run a Brax evaluator rely on
``_log_cumulative_reward_scalars`` for their eval/rollout_episode_reward-style
scalars. These tests call the helper directly and assert on the payload it
hands to ``wandb.log``, with the wandb transport stubbed out so no network call
happens.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import pytest
from jax import numpy as jnp
from omegaconf import OmegaConf

from track_mjx.agent import wandb_logging


@dataclass
class _FakeState:
    reward: float
    metrics: dict = field(default_factory=dict)
    done: float = 0.0
    obs: Any = None


class _FakeRun:
    """Stand-in for ``wandb.run``; ``id`` raises when ``id_raises`` is set."""

    def __init__(self, run_id, id_raises=False):
        self._id = run_id
        self._id_raises = id_raises

    @property
    def id(self):
        if self._id_raises:
            raise RuntimeError("no run id")
        return self._id


class _FakeEnvConfig:
    reward_terms: ClassVar[dict] = {"upright": None, "control": None}


class _FakeEnv:
    _config = _FakeEnvConfig()


def _payload(logged):
    payload = {}
    for entry, commit in logged:
        assert commit is False
        payload.update(entry)
    return payload


@pytest.fixture(autouse=True)
def _fresh_warned_terms(monkeypatch):
    """Give every test an empty warn-once set so test order cannot matter."""
    monkeypatch.setattr(wandb_logging, "_warned_reward_terms", set())
    monkeypatch.setattr(wandb_logging, "_warned_run_id", None)


def _warnings_for(caplog, term):
    return [
        r
        for r in caplog.records
        if f"Failed to log episode_reward_{term} scalar" in r.getMessage()
    ]


@pytest.fixture
def logged(monkeypatch):
    """Stub wandb.log to record payloads instead of hitting the network."""
    calls = []
    monkeypatch.setattr(
        wandb_logging.wandb,
        "log",
        lambda payload, commit=False: calls.append((payload, commit)),
    )
    return calls


def test_log_cumulative_reward_scalars_sums_across_rollout(logged):
    rollout = [
        _FakeState(
            reward=0.0, metrics={"rewards/upright": 0.0, "rewards/control": 0.0}
        ),
        _FakeState(
            reward=1.0, metrics={"rewards/upright": 0.5, "rewards/control": 0.2}
        ),
        _FakeState(
            reward=2.0, metrics={"rewards/upright": 1.5, "rewards/control": 0.3}
        ),
    ]

    wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 42, rollout)

    payload = _payload(logged)
    assert payload["eval/rollout_episode_reward"] == pytest.approx(3.0)
    assert payload["eval/rollout_episode_reward_upright"] == pytest.approx(2.0)
    assert payload["eval/rollout_episode_reward_control"] == pytest.approx(0.5)


def test_log_cumulative_reward_scalars_stops_at_episode_end(logged):
    """Sums stop at the first done step, like Brax's evaluator.

    The step on which ``done`` first becomes true is counted; every later step
    is not, including a later step whose ``done`` flag is clear again and one
    whose per-term value is NaN.
    """
    rollout = [
        _FakeState(
            reward=0.0, metrics={"rewards/upright": 0.0, "rewards/control": 0.0}
        ),
        _FakeState(
            reward=1.0, metrics={"rewards/upright": 0.5, "rewards/control": 0.2}
        ),
        _FakeState(
            reward=2.0,
            metrics={"rewards/upright": 1.5, "rewards/control": 0.3},
            done=1.0,
        ),
        _FakeState(
            reward=100.0,
            metrics={"rewards/upright": float("nan"), "rewards/control": 10.0},
            done=1.0,
        ),
        _FakeState(
            reward=100.0, metrics={"rewards/upright": 50.0, "rewards/control": 10.0}
        ),
    ]

    wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    payload = _payload(logged)
    assert payload["eval/rollout_episode_reward"] == pytest.approx(3.0)
    assert payload["eval/rollout_episode_reward_upright"] == pytest.approx(2.0)
    assert payload["eval/rollout_episode_reward_control"] == pytest.approx(0.5)


def test_log_cumulative_reward_scalars_done_on_first_step(logged):
    """A done flag on the first post-reset step counts only that step."""
    rollout = [
        _FakeState(
            reward=0.0, metrics={"rewards/upright": 0.0, "rewards/control": 0.0}
        ),
        _FakeState(
            reward=1.0,
            metrics={"rewards/upright": 0.5, "rewards/control": 0.2},
            done=1.0,
        ),
        _FakeState(
            reward=2.0, metrics={"rewards/upright": 1.5, "rewards/control": 0.3}
        ),
    ]

    wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    payload = _payload(logged)
    assert payload["eval/rollout_episode_reward"] == pytest.approx(1.0)
    assert payload["eval/rollout_episode_reward_upright"] == pytest.approx(0.5)
    assert payload["eval/rollout_episode_reward_control"] == pytest.approx(0.2)


def test_log_cumulative_reward_scalars_done_on_last_step(logged):
    """A done flag on the last step sums every post-reset step.

    The reset state's own done flag is ignored, as in Brax's evaluator.
    """
    rollout = [
        _FakeState(
            reward=0.0,
            metrics={"rewards/upright": 0.0, "rewards/control": 0.0},
            done=1.0,
        ),
        _FakeState(
            reward=1.0, metrics={"rewards/upright": 0.5, "rewards/control": 0.2}
        ),
        _FakeState(
            reward=2.0,
            metrics={"rewards/upright": 1.5, "rewards/control": 0.3},
            done=1.0,
        ),
    ]

    wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    payload = _payload(logged)
    assert payload["eval/rollout_episode_reward"] == pytest.approx(3.0)
    assert payload["eval/rollout_episode_reward_upright"] == pytest.approx(2.0)
    assert payload["eval/rollout_episode_reward_control"] == pytest.approx(0.5)


def test_log_cumulative_reward_scalars_skips_reset_only_rollout(logged):
    """A rollout with only the reset state (len<=1) must log nothing."""
    rollout = [_FakeState(reward=0.0, metrics={"rewards/upright": 0.0})]

    wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert logged == []


def test_log_cumulative_reward_scalars_skips_missing_metric_terms(logged, caplog):
    """A reward term absent from state.metrics is skipped with a warning."""
    rollout = [
        _FakeState(reward=0.0, metrics={}),
        _FakeState(reward=1.0, metrics={}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert _payload(logged) == {"eval/rollout_episode_reward": pytest.approx(1.0)}
    for term in ("upright", "control"):
        assert f"Failed to log episode_reward_{term} scalar" in caplog.text
    assert "KeyError" in caplog.text


def test_log_cumulative_reward_scalars_warns_once_per_term(logged, caplog):
    """A term that keeps failing warns on the first eval only."""
    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/control": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/control": 0.2}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 1, rollout)

    upright_warnings = _warnings_for(caplog, "upright")
    assert len(upright_warnings) == 1
    assert "KeyError" in upright_warnings[0].getMessage()
    assert [p for p, _ in logged].count(
        {"eval/rollout_episode_reward_control": pytest.approx(0.2)}
    ) == 2


def test_log_cumulative_reward_scalars_survives_none_metric(logged, caplog):
    """A None per-term value skips that term only, with a warning."""
    rollout = [
        _FakeState(
            reward=0.0, metrics={"rewards/upright": 0.0, "rewards/control": 0.0}
        ),
        _FakeState(
            reward=1.0, metrics={"rewards/upright": None, "rewards/control": 0.2}
        ),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert _payload(logged) == {
        "eval/rollout_episode_reward": pytest.approx(1.0),
        "eval/rollout_episode_reward_control": pytest.approx(0.2),
    }
    assert "Failed to log episode_reward_upright scalar" in caplog.text
    assert "ValueError" in caplog.text


def test_log_cumulative_reward_scalars_survives_non_iterable_reward_terms(
    logged, caplog
):
    """A reward_terms config that cannot be iterated skips per-term scalars."""

    class _BadConfig:
        reward_terms = 5

    class _BadEnv:
        _config = _BadConfig()

    rollout = [
        _FakeState(reward=0.0),
        _FakeState(reward=1.0),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_BadEnv(), 0, rollout)

    assert _payload(logged) == {"eval/rollout_episode_reward": pytest.approx(1.0)}
    assert "reward_terms" in caplog.text
    assert "TypeError" in caplog.text


def test_log_cumulative_reward_scalars_warns_again_for_a_new_failure(logged, caplog):
    """A different exception for the same term is warned about, not deduped."""
    missing = [
        _FakeState(reward=0.0, metrics={}),
        _FakeState(reward=1.0, metrics={}),
    ]
    unusable = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": None}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, missing)
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 1, unusable)

    messages = [r.getMessage() for r in _warnings_for(caplog, "upright")]
    assert len(messages) == 2
    assert "KeyError" in messages[0]
    assert "ValueError" in messages[1]


def test_log_cumulative_reward_scalars_warns_again_on_a_new_run(
    logged, caplog, monkeypatch
):
    """The warn-once bookkeeping resets when the wandb run changes."""
    rollout = [
        _FakeState(reward=0.0, metrics={}),
        _FakeState(reward=1.0, metrics={}),
    ]

    with caplog.at_level(logging.WARNING):
        monkeypatch.setattr(
            wandb_logging.wandb, "run", _FakeRun("run-1"), raising=False
        )
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 1, rollout)
        monkeypatch.setattr(
            wandb_logging.wandb, "run", _FakeRun("run-2"), raising=False
        )
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert len(_warnings_for(caplog, "upright")) == 2


@pytest.mark.parametrize("run", [None, _FakeRun("run-1", id_raises=True)])
def test_log_cumulative_reward_scalars_without_a_readable_run(
    logged, caplog, monkeypatch, run
):
    """An uninitialised wandb, or a run id that raises, must not break warnings."""
    rollout = [
        _FakeState(reward=0.0, metrics={}),
        _FakeState(reward=1.0, metrics={}),
    ]

    with caplog.at_level(logging.WARNING):
        monkeypatch.setattr(wandb_logging.wandb, "run", run, raising=False)
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 1, rollout)

    assert _payload(logged) == {"eval/rollout_episode_reward": pytest.approx(1.0)}
    assert len(_warnings_for(caplog, "upright")) == 1


@pytest.mark.parametrize("reward_terms", [None, {}])
def test_log_cumulative_reward_scalars_warns_once_when_no_terms(
    logged, caplog, reward_terms
):
    """An env with no reward terms warns once, instead of silently logging none."""

    class _NoTermsConfig:
        pass

    config = _NoTermsConfig()
    if reward_terms is not None:
        config.reward_terms = reward_terms

    class _NoTermsEnv:
        _config = config

    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_NoTermsEnv(), 0, rollout)
        wandb_logging._log_cumulative_reward_scalars(_NoTermsEnv(), 1, rollout)

    assert _payload(logged) == {"eval/rollout_episode_reward": pytest.approx(1.0)}
    no_terms = [r for r in caplog.records if "no reward terms" in r.getMessage()]
    assert len(no_terms) == 1


def test_log_cumulative_reward_scalars_warns_once_without_a_config(logged, caplog):
    """An env without a _config warns once, not on every eval."""

    class _BareEnv:
        pass

    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_BareEnv(), 0, rollout)
        wandb_logging._log_cumulative_reward_scalars(_BareEnv(), 1, rollout)

    assert _payload(logged) == {"eval/rollout_episode_reward": pytest.approx(1.0)}
    no_terms = [r for r in caplog.records if "no reward terms" in r.getMessage()]
    assert len(no_terms) == 1


def test_log_cumulative_reward_scalars_rejects_non_finite_done(logged, caplog):
    """A NaN done flag logs nothing, rather than being read as the episode end."""
    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}, done=float("nan")),
        _FakeState(reward=5.0, metrics={"rewards/upright": 2.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert logged == []
    assert "cannot read done flags: ValueError" in caplog.text
    assert "finite" in caplog.text


def test_log_cumulative_reward_scalars_rejects_string_reward_terms(logged, caplog):
    """A string reward_terms is not iterated per character."""

    class _StrConfig:
        reward_terms = "upright"

    class _StrEnv:
        _config = _StrConfig()

    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_StrEnv(), 0, rollout)

    assert _payload(logged) == {"eval/rollout_episode_reward": pytest.approx(1.0)}
    assert len(caplog.records) == 1
    assert "cannot read reward_terms: TypeError" in caplog.text


class _Unprintable:
    """A reward term whose str() raises, and optionally its repr() too."""

    def __init__(self, repr_raises):
        self._repr_raises = repr_raises

    def __str__(self):
        raise RuntimeError("no str")

    def __repr__(self):
        if self._repr_raises:
            raise RuntimeError("no repr")
        return "<unprintable term>"


@pytest.mark.parametrize(
    ("repr_raises", "label"),
    [(False, "<unprintable term>"), (True, "<unprintable reward term>")],
)
def test_log_cumulative_reward_scalars_survives_unprintable_term(
    logged, caplog, repr_raises, label
):
    """A term that cannot be formatted is skipped with a warning, never raised."""

    class _UnprintableConfig:
        reward_terms: ClassVar[list] = ["upright", _Unprintable(repr_raises)]

    class _UnprintableEnv:
        _config = _UnprintableConfig()

    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_UnprintableEnv(), 0, rollout)

    assert _payload(logged) == {
        "eval/rollout_episode_reward": pytest.approx(1.0),
        "eval/rollout_episode_reward_upright": pytest.approx(0.5),
    }
    assert f"Failed to log episode_reward_{label} scalar" in caplog.text


def test_log_cumulative_reward_scalars_survives_none_reward(logged, caplog):
    """A None reward skips the total with a warning; per-term scalars still log."""
    rollout = [
        _FakeState(
            reward=0.0, metrics={"rewards/upright": 0.0, "rewards/control": 0.0}
        ),
        _FakeState(
            reward=None, metrics={"rewards/upright": 0.5, "rewards/control": 0.2}
        ),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert _payload(logged) == {
        "eval/rollout_episode_reward_upright": pytest.approx(0.5),
        "eval/rollout_episode_reward_control": pytest.approx(0.2),
    }
    assert "Failed to log episode_reward scalar: ValueError" in caplog.text


def test_log_cumulative_reward_scalars_survives_total_log_failure(monkeypatch, caplog):
    """A wandb.log failure on the total key is warned; per-term keys still log."""
    calls = []

    def _raising_log(payload, commit=False):
        if "eval/rollout_episode_reward" in payload:
            raise RuntimeError("simulated wandb failure")
        calls.append((payload, commit))

    monkeypatch.setattr(wandb_logging.wandb, "log", _raising_log)

    rollout = [
        _FakeState(
            reward=0.0, metrics={"rewards/upright": 0.0, "rewards/control": 0.0}
        ),
        _FakeState(
            reward=1.0, metrics={"rewards/upright": 0.5, "rewards/control": 0.2}
        ),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert _payload(calls) == {
        "eval/rollout_episode_reward_upright": pytest.approx(0.5),
        "eval/rollout_episode_reward_control": pytest.approx(0.2),
    }
    assert "Failed to log episode_reward scalar: RuntimeError" in caplog.text


@dataclass
class _NoDoneState:
    reward: float
    metrics: dict = field(default_factory=dict)


def test_log_cumulative_reward_scalars_survives_states_without_done(logged, caplog):
    """States with no done flag log nothing, with a warning."""
    rollout = [
        _NoDoneState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _NoDoneState(reward=1.0, metrics={"rewards/upright": 0.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert logged == []
    assert "cannot read done flags: AttributeError" in caplog.text


def test_log_cumulative_reward_scalars_rejects_non_scalar_done(logged, caplog):
    """Done flags with a trailing dimension log nothing, with a warning."""
    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}, done=[0.0, 0.0]),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}, done=[0.0, 1.0]),
        _FakeState(reward=2.0, metrics={"rewards/upright": 1.5}, done=[0.0, 0.0]),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert logged == []
    assert "cannot read done flags: ValueError" in caplog.text
    assert "shape (2,)" in caplog.text


def test_log_cumulative_reward_scalars_survives_per_term_log_failure(
    monkeypatch, caplog
):
    """A wandb.log failure on the per-term write must not escape the helper.

    ``scripts/train.py`` wires ``rollout_logging_fn`` in as the PPO
    ``policy_params_fn``, which the ``ff_ppo`` and ``recurrent_ppo`` training
    loops invoke with no try/except of their own, so this helper must swallow
    wandb.log failures itself rather than let them propagate into a live
    training run.
    """

    def _raising_log(payload, commit=False):
        if any(key.startswith("eval/rollout_episode_reward_") for key in payload):
            raise RuntimeError("simulated wandb failure")

    monkeypatch.setattr(wandb_logging.wandb, "log", _raising_log)

    rollout = [
        _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}),
        _FakeState(reward=1.0, metrics={"rewards/upright": 0.5}),
    ]

    with caplog.at_level(logging.WARNING):
        wandb_logging._log_cumulative_reward_scalars(_FakeEnv(), 0, rollout)

    assert "Failed to log episode_reward_upright scalar" in caplog.text


def test_rollout_logging_fn_references_cumulative_reward_helper():
    """rollout_logging_fn's bytecode must reference the helper by name.

    ``co_names`` lists every global/attribute name a function's bytecode
    resolves at run time. This only checks that the name is referenced, not
    that the call is reached: it catches the helper being dropped from
    ``rollout_logging_fn`` entirely, which would silently stop the
    eval/rollout_episode_reward scalars from being logged.
    """
    assert (
        "_log_cumulative_reward_scalars"
        in wandb_logging.rollout_logging_fn.__code__.co_names
    )


class _RolloutEnvConfig:
    ctrl_dt = 0.02
    sim_dt = 0.002
    mocap_hz = 50
    reward_terms: ClassVar[dict] = {"upright": None}


class _RolloutEnv:
    """Minimal stand-in for the eval rollout environment."""

    _config = _RolloutEnvConfig()

    def _clip_length(self):
        return 4


def test_rollout_logging_fn_calls_cumulative_reward_helper(monkeypatch, logged):
    """rollout_logging_fn must call the helper with the rollout it collected.

    The stubs stand in for the jitted reset/step/inference functions, so this
    exercises the real control flow: if the call were unreachable, or were
    handed something other than the collected rollout, this fails.
    """
    calls = []
    monkeypatch.setattr(
        wandb_logging,
        "_log_cumulative_reward_scalars",
        lambda env, current_step, rollout: calls.append((env, current_step, rollout)),
    )

    def jit_reset(rng):
        return _FakeState(reward=0.0, metrics={"rewards/upright": 0.0}, obs={})

    def jit_step(state, ctrl):
        assert ctrl.shape == (3,)
        return _FakeState(
            reward=state.reward + 1.0,
            metrics={"rewards/upright": 0.5},
            obs={},
        )

    def jit_logging_inference_fn(params, obs, rng):
        extras = {"latent_mean": jnp.zeros(2), "latent_logvar": jnp.zeros(2)}
        return jnp.zeros((1, 3)), extras

    cfg = OmegaConf.create(
        {
            "network_config": {"arch_name": "intention"},
            "logging_config": {"log_histograms": False},
        }
    )
    env = _RolloutEnv()

    wandb_logging.rollout_logging_fn(
        env=env,
        jit_reset=jit_reset,
        jit_step=jit_step,
        cfg=cfg,
        model_path="unused",
        current_step=7,
        jit_logging_inference_fn=jit_logging_inference_fn,
        params=None,
        policy_params_fn_key=jax.random.PRNGKey(0),
        render_video=False,
    )

    assert len(calls) == 1
    spied_env, current_step, rollout = calls[0]
    assert spied_env is env
    assert current_step == 7
    assert [s.reward for s in rollout] == [0.0, 1.0, 2.0, 3.0, 4.0]
