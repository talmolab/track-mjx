"""The post-hoc checkpoint evaluator and the entry eval it shares with training.

``scripts/eval_dmpo_checkpoints.py`` restores saved checkpoints and runs the
training entry's own eval (``_run_entry_eval``) on them. These tests run on CPU
with toy networks and a stub env: they check which checkpoints are evaluated,
what is restored, and what is written, and that the shared eval produces the
gate keys from a state that holds unbatched leaves.
"""

import json
import os
import runpy
from types import SimpleNamespace

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.scripts._entries import SCRIPTS_DIR
from track_mjx.agent.dmpo.checkpoint import make_checkpointer, record_alignment, save
from track_mjx.agent.dmpo.config import LEGACY_ALIGNMENT, DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.obs_utils import init_dict_normalizer

_ENV_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "XLA_PYTHON_CLIENT_PREALLOCATE")
ENV_SPEC = {"obs_size": 53, "action_size": 37}


@pytest.fixture
def evaluator(monkeypatch):
    """Module globals of ``scripts/eval_dmpo_checkpoints.py`` (``main`` not run)."""
    for var in _ENV_VARS:
        if var in os.environ:
            monkeypatch.setenv(var, os.environ[var])
        else:
            monkeypatch.delenv(var, raising=False)
    return runpy.run_path(
        str(SCRIPTS_DIR / "eval_dmpo_checkpoints.py"), run_name="evaluator_under_test"
    )


def _template(env_spec=ENV_SPEC, proprio_width=3, dict_normalizer=True):
    """A TrainingState; by default with the dict normalizer the entry uses.

    ``dict_normalizer=False`` keeps the flat normalizer ``init_training_state``
    builds for these flat-obs networks, i.e. the template ``evaluate`` builds.
    """
    cfg = DMPOConfig(num_envs=4)
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(jax.random.key(0), nets, env_spec, cfg)
    if not dict_normalizer:
        return cfg, nets, state
    normalizer = init_dict_normalizer(
        {
            "proprioception": jnp.zeros((1, proprio_width)),
            "imitation_target": jnp.zeros((1, 5)),
        }
    )
    return cfg, nets, state._replace(normalizer_params=normalizer)


def _run_config(hydra_cfg=None, cfg=None):
    """The config a run launched with ``hydra_cfg`` saves with its checkpoints."""
    hydra_cfg = {} if hydra_cfg is None else hydra_cfg
    return record_alignment(json.loads(json.dumps(hydra_cfg)), cfg or DMPOConfig())


def _save_steps(ckpt_dir, template, steps, config=None):
    """Save one checkpoint per step, each with distinguishable values.

    Every normalizer leaf is saved as an array, ``std_eps`` included, as it is
    after a jitted training step (the template's ``std_eps`` is a Python float).
    ``config`` is the saved config item; by default the one a run launched with
    an empty hydra config saves.
    """
    config = _run_config() if config is None else config
    mgr = make_checkpointer(str(ckpt_dir))
    for step in steps:
        norm = template.normalizer_params
        if hasattr(norm, "proprioception"):
            norm = norm.replace(
                proprioception=norm.proprioception.replace(
                    mean=norm.proprioception.mean + step
                )
            )
        else:
            norm = norm.replace(mean=norm.mean + step)
        state = template._replace(
            policy_params=jax.tree.map(lambda x, d=step: x + d, template.policy_params),
            normalizer_params=jax.tree.map(jnp.asarray, norm),
            steps=jnp.asarray(step // 10, dtype=template.steps.dtype),
        )
        save(mgr, step, state, config=config)
    mgr.wait_until_finished()


def _first_leaf(tree):
    return np.asarray(jax.tree.leaves(tree)[0])


def test_checkpoint_steps_lists_only_finished_step_dirs(evaluator, tmp_path):
    for name in (
        "DMPONetwork_200",
        "DMPONetwork_100",
        "DMPONetwork_300.orbax-checkpoint-tmp",
        "other_7",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "wandb_state.json").write_text("{}")
    assert evaluator["checkpoint_steps"](tmp_path) == [100, 200]


def test_restore_eval_state_reads_policy_normalizer_and_steps(evaluator, tmp_path):
    _, _, template = _template()
    _save_steps(tmp_path, template, [100])
    state = evaluator["restore_eval_state"](template, tmp_path / "DMPONetwork_100")
    np.testing.assert_allclose(
        _first_leaf(state.policy_params), _first_leaf(template.policy_params) + 100
    )
    norm, like = state.normalizer_params, template.normalizer_params
    assert type(norm) is type(like)
    np.testing.assert_allclose(norm.proprioception.mean, like.proprioception.mean + 100)
    np.testing.assert_allclose(norm.proprioception.std, like.proprioception.std)
    # The configuration scalar saved as a float32 array is the template's value.
    assert norm.proprioception.std_eps == like.proprioception.std_eps
    assert int(state.steps) == 10
    # Items an eval does not read stay the template's.
    assert state.critic_params is template.critic_params


@pytest.mark.parametrize(
    ("kwargs", "what"),
    [
        ({"env_spec": {"obs_size": 54, "action_size": 37}}, "policy_params"),
        ({"proprio_width": 4}, "normalizer_params"),
    ],
)
def test_restore_eval_state_refuses_other_networks(evaluator, tmp_path, kwargs, what):
    _, _, template = _template()
    _save_steps(tmp_path, template, [100])
    _, _, other = _template(**kwargs)
    with pytest.raises(ValueError, match=f"{what} in .* does not match the networks"):
        evaluator["restore_eval_state"](other, tmp_path / "DMPONetwork_100")


def _fake_entry(cfg, nets, calls):
    def run_entry_eval(
        hydra_cfg, cfg_, K, nets_, recurrent, eval_env, spout_eval, state, step, key
    ):
        calls.append(
            {
                "step": step,
                "state": state,
                "key": key,
                "eval_env": eval_env,
                "recurrent": recurrent,
            }
        )
        return SimpleNamespace(
            metrics={
                "batch/gap_crossings_per_episode": step / 100,
                "episode_rewards/treat_collected": np.float32(1.5),
            },
            spout=None,
            vision_sensitivity=None,
            erc={"episode_length": 7},
        )

    return {
        "_resolve_run_setup": lambda h: SimpleNamespace(
            cfg=cfg, from_scratch=False, recurrent=False
        ),
        "_load_frozen_prior": lambda h, from_scratch: "prior",
        "_make_env": lambda h, prior, from_scratch: SimpleNamespace(
            base_env_wrapped="training-env"
        ),
        "_make_networks": lambda h, setup, prior, built: nets,
        "_env_spec": lambda built: ENV_SPEC,
        "_make_eval_env": lambda base: (f"eval({base})", None),
        "_run_entry_eval": run_entry_eval,
    }


def test_evaluate_runs_the_entry_eval_on_each_saved_checkpoint(evaluator, tmp_path):
    cfg, nets, template = _template(dict_normalizer=False)
    ckpt_dir = tmp_path / "ckpt"
    _save_steps(ckpt_dir, template, [100, 200])
    (ckpt_dir / "DMPONetwork_300.orbax-checkpoint-tmp").mkdir()
    out = tmp_path / "out" / "posthoc.jsonl"
    calls: list[dict] = []
    entry = _fake_entry(cfg, nets, calls)
    hydra_cfg = OmegaConf.create({})

    records = evaluator["evaluate"](entry, hydra_cfg, ckpt_dir, out, config_name="c")

    assert [c["step"] for c in calls] == [100, 200]
    assert calls[0]["eval_env"] == "eval(training-env)"
    np.testing.assert_allclose(
        _first_leaf(calls[1]["state"].policy_params),
        _first_leaf(template.policy_params) + 200,
    )
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines == records
    assert [r["env_steps"] for r in lines] == [100, 200]
    assert lines[1]["eval/batch/gap_crossings_per_episode"] == 2.0
    assert lines[0]["eval/episode_rewards/treat_collected"] == 1.5
    assert lines[0]["eval_seed"] == 0
    assert lines[0]["learner_steps"] == 10
    assert lines[0]["eval_episode_length"] == 7
    assert lines[0]["eval_num_envs"] == 4
    assert lines[0]["config_name"] == "c"
    assert lines[0]["config_diff"] == {}

    # Recorded (step, seed) pairs are not evaluated again; new seeds are.
    assert evaluator["evaluate"](entry, hydra_cfg, ckpt_dir, out) == []
    assert len(calls) == 2
    evaluator["evaluate"](entry, hydra_cfg, ckpt_dir, out, steps=[200], seeds=(0, 1))
    assert [(c["step"], int(np.asarray(c["key"])[-1])) for c in calls[2:]] == [(200, 1)]
    assert len(out.read_text().splitlines()) == 3


def test_evaluate_refuses_a_step_without_a_finished_checkpoint(evaluator, tmp_path):
    cfg, nets, template = _template(dict_normalizer=False)
    _save_steps(tmp_path, template, [100])
    (tmp_path / "DMPONetwork_300.orbax-checkpoint-tmp").mkdir()
    entry = _fake_entry(cfg, nets, [])
    with pytest.raises(FileNotFoundError, match=r"\[300\]"):
        evaluator["evaluate"](
            entry,
            OmegaConf.create({}),
            tmp_path,
            tmp_path / "o.jsonl",
            steps=[100, 300],
        )


def _launch(evaluator, tmp_path, saved_hydra, eval_hydra, **kwargs):
    """Save checkpoints of a run launched with ``saved_hydra``; evaluate them
    under ``eval_hydra``. Returns ``(calls, records)``."""
    cfg, nets, template = _template(dict_normalizer=False)
    ckpt_dir = tmp_path / "ckpt"
    _save_steps(ckpt_dir, template, [100, 200], config=_run_config(saved_hydra))
    calls: list[dict] = []
    records = evaluator["evaluate"](
        _fake_entry(cfg, nets, calls),
        OmegaConf.create(eval_hydra),
        ckpt_dir,
        tmp_path / "o.jsonl",
        **kwargs,
    )
    return calls, records


def test_evaluate_refuses_a_checkpoint_trained_under_another_config(
    evaluator, tmp_path
):
    saved = {"env_config": {"gap_width": 0.1, "walls": True}}
    run_under = {"env_config": {"gap_width": 0.2}}
    with pytest.raises(ValueError) as err:
        _launch(evaluator, tmp_path, saved, run_under)
    message = str(err.value)
    assert "env_config.gap_width: checkpoint=0.1 this config=0.2" in message
    assert "env_config.walls: checkpoint=True this config=<absent>" in message
    assert "[100, 200]" in message
    assert "++posthoc.allow_config_diff=true" in message
    assert not (tmp_path / "o.jsonl").exists()


def test_evaluate_writes_the_allowed_config_diff_into_each_record(evaluator, tmp_path):
    calls, records = _launch(
        evaluator,
        tmp_path,
        {"env_config": {"gap_width": 0.1}},
        {"env_config": {"gap_width": 0.2}},
        allow_config_diff=True,
    )
    assert [c["step"] for c in calls] == [100, 200]
    for record in records:
        assert record["config_diff"] == {
            "env_config.gap_width": {"checkpoint": 0.1, "config": 0.2}
        }


def test_evaluate_ignores_keys_that_differ_between_launch_and_eval(
    evaluator, tmp_path, monkeypatch
):
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/eval/machine/priors")
    monkeypatch.delenv("DMPO_CKPT_ROOT", raising=False)
    saved = {
        "checkpoint_dir": "/train/machine/ckpt/run",
        "logging_config": {"exp_name": "run", "project_name": "p"},
        "transfer": {
            "prior_checkpoint_path": "/train/machine/priors/gen1",
            "graft_ff_checkpoint": "/train/ckpts/src/DMPONetwork_5",
        },
    }
    run_under = {
        "checkpoint_dir": "/eval/machine/ckpt/run",
        "logging_config": {"exp_name": "other"},
        "posthoc": {"out": "o.jsonl", "seeds": [0]},
        "transfer": {
            "prior_checkpoint_path": "/eval/machine/priors/gen1",
            # DMPO_CKPT_ROOT unset: the configs' default root.
            "graft_ff_checkpoint": "dmpo_checkpoints/src/DMPONetwork_5",
        },
    }
    _, records = _launch(evaluator, tmp_path, saved, run_under)
    assert [r["config_diff"] for r in records] == [{}, {}]


def test_config_diff_compares_root_paths_by_what_follows_the_root(
    evaluator, monkeypatch
):
    monkeypatch.setenv("DMPO_PRIOR_ROOT", "/eval/priors/")
    diff = evaluator["config_diff"]
    saved = {"transfer": {"prior_checkpoint_path": "/train/priors/gen1"}}
    assert (
        diff(saved, {"transfer": {"prior_checkpoint_path": "/eval/priors/gen1"}}) == {}
    )
    assert diff(
        saved, {"transfer": {"prior_checkpoint_path": "/eval/priors/gen2"}}
    ) == {
        "transfer.prior_checkpoint_path": {
            "checkpoint": "/train/priors/gen1",
            "config": "/eval/priors/gen2",
        }
    }
    # A path outside every root is compared as it is.
    assert diff({"x": "/a/gen1"}, {"x": "/b/gen1"}) == {
        "x": {"checkpoint": "/a/gen1", "config": "/b/gen1"}
    }


def test_config_diff_reads_absent_alignment_fields_as_their_legacy_values(
    evaluator, tmp_path
):
    """A checkpoint saved before the alignment fields existed was trained by
    the legacy learner, as ``saved_alignment`` reads it."""
    legacy = DMPOConfig(**LEGACY_ALIGNMENT)
    cfg, nets, template = _template(dict_normalizer=False)
    ckpt_dir = tmp_path / "ckpt"
    _save_steps(ckpt_dir, template, [100], config={"train_config": {"n_step": 5}})
    entry = _fake_entry(legacy, nets, [])
    hydra_cfg = OmegaConf.create({"train_config": {"n_step": 5}})
    (record,) = evaluator["evaluate"](entry, hydra_cfg, ckpt_dir, tmp_path / "a.jsonl")
    assert record["config_diff"] == {}

    entry = _fake_entry(DMPOConfig(), nets, [])
    with pytest.raises(ValueError, match="bootstrap_on_truncation: checkpoint=False"):
        evaluator["evaluate"](entry, hydra_cfg, ckpt_dir, tmp_path / "b.jsonl")


def test_evaluate_refuses_a_checkpoint_without_a_config(evaluator, tmp_path):
    cfg, nets, template = _template(dict_normalizer=False)
    mgr = make_checkpointer(str(tmp_path))
    save(mgr, 100, template)
    mgr.wait_until_finished()
    with pytest.raises(ValueError, match="could not read the saved config"):
        evaluator["evaluate"](
            _fake_entry(cfg, nets, []),
            OmegaConf.create({}),
            tmp_path,
            tmp_path / "o.jsonl",
            allow_config_diff=True,
        )


def test_entry_defines_every_function_the_evaluator_calls(evaluator, kl_anchor_entry):
    assert evaluator["ENTRY_SCRIPT"] == SCRIPTS_DIR / "train_dmpo_highlvl_kl_anchor.py"
    for name in evaluator["ENTRY_FUNCTIONS"]:
        assert callable(kl_anchor_entry[name]), name


@flax.struct.dataclass
class _State:
    obs: dict
    data: dict
    done: jax.Array
    reward: jax.Array
    metrics: dict
    info: dict


class _StubEnv:
    """Batched env whose state has a shared leaf and an empty leaf.

    Env e earns reward 1 per step and ends its episode every ``e + 2`` steps;
    each env crosses a gap on the step its episode ends.
    """

    action_size = 3

    def reset(self, keys):
        n = keys.shape[0]
        return _State(
            obs={"proprioception": jnp.zeros((n, 4))},
            data={
                "t": jnp.zeros((n,)),
                "pool": jnp.zeros((9, 3)),
                "scratch": jnp.zeros((0, 0)),
            },
            done=jnp.zeros((n,)),
            reward=jnp.zeros((n,)),
            metrics={"rewards/treat_collected": jnp.zeros((n,))},
            info={"just_crossed_gap": jnp.zeros((n,))},
        )

    def step(self, st, _action):
        n = st.reward.shape[0]
        t = st.data["t"] + 1.0
        done = (t % (jnp.arange(n) + 2.0) == 0).astype(jnp.float32)
        ones = jnp.ones((n,))
        return st.replace(
            data={**st.data, "t": t, "pool": st.data["pool"] + 1.0},
            done=done,
            reward=ones,
            metrics={"rewards/treat_collected": ones},
            info={"just_crossed_gap": done},
        )


def test_entry_eval_reports_the_gate_keys(kl_anchor_entry):
    """The shared eval runs the rollout and returns the keys the run logs."""

    class _Mode:
        def mode(self):
            return jnp.zeros((3,))

    num_envs, episode_length = 4, 6
    ev = kl_anchor_entry["_run_entry_eval"](
        OmegaConf.create({"eval_render_config": {"episode_length": episode_length}}),
        DMPOConfig(num_envs=num_envs),
        50,
        SimpleNamespace(
            policy=SimpleNamespace(apply=lambda params, obs: _Mode()),
            recurrent_meta=None,
        ),
        False,
        _StubEnv(),
        None,
        SimpleNamespace(policy_params=None, normalizer_params=None, steps=0),
        1000,
        jax.random.key(0),
    )
    assert ev.erc == {"episode_length": episode_length}
    assert len(ev.rollout) == episode_length + 1
    assert ev.spout is None and ev.vision_sensitivity is None
    m = ev.metrics
    # Episodes of length 2, 3, 4, 5 complete 3 + 2 + 1 + 1 = 7 times in 6 steps,
    # with one crossing each.
    assert m["batch/n_complete_episodes"] == 7.0
    assert m["batch/gap_crossings_per_episode"] == 1.0
    # First episode of each env, censored at 6 steps: 1 treat per step.
    assert m["episode_rewards/treat_collected"] == pytest.approx((2 + 3 + 4 + 5) / 4)
