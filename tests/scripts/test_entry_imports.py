"""The DMPO entry scripts load cleanly, and their CPU-checkable helpers behave.

Loading an entry (see ``tests.scripts._entries.load_entry``) must not import the spout-maze
eval helpers: they are needed only when the built env is a
``RodentSpoutMazeVision``, and ``main()`` imports them in that case alone.
"""

import ast
import sys

import pytest
from omegaconf import OmegaConf

from tests.scripts._entries import ENTRIES, SCRIPTS_DIR, load_entry

SPOUT_EVAL = "vnl_playground.tasks.rodent.spout_eval"


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_entries_import(entry, monkeypatch):
    monkeypatch.delitem(sys.modules, SPOUT_EVAL, raising=False)
    namespace = load_entry(entry, monkeypatch)
    assert callable(namespace["main"])
    assert SPOUT_EVAL not in sys.modules


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_entries_resume_the_env_step_counter(entry):
    """A resumed run must continue its env-step count, not restart at 0.

    Checkpoints are saved at ``step=env_steps``; a counter restarted at 0
    trains ``num_timesteps`` more steps under step names that already exist.
    """
    tree = ast.parse((SCRIPTS_DIR / ENTRIES[entry][0]).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_training_loop"
    ]
    assert len(calls) == 1
    passed = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "start_env_steps" in passed
    assert ast.unparse(passed["start_env_steps"]) == "start_env_steps"
    assigned = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "start_env_steps" for t in node.targets
        )
    ]
    assert "int(ckpt_mgr.latest_step() or 0)" in assigned


# --- kl-anchor entry: spout-maze eval env --------------------------------------


class _StubEnv:
    """Innermost env of a stub wrapper stack."""


class _StubWrapper:
    def __init__(self, env):
        self.env = env


def test_spout_eval_is_not_imported_for_other_envs(kl_anchor_entry, monkeypatch):
    monkeypatch.delitem(sys.modules, SPOUT_EVAL, raising=False)
    stack = _StubWrapper(_StubWrapper(_StubEnv()))
    eval_env, spout_eval = kl_anchor_entry["_make_eval_env"](stack)
    assert eval_env is stack
    assert spout_eval is None
    assert SPOUT_EVAL not in sys.modules


def test_spout_maze_env_gets_the_metric_alias(kl_anchor_entry):
    spout_maze = pytest.importorskip("vnl_playground.tasks.rodent.spout_maze")

    class _SpoutStub(spout_maze.RodentSpoutMazeVision):
        def __init__(self):  # skip building the model
            pass

    stack = _StubWrapper(_SpoutStub())
    eval_env, spout_eval = kl_anchor_entry["_make_eval_env"](stack)
    assert spout_eval is sys.modules[SPOUT_EVAL]
    assert isinstance(eval_env, spout_eval.SpoutEvalMetricAlias)
    assert eval_env.env is stack


# --- kl-anchor entry: graft source consistency ---------------------------------


def _write_source_config(step_dir, config):
    import orbax.checkpoint as ocp

    ocp.Checkpointer(ocp.JsonCheckpointHandler()).save(
        step_dir / "config", args=ocp.args.JsonSave(config)
    )


def test_graft_keys_include_critic_and_support(kl_anchor_entry, tmp_path):
    keys = kl_anchor_entry["_GRAFT_CONFIG_KEYS"]
    assert ("network_config", "critic_use_proprio", False) in keys
    assert ("train_config", "vmin", -150.0) in keys
    assert ("train_config", "vmax", 150.0) in keys

    check = kl_anchor_entry["_check_graft_source_config"]
    step_dir = tmp_path / "DMPONetwork_0"
    source = {
        "network_config": {"critic_use_proprio": True},
        "train_config": {"vmin": 0.0, "vmax": 20},
    }
    _write_source_config(step_dir, source)

    # Same function; the float-valued keys compare as floats (20 == 20.0).
    same = {
        "network_config": {"critic_use_proprio": True},
        "train_config": {"vmin": 0, "vmax": 20.0},
    }
    check(str(step_dir), OmegaConf.to_container(OmegaConf.create(same)))
    with pytest.raises(ValueError) as err:
        check(str(step_dir), {})  # every key falls back to its default
    for key in (
        "network_config.critic_use_proprio",
        "train_config.vmin",
        "train_config.vmax",
    ):
        assert key in str(err.value)
