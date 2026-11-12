"""CPU unit tests for the kl-anchor entry's pure config checks.

* ``_parse_slow_core_cfg``: ``network_config.policy_head_slow_core`` keys
  (``cell``, ``forget_bias``), their validation and defaults;
* ``_check_maze_critic_priv_horizon``: the env's ``episode_length`` (what its
  ``step_count / episode_length`` critic entry is normalised by) must equal the
  horizon the training wrapper truncates at;
* ``_check_maze_critic_priv_dim``: ``network_config.critic_privileged_dim``
  must be the width the full maze writes (0 when it writes none);
* ``_make_eval_env`` on the full maze: the visited-cell mirror only when the
  env carries the critic vector.
"""

import sys
import types

import pytest
from omegaconf import OmegaConf

BASE = {"hidden_size": 512, "tick_stride": 10}
MAZE_EVAL = "vnl_playground.tasks.rodent.maze_eval"


# --- policy_head_slow_core -----------------------------------------------------


def test_omitted_optional_keys_keep_the_old_behaviour(kl_anchor_entry):
    out = kl_anchor_entry["_parse_slow_core_cfg"](dict(BASE))
    assert out == {
        "hidden_size": 512,
        "tick_stride": 10,
        "store_dtype": "float16",
        "cell": "gru",
        "forget_bias": 0.0,
    }


def test_store_dtype_and_cell_and_forget_bias_pass_through(kl_anchor_entry):
    parse = kl_anchor_entry["_parse_slow_core_cfg"]
    out = parse({**BASE, "store_dtype": "float32", "cell": "lstm", "forget_bias": 1.0})
    assert (out["store_dtype"], out["cell"], out["forget_bias"]) == (
        "float32",
        "lstm",
        1.0,
    )
    assert parse({**BASE, "cell": "gru", "forget_bias": 0})["forget_bias"] == 0.0


@pytest.mark.parametrize("cell", ["foo", "GRU", "rnn", None])
def test_unknown_cell_is_rejected(kl_anchor_entry, cell):
    with pytest.raises(ValueError, match="cell must be 'gru' or 'lstm'"):
        kl_anchor_entry["_parse_slow_core_cfg"]({**BASE, "cell": cell})


def test_nonzero_forget_bias_needs_the_lstm(kl_anchor_entry):
    parse = kl_anchor_entry["_parse_slow_core_cfg"]
    with pytest.raises(
        ValueError, match=r"forget_bias=1\.0 only applies to cell='lstm'"
    ):
        parse({**BASE, "forget_bias": 1.0})
    with pytest.raises(ValueError, match="only applies to cell='lstm'"):
        parse({**BASE, "cell": "gru", "forget_bias": -0.5})
    assert parse({**BASE, "cell": "lstm", "forget_bias": -0.5})["forget_bias"] == -0.5


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_forget_bias_is_rejected(kl_anchor_entry, bad):
    with pytest.raises(ValueError, match="finite"):
        kl_anchor_entry["_parse_slow_core_cfg"](
            {**BASE, "cell": "lstm", "forget_bias": bad}
        )


def test_unknown_and_missing_keys_are_rejected(kl_anchor_entry):
    parse = kl_anchor_entry["_parse_slow_core_cfg"]
    with pytest.raises(ValueError, match=r"unknown keys \['bogus'\]"):
        parse({**BASE, "bogus": 1})
    with pytest.raises(ValueError, match=r"missing keys \['tick_stride'\]"):
        parse({"hidden_size": 512})


def _slow_core_hydra_cfg(slow_core: dict) -> OmegaConf:
    """A minimal slow-core arm: the keys ``_resolve_run_setup`` reads."""
    return OmegaConf.create(
        {
            "train_config": {
                "rnn_bptt_length": 4,
                "n_step": 5,
                "sequence_length": 4 * slow_core["tick_stride"] + 5,
                "store_next_observation": False,
                "use_n_step": True,
            },
            "kl_anchor": {"alpha_anchor": 0.0, "w_anchor": 0.5},
            "network_config": {"policy_head_slow_core": slow_core},
        }
    )


@pytest.mark.parametrize(
    ("slow_core", "cell", "bias"),
    [
        ({"hidden_size": 8, "tick_stride": 2}, "gru", 0.0),
        (
            {"hidden_size": 8, "tick_stride": 2, "cell": "lstm", "forget_bias": 1.0},
            "lstm",
            1.0,
        ),
    ],
)
def test_run_setup_carries_the_slow_core_cell(kl_anchor_entry, slow_core, cell, bias):
    setup = kl_anchor_entry["_resolve_run_setup"](_slow_core_hydra_cfg(slow_core))
    assert setup.slow_core and setup.recurrent and not setup.gru_head
    assert (setup.slow_cell, setup.slow_forget_bias) == (cell, bias)


def test_run_setup_leaves_the_slow_core_fields_unset_off_the_slow_core(
    kl_anchor_entry,
):
    hydra_cfg = OmegaConf.create(
        {
            "train_config": {},
            "kl_anchor": {"alpha_anchor": 0.0, "w_anchor": 0.5},
            "network_config": {},
        }
    )
    setup = kl_anchor_entry["_resolve_run_setup"](hydra_cfg)
    assert (setup.slow_cell, setup.slow_forget_bias) == (None, None)


# --- maze critic vector: horizon ------------------------------------------------


def _env(flag, episode_length):
    return types.SimpleNamespace(
        _maze_critic_priv=flag,
        _config=types.SimpleNamespace(episode_length=episode_length),
    )


def test_horizon_check_passes_when_equal(kl_anchor_entry):
    kl_anchor_entry["_check_maze_critic_priv_horizon"](_env(True, 2000), 2000)


def test_horizon_check_raises_when_the_env_default_differs_from_the_wrapper(
    kl_anchor_entry,
):
    # An env_config without episode_length: the env keeps its default 1000 while
    # the wrapper truncates at unroll_length (here 2000).
    with pytest.raises(
        ValueError, match=r"episode_length=1000 differs .* horizon 2000"
    ):
        kl_anchor_entry["_check_maze_critic_priv_horizon"](_env(True, 1000), 2000)


def test_horizon_check_is_a_no_op_without_the_flag_or_on_other_envs(kl_anchor_entry):
    check = kl_anchor_entry["_check_maze_critic_priv_horizon"]
    check(_env(False, 1000), 2000)
    check(types.SimpleNamespace(), 2000)  # no flag attribute (spout, others)


# --- maze critic vector: width ---------------------------------------------------


@pytest.fixture
def maze_stubs():
    """Full-maze and spout-maze envs that skip building their models."""
    maze = pytest.importorskip(
        "vnl_playground.tasks.rodent.maze_forage_vision",
        reason="needs the maze-forage env, which the pinned vnl-playground lacks",
    )
    spout = pytest.importorskip(
        "vnl_playground.tasks.rodent.spout_maze",
        reason="needs the spout-maze env, which the pinned vnl-playground lacks",
    )

    class _MazeStub(maze.MazeForageVision):
        def __init__(self, flag, dim=264):
            self._maze_critic_priv = flag
            self._critic_priv_dim = dim
            self._config = types.SimpleNamespace(ctrl_dt=0.01)

    class _SpoutStub(spout.RodentSpoutMazeVision):
        def __init__(self):
            self._maze_critic_priv = False

    return _MazeStub, _SpoutStub


class _Wrapper:
    def __init__(self, env):
        self.env = env


def test_critic_dim_must_be_the_width_the_maze_writes(kl_anchor_entry, maze_stubs):
    maze_stub, _ = maze_stubs
    check = kl_anchor_entry["_check_maze_critic_priv_dim"]
    check(maze_stub(True, 264), 264)
    with pytest.raises(ValueError, match=r"critic_privileged_dim=200 .* width 264"):
        check(maze_stub(True, 264), 200)
    # The env computes a vector no critic would read.
    with pytest.raises(ValueError, match=r"critic_privileged_dim=0 .* width 264"):
        check(maze_stub(True, 264), 0)


def test_critic_dim_needs_the_flag_on_the_full_maze(kl_anchor_entry, maze_stubs):
    maze_stub, _ = maze_stubs
    check = kl_anchor_entry["_check_maze_critic_priv_dim"]
    check(maze_stub(False), 0)
    with pytest.raises(ValueError, match="maze_critic_priv is off"):
        check(maze_stub(False), 264)


def test_critic_dim_check_skips_the_spout_maze_and_other_envs(
    kl_anchor_entry, maze_stubs
):
    _, spout_stub = maze_stubs
    check = kl_anchor_entry["_check_maze_critic_priv_dim"]
    check(spout_stub(), 8)  # writes its own critic vector
    check(types.SimpleNamespace(), 8)


# --- full-maze eval env ---------------------------------------------------------


def test_full_maze_eval_env_mirrors_visited_cells_only_with_the_critic_vector(
    kl_anchor_entry, maze_stubs, monkeypatch
):
    maze_stub, spout_stub = maze_stubs
    monkeypatch.delitem(sys.modules, MAZE_EVAL, raising=False)
    make_eval_env = kl_anchor_entry["_make_eval_env"]

    stack = _Wrapper(_Wrapper(maze_stub(False)))
    assert make_eval_env(stack) == (stack, None)
    assert MAZE_EVAL not in sys.modules

    stack = _Wrapper(maze_stub(True))
    eval_env, spout_eval = make_eval_env(stack)
    maze_eval = sys.modules[MAZE_EVAL]
    assert isinstance(eval_env, maze_eval.MazeEvalMetricAlias)
    assert eval_env.env is stack and spout_eval is None
    assert kl_anchor_entry["_full_maze_env"](eval_env) is stack.env
    # The spout maze subclasses the full maze but is not one.
    assert kl_anchor_entry["_full_maze_env"](_Wrapper(spout_stub())) is None
