"""CPU unit tests for the kl-anchor entry's pure config checks.

* ``_parse_slow_core_cfg``: ``network_config.policy_head_slow_core`` keys
  (``cell``, ``forget_bias``), their validation and defaults.
"""

import pytest
from omegaconf import OmegaConf

BASE = {"hidden_size": 512, "tick_stride": 10}


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
