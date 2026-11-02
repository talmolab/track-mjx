"""Tests for the DMPO-private dictionary-observation helpers."""

import dataclasses
import inspect

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from track_mjx.agent.dmpo import obs_utils


def _flat_obs(batch=(), imit=6, prop=4, seed=0):
    """Random flat observation dict with a non-trivial mean in each half."""
    rng = np.random.default_rng(seed)
    return {
        "imitation_target": jnp.asarray(
            rng.normal(3.0, 2.0, (*batch, imit)), dtype=jnp.float32
        ),
        "proprioception": jnp.asarray(
            rng.normal(-1.0, 0.5, (*batch, prop)), dtype=jnp.float32
        ),
    }


# --------------------------------------------------------------------------
# flatten_obs_dict: batch rank comes from the proprioception leaf
# --------------------------------------------------------------------------


def test_flatten_nested_batch_rank_uses_proprioception():
    obs = {
        "imitation_target": {"joint": jnp.zeros((5, 7)), "wrist": jnp.zeros((5, 3))},
        "proprioception": jnp.zeros((5, 277)),
    }
    flat = obs_utils.flatten_obs_dict(obs)
    assert flat["imitation_target"].shape == (5, 10)
    assert flat["proprioception"].shape == (5, 277)


def test_flatten_nested_unbatched_proprioception_ravels_every_leaf():
    # Unbatched proprioception means no batch axes at all, so a leading axis
    # that two imitation leaves happen to share is a feature axis.
    obs = {
        "imitation_target": {"joint": jnp.zeros((5, 7)), "wrist": jnp.zeros((5, 3))},
        "proprioception": jnp.zeros((277,)),
    }
    flat = obs_utils.flatten_obs_dict(obs)
    assert flat["imitation_target"].shape == (50,)


def test_flatten_nested_leaves_sharing_leading_feature_axes():
    # Both leaves share their first two axes (5, 5), but only the first is a
    # batch axis; guessing the batch rank from the leaves would give (5, 5, 5).
    obs = {
        "imitation_target": {"a": jnp.zeros((5, 5, 3)), "b": jnp.zeros((5, 5, 2))},
        "proprioception": jnp.zeros((5, 277)),
    }
    flat = obs_utils.flatten_obs_dict(obs)
    assert flat["imitation_target"].shape == (5, 25)


def test_flatten_extra_batch_axes_follow_proprioception():
    obs = {
        "imitation_target": jnp.zeros((2, 5, 18, 5)),
        "proprioception": jnp.zeros((2, 5, 277)),
    }
    flat = obs_utils.flatten_obs_dict(obs)
    assert flat["imitation_target"].shape == (2, 5, 90)
    assert flat["proprioception"].shape == (2, 5, 277)


def test_flatten_nested_proprioception_keeps_batch_prefix():
    batched = {
        "imitation_target": jnp.zeros((7, 5)),
        "proprioception": {"a": jnp.zeros((7, 2)), "b": jnp.zeros((7, 3))},
    }
    flat = obs_utils.flatten_obs_dict(batched)
    assert flat["proprioception"].shape == (7, 5)
    unbatched = {
        "imitation_target": jnp.zeros((5,)),
        "proprioception": {"a": jnp.zeros((2,)), "b": jnp.zeros((3,))},
    }
    assert obs_utils.flatten_obs_dict(unbatched)["proprioception"].shape == (5,)


def test_flatten_concatenates_leaves_in_tree_order():
    obs = {
        "imitation_target": {"b": jnp.full((2, 2), 2.0), "a": jnp.full((2, 1), 1.0)},
        "proprioception": jnp.zeros((2, 3)),
    }
    flat = obs_utils.flatten_obs_dict(obs)
    np.testing.assert_array_equal(
        np.asarray(flat["imitation_target"]), [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]
    )


def test_flatten_mismatched_batch_prefix_raises():
    obs = {
        "imitation_target": {"a": jnp.zeros((4, 3)), "b": jnp.zeros((4, 2))},
        "proprioception": jnp.zeros((5, 277)),
    }
    with pytest.raises(ValueError, match="batch"):
        obs_utils.flatten_obs_dict(obs)


def test_flatten_unwraps_state_and_maps_task_obs_to_imitation_target():
    nested = {
        "state": {
            "proprioception": {"a": jnp.ones(2), "b": jnp.ones(3)},
            "task_obs": jnp.ones(5),
        },
        "vision": jnp.ones((8, 8, 2)),
    }
    flat = obs_utils.flatten_obs_dict(nested)
    assert set(flat) == {"imitation_target", "proprioception", "vision"}
    assert flat["proprioception"].shape == (5,)
    assert flat["imitation_target"].shape == (5,)
    assert flat["vision"].shape == (8, 8, 2)


def test_flatten_vision_nested_inside_state():
    nested = {
        "state": {
            "proprioception": jnp.ones(4),
            "imitation_target": jnp.ones(6),
            "vision": jnp.ones((5, 5, 3)),
        }
    }
    assert obs_utils.flatten_obs_dict(nested)["vision"].shape == (5, 5, 3)


def test_flatten_without_reference_obs_gives_zero_width_half():
    batched = obs_utils.flatten_obs_dict({"proprioception": jnp.zeros((3, 4, 9))})
    assert batched["imitation_target"].shape == (3, 4, 0)
    unbatched = obs_utils.flatten_obs_dict({"proprioception": jnp.zeros((9,))})
    assert unbatched["imitation_target"].shape == (0,)


def test_flatten_is_jittable_on_nested_input():
    obs = {
        "imitation_target": {"joint": jnp.ones((5, 7)), "wrist": jnp.ones((5, 3))},
        "proprioception": jnp.ones((5, 11)),
    }
    flat = jax.jit(obs_utils.flatten_obs_dict)(obs)
    assert flat["imitation_target"].shape == (5, 10)


# --------------------------------------------------------------------------
# get_dmpo_obs_sizes
# --------------------------------------------------------------------------


def test_vision_size_not_multiplied_by_batch_dims():
    obs = {
        "state": {
            "proprioception": jnp.zeros((4, 277)),
            "imitation_target": jnp.zeros((4, 10)),
            "vision": jnp.zeros((4, 32, 32, 2)),
        }
    }
    sizes = obs_utils.get_dmpo_obs_sizes(obs)
    assert sizes["vision"] == 32 * 32 * 2
    assert sizes == {
        "imitation_target": 10,
        "proprioception": 277,
        "vision": 32 * 32 * 2,
    }


@pytest.mark.parametrize(
    "vision_shape", [(32, 32, 2), (4, 32, 32, 2), (3, 4, 32, 32, 2)]
)
def test_vision_size_is_hwc_product_for_any_leading_dims(vision_shape):
    obs = {
        "proprioception": jnp.zeros((*vision_shape[:-3], 277)),
        "imitation_target": jnp.zeros((*vision_shape[:-3], 10)),
        "vision": jnp.zeros(vision_shape),
    }
    assert obs_utils.get_dmpo_obs_sizes(obs)["vision"] == 32 * 32 * 2


def test_obs_sizes_preserves_task_obs_key_name():
    obs = {"state": {"proprioception": jnp.ones(277), "task_obs": jnp.ones(640)}}
    assert obs_utils.get_dmpo_obs_sizes(obs) == {
        "task_obs": 640,
        "proprioception": 277,
    }


def test_obs_sizes_of_flat_dict_matches_flatten():
    assert obs_utils.get_dmpo_obs_sizes(_flat_obs()) == {
        "imitation_target": 6,
        "proprioception": 4,
    }


def test_obs_sizes_flatten_nested_halves():
    obs = {
        "imitation_target": {"joint": jnp.zeros((5, 7)), "wrist": jnp.zeros((5, 3))},
        "proprioception": jnp.zeros((5, 277)),
    }
    assert obs_utils.get_dmpo_obs_sizes(obs) == {
        "imitation_target": 10,
        "proprioception": 277,
    }


def test_obs_sizes_omit_absent_reference_half():
    assert obs_utils.get_dmpo_obs_sizes({"proprioception": jnp.zeros((2, 9))}) == {
        "proprioception": 9
    }


def test_obs_sizes_raise_on_malformed_input():
    with pytest.raises(KeyError, match="Expected 'state' key"):
        obs_utils.get_dmpo_obs_sizes({"unexpected_key": jnp.ones(3)})


# --------------------------------------------------------------------------
# DictRunningStatisticsState and the normalizer
# --------------------------------------------------------------------------


def test_state_is_a_flax_struct_with_one_entry_per_half():
    state = obs_utils.init_dict_normalizer(_flat_obs())
    assert dataclasses.is_dataclass(state)
    assert [f.name for f in dataclasses.fields(state)] == [
        "imitation_target",
        "proprioception",
    ]
    assert set(flax.serialization.to_state_dict(state)) == {
        "imitation_target",
        "proprioception",
    }
    assert state.imitation_target.mean.shape == (6,)
    assert state.proprioception.mean.shape == (4,)


def test_imitation_target_half_round_trips():
    template = _flat_obs()
    state = obs_utils.init_dict_normalizer(template)
    assert isinstance(state, obs_utils.DictRunningStatisticsState)
    batch = _flat_obs(batch=(256,), seed=1)
    state = obs_utils.update_dict_normalizer(state, batch)
    normed = obs_utils.normalize_dict_obs(batch, state)
    assert set(normed) == {"imitation_target", "proprioception"}
    for key in ("imitation_target", "proprioception"):
        out = np.asarray(normed[key])
        np.testing.assert_allclose(out.mean(0), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.std(0), 1.0, atol=1e-3)


def test_normalizer_round_trip_on_state_wrapped_nested_obs_with_extra_batch_axes():
    def make(batch, seed):
        rng = np.random.default_rng(seed)
        return {
            "state": {
                "task_obs": {
                    "joint": jnp.asarray(rng.normal(5.0, 3.0, (*batch, 7))),
                    "wrist": jnp.asarray(rng.normal(-2.0, 1.0, (*batch, 3))),
                },
                "proprioception": jnp.asarray(rng.normal(1.0, 4.0, (*batch, 12))),
            }
        }

    state = obs_utils.init_dict_normalizer(make((), 0))
    assert state.imitation_target.mean.shape == (10,)
    assert state.proprioception.mean.shape == (12,)
    batch = make((8, 32), 1)
    state = obs_utils.update_dict_normalizer(state, batch)
    normed = obs_utils.normalize_dict_obs(batch, state)
    for key in ("imitation_target", "proprioception"):
        out = np.asarray(normed[key])
        assert out.shape[:2] == (8, 32)
        np.testing.assert_allclose(
            out.reshape(-1, out.shape[-1]).mean(0), 0.0, atol=1e-5
        )


def test_update_forwards_pmap_axis_name():
    template = _flat_obs()
    batch = _flat_obs(batch=(64,), seed=2)
    expected = obs_utils.update_dict_normalizer(
        obs_utils.init_dict_normalizer(template), batch
    )
    state0 = obs_utils.init_dict_normalizer(template)
    per_device = jax.tree.map(lambda x: x[None], batch)
    pmapped = jax.pmap(
        lambda obs: obs_utils.update_dict_normalizer(
            state0, obs, pmap_axis_name="devices"
        ),
        axis_name="devices",
    )(per_device)
    for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(pmapped)):
        np.testing.assert_allclose(np.asarray(got)[0], np.asarray(want), rtol=1e-5)


def test_zero_width_reference_half_is_a_no_op_through_the_normalizer():
    obs = {"proprioception": _flat_obs(batch=(16,))["proprioception"]}
    state = obs_utils.init_dict_normalizer(obs)
    assert state.imitation_target.mean.shape == (0,)
    state = obs_utils.update_dict_normalizer(state, obs)
    normed = obs_utils.normalize_dict_obs(obs, state)
    assert normed["imitation_target"].shape == (16, 0)
    np.testing.assert_allclose(
        np.asarray(normed["proprioception"]).mean(0), 0.0, atol=1e-5
    )


def test_normalize_rescales_uint8_vision():
    obs = {**_flat_obs(), "vision": jnp.full((4, 4, 3), 255, dtype=jnp.uint8)}
    state = obs_utils.init_dict_normalizer(obs)
    normed = obs_utils.normalize_dict_obs(obs, state)
    assert normed["vision"].dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(normed["vision"]), 1.0)


def test_normalize_passes_float_vision_through():
    obs = {**_flat_obs(), "vision": jnp.full((4, 4, 3), 0.5, dtype=jnp.float32)}
    state = obs_utils.init_dict_normalizer(obs)
    normed = obs_utils.normalize_dict_obs(obs, state)
    np.testing.assert_allclose(np.asarray(normed["vision"]), 0.5)


def test_normalize_is_jittable_on_nested_input():
    obs = {
        "imitation_target": {"joint": jnp.ones((5, 7)), "wrist": jnp.ones((5, 3))},
        "proprioception": jnp.ones((5, 11)),
    }
    state = obs_utils.init_dict_normalizer(obs)
    normed = jax.jit(obs_utils.normalize_dict_obs)(obs, state)
    assert normed["imitation_target"].shape == (5, 10)


# --------------------------------------------------------------------------
# documentation
# --------------------------------------------------------------------------


def test_docstrings_make_no_same_keys_claim():
    assert "same keys" not in obs_utils.__doc__.lower()
    for name, obj in vars(obs_utils).items():
        if name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != obs_utils.__name__:
            continue
        assert "same keys" not in (inspect.getdoc(obj) or "").lower(), name
