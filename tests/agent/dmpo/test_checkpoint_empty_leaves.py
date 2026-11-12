"""Checkpoints of a state whose normalizer holds zero-size arrays.

An observation key of width zero gives the running normalizer arrays with no
elements, which orbax refuses to write. The save leaves them out and records
their shape and dtype; restore puts them back.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from track_mjx.agent.dmpo.checkpoint import (
    EMPTY_LEAVES_ITEM,
    load_policy,
    load_train_state_items_numpy,
    make_checkpointer,
    restore,
    save,
)
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.obs_utils import init_dict_normalizer


def _state(rng, env_spec, proprio_width):
    cfg = DMPOConfig()
    nets = make_dmpo_networks(env_spec["obs_size"], env_spec["action_size"], cfg)
    state = init_training_state(rng, nets, env_spec, cfg)
    obs = {
        "proprioception": jnp.zeros((2, proprio_width)),
        "imitation_target": jnp.ones((2, 5)),
    }
    return state._replace(normalizer_params=init_dict_normalizer(obs))


def _equal(a, b):
    return all(
        jax.tree.leaves(
            jax.tree.map(
                lambda x, y: (
                    np.asarray(x).shape == np.asarray(y).shape
                    and np.asarray(x).dtype == np.asarray(y).dtype
                    and bool(np.array_equal(np.asarray(x), np.asarray(y)))
                ),
                a,
                b,
            )
        )
    )


def _layout(root: pathlib.Path) -> dict[str, bytes | None]:
    """Every file under ``root``: its name, and its bytes where they are stable.

    The write stamp, the ocdbt manifest and the content-addressed chunk names
    vary from save to save; the tree metadata and the file layout do not.
    """
    out: dict[str, bytes | None] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "_CHECKPOINT_METADATA":
            continue
        rel = p.relative_to(root)
        if "d" in rel.parts[:-1]:
            out[str(rel.parent)] = None
        elif p.name == "manifest.ocdbt":
            out[str(rel)] = None
        else:
            out[str(rel)] = p.read_bytes()
    return out


def test_empty_normalizer_leaf_round_trips(tmp_path, rng, env_spec):
    state = _state(rng, env_spec, proprio_width=0)
    assert state.normalizer_params.proprioception.mean.size == 0
    state = state._replace(
        normalizer_params=state.normalizer_params.replace(
            imitation_target=state.normalizer_params.imitation_target.replace(
                mean=jnp.arange(5, dtype=jnp.float32)
            )
        )
    )

    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=3, state=state)
    mgr.wait_until_finished()

    restored = restore(mgr, state_template=state)
    assert restored is not None
    assert _equal(state.normalizer_params, restored.normalizer_params)
    assert _equal(state.policy_params, restored.policy_params)
    assert _equal(state.critic_params, restored.critic_params)
    assert restored.normalizer_params.proprioception.mean.shape == (0,)
    assert int(restored.steps) == int(state.steps)

    policy = load_policy(mgr, state.policy_params)
    assert _equal(state.policy_params, policy)


def test_empty_policy_leaf_round_trips(tmp_path, rng, env_spec):
    """A zero-size leaf in the policy parameters comes back through the policy
    item (``load_policy``) as well as through the train state."""
    state = _state(rng, env_spec, proprio_width=3)
    params = state.policy_params
    empty_head = {"bias": jnp.zeros((0,), jnp.float32)}
    state = state._replace(
        policy_params={**params, "params": {**params["params"], "empty": empty_head}}
    )
    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=2, state=state)
    mgr.wait_until_finished()

    policy = load_policy(mgr, state.policy_params)
    assert policy["params"]["empty"]["bias"].shape == (0,)
    assert _equal(state.policy_params, policy)
    assert _equal(state.policy_params, restore(mgr, state).policy_params)
    loaded = load_train_state_items_numpy(
        tmp_path / "DMPONetwork_2", items=("policy_params",)
    )
    assert loaded["policy_params"]["params"]["empty"]["bias"].shape == (0,)


def test_empty_leaf_comes_back_from_a_shape_only_template(tmp_path, rng, env_spec):
    state = _state(rng, env_spec, proprio_width=0)
    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=1, state=state)
    mgr.wait_until_finished()

    template = state._replace(
        normalizer_params=jax.tree.map(
            lambda x: (
                jax.ShapeDtypeStruct(x.shape, x.dtype)
                if getattr(x, "size", 1) == 0
                else x
            ),
            state.normalizer_params,
        )
    )
    restored = restore(mgr, state_template=template)
    mean = restored.normalizer_params.proprioception.mean
    assert mean.shape == (0,) and mean.dtype == jnp.float32


def test_numpy_loader_returns_the_empty_leaf(tmp_path, rng, env_spec):
    state = _state(rng, env_spec, proprio_width=0)
    mgr = make_checkpointer(str(tmp_path))
    save(mgr, step=7, state=state)
    mgr.wait_until_finished()

    loaded = load_train_state_items_numpy(tmp_path / "DMPONetwork_7")
    norm = loaded["normalizer_params"]
    assert isinstance(norm["proprioception"]["mean"], np.ndarray)
    assert norm["proprioception"]["mean"].shape == (0,)
    assert norm["proprioception"]["std"].shape == (0,)
    assert norm["imitation_target"]["mean"].shape == (5,)


def test_state_without_empty_leaves_is_saved_unchanged(tmp_path, rng, env_spec):
    state = _state(rng, env_spec, proprio_width=4)
    config = {"a": 1, "b": [1, 2]}

    mgr = make_checkpointer(str(tmp_path / "new"))
    save(mgr, step=5, state=state, config=config)
    mgr.wait_until_finished()

    # What ``save`` wrote before empty leaves were handled.
    reference = make_checkpointer(str(tmp_path / "reference"))
    reference.save(
        5,
        args=ocp.args.Composite(
            policy=ocp.args.StandardSave(state.policy_params),
            train_state=ocp.args.StandardSave(state._asdict()),
            config=ocp.args.JsonSave(config),
        ),
    )
    reference.wait_until_finished()

    new_dir = tmp_path / "new" / "DMPONetwork_5"
    ref_dir = tmp_path / "reference" / "DMPONetwork_5"
    assert not (new_dir / EMPTY_LEAVES_ITEM).exists()
    assert sorted(p.name for p in new_dir.iterdir()) == sorted(
        p.name for p in ref_dir.iterdir()
    )
    assert _layout(new_dir) == _layout(ref_dir)
    keys = ("policy_params", "critic_params", "normalizer_params", "steps")
    assert _equal(
        load_train_state_items_numpy(new_dir, items=keys),
        load_train_state_items_numpy(ref_dir, items=keys),
    )

    restored = restore(mgr, state_template=state)
    assert _equal(state.normalizer_params, restored.normalizer_params)
    assert _equal(state.critic_params, restored.critic_params)
