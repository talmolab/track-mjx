"""Warm-start graft: load a feed-forward DMPO checkpoint into a SUPERSET tree.

A memory-augmented or privileged-critic variant can warm-start from a
feed-forward (FF) checkpoint while building networks whose param trees are
supersets of it:

  * slow-core policy (`networks_kl_anchor_slow`): the FF policy tree with
    `policy_head/hidden_0/kernel` widened by H trailing rows (the slow state
    is appended LAST) plus a new `policy_head/slow_core` subtree;
  * privileged critic (`networks_vision_bottleneck`, `privileged_dim > 0`):
    the privileged vector is concatenated LAST, so the first critic Dense
    kernel (`params/Dense_0/kernel`) gains trailing rows.

`flax.serialization.from_state_dict` does not shape-check leaves, so loading
the FF checkpoint into such a tree would silently keep the OLD (narrow) shape.
`graft_params` is the explicit, shape-checked replacement. Its widened rows
are ZERO (not the target's random init): that is what makes the grafted
policy equal the FF policy, and the grafted critic equal the source critic,
at step 0 -- the extra inputs reach the output only through those rows.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping
from typing import Any

import flax.serialization as ser
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.core import FrozenDict, freeze
from flax.traverse_util import flatten_dict, unflatten_dict

from track_mjx.agent.dmpo.checkpoint import load_train_state_items_numpy

log = logging.getLogger(__name__)

# Items the policy side cannot do without: online policy, its target, and the
# observation normalizer it was trained under (a policy is useless without
# its running stats). A checkpoint missing any of these is an error.
_REQUIRED_ITEMS = ("policy_params", "target_policy_params", "normalizer_params")
# Critic items are optional: a checkpoint without them leaves the critic at
# its fresh init (reported, not fatal).
_OPTIONAL_ITEMS = ("critic_params", "target_critic_params")


def _path_str(path: tuple) -> str:
    return "/".join(str(p) for p in path)


def _shape_dtype(leaf: Any) -> tuple[tuple, np.dtype]:
    """(shape, dtype) of an array leaf or a Python scalar (shape (), the
    dtype jax would give it) -- never an AttributeError on a bare float."""
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return tuple(leaf.shape), leaf.dtype
    arr = np.asarray(leaf)
    return tuple(arr.shape), jax.dtypes.canonicalize_dtype(arr.dtype)


def _is_row_widening(t_shape: tuple, s_shape: tuple) -> bool:
    """True iff the target is the source plus extra ROWS (axis 0) of a rank-2
    kernel: same rank, same trailing dim, strictly more rows."""
    return (
        len(t_shape) == 2
        and len(s_shape) == 2
        and t_shape[0] > s_shape[0]
        and t_shape[1:] == s_shape[1:]
    )


def graft_params(target: Any, source: Any) -> tuple[Any, dict]:
    """Graft `source` params into `target`'s (superset) param tree.

    Both trees are compared leaf-by-leaf on their full paths (e.g.
    ``params/policy_head/hidden_0/kernel``); the nesting must be like for like
    (a flax variables dict with its top-level ``"params"`` key on both sides).
    Per leaf path:

      * in both, equal shapes -> the source value (cast to the target dtype);
      * in both, a rank-2 ``kernel`` whose target has MORE rows (axis 0) and
        the same trailing dim -> zeros of the target shape with the source
        copied into the first rows. The extra rows are zero, not the target's
        init, so the new inputs contribute nothing at step 0;
      * only in the target -> the target's own value (its init) is kept;
      * only in the source, or any other shape mismatch (a non-kernel leaf
        that changed shape, a kernel that shrank or changed its trailing dim,
        a rank change) -> ``ValueError``. All offending paths are listed in
        one error.

    Inputs are not modified. Source leaves may be numpy (the loader's
    output) or jax arrays; the grafted leaves are jax arrays. The result has
    the target's container type (plain dict or ``FrozenDict``).

    This is for PARAM trees. The observation normalizer is a
    ``DictRunningStatisticsState`` (a flax struct with a split ``count`` and
    a Python-float ``std_eps``), not a nested dict of arrays: use
    `graft_normalizer` for it.

    Returns:
      ``(grafted, report)`` with ``report = {"copied": n, "widened": [paths],
      "new": [paths]}``: ``copied`` counts equal-shape leaves taken from the
      source (widened leaves are listed under ``widened``, not counted in
      ``copied``); ``widened`` / ``new`` are ``"a/b/c"`` path strings.
    """
    t_flat = flatten_dict(target)
    s_flat = flatten_dict(source)

    errors: list[str] = []
    out: dict[tuple, Any] = {}
    copied = 0
    widened: list[str] = []
    new: list[str] = []

    for path, t_leaf in t_flat.items():
        if path not in s_flat:
            out[path] = t_leaf
            new.append(_path_str(path))
            continue
        s_leaf = np.asarray(s_flat[path])
        t_shape, dtype = _shape_dtype(t_leaf)
        s_shape = tuple(s_leaf.shape)
        if t_shape == s_shape:
            out[path] = jnp.asarray(s_leaf, dtype=dtype)
            copied += 1
        elif path[-1] == "kernel" and _is_row_widening(t_shape, s_shape):
            padded = np.zeros(t_shape, dtype=dtype)
            padded[: s_shape[0]] = s_leaf
            out[path] = jnp.asarray(padded)
            widened.append(_path_str(path))
        else:
            errors.append(
                f"{_path_str(path)}: shape mismatch (target {t_shape}, source "
                f"{s_shape}); only a rank-2 'kernel' may grow, and only along "
                "axis 0 with identical trailing dims"
            )

    for path in s_flat:
        if path not in t_flat:
            errors.append(
                f"{_path_str(path)}: present in the source but not in the "
                "target (source-only leaves cannot be dropped silently)"
            )

    if errors:
        raise ValueError(
            f"graft_params: {len(errors)} incompatible leaf path(s):\n  "
            + "\n  ".join(errors)
        )

    grafted = unflatten_dict(out)
    if isinstance(target, FrozenDict):
        grafted = freeze(grafted)
    return grafted, {"copied": copied, "widened": widened, "new": new}


def _is_config_scalar(path: tuple, leaf: Any) -> bool:
    """A normalizer leaf that is configuration, not a statistic: ``std_eps``
    (or any non-array leaf). It must MATCH between source and target."""
    return path[-1] == "std_eps" or not (
        hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    )


def graft_normalizer(target_state: Any, source: Any) -> Any:
    """Graft a loaded observation-normalizer state into the target's type.

    ``load_graft_source`` returns ``normalizer_params`` as a nested numpy
    dict (the flax state-dict of the saved ``DictRunningStatisticsState``).
    Downstream code dispatches on ``isinstance(..., DictRunningStatisticsState)``
    (``rollout._normalize_obs``), so a raw dict must not reach the trainer;
    and ``from_state_dict`` alone does not shape-check. This rebuilds a real
    ``type(target_state)`` -- e.g. from ``init_dict_normalizer`` with the
    obs template the trainer uses -- with every leaf checked:

      * array leaves (``mean``, ``std``, ``summed_variance``, the split
        ``count/hi`` and ``count/lo``): the shape must match the target
        exactly (a changed proprio / task-obs dim raises), the value is the
        source's, cast to the target's dtype;
      * ``std_eps`` (configuration, and any other non-array leaf): the
        target's value is kept when the source's equals it, ``ValueError``
        when it differs (different normalization hyperparameters would break
        step-0 equality);
      * a path in only one side -> ``ValueError`` (a normalizer with a
        different key set means a different observation layout; nothing is
        dropped or left at init silently).

    Every offending path (``"proprioception/mean"``) is named in one
    ``ValueError``. Static fields (``mode``) come from the target. ``source``
    may be the loader's nested dict or a flax struct state.
    """
    t_flat = flatten_dict(ser.to_state_dict(target_state))
    s_sd = source if isinstance(source, Mapping) else ser.to_state_dict(source)
    s_flat = flatten_dict(s_sd)

    errors: list[str] = []
    out: dict[tuple, Any] = {}

    for path, t_leaf in t_flat.items():
        name = _path_str(path)
        if path not in s_flat:
            errors.append(
                f"{name}: present in the target normalizer but missing from "
                "the source (the checkpoint has no statistics for it)"
            )
            continue
        s_leaf = np.asarray(s_flat[path])
        if _is_config_scalar(path, t_leaf):
            if s_leaf.shape != ():
                errors.append(
                    f"{name}: expected a scalar, source has shape {s_leaf.shape}"
                )
            elif not bool(np.asarray(t_leaf).astype(s_leaf.dtype) == s_leaf):
                errors.append(
                    f"{name}: value differs (target {t_leaf!r}, source "
                    f"{s_leaf.item()!r}); the normalization would change"
                )
            else:
                out[path] = t_leaf
            continue
        t_shape, dtype = _shape_dtype(t_leaf)
        if tuple(s_leaf.shape) != t_shape:
            errors.append(
                f"{name}: shape mismatch (target {t_shape}, source "
                f"{tuple(s_leaf.shape)})"
            )
            continue
        out[path] = jnp.asarray(s_leaf.astype(dtype))

    for path in s_flat:
        if path not in t_flat:
            errors.append(
                f"{_path_str(path)}: present in the source normalizer but not "
                "in the target"
            )

    if errors:
        raise ValueError(
            f"graft_normalizer: {len(errors)} incompatible leaf path(s):\n  "
            + "\n  ".join(errors)
        )
    return ser.from_state_dict(target_state, unflatten_dict(out))


def _train_state_item_names(ckpt_step_dir: pathlib.Path) -> set[str]:
    """Top-level keys saved in a step dir's ``train_state`` item (metadata
    only, no array reads)."""
    path = ckpt_step_dir / "train_state"
    if not path.is_dir():
        raise FileNotFoundError(f"no train_state item at {path}")
    meta_tree = ocp.PyTreeCheckpointer().metadata(str(path)).item_metadata.tree
    return set(meta_tree)


def load_graft_source(ckpt_step_dir: str | pathlib.Path) -> dict:
    """Load the graftable items of a DMPO checkpoint as host numpy pytrees.

    Returns a dict with those of ``{policy_params, target_policy_params,
    critic_params, target_critic_params, normalizer_params}`` that exist in
    ``ckpt_step_dir/train_state``, via
    ``checkpoint.load_train_state_items_numpy`` with an explicit item list. The
    param items keep the checkpoint's nesting (a top-level ``"params"`` key,
    as in a flax init), which is what `graft_params` expects.
    ``normalizer_params`` comes back as the nested numpy state-dict of the
    saved ``DictRunningStatisticsState``; turn it into a real state with
    `graft_normalizer` (not `graft_params`).

    ``policy_params``, ``target_policy_params`` and ``normalizer_params`` are
    required (KeyError naming what is missing). The critic items are
    optional: absent ones are logged as a warning and omitted from the
    result, and the caller keeps the critic at its fresh init.
    """
    ckpt_step_dir = pathlib.Path(ckpt_step_dir)
    available = _train_state_item_names(ckpt_step_dir)
    present_optional = tuple(k for k in _OPTIONAL_ITEMS if k in available)
    absent_optional = [k for k in _OPTIONAL_ITEMS if k not in available]
    if absent_optional:
        log.warning(
            "graft source %s has no %s; the critic stays at its fresh init.",
            ckpt_step_dir,
            absent_optional,
        )
    return load_train_state_items_numpy(
        ckpt_step_dir, items=_REQUIRED_ITEMS + present_optional
    )
