"""flashbax-backed n-step trajectory replay for DMPO.

vnl-ray's DMPO uses n=50 step trajectories with uniform sampling. We wrap
``flashbax.make_trajectory_buffer`` to expose the same lifecycle (``init`` /
``add`` / ``sample`` / ``can_sample``) that Acme's reverb adder/sampler used.
The returned object is the bare flashbax ``TrajectoryBuffer`` -- pass-through,
no extra abstraction.

Observed flashbax 0.1.3 contract (verified against the installed library):

* ``buf.init(template)`` takes a single unbatched transition (pytree of
  per-leaf shape ``[...]``) and returns a ``TrajectoryBufferState``.
* ``buf.add(state, batch)`` expects per-leaf shape
  ``[add_batch_size, T, ...]`` -- batched over envs and time.
* ``buf.sample(state, rng)`` returns a ``TrajectoryBufferSample`` whose
  ``.experience`` field has per-leaf shape
  ``[sample_batch_size, sample_sequence_length, ...]``.
* ``buf.can_sample(state)`` flips to True once at least
  ``min_length_time_axis`` time steps per env have been added.

``make_replay(..., add_sequence_length=T)`` swaps in an in-place ``add`` for
fixed-length batches (see ``_make_inplace_add``);
``replay_add_sequence_length(cfg, max_size)`` resolves T from
``DMPOConfig.replay_inplace_add``, so an entry point builds its buffer with::

    max_size = max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs)
    rb = make_replay(
        max_size=max_size,
        ...,
        add_sequence_length=replay_add_sequence_length(cfg, max_size),
    )
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flashbax as fbx
import jax

if TYPE_CHECKING:
    from track_mjx.agent.dmpo.config import DMPOConfig


def make_replay(
    max_size: int,
    min_size: int,
    sequence_length: int,
    sample_batch_size: int,
    add_batch_size: int,
    period: int = 1,
    add_sequence_length: int | None = None,
):
    """Construct a flashbax trajectory buffer.

    Args:
      max_size: max time-axis length per env (maps to flashbax
        ``max_length_time_axis``). Note flashbax counts per-env, so to match
        vnl-ray's ``max_replay_size`` divide by ``num_envs``.
      min_size: minimum time-axis length per env before sampling is allowed
        (maps to flashbax ``min_length_time_axis``). Same per-env caveat as
        ``max_size``.
      sequence_length: length of sampled trajectories along the time axis
        (maps to ``sample_sequence_length``). Use ``cfg.n_step + 1`` so the
        learner sees ``n_step`` transitions plus the bootstrapping next state.
      sample_batch_size: number of trajectories per training batch.
      add_batch_size: number of parallel envs feeding the buffer (i.e. the
        leading dimension of every ``add`` call).
      period: stride between sampled-trajectory start indices (1 = uniform
        sampling over every valid start position).
      add_sequence_length: the time length T of EVERY ``add`` batch (the
        rollout unroll). When given and ``max_size % T == 0``, ``add`` writes
        with ``dynamic_update_slice`` instead of flashbax's
        ``experience.at[:, indices].set``. Same values at the same positions
        (``current_index`` only ever takes multiples of T, so a write never
        wraps), but XLA does it in place: the scatter makes XLA transpose each
        WHOLE replay leaf to time-major and back on every add, a transient the
        size of the largest leaf, which can exhaust device memory on a deep
        replay. An ``add`` with any other T raises at trace time. When
        ``max_size`` is not a multiple of T this keeps flashbax's ``add``;
        ``replay_add_sequence_length`` refuses that case instead. None
        (default) keeps flashbax's ``add``.

    Returns:
      A flashbax ``TrajectoryBuffer`` exposing ``init``, ``add``, ``sample``,
      and ``can_sample`` as pure functions of state.
    """
    buf = fbx.make_trajectory_buffer(
        max_length_time_axis=max_size,
        min_length_time_axis=min_size,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sequence_length,
        period=period,
    )
    if add_sequence_length is None:
        return buf
    if int(add_sequence_length) < 1:
        raise ValueError(
            f"add_sequence_length must be >= 1 or None, got {add_sequence_length}."
        )
    if max_size % int(add_sequence_length) != 0:
        return buf
    return dataclasses.replace(buf, add=_make_inplace_add(int(add_sequence_length)))


def replay_add_sequence_length(cfg: DMPOConfig, max_size: int) -> int | None:
    """The ``add_sequence_length`` to pass to ``make_replay`` for ``cfg``.

    None when ``cfg.replay_inplace_add`` is False (flashbax's ``add``,
    unchanged), else ``cfg.unroll_length``: every add of the fused training
    step is one unroll.

    Args:
      cfg: the run's ``DMPOConfig``.
      max_size: the per-env replay depth the buffer is built with (the
        ``max_size`` passed to ``make_replay``).

    Raises:
      ValueError: when ``replay_inplace_add`` is not a bool, or is True with a
        ``max_size`` that is not a multiple of ``unroll_length`` (the in-place
        write could then wrap, and silently keeping the scatter add would
        hide the memory saving the switch asks for).
    """
    flag = cfg.replay_inplace_add
    if not isinstance(flag, bool):
        raise ValueError(
            f"unknown replay_inplace_add={flag!r}; valid values: True, False"
        )
    if not flag:
        return None
    seq_len = int(cfg.unroll_length)
    if seq_len < 1 or int(max_size) % seq_len != 0:
        raise ValueError(
            "train_config.replay_inplace_add needs the per-env replay depth "
            f"{int(max_size)} to be a multiple of unroll_length {seq_len}; "
            "adjust max_replay_size (depth = max_replay_size // num_envs) or "
            "set replay_inplace_add=false."
        )
    return seq_len


def _make_inplace_add(seq_len: int):
    """flashbax ``add`` for fixed-length batches, written as in-place slices.

    Equivalent to ``flashbax.buffers.trajectory_buffer.add`` whenever every add
    has time length ``seq_len`` and the time axis is a multiple of it: the
    write index then starts at 0 and advances by ``seq_len`` mod the axis
    length, so ``[current_index, current_index + seq_len)`` never wraps.
    """

    def add(state, batch):
        if jax.tree.structure(batch) != jax.tree.structure(state.experience):
            raise ValueError("replay add: batch and buffer pytrees differ.")
        exp_leaves = jax.tree.leaves(state.experience)
        max_len = exp_leaves[0].shape[1]
        for b, e in zip(jax.tree.leaves(batch), exp_leaves):
            if b.shape[1] != seq_len:
                raise ValueError(
                    f"replay add: batch time length {b.shape[1]} != "
                    f"add_sequence_length {seq_len}; the in-place add needs "
                    "every add to have the same length."
                )
            if (
                b.shape[0] != e.shape[0]
                or b.shape[2:] != e.shape[2:]
                or b.dtype != e.dtype
            ):
                raise ValueError(
                    f"replay add: batch leaf {b.shape} {b.dtype} does not fit "
                    f"buffer leaf {e.shape} {e.dtype}."
                )
        start = state.current_index
        new_experience = jax.tree.map(
            lambda e, b: jax.lax.dynamic_update_slice_in_dim(e, b, start, axis=1),
            state.experience,
            batch,
        )
        new_index = state.current_index + seq_len
        return state.replace(
            experience=new_experience,
            current_index=new_index % max_len,
            is_full=state.is_full | (new_index >= max_len),
        )

    return add
