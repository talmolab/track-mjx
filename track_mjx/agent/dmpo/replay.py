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
"""
import flashbax as fbx


def make_replay(
    max_size: int,
    min_size: int,
    sequence_length: int,
    sample_batch_size: int,
    add_batch_size: int,
    period: int = 1,
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

    Returns:
      A flashbax ``TrajectoryBuffer`` exposing ``init``, ``add``, ``sample``,
      and ``can_sample`` as pure functions of state.
    """
    return fbx.make_trajectory_buffer(
        max_length_time_axis=max_size,
        min_length_time_axis=min_size,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sequence_length,
        period=period,
    )
