"""DMPO training entry point.

Single-device JAX (no pmap for first cut), Hydra-driven, structured loosely after
``track_mjx.train``/``track_mjx.agent.ff_ppo.ppo`` but slimmed down for the
DMPO rollout/replay/sgd interleave. Eval and checkpointing live in
``train_dmpo_eval`` and ``checkpoint``.

Loop:
    1. Roll out ``num_envs * unroll_length`` env steps via ``collect_rollout``.
    2. Push the trajectory to a flashbax replay buffer.
    3. Once ``can_sample`` flips, run ``K`` SGD updates per rollout, where
       ``K = max(1, samples_per_insert * unroll_length * num_envs / batch_size)``.
    4. Periodically log metrics to wandb (with print fallback).
"""

import os

# JAX/MuJoCo backend hints, mirroring track_mjx.train.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from track_mjx.agent.dmpo.config import DMPOConfig

# ---------------------------------------------------------------------------
# vnl-playground env adapter.
#
# The ``track_mjx.agent.dmpo.rollout.collect_rollout`` contract assumes
# ``env.step(state, action)`` returns ``(new_state, reward)``. vnl-playground's
# mujoco_playground-derived envs return a single ``State`` whose ``.reward``
# field carries the per-step reward. We wrap once at the registry boundary so
# the rollout sees the (state, reward) tuple it expects, without monkey-
# patching the env or branching the rollout.
# ---------------------------------------------------------------------------


class _VnlPlaygroundEnvAdapter:
    """Wrap a vnl-playground env so ``step`` returns ``(state, reward)``.

    ``pre_batched`` signals the rollout/eval that this env stack already
    handles vmapping internally (e.g. via brax VmapWrapper) and that the
    caller must NOT outer-vmap reset/step. Required for the binocular
    vision render wrapper, whose mjx renderer is allocated once with a
    fixed nworld and operates on already-batched mjx.Data.
    """

    def __init__(self, env, pre_batched: bool = False):
        self._env = env
        self.pre_batched = pre_batched

    @property
    def action_size(self) -> int:
        return self._env.action_size

    @property
    def observation_size(self):
        return self._env.observation_size

    def reset(self, rng):
        return self._env.reset(rng)

    def step(self, state, action):
        new_state = self._env.step(state, action)
        return new_state, new_state.reward


def _filter_dmpo_kwargs(d: dict) -> dict:
    """Drop keys that aren't DMPOConfig fields (defensive against stray YAML keys)."""
    valid = set(DMPOConfig.__dataclass_fields__.keys())
    return {k: v for k, v in d.items() if k in valid}


def compute_num_updates(cfg: DMPOConfig) -> int:
    """Number of SGD updates per rollout to maintain the configured samples-per-insert ratio.

    samples_per_insert (Acme/Reverb convention) = samples_drawn / inserts.

    In a synchronous train loop:
      samples_drawn = num_updates * batch_size
      inserts        = unroll_length * num_envs
    so num_updates = samples_per_insert * inserts / batch_size.

    Floored at 1: at least one SGD step per rollout, regardless of the ratio.
    """
    return max(
        1,
        int(cfg.samples_per_insert * cfg.unroll_length * cfg.num_envs / cfg.batch_size),
    )
