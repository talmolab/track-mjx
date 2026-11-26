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

import difflib
import os
from collections.abc import Collection

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


# Keys an entry script reads from ``train_config`` itself, not DMPOConfig fields.
ENTRY_TRAIN_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "iters_per_chunk",  # SGD-chunk size of the training loop
        "episode_length",  # carried by the spout arms; equals env_config's
        "seed",  # imitation entry: PRNG seed
        "action_repeat",  # imitation entry: env action repeat
        "allow_alignment_change_on_resume",  # resume override of every entry
    }
)

# The PPO hyperparameter block of rodent-full-clips, which the intention
# imitation config inherits and merges its DMPO block into.
INHERITED_PPO_TRAIN_KEYS: frozenset[str] = frozenset(
    {
        "num_minibatches",
        "num_updates_per_batch",
        "learning_rate",
        "clipping_epsilon",
        "discounting",
        "entropy_cost",
        "reward_scaling",
        "normalize_observations",
        "deterministic_eval",
        "grad_clip_threshold",
        "vf_loss_coefficient",
        "get_activation",
    }
)


def _filter_dmpo_kwargs(d: dict, ignore: Collection[str] = frozenset()) -> dict:
    """The ``DMPOConfig`` keyword arguments of a ``train_config`` dict.

    Keys in ``ENTRY_TRAIN_CONFIG_KEYS`` and ``ignore`` are read elsewhere and
    left out. Any other key that is not a DMPOConfig field is refused: dropping
    it would train on the default, so a misspelt alignment switch or
    hyperparameter would go unnoticed.

    Args:
      d: the ``train_config`` mapping.
      ignore: further keys the caller reads itself or knows to be inert.

    Raises:
      ValueError: naming every unknown key and its closest field names.
    """
    valid = set(DMPOConfig.__dataclass_fields__.keys())
    known_elsewhere = ENTRY_TRAIN_CONFIG_KEYS | set(ignore)
    unknown = sorted(k for k in d if k not in valid and k not in known_elsewhere)
    if unknown:
        lines = []
        for key in unknown:
            close = difflib.get_close_matches(str(key), sorted(valid), n=3)
            hint = f" (did you mean {', '.join(map(repr, close))}?)" if close else ""
            lines.append(f"  {key!r}{hint}")
        raise ValueError(
            "train_config has keys that DMPOConfig does not take; a run would "
            "train on the default for each:\n"
            + "\n".join(lines)
            + "\nCheck the spelling, or the installed track-mjx may predate the field."
        )
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
