"""Evaluate saved DMPO checkpoints with the high-level training entry's own eval.

A run of ``scripts/train_dmpo_highlvl_kl_anchor.py`` saves a
``DMPONetwork_<env_steps>`` checkpoint right after each of its evals. This
script builds the env, the eval env and the networks with that entry's own
build functions, restores each checkpoint's policy parameters, observation
normalizer and step counter, and runs the entry's eval function
(``_run_entry_eval``) on them. Every evaluated checkpoint becomes one JSON line
holding the keys the in-run eval logs to wandb (``eval/<key>``,
``spout/<key>``, ``eval/vision_sensitivity``) plus ``env_steps``,
``eval_seed`` and the eval settings.

Nothing is written next to the checkpoints, no wandb run is started and no
eval video is rendered.

Usage:
    python scripts/eval_dmpo_checkpoints.py --config-name=<group>/<arm> \\
        ++checkpoint_dir=<directory holding DMPONetwork_<step>> \\
        ++posthoc.out=<results.jsonl> \\
        [++posthoc.steps=[<step>,...]] [++posthoc.seeds=[0]]

``checkpoint_dir`` is the training entry's own key, so the same config and
overrides that launched a run point this script at that run's checkpoints.
Without ``posthoc.steps`` every finished checkpoint in it is evaluated.
(step, seed) pairs already in ``posthoc.out`` are skipped, so a re-run after
more checkpoints are saved evaluates only the new ones.

The eval PRNG key is ``PRNGKey(seed)`` for every checkpoint, so the
checkpoints of a run are compared on the same env resets. The in-run eval
draws its key from the training key chain, so a post-hoc value is another
sample of the logged quantity, not a reproduction of the logged number.
"""

from __future__ import annotations

import json
import logging
import os
import re
import runpy
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.serialization
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent.dmpo.checkpoint import STEP_PREFIX, load_train_state_items_numpy
from track_mjx.agent.dmpo.config import resolve_sgd_steps_per_rollout
from track_mjx.agent.dmpo.learner import init_training_state
from track_mjx.agent.dmpo.warm_start_graft import graft_normalizer

log = logging.getLogger(__name__)

ENTRY_SCRIPT = Path(__file__).resolve().parent / "train_dmpo_highlvl_kl_anchor.py"

# The entry functions this script calls, in call order.
ENTRY_FUNCTIONS = (
    "_resolve_run_setup",
    "_load_frozen_prior",
    "_make_env",
    "_make_networks",
    "_env_spec",
    "_make_eval_env",
    "_run_entry_eval",
)

# The TrainingState fields an eval reads (see the entry's ``_run_entry_eval``).
EVAL_ITEMS = ("policy_params", "normalizer_params", "steps")

_STEP_DIR = re.compile(rf"^{re.escape(STEP_PREFIX)}_(\d+)$")


def checkpoint_steps(ckpt_dir: str | Path) -> list[int]:
    """Steps of the finished checkpoints in ``ckpt_dir``, ascending.

    A finished checkpoint is a ``DMPONetwork_<step>`` directory. Orbax writes a
    checkpoint under a temporary name and renames it when the write is
    complete, so a checkpoint still being written is not listed.
    """
    steps = []
    for path in Path(ckpt_dir).iterdir():
        match = _STEP_DIR.match(path.name)
        if match and path.is_dir():
            steps.append(int(match.group(1)))
    return sorted(steps)


def _check_like(restored: Any, like: Any, name: str, where: Path) -> None:
    """Raise unless every leaf of ``restored`` has the shape and dtype of ``like``'s."""
    like_leaves = jax.tree_util.tree_flatten_with_path(like)[0]
    got_leaves, got_def = jax.tree.flatten(restored)
    if got_def != jax.tree.structure(like):
        raise ValueError(f"{name} in {where} does not have the networks' tree")
    bad = [
        f"{jax.tree_util.keystr(path)}: saved {np.shape(got)} {np.asarray(got).dtype}, "
        f"expected {np.shape(want)} {np.asarray(want).dtype}"
        for (path, want), got in zip(like_leaves, got_leaves, strict=True)
        if np.shape(got) != np.shape(want)
        or np.asarray(got).dtype != np.asarray(want).dtype
    ]
    if bad:
        raise ValueError(
            f"{name} in {where} does not match the networks this config builds:\n  "
            + "\n  ".join(bad)
        )


def restore_eval_state(template: Any, step_dir: str | Path) -> Any:
    """``template`` with the ``EVAL_ITEMS`` of the checkpoint at ``step_dir``.

    The policy parameters and the step counter must match the template's
    shapes and dtypes. The normalizer goes through ``graft_normalizer``, the
    entry's own normalizer loader: array statistics are shape-checked and take
    the template's dtype, and configuration scalars (``std_eps``) must equal
    the template's.

    Args:
        template: A ``TrainingState`` of the networks the config builds.
        step_dir: A ``DMPONetwork_<step>`` directory.

    Raises:
        ValueError: A restored leaf does not fit the template, i.e. the
            checkpoint was trained with other networks or observations.
    """
    step_dir = Path(step_dir)
    saved = load_train_state_items_numpy(step_dir, items=EVAL_ITEMS)
    fields = {}
    for name in ("policy_params", "steps"):
        like = getattr(template, name)
        restored = flax.serialization.from_state_dict(like, saved[name])
        _check_like(restored, like, name, step_dir)
        fields[name] = jax.tree.map(jnp.asarray, restored)
    try:
        fields["normalizer_params"] = graft_normalizer(
            template.normalizer_params, saved["normalizer_params"]
        )
    except ValueError as e:
        raise ValueError(
            f"normalizer_params in {step_dir} does not match the networks this "
            f"config builds: {e}"
        ) from e
    return template._replace(**fields)


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (np.ndarray, jax.Array)):
        return np.asarray(x).tolist()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def eval_record(ev: Any, step: int, seed: int, extra: dict) -> dict:
    """One JSON line: ``extra`` plus the keys the in-run eval logs to wandb."""
    record = {"env_steps": int(step), "eval_seed": int(seed), **extra}
    if ev.metrics is not None:
        record.update({f"eval/{k}": v for k, v in ev.metrics.items()})
    if ev.spout is not None:
        record.update({f"spout/{k}": v for k, v in ev.spout.items()})
    if ev.vision_sensitivity is not None:
        record["eval/vision_sensitivity"] = ev.vision_sensitivity
    return record


def recorded(out_path: str | Path) -> set[tuple[int, int]]:
    """The (env_steps, eval_seed) pairs already in ``out_path``."""
    path = Path(out_path)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            done.add((int(rec["env_steps"]), int(rec["eval_seed"])))
    return done


def evaluate(
    entry: dict,
    hydra_cfg: DictConfig,
    ckpt_dir: str | Path,
    out_path: str | Path,
    steps: list[int] | None = None,
    seeds: tuple[int, ...] = (0,),
    config_name: str | None = None,
) -> list[dict]:
    """Evaluate the checkpoints of one run and append one JSON line per eval.

    Args:
        entry: Module globals of the training entry (``ENTRY_SCRIPT``).
        hydra_cfg: The config the run was launched with.
        ckpt_dir: The run's checkpoint directory.
        out_path: JSON-lines file the results are appended to.
        steps: Checkpoint steps to evaluate; None for every finished one.
        seeds: Eval PRNG seeds; each (step, seed) pair is one eval.
        config_name: Recorded in each line when given.

    Returns:
        The records written by this call.
    """
    ckpt_dir = Path(ckpt_dir)
    present = checkpoint_steps(ckpt_dir)
    if steps is None:
        steps = present
    missing = sorted(set(steps) - set(present))
    if missing:
        raise FileNotFoundError(
            f"no finished checkpoint for steps {missing} in {ckpt_dir} "
            f"(present: {present})"
        )
    done = recorded(out_path)
    todo = [(s, seed) for s in sorted(steps) for seed in seeds if (s, seed) not in done]
    log.info(
        "posthoc eval: %d checkpoint(s) in %s, %d eval(s) to run, %d already in %s",
        len(present),
        ckpt_dir,
        len(todo),
        len(done),
        out_path,
    )
    if not todo:
        return []

    setup = entry["_resolve_run_setup"](hydra_cfg)
    cfg = setup.cfg
    prior = entry["_load_frozen_prior"](hydra_cfg, setup.from_scratch)
    built = entry["_make_env"](hydra_cfg, prior, setup.from_scratch)
    nets = entry["_make_networks"](hydra_cfg, setup, prior, built)
    template = init_training_state(
        jax.random.PRNGKey(0), nets, entry["_env_spec"](built), cfg
    )
    eval_env, spout_eval = entry["_make_eval_env"](built.base_env_wrapped)
    K = resolve_sgd_steps_per_rollout(cfg)

    records = []
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for step, seed in todo:
        step_dir = ckpt_dir / f"{STEP_PREFIX}_{step}"
        state = restore_eval_state(template, step_dir)
        t0 = time.time()
        ev = entry["_run_entry_eval"](
            hydra_cfg,
            cfg,
            K,
            nets,
            setup.recurrent,
            eval_env,
            spout_eval,
            state,
            step,
            jax.random.PRNGKey(seed),
        )
        extra = {
            "checkpoint": str(step_dir),
            "learner_steps": int(np.asarray(state.steps)),
            "eval_num_envs": int(cfg.num_envs),
            "eval_episode_length": int(ev.erc.get("episode_length", 1000)),
            "eval_policy": "mode",
            "eval_wall_s": round(time.time() - t0, 1),
        }
        if config_name is not None:
            extra["config_name"] = config_name
        record = eval_record(ev, step, seed, extra)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, default=_jsonable) + "\n")
        log.info(
            "posthoc eval written: env_steps=%d seed=%d -> %s", step, seed, out_path
        )
        records.append(record)
    return records


@hydra.main(
    config_path="../track_mjx/config/dmpo",
    config_name="run_gap/velocity_only_kl_anchor",
    version_base=None,
)
def main(hydra_cfg: DictConfig) -> None:
    posthoc = OmegaConf.to_container(hydra_cfg.get("posthoc", {}), resolve=True) or {}
    out = posthoc.get("out")
    if not out:
        raise ValueError("set ++posthoc.out=<results.jsonl>")
    steps = posthoc.get("steps")
    seeds = tuple(int(s) for s in posthoc.get("seeds", [0]))
    entry = runpy.run_path(str(ENTRY_SCRIPT), run_name=ENTRY_SCRIPT.stem)
    from hydra.core.hydra_config import HydraConfig

    evaluate(
        entry,
        hydra_cfg,
        ckpt_dir=str(hydra_cfg.checkpoint_dir),
        out_path=str(out),
        steps=None if steps is None else [int(s) for s in steps],
        seeds=seeds,
        config_name=HydraConfig.get().job.config_name,
    )


if __name__ == "__main__":
    main()
