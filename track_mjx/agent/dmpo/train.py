"""DMPO training entry point.

Single-device JAX (no pmap for first cut), Hydra-driven, structured loosely after
``track_mjx.train``/``track_mjx.agent.ff_ppo.ppo`` but slimmed down for the
DMPO rollout/replay/sgd interleave. Eval and checkpointing land in Task 15.

Loop:
    1. Roll out ``num_envs * unroll_length`` env steps via ``collect_rollout``.
    2. Push the trajectory to a flashbax replay buffer.
    3. Once ``can_sample`` flips, run ``K`` SGD updates per rollout, where
       ``K = max(1, unroll_length * num_envs / (batch_size * samples_per_insert))``.
    4. Periodically log metrics to wandb (with print fallback).
"""

import logging
import os
import time

# JAX/MuJoCo backend hints, mirroring track_mjx.train.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import (
    init_training_state,
    make_optimizers,
    sgd_step,
)
from track_mjx.agent.dmpo.networks import make_dmpo_networks
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.rollout import collect_rollout

log = logging.getLogger(__name__)

try:
    import wandb

    _WANDB_IMPORTED = True
except ImportError:  # pragma: no cover - optional dep
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


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
    """Wrap a vnl-playground env so ``step`` returns ``(state, reward)``."""

    def __init__(self, env):
        self._env = env

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


def _log_metrics(metrics: dict, env_steps: int) -> None:
    """Cast metrics to floats and emit to wandb AND stdout.

    We always emit a compact one-liner via ``log.info`` so smoke tests and
    tail-based observers can see metric flow even when wandb is online.
    Wandb gets the same payload when active.
    """
    payload = {k: float(v) for k, v in metrics.items()}
    payload["env_steps"] = int(env_steps)
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.log(payload, step=int(env_steps))
    # Always log a tail-friendly one-liner; matches the "step=" pattern that
    # smoke tests grep for.
    log.info(
        "step=%d "
        + " ".join(f"{k}={v:.4g}" for k, v in payload.items() if k != "env_steps"),
        int(env_steps),
    )


def _filter_dmpo_kwargs(d: dict) -> dict:
    """Drop keys that aren't DMPOConfig fields (defensive against stray YAML keys)."""
    valid = set(DMPOConfig.__dataclass_fields__.keys())
    return {k: v for k, v in d.items() if k in valid}


@hydra.main(config_path="../../config", config_name="rodent-dmpo", version_base=None)
def main(hydra_cfg: DictConfig):
    """DMPO training driver."""
    raw_train_cfg = OmegaConf.to_container(hydra_cfg.train_config, resolve=True)
    cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    seed = int(hydra_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    # 0. Optional wandb. Wrapped so a missing API key / offline mode doesn't
    # abort the smoke test.
    if _WANDB_IMPORTED:
        try:
            wandb.init(
                project=str(hydra_cfg.get("wandb_project", "dmpo-rodent")),
                config=OmegaConf.to_container(hydra_cfg, resolve=True),
                mode=os.environ.get("WANDB_MODE", "online"),
                reinit=True,
            )
        except Exception as e:  # pragma: no cover - depends on env
            log.warning("wandb.init failed (%s); falling back to stdout logs.", e)

    # 1. Load env via vnl-playground registry.
    env_name = str(hydra_cfg.get("env_name", "RodentRunGap"))
    log.info("Loading vnl-playground env: %s", env_name)
    from vnl_playground import registry as vp_registry  # type: ignore

    raw_env = vp_registry.load(env_name)
    env = _VnlPlaygroundEnvAdapter(raw_env)

    obs_size = int(env.observation_size)
    action_size = int(env.action_size)
    env_spec = {"obs_size": obs_size, "action_size": action_size}
    log.info("env_spec: obs_size=%d action_size=%d", obs_size, action_size)

    # 2. Networks + training state + optimizers.
    nets = make_dmpo_networks(obs_size, action_size, cfg)
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    # 3. Replay (flashbax, per-env time axis).
    transition_template = {
        "observation": jnp.zeros((obs_size,), dtype=jnp.float32),
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": jnp.zeros((obs_size,), dtype=jnp.float32),
    }
    rb = make_replay(
        # flashbax counts per-env; vnl-ray's max_replay_size is global.
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    # 4. Rollout: pass ``nets.policy.apply`` (a stable, hashable callable on the
    # nn.Module) plus ``state.policy_params`` (data) into ``collect_rollout``.
    # Threading params through as data instead of via a per-iteration closure
    # keeps the inner ``lax.scan`` trace stable across loop iterations -- the
    # apply function is the same Python object every call, and JAX treats the
    # changing params as ordinary input arrays. (See Brax PPO for the same
    # pattern.)
    #
    # We wrap the whole rollout in a jit so the scan body is compiled once and
    # reused on every iteration. ``env`` and ``policy_apply`` are static
    # (closed over); ``policy_params``, ``rng`` are the only live inputs.
    policy_apply = nets.policy.apply
    _num_envs_static = cfg.num_envs
    _unroll_static = cfg.unroll_length

    @jax.jit
    def jit_collect_rollout(policy_params, rng):
        return collect_rollout(
            env,
            policy_apply,
            policy_params,
            rng,
            num_envs=_num_envs_static,
            num_steps=_unroll_static,
        )

    # 5. SGD step jit. Static args: nets, optimizers, cfg are baked in.
    def _sgd_step_unbound(s, b):
        return sgd_step(s, b, nets, optimizers, cfg)

    jit_sgd_step = jax.jit(_sgd_step_unbound)

    # 6. Main loop.
    total_env_steps = 0
    last_log_step = 0
    metrics: dict = {}
    first_sgd = True
    t0 = time.time()
    log.info(
        "Starting DMPO loop: num_envs=%d unroll_length=%d batch_size=%d "
        "min_replay=%d num_timesteps=%d",
        cfg.num_envs,
        cfg.unroll_length,
        cfg.batch_size,
        cfg.min_replay_size,
        cfg.num_timesteps,
    )

    first_rollout = True
    while total_env_steps < cfg.num_timesteps:
        # 6a. Rollout (jitted; first call compiles the scan body, subsequent
        # calls reuse it -- we only pay the trace cost once).
        rng, k_roll = jax.random.split(rng)
        if first_rollout:
            log.info("Compiling collect_rollout (first call only)...")
            roll_compile_start = time.time()
            traj, _final_state = jit_collect_rollout(state.policy_params, k_roll)
            jax.block_until_ready(traj["observation"])
            log.info(
                "collect_rollout compiled + first call done in %.1fs",
                time.time() - roll_compile_start,
            )
            first_rollout = False
        else:
            traj, _final_state = jit_collect_rollout(state.policy_params, k_roll)
        # NB: discount carries (1 - done) but vnl-ray applies γ inside the
        # learner's Bellman target; cfg.discount is not multiplied in here.
        # If parity ever drifts, multiply by cfg.discount before push.
        rb_state = rb.add(rb_state, traj)
        total_env_steps += cfg.num_envs * cfg.unroll_length

        if not bool(rb.can_sample(rb_state)):
            if total_env_steps - last_log_step >= cfg.log_every_steps:
                log.info(
                    "warming replay: %d env steps (need ~%d)",
                    total_env_steps,
                    cfg.min_replay_size,
                )
                last_log_step = total_env_steps
            continue

        # 6b. SGD updates. Match vnl-ray's samples-per-insert ratio.
        num_updates = max(
            1,
            int(
                cfg.unroll_length
                * cfg.num_envs
                / (cfg.batch_size * cfg.samples_per_insert)
            ),
        )
        for _ in range(num_updates):
            rng, k_sample = jax.random.split(rng)
            sample = rb.sample(rb_state, k_sample)
            batch = sample.experience
            if first_sgd:
                log.info(
                    "Compiling sgd_step (first call may take 5-10 minutes "
                    "on RodentRunGap; subsequent calls reuse the cache)..."
                )
                compile_start = time.time()
                state, metrics = jit_sgd_step(state, batch)
                # Block until the compile + first execution finishes, so the
                # timing log below is accurate. Touching one leaf is enough.
                jax.block_until_ready(metrics["policy_loss"])
                log.info(
                    "sgd_step compiled + first call done in %.1fs",
                    time.time() - compile_start,
                )
                first_sgd = False
            else:
                state, metrics = jit_sgd_step(state, batch)

        # 6c. Log.
        if total_env_steps - last_log_step >= cfg.log_every_steps:
            elapsed = max(time.time() - t0, 1e-6)
            metrics_to_log = {
                **{k: v for k, v in metrics.items()},
                "steps_per_sec": total_env_steps / elapsed,
                "num_updates_per_rollout": num_updates,
            }
            _log_metrics(metrics_to_log, total_env_steps)
            last_log_step = total_env_steps

    log.info("Training complete: %d env steps", total_env_steps)
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
