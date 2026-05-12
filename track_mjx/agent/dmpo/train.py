"""DMPO training entry point.

Single-device JAX (no pmap for first cut), Hydra-driven, structured loosely after
``track_mjx.train``/``track_mjx.agent.ff_ppo.ppo`` but slimmed down for the
DMPO rollout/replay/sgd interleave. Eval and checkpointing land in Task 15.

Loop:
    1. Roll out ``num_envs * unroll_length`` env steps via ``collect_rollout``.
    2. Push the trajectory to a flashbax replay buffer.
    3. Once ``can_sample`` flips, run ``K`` SGD updates per rollout, where
       ``K = max(1, samples_per_insert * unroll_length * num_envs / batch_size)``.
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

from track_mjx.agent.dmpo.action_utils import bind
from track_mjx.agent.dmpo.checkpoint import (
    make_checkpointer,
    restore as restore_ckpt,
    save as save_ckpt,
)
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
        int(
            cfg.samples_per_insert
            * cfg.unroll_length
            * cfg.num_envs
            / cfg.batch_size
        ),
    )


def _eval_episodic_return(
    env,
    policy_apply,
    policy_params,
    rng,
    num_eval_envs: int = 64,
    num_steps: int = 1000,
) -> float:
    """Deterministic eval rollout for a fixed number of steps.

    Distinct from the training rollout in two ways:
      1. Uses ``dist.mode()`` (no sampling) so we measure greedy policy
         performance, not exploration-flavored returns.
      2. Bounds the action via ``bind()`` (= clipped tanh, matching training)
         rather than bare ``jnp.tanh`` -- we never need the raw pre-tanh value
         for eval (no MPO loss to feed), but we do want the same open-interval
         guarantee the env wrapper expects.

    Returns the mean over ``num_eval_envs`` of the per-env summed reward across
    ``num_steps`` steps. Episodes that ``done`` mid-rollout still accumulate
    reward; this matches a "fixed-budget return" rather than a strict episodic
    return, which is fine for relative tracking across training and avoids
    needing to mask post-done rewards.
    """
    # See rollout.collect_rollout: if env is pre-batched (vision), skip
    # outer vmap on reset/step.
    pre_batched = bool(getattr(env, "pre_batched", False))

    rng, k_reset = jax.random.split(rng)
    reset_keys = jax.random.split(k_reset, num_eval_envs)
    if pre_batched:
        state = env.reset(reset_keys)
    else:
        state = jax.vmap(env.reset)(reset_keys)

    def step_fn(carry, _):
        st, total = carry
        # Deterministic action: mode of the action distribution.
        raw_action = jax.vmap(lambda o: policy_apply(policy_params, o).mode())(st.obs)
        # Use the same bind() that training uses, so eval's action distribution
        # matches what the env saw at train time (strictly inside (-1, 1) instead
        # of tanh's open closure on ±1, which can break atanh-using wrappers).
        bound_action = bind(raw_action)
        if pre_batched:
            new_st, reward = env.step(st, bound_action)
        else:
            new_st, reward = jax.vmap(env.step)(st, bound_action)
        return (new_st, total + reward), None

    (_, total_return), _ = jax.lax.scan(
        step_fn,
        (state, jnp.zeros(num_eval_envs)),
        None,
        length=num_steps,
    )
    return float(jnp.mean(total_return))


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

    # 1. Load env. Two paths:
    #    - flat-obs (default): vnl-playground registry, standard RodentRunGap.
    #    - vision (use_vision=True): raw env via tasks.load + PriorHighLevelWrapper.
    transfer_mode = str(hydra_cfg.get("transfer", {}).get("mode", ""))
    use_prior_decoder = transfer_mode == "prior_decoder"
    use_from_scratch = transfer_mode == "from_scratch"
    use_vision = use_prior_decoder or use_from_scratch
    _VALID_TRANSFER_MODES = ("", "prior_decoder", "from_scratch")
    if transfer_mode not in _VALID_TRANSFER_MODES:
        raise ValueError(
            f"Unknown transfer.mode={transfer_mode!r}; "
            f"expected one of {_VALID_TRANSFER_MODES}"
        )
    obs_size = None
    obs_size_dict = None
    vision_shape = None

    env_name = str(hydra_cfg.get("env_name", "RodentRunGap"))
    if use_vision:
        log.info("Loading vision-enabled env via tasks.load: %s", env_name)
        from vnl_playground import tasks  # type: ignore
        from vnl_playground.tasks.wrappers import (  # type: ignore
            PriorHighLevelWrapper,
        )
        from vnl_playground.tasks.prior_utils import (  # type: ignore
            load_prior_checkpoint,
            make_decoder_inference_fn as make_prior_decoder_fn,
            make_prior_inference_fn,
        )

        if use_prior_decoder:
            prior_ckpt_path = str(hydra_cfg.transfer.prior_checkpoint_path)
            prior_ckpt_step = hydra_cfg.transfer.get("prior_checkpoint_step", None)
            log.info("Loading prior checkpoint from: %s", prior_ckpt_path)
            (
                _enc_params,
                prior_params,
                decoder_params,
                normalizer_params,
                prior_cfg,
            ) = load_prior_checkpoint(prior_ckpt_path, prior_ckpt_step)
            latent_size = int(prior_cfg["network_config"]["intention_size"])
            log.info("Prior intention_size=%d", latent_size)

            prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_cfg)
            decoder_fn = make_prior_decoder_fn(
                decoder_params, normalizer_params, prior_cfg
            )
        else:
            log.info("from_scratch mode: skipping prior/decoder load.")
            prior_fn = None
            decoder_fn = None
            latent_size = None

        env_args = (
            OmegaConf.to_container(hydra_cfg.get("env_config", {}), resolve=True)
            or {}
        )
        # ``env_config`` carries an ``env_name`` field for bookkeeping, but the
        # underlying env's config_dict is locked and won't accept it. Drop it
        # (and ``flatten_obs``, which is a tasks.load arg, not an env field)
        # before forwarding the rest as config_overrides.
        env_args = {
            k: v
            for k, v in env_args.items()
            if k not in ("env_name", "flatten_obs")
        }
        base_env = tasks.load(
            env_name,
            flatten_obs=False,
            config_overrides=env_args,
        )
        # Capture renderer-needed model handles BEFORE high-level wrappers
        # stack on top — mj_model/mjx_model live on the base env (or its
        # immediate ``.env`` attribute), not on the wrapped result.
        raw_env = base_env
        mj_model = getattr(raw_env, "mj_model", None) or getattr(
            getattr(raw_env, "env", None), "mj_model", None
        )
        mjx_model = getattr(raw_env, "mjx_model", None) or getattr(
            getattr(raw_env, "env", None), "mjx_model", None
        )
        if mj_model is None or mjx_model is None:
            raise RuntimeError(
                "Could not find mj_model/mjx_model on base env for vision rendering"
            )
        n_eye_actuators = getattr(
            base_env.env if hasattr(base_env, "env") else base_env,
            "n_eye_actuators",
            0,
        )

        from vnl_playground.tasks.wrappers import EndToEndWrapper  # type: ignore

        if use_from_scratch:
            base_env = EndToEndWrapper(
                base_env,
                highlvl_obs_key=str(
                    hydra_cfg.transfer.get("highlvl_obs_key", "task_obs")
                ),
                decoder_obs_key=str(
                    hydra_cfg.transfer.get("decoder_obs_key", "proprioception")
                ),
            )
        elif use_prior_decoder:
            base_env = PriorHighLevelWrapper(
                base_env,
                prior_fn,
                decoder_fn,
                latent_size,
                highlvl_obs_key=str(
                    hydra_cfg.transfer.get("highlvl_obs_key", "task_obs")
                ),
                decoder_obs_key=str(
                    hydra_cfg.transfer.get("decoder_obs_key", "proprioception")
                ),
                pass_vision=True,
                pass_task_obs=True,
                deterministic_prior=bool(
                    hydra_cfg.transfer.get("deterministic_prior", True)
                ),
                noise_logvar=float(hydra_cfg.transfer.get("noise_logvar", -2.0)),
                n_eye_actuators=n_eye_actuators,
            )

        # Add ray-traced binocular vision rendering on top of the high-level
        # wrapper. ``_inject_vision`` populates state.obs["vision"] in-place,
        # replacing the placeholder zeros that the wrapper emitted.
        #
        # The mjx renderer is allocated once with a fixed nworld and operates
        # on a pre-batched mjx.Data. Outer-vmapping the render wrapper would
        # collapse the batch dim and produce shape mismatches. So we slot a
        # brax ``wrap_for_brax_training`` between the high-level wrapper and
        # the render wrapper -- this adds VmapWrapper + EpisodeWrapper +
        # AutoResetWrapper, so the env stack handles batching internally and
        # the rollout's outer vmap is bypassed via ``pre_batched=True``.
        from mujoco_playground._src import wrapper as mp_wrapper  # type: ignore
        from vnl_playground.tasks.rodent.vision_jax import (  # type: ignore
            BinocularVisionRenderWrapper,
        )

        episode_length = int(
            hydra_cfg.env_config.get(
                "episode_length",
                hydra_cfg.train_config.get("unroll_length", 1000),
            )
        )
        action_repeat = int(hydra_cfg.env_config.get("action_repeat", 1))
        base_env = mp_wrapper.wrap_for_brax_training(
            base_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            full_reset=False,
        )

        vision_width = int(hydra_cfg.env_config.get("vision_width", 32))
        vision_height = int(hydra_cfg.env_config.get("vision_height", 32))
        grayscale = bool(hydra_cfg.env_config.get("grayscale", True))
        left_camera = str(
            hydra_cfg.env_config.get("left_camera_name", "eye_left-rodent")
        )
        right_camera = str(
            hydra_cfg.env_config.get("right_camera_name", "eye_right-rodent")
        )
        base_env = BinocularVisionRenderWrapper(
            base_env,
            mj_model=mj_model,
            mjx_model=mjx_model,
            width=vision_width,
            height=vision_height,
            grayscale=grayscale,
            left_camera_name=left_camera,
            right_camera_name=right_camera,
            render_depth=False,
            use_textures=bool(hydra_cfg.env_config.get("use_textures", False)),
            use_shadows=bool(hydra_cfg.env_config.get("use_shadows", False)),
            eye_dropout_rate=float(
                hydra_cfg.env_config.get("eye_dropout_rate", 0.0)
            ),
            eval_eye_mode=str(
                hydra_cfg.env_config.get("eval_eye_mode", "binocular")
            ),
        )

        env = _VnlPlaygroundEnvAdapter(base_env, pre_batched=True)
        # env.observation_size is now a dict (e.g. {"imitation_target": int,
        # "proprioception": 0}); env.vision_shape is (H, W, 2C).
        obs_size_dict = dict(env.observation_size)
        action_size = int(env.action_size)
        vision_shape = tuple(
            getattr(
                base_env,
                "vision_shape",
                getattr(base_env.env, "vision_shape", (32, 32, 2)),
            )
        )
        log.info(
            "env_spec (vision): vision_shape=%s task_obs_size=%d action_size=%d",
            vision_shape,
            obs_size_dict.get("imitation_target", 0),
            action_size,
        )
        # The wrapper's process_state emits THREE leaves in pass_vision +
        # pass_task_obs mode: ``vision``, ``imitation_target``, and a zero-
        # length ``proprioception``. We carry the zero-length leaf in the
        # template so flashbax's structural check matches; the network only
        # consumes vision + imitation_target.
        proprio_size = int(obs_size_dict.get("proprioception", 0))
        if use_from_scratch and proprio_size == 0:
            raise RuntimeError(
                "from_scratch mode requires real proprioception in the obs "
                "(EndToEndWrapper should expose proprio_size > 0)."
            )
        proprio_template = jnp.zeros((proprio_size,), dtype=jnp.float32)
        env_spec = {
            "obs_template": {
                "vision": jnp.zeros(vision_shape, dtype=jnp.float32),
                "imitation_target": jnp.zeros(
                    (obs_size_dict.get("imitation_target", 0),), dtype=jnp.float32
                ),
                "proprioception": proprio_template,
            },
            "action_size": action_size,
        }
    else:
        log.info("Loading vnl-playground env via registry: %s", env_name)
        from vnl_playground import registry as vp_registry  # type: ignore

        raw_env = vp_registry.load(env_name)
        env = _VnlPlaygroundEnvAdapter(raw_env)
        obs_size = int(env.observation_size)
        action_size = int(env.action_size)
        env_spec = {"obs_size": obs_size, "action_size": action_size}
        log.info("env_spec: obs_size=%d action_size=%d", obs_size, action_size)

    # 2. Networks + training state + optimizers.
    if use_vision:
        if use_from_scratch:
            from track_mjx.agent.dmpo.networks_vision_scratch import (
                make_dmpo_vision_scratch_networks,
            )
            nets = make_dmpo_vision_scratch_networks(
                task_obs_size=obs_size_dict["imitation_target"],
                proprio_size=obs_size_dict.get("proprioception", 0),
                action_size=action_size,
                vision_shape=vision_shape,
                cfg=cfg,
                cnn_feature_size=int(
                    hydra_cfg.network_config.get("vision_feature_size", 32)
                ),
                cnn_channels=tuple(
                    hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])
                ),
                mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
                shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared")
                == "shared",
            )
        else:
            from track_mjx.agent.dmpo.networks_vision import make_dmpo_vision_networks
            nets = make_dmpo_vision_networks(
                task_obs_size=obs_size_dict["imitation_target"],
                action_size=action_size,
                vision_shape=vision_shape,
                cfg=cfg,
                cnn_feature_size=int(
                    hydra_cfg.network_config.get("vision_feature_size", 32)
                ),
                cnn_channels=tuple(
                    hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])
                ),
                mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
                shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared")
                == "shared",
            )
    else:
        nets = make_dmpo_networks(obs_size, action_size, cfg)
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    # 2a. Checkpoint manager + restore-on-resume.
    # ``state`` is used as the template for orbax's StandardRestore, so the
    # template must already be a fully realized TrainingState.
    ckpt_dir = str(hydra_cfg.get("checkpoint_dir", "./checkpoints/dmpo"))
    ckpt_mgr = make_checkpointer(ckpt_dir)
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    if restored is not None:
        log.info(
            "Restored DMPO checkpoint from %s at training step %d",
            ckpt_dir,
            int(restored.steps),
        )
        state = restored

    # 3. Replay (flashbax, per-env time axis).
    if use_vision:
        proprio_size_for_replay = int(obs_size_dict.get("proprioception", 0))
        obs_template_for_replay = {
            "vision": jnp.zeros(vision_shape, dtype=jnp.float32),
            "imitation_target": jnp.zeros(
                (obs_size_dict.get("imitation_target", 0),), dtype=jnp.float32
            ),
            "proprioception": jnp.zeros(
                (proprio_size_for_replay,), dtype=jnp.float32
            ),
        }
        transition_template = {
            "observation": obs_template_for_replay,
            "action": jnp.zeros((action_size,), dtype=jnp.float32),
            "reward": jnp.zeros((), dtype=jnp.float32),
            "discount": jnp.zeros((), dtype=jnp.float32),
            "next_observation": obs_template_for_replay,
        }
    else:
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
    def jit_collect_rollout(policy_params, rng, init_state):
        return collect_rollout(
            env,
            policy_apply,
            policy_params,
            rng,
            num_envs=_num_envs_static,
            num_steps=_unroll_static,
            init_state=init_state,
        )

    # 5. SGD step jit. Static args: nets, optimizers, cfg are baked in.
    def _sgd_step_unbound(s, b):
        return sgd_step(s, b, nets, optimizers, cfg)

    jit_sgd_step = jax.jit(_sgd_step_unbound)

    # 6. Main loop.
    total_env_steps = 0
    last_log_step = 0
    last_eval_step = 0
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
    second_rollout = True  # see two-trace note below.
    env_state = None  # None on first iteration -> reset; persisted thereafter.
    while total_env_steps < cfg.num_timesteps:
        # 6a. Rollout (jitted). NOTE: jit_collect_rollout traces twice in this
        # design -- once for the init_state=None reset path (first call) and
        # once for the init_state=pytree resume path (second call onward). JAX
        # treats None and a pytree as distinct structural types, so this is
        # unavoidable without restructuring. Both traces are logged below so a
        # heavy MJX/vision compile at iter 2 is not mistaken for a hang.
        # Subsequent iterations reuse the resume-path trace.
        # env_state is threaded forward so the env's auto-reset wrapper is
        # the only thing that resets episodes; without this the buffer would
        # only see steps 1..unroll_length of post-reset episodes (Issue I in
        # the followup plan).
        rng, k_roll = jax.random.split(rng)
        if first_rollout:
            log.info("Compiling collect_rollout (first call, reset path)...")
            roll_compile_start = time.time()
            traj, env_state = jit_collect_rollout(
                state.policy_params, k_roll, env_state
            )
            jax.block_until_ready(traj["observation"])
            log.info(
                "collect_rollout (reset path) compiled + first call done in %.1fs",
                time.time() - roll_compile_start,
            )
            first_rollout = False
        elif second_rollout:
            log.info(
                "Compiling collect_rollout (second call, resume path)..."
            )
            roll_compile_start = time.time()
            traj, env_state = jit_collect_rollout(
                state.policy_params, k_roll, env_state
            )
            jax.block_until_ready(traj["observation"])
            log.info(
                "collect_rollout (resume path) compiled + second call done in %.1fs",
                time.time() - roll_compile_start,
            )
            second_rollout = False
        else:
            traj, env_state = jit_collect_rollout(
                state.policy_params, k_roll, env_state
            )
        # discount stored in the buffer is just the (1 - done) mask;
        # γ is applied by compute_categorical_target inside the learner.
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

        # 6b. SGD updates. Match Acme/Reverb samples_per_insert convention
        # (samples_drawn / inserts == cfg.samples_per_insert). See
        # compute_num_updates docstring.
        num_updates = compute_num_updates(cfg)
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

        # 6d. Eval + checkpoint. We run on the training env (cheap reset) with
        # a smaller batch and deterministic actions. The checkpoint save uses
        # the SGD step counter, NOT env steps, since ``state.steps`` is what
        # restore restores.
        if total_env_steps - last_eval_step >= cfg.eval_every_steps:
            rng, k_eval = jax.random.split(rng)
            # Vision configs (BinocularVisionRenderWrapper) have a fixed
            # nworld matching cfg.num_envs and require eval to use the same
            # batch. Non-vision has no such constraint, so cap at 64 to keep
            # eval cost bounded (was the prior default before commit aa2abd9).
            num_eval_envs = cfg.num_envs if use_vision else min(cfg.num_envs, 64)
            ep_return = _eval_episodic_return(
                env,
                nets.policy.apply,
                state.policy_params,
                k_eval,
                num_eval_envs=num_eval_envs,
                num_steps=1000,
            )
            log.info("eval/episode_return=%.3f", ep_return)
            if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "eval/episode_return": ep_return,
                        "env_steps": int(total_env_steps),
                    },
                    step=int(total_env_steps),
                )
            save_ckpt(ckpt_mgr, int(state.steps), state)
            last_eval_step = total_env_steps

    # Wait for any in-flight async checkpoint save to settle before exiting.
    ckpt_mgr.wait_until_finished()
    log.info("Training complete: %d env steps", total_env_steps)
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
