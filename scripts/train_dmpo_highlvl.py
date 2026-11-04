"""DMPO entry for high-level prior+decoder transfer on vnl-playground tasks.

The DMPO counterpart of the PPO high-level transfer entry: loads a frozen
SCAMPER prior + decoder, wraps the env in `PriorHighLevelWrapper` +
`BinocularVisionRenderWrapper`, builds DMPO bottleneck vision networks
(matched to PPO's `binocular_shared_vision_task_obs`), and runs the DMPO
training loop. The policy outputs a residual on the frozen prior's latent.

Usage:
    python scripts/train_dmpo_highlvl.py --config-name=<group>/<arm>
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import vnl_playground.naccdmax_patch  # noqa: F401  # naccdmax default
from omegaconf import DictConfig, OmegaConf
from vnl_playground import tasks
from vnl_playground.tasks.wrappers import PriorHighLevelWrapper

from track_mjx.agent.dmpo.checkpoint import make_checkpointer
from track_mjx.agent.dmpo.checkpoint import restore as restore_ckpt
from track_mjx.agent.dmpo.checkpoint import save as save_ckpt
from track_mjx.agent.dmpo.config import (
    DMPOConfig,
    realized_ratios,
    resolve_sgd_steps_per_rollout,
)
from track_mjx.agent.dmpo.learner import init_training_state, make_optimizers
from track_mjx.agent.dmpo.networks_vision_bottleneck import (
    make_dmpo_bottleneck_vision_networks,
)
from track_mjx.agent.dmpo.prior_utils import (
    load_prior_checkpoint,
    make_prior_inference_fn,
)
from track_mjx.agent.dmpo.prior_utils import (
    make_decoder_inference_fn as make_prior_decoder_fn,
)
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.schedules import env_steps_estimate, reward_anneal_lambda
from track_mjx.agent.dmpo.train import (
    _filter_dmpo_kwargs,
    _VnlPlaygroundEnvAdapter,
)
from track_mjx.agent.dmpo.train_dmpo_eval import (
    compute_batch_rollout_metrics,
    compute_rollout_metrics,
    compute_vision_sensitivity,
    render_eval_video,
    run_eval_rollout_envzero,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    detect_git_sha,
    make_run_id,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    load_wandb_state as load_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.train_dmpo_logging import (
    save_wandb_state as save_dmpo_wandb_state,
)
from track_mjx.agent.dmpo.training_loop import run as run_training_loop

log = logging.getLogger(__name__)

try:
    import wandb

    _WANDB_IMPORTED = True
except ImportError:
    _WANDB_IMPORTED = False
    wandb = None  # type: ignore


def _load_prior_and_decoder(hydra_cfg: DictConfig):
    """Load the frozen SCAMPER prior + decoder.

    Returns (prior_fn, decoder_fn, latent_size, prior_cfg).
    """
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
    log.info("Prior loaded. intention_size=%d", latent_size)
    prior_fn = make_prior_inference_fn(prior_params, normalizer_params, prior_cfg)
    decoder_fn = make_prior_decoder_fn(decoder_params, normalizer_params, prior_cfg)
    return prior_fn, decoder_fn, latent_size, prior_cfg


def _build_env(hydra_cfg: DictConfig, prior_fn, decoder_fn, latent_size):
    """Load env, wrap in PriorHighLevelWrapper + BinocularVisionRenderWrapper.

    Returns (env_adapter, base_env, mj_model, mjx_model, n_eye_actuators,
    vision_shape).
    """
    env_name = str(hydra_cfg.env_name)
    env_args = (
        OmegaConf.to_container(hydra_cfg.get("env_config", {}), resolve=True) or {}
    )
    # Filter to only keys that are valid in the env's default config. This drops
    # YAML-only metadata (env_name, task_name, walker_name, render_camera_name,
    # flatten_obs) that the env's ConfigDict doesn't accept.
    valid_keys = set(tasks.get_default_config(env_name).keys())
    env_args = {k: v for k, v in env_args.items() if k in valid_keys}
    base_env = tasks.load(env_name, flatten_obs=False, config_overrides=env_args)
    raw_env = base_env.env if hasattr(base_env, "env") else base_env
    mj_model = getattr(raw_env, "mj_model", None)
    mjx_model = getattr(raw_env, "mjx_model", None)
    if mj_model is None or mjx_model is None:
        raise RuntimeError(
            "Could not find mj_model/mjx_model on base env for vision rendering"
        )
    n_eye_actuators = int(getattr(raw_env, "n_eye_actuators", 0))
    if n_eye_actuators > 0:
        log.info("Actuable eyes: %d eye actuators bypass decoder", n_eye_actuators)

    base_env = PriorHighLevelWrapper(
        base_env,
        prior_fn,
        decoder_fn,
        latent_size,
        highlvl_obs_key=str(hydra_cfg.transfer.get("highlvl_obs_key", "task_obs")),
        lowlvl_obs_key=str(hydra_cfg.transfer.get("decoder_obs_key", "proprioception")),
        pass_vision=True,
        pass_task_obs=True,
        deterministic_prior=bool(hydra_cfg.transfer.get("deterministic_prior", True)),
        noise_logvar=float(hydra_cfg.transfer.get("noise_logvar", -2.0)),
        n_eye_actuators=n_eye_actuators,
    )

    # Wrap for training (brax stack plus the truncation stash), then add
    # binocular vision rendering.
    from vnl_playground.tasks.rodent.vision_jax import (
        BinocularVisionRenderWrapper,
    )
    from vnl_playground.tasks.wrappers_truncation import wrap_for_dmpo_training

    episode_length = int(
        hydra_cfg.env_config.get(
            "episode_length", hydra_cfg.train_config.get("unroll_length", 1000)
        )
    )
    action_repeat = int(hydra_cfg.env_config.get("action_repeat", 1))
    base_env = wrap_for_dmpo_training(
        base_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        full_reset=False,
    )
    base_env = BinocularVisionRenderWrapper(
        base_env,
        mj_model=mj_model,
        mjx_model=mjx_model,
        width=int(hydra_cfg.env_config.get("vision_width", 32)),
        height=int(hydra_cfg.env_config.get("vision_height", 32)),
        grayscale=bool(hydra_cfg.env_config.get("grayscale", True)),
        left_camera_name=str(
            hydra_cfg.env_config.get("left_camera_name", "eye_left-rodent")
        ),
        right_camera_name=str(
            hydra_cfg.env_config.get("right_camera_name", "eye_right-rodent")
        ),
        render_depth=False,
        use_textures=bool(hydra_cfg.env_config.get("use_textures", False)),
        use_shadows=bool(hydra_cfg.env_config.get("use_shadows", False)),
        eye_dropout_rate=float(hydra_cfg.env_config.get("eye_dropout_rate", 0.0)),
        eval_eye_mode=str(hydra_cfg.env_config.get("eval_eye_mode", "binocular")),
    )

    env_adapter = _VnlPlaygroundEnvAdapter(base_env, pre_batched=True)
    vision_shape = tuple(
        getattr(
            base_env,
            "vision_shape",
            getattr(base_env.env, "vision_shape", (32, 32, 2)),
        )
    )
    return env_adapter, base_env, mj_model, mjx_model, n_eye_actuators, vision_shape


@hydra.main(
    config_path="../track_mjx/config/dmpo",
    config_name="run_gap/velocity_only",
    version_base=None,
)
def main(hydra_cfg: DictConfig):
    """DMPO entry — prior + decoder transfer (binocular vision)."""
    raw_train_cfg = OmegaConf.to_container(hydra_cfg.train_config, resolve=True)
    cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    iters_per_chunk = int(hydra_cfg.train_config.get("iters_per_chunk", 32))
    cfg_dict = OmegaConf.to_container(hydra_cfg, resolve=True)
    seed = int(hydra_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)

    config_name = str(
        hydra_cfg.get("logging_config", {}).get(
            "exp_name", hydra_cfg.get("env_name", "dmpo-highlvl")
        )
    )
    git_sha = detect_git_sha(Path(__file__).resolve().parents[1])
    run_id = make_run_id(config_name, seed, git_sha)
    log.info("wandb run_id=%s", run_id)

    ckpt_dir = str(hydra_cfg.get("checkpoint_dir", "./checkpoints/dmpo_highlvl"))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    existing = load_dmpo_wandb_state(ckpt_dir)
    if _WANDB_IMPORTED:
        try:
            wandb.init(
                project=str(
                    hydra_cfg.get("logging_config", {}).get(
                        "project_name", "dmpo-rodent"
                    )
                ),
                config=cfg_dict,
                mode=os.environ.get("WANDB_MODE", "online"),
                id=existing["wandb_run_id"] if existing else run_id,
                name=existing["wandb_run_id"] if existing else run_id,
                resume="must" if existing else "allow",
                group=str(
                    hydra_cfg.get("logging_config", {}).get(
                        "group_name", hydra_cfg.get("env_name", "dmpo-highlvl")
                    )
                ),
                notes=str(hydra_cfg.get("logging_config", {}).get("notes", "")),
                reinit=True,
            )
            save_dmpo_wandb_state(
                ckpt_dir,
                run_id if not existing else existing["wandb_run_id"],
            )
        except Exception as e:
            log.warning("wandb.init failed (%s); continuing without wandb.", e)

    # --- 1. Load prior + decoder ---
    prior_fn, decoder_fn, latent_size, _prior_cfg = _load_prior_and_decoder(hydra_cfg)

    # --- 2. Build env (PriorHighLevelWrapper + BinocularVisionRenderWrapper) ---
    env, base_env, mj_model, mjx_model, n_eye_actuators, vision_shape = _build_env(
        hydra_cfg,
        prior_fn,
        decoder_fn,
        latent_size,
    )
    obs_size_dict = dict(env.observation_size)
    action_size = int(env.action_size)
    log.info(
        "env_spec: action_size=%d (latent=%d + eye=%d), task_obs_size=%d, vision_shape=%s",
        action_size,
        latent_size,
        n_eye_actuators,
        obs_size_dict.get("imitation_target", 0),
        vision_shape,
    )

    # --- 3. Build bottleneck DMPO networks ---
    nets = make_dmpo_bottleneck_vision_networks(
        task_obs_size=obs_size_dict.get("imitation_target", 0),
        action_size=action_size,
        vision_shape=tuple(vision_shape),
        cfg=cfg,
        vision_latent_size=int(hydra_cfg.network_config.get("vision_latent_size", 16)),
        cnn_feature_size=int(hydra_cfg.network_config.get("vision_feature_size", 32)),
        cnn_channels=tuple(
            hydra_cfg.network_config.get("vision_channels", [4, 8, 16, 32])
        ),
        mono_channels=1 if hydra_cfg.env_config.get("grayscale", True) else 3,
        shared_weights=hydra_cfg.network_config.get("binocular_mode", "shared")
        == "shared",
        fusion_hidden_layer_sizes=tuple(
            hydra_cfg.network_config.get("fusion_hidden_layer_sizes", [256, 256, 256])
        ),
        decoder_hidden_layer_sizes=tuple(
            hydra_cfg.network_config.get("decoder_hidden_layer_sizes", [512, 512, 512])
        ),
        value_hidden_layer_sizes=tuple(
            hydra_cfg.network_config.get(
                "value_hidden_layer_sizes", [512, 512, 512, 512]
            )
        ),
    )

    # --- 4. Training state, optimizers, replay ---
    env_spec = {
        "obs_template": {
            "vision": jnp.zeros(tuple(vision_shape), dtype=jnp.float32),
            "imitation_target": jnp.zeros(
                (obs_size_dict.get("imitation_target", 0),), dtype=jnp.float32
            ),
            "proprioception": jnp.zeros(
                (obs_size_dict.get("proprioception", 0),), dtype=jnp.float32
            ),
        },
        "action_size": action_size,
    }
    transition_template = {
        "observation": env_spec["obs_template"],
        "action": jnp.zeros((action_size,), dtype=jnp.float32),
        "reward": jnp.zeros((), dtype=jnp.float32),
        "discount": jnp.zeros((), dtype=jnp.float32),
        "next_observation": env_spec["obs_template"],
    }
    # Post-step time-out flag, stored when the learner bootstraps through time
    # limits (the rollout writes it under the same switch).
    if cfg.bootstrap_on_truncation:
        transition_template["truncation"] = jnp.zeros((), dtype=jnp.float32)

    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, cfg)
    optimizers = make_optimizers(cfg)

    ckpt_mgr = make_checkpointer(ckpt_dir)
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    # Checkpoints are saved at `step=total_env_steps`, so the manager's latest
    # step IS the env-step count to resume the training-loop counter from.
    # Without it a resumed run restarts counting at 0 and trains
    # `num_timesteps` MORE steps under colliding checkpoint names.
    start_env_steps = 0
    if restored is not None:
        start_env_steps = int(ckpt_mgr.latest_step() or 0)
        log.info(
            "Restored DMPO checkpoint: sgd_step=%d env_steps=%d (resuming counter)",
            int(restored.steps),
            start_env_steps,
        )
        state = restored

    rb = make_replay(
        max_size=max(cfg.sequence_length + 1, cfg.max_replay_size // cfg.num_envs),
        min_size=max(cfg.sequence_length + 1, cfg.min_replay_size // cfg.num_envs),
        sequence_length=cfg.sequence_length,
        sample_batch_size=cfg.batch_size,
        add_batch_size=cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    K = resolve_sgd_steps_per_rollout(cfg)
    log.info(
        "DMPO highlvl: K=%d SGD updates per rollout | %s",
        K,
        " ".join(f"{k}={v:.4g}" for k, v in realized_ratios(cfg, K).items()),
    )
    # Only warn when K was DERIVED. A pinned `sgd_steps_per_rollout` outside the
    # 30..80 band is the whole point of arms like a Ray-throughput-parity one
    # (K=324 at batch 1024), and a "check samples_per_insert" warning there
    # would be actively misleading -- that knob is unread when K is pinned.
    if cfg.sgd_steps_per_rollout is None and (K > 80 or K < 30):
        log.warning(
            "K=%d outside the validated 30..80 band. Check num_envs / batch_size / "
            "samples_per_insert / unroll_length consistency.",
            K,
        )

    # --- 5. Callbacks: wandb, checkpointing, eval ---
    def wandb_log_cb(payload: dict, env_steps: int) -> None:
        if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
            wandb.log(payload, step=int(env_steps))

    def ckpt_save_cb(state: object, env_steps: int) -> None:
        save_ckpt(ckpt_mgr, int(env_steps), state, config=cfg_dict)

    erc_enable = bool(hydra_cfg.get("eval_render_config", {}).get("enable", True))

    def eval_cb(state: object, env_steps: int, k_eval: jax.Array) -> None:
        if not erc_enable:
            return
        try:
            erc_raw = hydra_cfg.get("eval_render_config", {})
            erc = OmegaConf.to_container(erc_raw, resolve=True) if erc_raw else {}
            eval_episode_length = int(erc.get("episode_length", 1000))
            # [2] is the UN-remixed all-env batch metric dict, [3] the raw allenv
            # arrays. The batch metrics are recomputed from allenv with the
            # reward-anneal lambda so the reported reward is the one the replay
            # buffer stored (see remix_eval_reward).
            rollout, term_events, _batch_rm_unmixed, allenv = run_eval_rollout_envzero(
                env=base_env,
                policy_apply=nets.policy.apply,
                params=state.policy_params,
                rng=k_eval,
                episode_length=eval_episode_length,
                num_envs=cfg.num_envs,
            )
            # lambda from state.steps via the SAME expression the fused training
            # step uses (`reward_anneal_lambda(env_steps_estimate(state.steps,
            # cfg, K), cfg)` in train_dmpo_step.make_fused_train_step) -- the
            # host env_steps int is a different quantity and would drift from
            # the buffer.
            remix_key = getattr(cfg, "reward_anneal_sparse_key", None)
            remix_lambda = (
                float(
                    reward_anneal_lambda(env_steps_estimate(state.steps, cfg, K), cfg)
                )
                if remix_key is not None
                else None
            )
            if (
                rollout
                and _WANDB_IMPORTED
                and wandb is not None
                and wandb.run is not None
            ):
                rollout_metrics = compute_rollout_metrics(
                    rollout, remix_key, remix_lambda
                )
                rollout_metrics.update(
                    compute_batch_rollout_metrics(allenv, remix_key, remix_lambda)
                )
                wandb.log(
                    {f"eval/{k}": v for k, v in rollout_metrics.items()}
                    | {"env_steps": int(env_steps)},
                    step=int(env_steps),
                    commit=False,
                )
            if rollout and "vision" in rollout[0].obs:
                mid = len(rollout) // 2
                sens = compute_vision_sensitivity(
                    nets.policy.apply,
                    state.policy_params,
                    rollout[mid].obs,
                    k_eval,
                )
                if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                    wandb.log(
                        {"eval/vision_sensitivity": sens, "env_steps": int(env_steps)},
                        step=int(env_steps),
                        commit=False,
                    )
            if mj_model is not None and rollout:
                video_path = Path(ckpt_dir) / f"eval_{int(env_steps)}.mp4"
                hud_cfg = erc.get("hud") if isinstance(erc, dict) else None
                rew_cfg_raw = hydra_cfg.get("env_config", {}).get("reward_terms", None)
                rew_cfg = (
                    OmegaConf.to_container(rew_cfg_raw, resolve=True)
                    if rew_cfg_raw is not None
                    else None
                )
                render_eval_video(
                    rollout,
                    mj_model,
                    video_path,
                    fps=int(erc.get("fps", 50)),
                    height=int(erc.get("height", 480)),
                    width=int(erc.get("width", 640)),
                    camera=str(erc.get("camera", "close_profile-rodent")),
                    hud_config=hud_cfg,
                    reward_config=rew_cfg,
                    termination_events=term_events,
                    reward_remix=(
                        {"sparse_key": remix_key, "lambda": remix_lambda}
                        if remix_key is not None
                        else None
                    ),
                )
                if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
                    wandb.log(
                        {"videos/eval": wandb.Video(str(video_path), format="mp4")},
                        step=int(env_steps),
                    )
        except Exception as e:
            log.warning("Eval render failed: %s", e, exc_info=True)
        finally:
            import gc

            gc.collect()

    # --- 6. Run training ---
    rng, k_run = jax.random.split(rng)
    state, env_state, rb_state, last_metrics = run_training_loop(
        env=env,
        nets=nets,
        optimizers=optimizers,
        rb=rb,
        cfg=cfg,
        K=K,
        iters_per_chunk=iters_per_chunk,
        rng=k_run,
        state=state,
        env_state=None,
        rb_state=rb_state,
        eval_callback=eval_cb,
        wandb_log_callback=wandb_log_cb,
        ckpt_mgr=ckpt_mgr,
        ckpt_save_callback=ckpt_save_cb,
        cfg_dict=cfg_dict,
        start_env_steps=start_env_steps,
    )

    ckpt_mgr.wait_until_finished()
    log.info(
        "Training complete: final policy_loss=%.4g critic_loss=%.4g",
        float(last_metrics.get("policy_loss", 0.0)),
        float(last_metrics.get("critic_loss", 0.0)),
    )
    if _WANDB_IMPORTED and wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
