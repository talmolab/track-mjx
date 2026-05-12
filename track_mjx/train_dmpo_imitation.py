"""Track-mjx DMPO training entry for rodent imitation with intention
encoder-decoder policy (Approach A: deterministic latent).

Mirrors track_mjx/train.py's PPO orchestration: dict-obs running-statistics
normalization, train/test split, domain randomization, eval rollout video
via wandb_logging.rollout_logging_fn, and run-state restore. Uses DMPO
algorithm via track_mjx.agent.dmpo.training_loop.run.

Plan: ClaudeCode_PromptHistory/2026-05-05-2-dmpo-imitation-intention.
Spec: ditto.

Usage:
    cd /home/talmolab/Desktop/SalkResearch/track-mjx
    source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
    python -m track_mjx.train_dmpo_imitation \\
        --config-name=rodent-dmpo-imitation-intention
"""
from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf
from vnl_playground import registry

from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker
from track_mjx.agent.dmpo.checkpoint import (
    make_checkpointer,
    restore as restore_ckpt,
    save as save_ckpt,
)
from track_mjx.agent.dmpo.config import DMPOConfig
from track_mjx.agent.dmpo.learner import init_training_state, make_optimizers
from track_mjx.agent.dmpo.networks_intention import (
    IntentionDMPOPolicy,
    make_dmpo_intention_networks,
)
from track_mjx.agent.dmpo.replay import make_replay
from track_mjx.agent.dmpo.train import (
    _VnlPlaygroundEnvAdapter,
    _filter_dmpo_kwargs,
)
from track_mjx.agent.dmpo.training_loop import run as run_training_loop
from track_mjx.agent.observation_utils import normalize_dict_obs
from track_mjx.config import utils as cfg_utils

log = logging.getLogger(__name__)


def _setup_environment() -> None:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _make_dmpo_logging_inference_fn(policy_module, normalizer_params):
    """Adapter: wrap DMPO policy.apply as PPO's (params, obs, key) -> (action, extras).

    PPO's wandb_logging.rollout_logging_fn expects extras["latent_mean"] and
    extras["latent_logvar"]. Under Approach A the decoder uses z=mean and
    logvar is uninformative but exposed for histogram plotting.
    """
    def policy_fn(params: Any, obs: dict, key: jax.Array):
        norm_obs = normalize_dict_obs(obs, normalizer_params)
        # Action: deterministic mode for eval rollouts.
        dist = policy_module.apply(params, norm_obs)
        action = dist.mode()
        # Latents: side-channel via the dedicated `encode` method.
        latent_mean, latent_logvar = policy_module.apply(
            params, norm_obs, method=IntentionDMPOPolicy.encode
        )
        extras = {
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,
        }
        return action, extras
    return policy_fn


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="rodent-dmpo-imitation-intention",
)
def main(hydra_cfg: DictConfig) -> None:
    """DMPO training entry — rodent imitation with intention encoder-decoder."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        log.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        log.info("Not using GPUs")

    # ---- 1. Resolve hydra config (paths, ConfigDict for env) ----
    hydra_cfg, cfg_dict, env_cfg_ml = cfg_utils.prepare_config(hydra_cfg)
    env_name = str(hydra_cfg.env_config.env_name)

    # ---- 2. DMPOConfig from train_config block ----
    raw_train_cfg = OmegaConf.to_container(
        hydra_cfg.train_setup.train_config, resolve=True
    )
    dmpo_cfg = DMPOConfig(**_filter_dmpo_kwargs(raw_train_cfg))
    iters_per_chunk = int(raw_train_cfg.get("iters_per_chunk", 8))
    seed = int(raw_train_cfg.get("seed", 0))
    rng = jax.random.PRNGKey(seed)
    log.info("iters_per_chunk=%d", iters_per_chunk)

    # ---- 3. Run-state restore + checkpoint manager ----
    run_id, checkpoint_path, existing_run_state = (
        checkpointing.load_from_run_state(hydra_cfg)
    )
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    mgr_options = ocp.CheckpointManagerOptions(
        create=True, step_prefix="DMPONetwork"
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # ---- 4. Reference clips + train/test split ----
    log.info(f"Loading reference clips: {hydra_cfg.env_config.reference_data_path}")
    reference_clips = registry.load_reference_clips(
        env_name,
        data_path=hydra_cfg.env_config.reference_data_path,
        n_frames_per_clip=hydra_cfg.env_config.clip_length,
        keep_clips_idx=hydra_cfg.env_config.get("keep_clips_idx", None),
    )
    key_split, _ = jax.random.split(jax.random.PRNGKey(seed))
    train_clips, test_clips = reference_clips.split(
        train_ratio=hydra_cfg.train_setup.train_subset_ratio,
        seed=key_split,
    )
    train_env = registry.load(
        env_name, config=env_cfg_ml, clips=train_clips, flatten_obs=False
    )
    test_env = registry.load(
        env_name, config=env_cfg_ml, clips=test_clips, flatten_obs=False
    )

    # Compute episode length (mirrors train.py).
    clip_length = hydra_cfg.env_config.clip_length
    if clip_length is None:
        clip_length = reference_clips.qpos.shape[1]
    steps_per_frame = (1 / hydra_cfg.env_config.mocap_hz) / hydra_cfg.env_config.ctrl_dt
    episode_length = int(
        (clip_length
         - hydra_cfg.env_config.start_frame_range[-1]
         - hydra_cfg.env_config.reference_length)
        * steps_per_frame
    )
    log.info("episode_length=%d", episode_length)

    # ---- 5. Wrap env for brax training (vmap + auto-reset + DR) ----
    # domain_randomization_maker returns (model, rng) -> (model, in_axes), but
    # wrap_for_brax_training's BraxDomainRandomizationVmapWrapper calls
    # randomization_fn(model) with a single argument. Bind a fresh rng split
    # so the function signature matches (mirrors PPO's ppo.py:548 pattern).
    v_randomization_fn = None
    if hydra_cfg.env_config.domain_randomization.use_domain_randomization:
        rng, dr_rng = jax.random.split(rng)
        dr_base_fn = domain_randomization_maker(
            floor_friction=hydra_cfg.env_config.domain_randomization.floor_friction,
            static_friction_scale=hydra_cfg.env_config.domain_randomization.static_friction_scale,
            armature_scale=hydra_cfg.env_config.domain_randomization.armature_scale,
            com_jitter=hydra_cfg.env_config.domain_randomization.com_jitter,
            link_mass_scale=hydra_cfg.env_config.domain_randomization.link_mass_scale,
            torso_mass_jitter=hydra_cfg.env_config.domain_randomization.torso_mass_jitter,
            qpos0_jitter=hydra_cfg.env_config.domain_randomization.qpos0_jitter,
        )
        dr_rng_split = jax.random.split(
            dr_rng, dmpo_cfg.num_envs
        )
        v_randomization_fn = functools.partial(dr_base_fn, rng=dr_rng_split)
    wrapped_train_env = playground_wrappers.wrap_for_brax_training(
        train_env,
        episode_length=episode_length,
        action_repeat=int(raw_train_cfg.get("action_repeat", 1)),
        full_reset=False,
        randomization_fn=v_randomization_fn,
    )
    dmpo_train_env = _VnlPlaygroundEnvAdapter(wrapped_train_env, pre_batched=True)

    # ---- 6. Networks ----
    # observation_size on the raw env (with flatten_obs=False) may be a scalar
    # or a dict depending on the registry/env version. Read from the adapter
    # (which proxies wrapped_train_env) and cast to a plain dict.
    raw_obs_size = dmpo_train_env.observation_size
    if isinstance(raw_obs_size, dict):
        obs_sizes = dict(raw_obs_size)
    else:
        # Fallback: derive sizes from a sample reset observation
        import jax.random as jr
        _sample_state = jax.jit(train_env.reset)(jr.PRNGKey(0))
        from track_mjx.agent.observation_utils import get_obs_sizes
        obs_sizes = get_obs_sizes(_sample_state.obs)
    action_size = int(train_env.action_size)
    log.info(
        "obs_sizes=%s action_size=%d", obs_sizes, action_size,
    )
    nets = make_dmpo_intention_networks(
        obs_sizes=obs_sizes,
        action_size=action_size,
        cfg=dmpo_cfg,
        network_cfg=OmegaConf.to_container(hydra_cfg.network_config, resolve=True),
    )

    # ---- 7. TrainingState (with normalizer) ----
    env_spec = {
        "obs_template": {
            "imitation_target": jnp.zeros(
                (obs_sizes["imitation_target"],), jnp.float32
            ),
            "proprioception": jnp.zeros(
                (obs_sizes["proprioception"],), jnp.float32
            ),
        },
        "action_size": action_size,
    }
    rng, k_state = jax.random.split(rng)
    state = init_training_state(k_state, nets, env_spec, dmpo_cfg)
    optimizers = make_optimizers(dmpo_cfg)

    # ---- 8. Restore from checkpoint if present ----
    restored = restore_ckpt(ckpt_mgr, state_template=state)
    if restored is not None:
        log.info(
            "Restored DMPO checkpoint at training step %d", int(restored.steps)
        )
        state = restored

    # ---- 9. Replay (flashbax, dict obs) ----
    transition_template = {
        "observation": env_spec["obs_template"],
        "action": jnp.zeros((action_size,), jnp.float32),
        "reward": jnp.zeros((), jnp.float32),
        "discount": jnp.zeros((), jnp.float32),
        "next_observation": env_spec["obs_template"],
    }
    rb = make_replay(
        max_size=max(
            dmpo_cfg.sequence_length + 1,
            dmpo_cfg.max_replay_size // dmpo_cfg.num_envs,
        ),
        min_size=max(
            dmpo_cfg.sequence_length + 1,
            dmpo_cfg.min_replay_size // dmpo_cfg.num_envs,
        ),
        sequence_length=dmpo_cfg.sequence_length,
        sample_batch_size=dmpo_cfg.batch_size,
        add_batch_size=dmpo_cfg.num_envs,
        period=1,
    )
    rb_state = rb.init(transition_template)

    # ---- 10. K (SGD updates per fused_step) ----
    K = max(
        1,
        int(
            dmpo_cfg.unroll_length * dmpo_cfg.num_envs
            / (dmpo_cfg.batch_size * dmpo_cfg.samples_per_insert)
        ),
    )
    log.info("DMPO imitation: K=%d SGD updates per rollout", K)

    # ---- 11. Wandb init (PPO convention) ----
    wandb_logging.initialize_wandb_logging(
        logging_cfg=hydra_cfg.logging_config,
        cfg=hydra_cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )
    if existing_run_state is None and wandb.run is not None:
        checkpointing.save_run_state(
            cfg=hydra_cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    # ---- 12. Build rollout env for eval render (start_frame=0) ----
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = registry.load(
        env_name, config=rollout_cfg, clips=None, flatten_obs=False
    )
    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    # ---- 13. Callbacks ----
    def wandb_log_cb(payload: dict, env_steps: int) -> None:
        # PPO's wandb_progress expects num_steps as first arg + metrics dict;
        # strip env_steps from the payload to match.
        metrics = {k: v for k, v in payload.items() if k != "env_steps"}
        wandb_logging.wandb_progress(env_steps, metrics)

    def ckpt_save_cb(state, env_steps: int) -> None:
        save_ckpt(ckpt_mgr, int(env_steps), state, config=cfg_dict)

    def eval_render_cb(state, env_steps: int, k_eval: jax.Array) -> None:
        try:
            inference_fn = jax.jit(
                _make_dmpo_logging_inference_fn(nets.policy, state.normalizer_params)
            )
            wandb_logging.rollout_logging_fn(
                env=rollout_env,
                jit_reset=jit_reset,
                jit_step=jit_step,
                cfg=hydra_cfg,
                model_path=checkpoint_path,
                current_step=int(env_steps),
                jit_logging_inference_fn=inference_fn,
                params=state.policy_params,
                policy_params_fn_key=k_eval,
                render_video=True,
            )
        except Exception as e:
            log.warning("Eval render failed: %s", e, exc_info=True)
        finally:
            import gc
            gc.collect()

    # ---- 14. Run training ----
    rng, k_run = jax.random.split(rng)
    state, env_state, rb_state, last_metrics = run_training_loop(
        env=dmpo_train_env,
        nets=nets,
        optimizers=optimizers,
        rb=rb,
        cfg=dmpo_cfg,
        K=K,
        iters_per_chunk=iters_per_chunk,
        rng=k_run,
        state=state,
        env_state=None,
        rb_state=rb_state,
        eval_callback=eval_render_cb,
        wandb_log_callback=wandb_log_cb,
        ckpt_mgr=ckpt_mgr,
        ckpt_save_callback=ckpt_save_cb,
        cfg_dict=cfg_dict,
    )

    ckpt_mgr.wait_until_finished()
    log.info(
        "Training complete: final policy_loss=%.4g critic_loss=%.4g",
        float(last_metrics.get("policy_loss", 0.0)),
        float(last_metrics.get("critic_loss", 0.0)),
    )

    try:
        checkpointing.cleanup_run_state(hydra_cfg)
        log.info("Cleaned up run state.")
    except Exception as e:
        log.warning("Failed to cleanup run state: %s", e)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
