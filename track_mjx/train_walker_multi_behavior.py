"""Step 1 training: Multi-behavior MLP + PPO for PlanarWalker.

Trains a conditional policy pi(a | s, mode_onehot) with 4 behavior modes.
Uses Brax PPO with standard MLP networks (no encoder-decoder).

Usage:
    cd /home/talmolab/Desktop/SalkResearch/track-mjx
    source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
    python track_mjx/train_walker_multi_behavior.py
    # or with overrides:
    python track_mjx/train_walker_multi_behavior.py env_config.fixed_mode=0
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import gc
import logging
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import orbax.checkpoint as ocp
from brax.training.agents.ppo.train import train as brax_ppo_train
from brax.training.agents.ppo import networks as brax_ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf

from vnl_playground.tasks.walker import consts
from vnl_playground.tasks.walker.multi_behavior import (
    MultiBehaviorWalker,
    default_config as walker_default_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reverse lookup: mode index -> name
MODE_NAMES = {v: k for k, v in consts.BEHAVIOR_MODES.items()}


def _run_eval_rollout(jit_reset, jit_step, inference_fn, params, episode_length, rng):
    """Run a single-env evaluation rollout with termination/auto-reset.

    Returns:
        rollout: list of states (may span multiple episodes)
        termination_events: list of (frame_index, reason_string)
    """
    _, reset_rng, act_rng = jax.random.split(rng, 3)
    state = jit_reset(reset_rng)
    rollout = [state]
    termination_events = []

    for _ in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        action, _ = inference_fn(params, state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)

        if float(state.done) > 0.5:
            termination_events.append((len(rollout) - 1, "done"))
            _, reset_rng = jax.random.split(act_rng)
            state = jit_reset(reset_rng)
            rollout.append(state)

    return rollout, termination_events


def render_walker_video(
    rollout,
    mj_model,
    mj_data,
    renderer,
    video_path,
    fps=50,
    termination_events=None,
    termination_fade_seconds=0.5,
):
    """Render a walker rollout to MP4 with behavior mode text overlay."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    try:
        camera.trackbodyid = mj_model.body("torso").id
    except Exception:
        camera.trackbodyid = 1
    camera.distance = 3.0
    camera.azimuth = 90
    camera.elevation = -20
    camera.lookat[:] = [0, 0, 0.8]

    scene_option = mujoco.MjvOption()

    termination_dict = {}
    if termination_events:
        termination_dict = {idx: reason for idx, reason in termination_events}

    with imageio.get_writer(video_path, fps=fps) as writer:
        for i, state in enumerate(rollout):
            mj_data.qpos = np.array(state.data.qpos)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera, scene_option=scene_option)
            frame = renderer.render().copy()

            # Overlay behavior mode label
            mode_idx = int(state.metrics.get("mode_idx", -1))
            mode_name = MODE_NAMES.get(mode_idx, f"mode_{mode_idx}")
            label = f"Mode: {mode_name}"
            # Background rectangle for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (8, 8), (16 + tw, 16 + th + 8), (0, 0, 0), -1)
            cv2.putText(
                frame, label, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

            # Termination overlay with fade
            if i in termination_dict:
                reason = termination_dict[i]
                term_label = f"Terminated: {reason}"
                cv2.putText(
                    frame, term_label, (10, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )
                writer.append_data(frame)
                n_fade = int(fps * termination_fade_seconds)
                for t in range(n_fade):
                    fade = 1 / (1 + np.exp(10 * (t / n_fade - 0.5)))
                    writer.append_data((frame * fade).astype(np.uint8))
            else:
                writer.append_data(frame)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="walker-multi-behavior",
)
def main(cfg: DictConfig) -> None:
    """Train multi-behavior walker with Brax PPO."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    try:
        n_devices = jax.device_count(backend="gpu")
        logger.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logger.info("No GPU, using CPU")

    # Create environment config
    env_cfg = walker_default_config()
    for key in [
        "sim_dt", "ctrl_dt", "episode_length", "mujoco_impl",
        "nconmax", "njmax", "mode_switch_prob", "fixed_mode",
    ]:
        if key in cfg.env_config:
            val = cfg.env_config[key]
            if val is not None or key == "fixed_mode":
                env_cfg[key] = val

    # Create train and eval environments
    env = MultiBehaviorWalker(config=env_cfg)
    eval_env = MultiBehaviorWalker(config=env_cfg)

    # Generate run ID and checkpoint path (absolute for Orbax)
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    checkpoint_path = Path(cfg.checkpoint.save_dir).resolve() / run_id
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Setup wandb
    if cfg.logging_config.log_to_wandb:
        import wandb
        wandb.init(
            project=cfg.logging_config.project_name,
            group=cfg.logging_config.group_name,
            name=f"{cfg.logging_config.exp_name}_{run_id}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def progress_fn(num_steps, metrics):
        logger.info(
            f"Step {num_steps}: "
            f"reward={metrics.get('eval/episode_reward', 0):.3f}"
        )
        if cfg.logging_config.log_to_wandb:
            import wandb
            wandb.log({"step": num_steps, **metrics})

    # Setup checkpoint manager
    ckpt_mgr = ocp.CheckpointManager(
        str(checkpoint_path),
        options=ocp.CheckpointManagerOptions(create=True),
    )

    # Network factory with configurable hidden sizes
    network_factory = functools.partial(
        brax_ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(
            cfg.network_config.policy_hidden_layer_sizes
        ),
        value_hidden_layer_sizes=tuple(
            cfg.network_config.value_hidden_layer_sizes
        ),
    )

    # --- Eval video rendering setup ---
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    renderer_obj = mujoco.Renderer(
        mj_model,
        height=cfg.get("render_config", {}).get("render_height", 480),
        width=cfg.get("render_config", {}).get("render_width", 640),
    )
    render_fps = cfg.get("render_config", {}).get("render_fps", 50)
    episode_length = cfg.env_config.episode_length

    # Build inference fn for eval rollouts (uses unwrapped eval_env)
    normalize = lambda x, y: x
    if cfg.train_config.normalize_observations:
        normalize = running_statistics.normalize

    ppo_network = network_factory(
        eval_env.observation_size,
        eval_env.action_size,
        preprocess_observations_fn=normalize,
    )

    def _make_logging_inference_fn(ppo_networks):
        def make_logging_policy(deterministic=False):
            policy_network = ppo_networks.policy_network
            parametric_action_distribution = ppo_networks.parametric_action_distribution

            def logging_policy(params, observations, key_sample):
                param_subset = (params[0], params[1])
                logits = policy_network.apply(*param_subset, observations)
                if deterministic:
                    return jp.array(parametric_action_distribution.mode(logits)), {}
                raw = parametric_action_distribution.sample_no_postprocessing(
                    logits, key_sample
                )
                log_prob = parametric_action_distribution.log_prob(logits, raw)
                post = parametric_action_distribution.postprocess(raw)
                return jp.array(post), {"log_prob": log_prob, "raw_action": raw}

            return logging_policy
        return make_logging_policy

    jit_logging_inference_fn = jax.jit(
        _make_logging_inference_fn(ppo_network)(deterministic=True)
    )
    jit_eval_reset = jax.jit(eval_env.reset)
    jit_eval_step = jax.jit(eval_env.step)

    orbax_checkpointer = ocp.PyTreeCheckpointer()

    def policy_params_fn(current_step, make_policy, params):
        """Render eval video and save checkpoint at each eval step."""
        logger.info(f"Eval callback at step {current_step}")

        # Run eval rollout
        eval_rng = jax.random.PRNGKey(current_step)
        rollout, termination_events = _run_eval_rollout(
            jit_eval_reset, jit_eval_step, jit_logging_inference_fn,
            params, episode_length, eval_rng,
        )

        # Log per-step reward metrics
        reward_keys = [k for k in rollout[0].metrics.keys() if k.startswith("reward/")]
        if cfg.logging_config.log_to_wandb:
            import wandb
            for metric_name in reward_keys:
                values = [float(s.metrics[metric_name]) for s in rollout]
                table = wandb.Table(
                    data=[[i, v] for i, v in enumerate(values)],
                    columns=["frame", metric_name],
                )
                wandb.log(
                    {
                        f"eval/rollout_{metric_name}": wandb.plot.line(
                            table, "frame", metric_name, title=metric_name
                        )
                    },
                    commit=False,
                )

        # Render video
        video_path = str(checkpoint_path / f"{current_step}.mp4")
        try:
            render_walker_video(
                rollout, mj_model, mj_data, renderer_obj, video_path,
                fps=render_fps, termination_events=termination_events,
            )
            if cfg.logging_config.log_to_wandb:
                import wandb
                wandb.log(
                    {"videos/rollout": wandb.Video(video_path, format="mp4")},
                    commit=False,
                )
            logger.info(f"Eval video saved: {video_path}")
        except mujoco.FatalError as e:
            logger.warning(f"Video rendering failed: {e}")

        # Save checkpoint
        try:
            from flax.training import orbax_utils
            save_args = orbax_utils.save_args_from_target(params)
            ckpt_step_path = checkpoint_path / f"{current_step}"
            orbax_checkpointer.save(
                str(ckpt_step_path), params, force=True, save_args=save_args,
            )
            logger.info(f"Checkpoint saved at step {current_step}")
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")

        del rollout
        gc.collect()

    # Train
    make_inference_fn, params, metrics = brax_ppo_train(
        environment=env,
        eval_env=eval_env,
        episode_length=cfg.env_config.episode_length,
        num_timesteps=cfg.train_config.num_timesteps,
        num_envs=cfg.train_config.num_envs,
        batch_size=cfg.train_config.batch_size,
        num_minibatches=cfg.train_config.num_minibatches,
        num_updates_per_batch=cfg.train_config.num_updates_per_batch,
        learning_rate=cfg.train_config.learning_rate,
        entropy_cost=cfg.train_config.entropy_cost,
        discounting=cfg.train_config.discounting,
        unroll_length=cfg.train_config.unroll_length,
        seed=cfg.train_config.seed,
        normalize_observations=cfg.train_config.normalize_observations,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=functools.partial(playground_wrappers.wrap_for_brax_training),
        num_evals=int(
            cfg.train_config.num_timesteps / cfg.checkpoint.save_every
        ),
    )

    # Save final checkpoint
    ckpt_mgr.save(
        cfg.train_config.num_timesteps,
        args=ocp.args.StandardSave(params),
    )
    logger.info(f"Final checkpoint saved to {checkpoint_path}")

    if cfg.logging_config.log_to_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
