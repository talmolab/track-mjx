"""
Rollout logging for prior training.

This module provides logging functionality for prior training, including:
- Latent statistics (encoder mean/logvar, prior mean/logvar)
- KL divergence between encoder and prior over rollout
- Video rendering (optional, since encoder+decoder is pretrained)
"""

import logging

import imageio
import jax
import jax.numpy as jnp
import mujoco
import wandb
from brax.training.acme import running_statistics
from omegaconf import DictConfig

from track_mjx.agent.mlp_prior import prior_networks
from track_mjx.agent.mlp_prior import losses


def prior_training_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg: DictConfig,
    model_path: str,
    current_step: int,
    params,
    policy_params_fn_key,
    reset_key: jax.Array,
    render_video: bool = False,
) -> None:
    """Rollout logging for prior training - logs latent statistics.

    Unlike distillation training, we don't render encoder+decoder rollouts
    since they are pretrained. We only log latent statistics for monitoring
    how well the prior is learning to match the encoder.

    Args:
        env: Environment for rollout.
        jit_reset: JIT-compiled reset function.
        jit_step: JIT-compiled step function.
        cfg: Configuration.
        model_path: Path for saving videos.
        current_step: Current training step.
        params: Policy parameters (normalizer, {encoder, decoder, prior}).
        policy_params_fn_key: Random key for rollout.
        reset_key: Random key for environment reset (shared with other evaluators).
        render_video: Whether to render video (default False for prior training).
    """
    physics_step_per_control_step = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_step_per_control_step
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_frame)

    # Extract network config
    latent_size = cfg.network_config.intention_size
    encoder_hidden_layer_sizes = tuple(cfg.network_config.encoder_layer_sizes)
    decoder_hidden_layer_sizes = tuple(cfg.network_config.decoder_layer_sizes)
    prior_hidden_layer_sizes = tuple(cfg.network_config.get("prior_layer_sizes", [1024, 1024]))
    reference_obs_size = cfg.network_config.reference_obs_size
    proprioceptive_obs_size = cfg.network_config.proprioceptive_obs_size
    action_size = cfg.network_config.action_size

    # Setup normalization
    normalize = lambda x, y: x
    if cfg.train_setup.train_config.normalize_observations:
        normalize = running_statistics.normalize

    # Extract params
    normalizer_params, network_params = params
    encoder_params = network_params["params"]["encoder"]
    decoder_params = network_params["params"]["decoder"]
    prior_params = network_params["params"]["prior"]

    # Create encoder+decoder policy for rollout
    policy = prior_networks.make_encoder_decoder_inference_fn(
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        normalizer_params=normalizer_params,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        latent_size=latent_size,
        action_size=action_size,
        reference_obs_size=reference_obs_size,
        proprioceptive_obs_size=proprioceptive_obs_size,
        deterministic=True,
    )
    jit_policy = jax.jit(policy)

    # Create encoder and prior apply functions for getting distributions
    encoder_apply_fn = prior_networks.make_encoder_apply_fn(
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        latent_size=latent_size,
        reference_obs_size=reference_obs_size,
    )

    from track_mjx.agent.mlp_prior.prior_networks import Prior
    prior_module = Prior(
        layer_sizes=list(prior_hidden_layer_sizes),
        latents=latent_size,
    )

    # Run rollout and collect latent statistics
    _, act_rng = jax.random.split(policy_params_fn_key)

    # Use provided reset_key for environment reset (shared with other evaluators)
    state = jit_reset(reset_key)
    rollout = [state]
    encoder_means = []
    encoder_logvars = []
    prior_means = []
    prior_logvars = []

    for i in range(episode_length):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs

        # Get action and encoder latents from policy
        ctrl, extras = jit_policy(obs, act_rng)
        ctrl = jnp.squeeze(ctrl, axis=0) if ctrl.shape[0] == 1 else ctrl

        # Collect encoder latent statistics from extras
        encoder_means.append(extras["latent_mean"])
        encoder_logvars.append(extras["latent_logvar"])

        # Get prior latent statistics
        # Normalize observations
        normalized_obs = running_statistics.normalize(obs, normalizer_params)
        proprio_obs = normalized_obs[..., reference_obs_size:]
        prior_mean, prior_logvar = prior_module.apply({"params": prior_params}, proprio_obs)
        prior_means.append(prior_mean)
        prior_logvars.append(prior_logvar)

        state = jit_step(state, ctrl)
        rollout.append(state)

    # Stack latents
    encoder_means = jnp.stack(encoder_means)
    encoder_logvars = jnp.stack(encoder_logvars)
    prior_means = jnp.stack(prior_means)
    prior_logvars = jnp.stack(prior_logvars)

    # Compute per-dimension statistics
    encoder_means_mean = jnp.mean(encoder_means, axis=0)
    encoder_means_std = jnp.std(encoder_means, axis=0)
    encoder_logvars_mean = jnp.mean(encoder_logvars, axis=0)
    encoder_logvars_std = jnp.std(encoder_logvars, axis=0)

    prior_means_mean = jnp.mean(prior_means, axis=0)
    prior_means_std = jnp.std(prior_means, axis=0)
    prior_logvars_mean = jnp.mean(prior_logvars, axis=0)
    prior_logvars_std = jnp.std(prior_logvars, axis=0)

    # Compute KL divergence over rollout
    kl_per_timestep = []
    for t in range(len(encoder_means)):
        kl = losses.compute_encoder_prior_kl_loss(
            encoder_means[t:t+1],
            encoder_logvars[t:t+1],
            prior_means[t:t+1],
            prior_logvars[t:t+1],
        )
        kl_per_timestep.append(float(kl))

    # Log per-dimension latent statistics
    for i in range(encoder_means_mean.shape[-1]):
        wandb.log(
            {
                # Encoder latent statistics
                f"latents/encoder_means_mean{i}": float(encoder_means_mean[i]),
                f"latents/encoder_means_std{i}": float(encoder_means_std[i]),
                f"latents/encoder_logvars_mean{i}": float(encoder_logvars_mean[i]),
                f"latents/encoder_logvars_std{i}": float(encoder_logvars_std[i]),
                # Prior latent statistics
                f"latents/prior_means_mean{i}": float(prior_means_mean[i]),
                f"latents/prior_means_std{i}": float(prior_means_std[i]),
                f"latents/prior_logvars_mean{i}": float(prior_logvars_mean[i]),
                f"latents/prior_logvars_std{i}": float(prior_logvars_std[i]),
            },
            commit=False,
        )

    # Optionally render video (encoder+decoder rollout)
    if render_video:
        render_fps = cfg.render_config.render_fps
        camera_name = f"{cfg.render_config.render_camera_name}-ghost"

        video_path = f"{model_path}/{current_step}_encoder_decoder.mp4"
        try:
            with imageio.get_writer(video_path, fps=render_fps) as writer:
                video = env.render(
                    rollout,
                    camera=camera_name,
                    height=480,
                    width=640,
                )
                for frame in video:
                    writer.append_data(frame)

            wandb.log(
                {"videos/encoder_decoder_rollout": wandb.Video(video_path, format="mp4")},
                commit=False,
            )
        except mujoco.FatalError as e:
            logging.warning(f"Video rendering failed due to MuJoCo error: {e}. Skipping video for this iteration.")
