"""Freeloop rollout evaluation for VQ-VAE Prior distillation.

This module provides evaluation utilities for testing the trained Prior
network in "freeloop" mode - where the Prior controls the rodent without
any reference trajectory input.

Freeloop evaluation:
1. Reset environment to random clip position
2. Prior generates z_p from proprio only (no trajectory!)
3. Quantize z_p to z_q using frozen codebook
4. Decoder generates action from z_q + proprio
5. Repeat until termination or max_steps

This tests whether the Prior has learned to predict encoder outputs
well enough to control the rodent autonomously.

Reference: track_mjx/agent/mlp_distill/prior_rollout.py
"""

import functools
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from absl import logging
from brax.training import distribution
from brax.training.acme import running_statistics

from vq_intention_network import Decoder
from vq_prior_networks import VQPrior
from vq_losses import compute_codebook_metrics
from track_mjx.agent.observation_utils import flatten_obs_dict


def run_freeloop_rollout(
    env: Any,
    prior_params: tuple[Any, Any],  # (normalizer_params, prior_params)
    decoder_params: dict[str, Any],
    codebook: jnp.ndarray,
    prior_module: VQPrior,
    decoder_module: Decoder,
    parametric_action_distribution: Any,
    reference_obs_size: int,
    max_steps: int,
    seed: int,
    quantize_prior: bool = True,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run a single freeloop rollout using Prior + Decoder.

    The Prior generates actions without reference trajectories by:
    1. Predicting z_p from proprio alone
    2. Quantizing z_p to nearest codebook entry (optional)
    3. Decoding z_q + proprio to action

    Args:
        env: MuJoCo environment (unwrapped).
        prior_params: Tuple of (normalizer_params, prior_weights).
        decoder_params: Frozen decoder parameters.
        codebook: Frozen VQ-VAE codebook [num_codes, latent_dim].
        prior_module: VQPrior Flax module.
        decoder_module: Decoder Flax module.
        parametric_action_distribution: Action distribution.
        reference_obs_size: Size of reference trajectory in observations.
        max_steps: Maximum steps before termination.
        seed: Random seed for environment reset.
        quantize_prior: If True, quantize prior output to codebook.
        deterministic: If True, use mean action (no sampling).

    Returns:
        Dictionary with:
        - states: List of environment states.
        - actions: Array of actions taken.
        - z_p: Prior outputs at each step.
        - z_q: Quantized codes (if quantize_prior=True).
        - indices: Code indices used.
        - terminated: Whether rollout terminated early.
        - survival_steps: Number of steps before termination.
        - rewards: Per-step rewards.
    """
    normalizer_params, prior_weights = prior_params

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    reset_rng, rng = jax.random.split(rng)

    # Reset to random clip start
    state = jit_reset(reset_rng)

    states = [state]
    actions = []
    z_ps = []
    z_qs = []
    indices_list = []
    rewards = []

    for step in range(max_steps):
        # Extract proprioceptive observation from dict (normalize it)
        obs = state.obs
        flat_obs = flatten_obs_dict(obs)
        proprio = flat_obs["proprioception"]

        # Normalize proprio
        proprio_normalized = running_statistics.normalize(
            proprio, normalizer_params
        )

        # Prior predicts z_p from proprio
        z_p = prior_module.apply({"params": prior_weights}, proprio_normalized)
        z_ps.append(z_p)

        # Optionally quantize to nearest codebook entry
        if quantize_prior:
            # Compute distances to all codebook entries
            distances = jnp.sum((z_p[None, :] - codebook) ** 2, axis=-1)
            idx = jnp.argmin(distances)
            z_q = codebook[idx]
            indices_list.append(int(idx))
        else:
            z_q = z_p
            indices_list.append(-1)
        z_qs.append(z_q)

        # Decode to action
        decoder_input = jnp.concatenate([z_q, proprio_normalized], axis=-1)
        action_logits, _ = decoder_module.apply(
            {"params": decoder_params}, decoder_input
        )

        # Get action from distribution
        if deterministic:
            action = parametric_action_distribution.mode(action_logits)
        else:
            rng, sample_key = jax.random.split(rng)
            raw_action = parametric_action_distribution.sample_no_postprocessing(
                action_logits, sample_key
            )
            action = parametric_action_distribution.postprocess(raw_action)

        # Step environment
        next_state = jit_step(state, action)

        states.append(next_state)
        actions.append(action)
        rewards.append(float(next_state.reward))

        # Check termination
        if next_state.done:
            break

        state = next_state

    return {
        "states": states,
        "actions": jnp.stack(actions) if actions else jnp.zeros((0, env.action_size)),
        "z_p": jnp.stack(z_ps) if z_ps else jnp.zeros((0, codebook.shape[-1])),
        "z_q": jnp.stack(z_qs) if z_qs else jnp.zeros((0, codebook.shape[-1])),
        "indices": jnp.array(indices_list),
        "terminated": step < max_steps - 1,
        "survival_steps": len(states) - 1,
        "rewards": jnp.array(rewards) if rewards else jnp.zeros((0,)),
        "total_reward": float(sum(rewards)) if rewards else 0.0,
    }


class VQPriorFreelloopEvaluator:
    """Evaluator for freeloop rollouts using trained Prior network.

    Runs multiple freeloop rollouts where the Prior controls the rodent
    without reference trajectories. Logs metrics and videos to wandb.

    Attributes:
        env: MuJoCo environment.
        prior_module: VQPrior Flax module.
        decoder_module: Decoder Flax module.
        parametric_action_distribution: Action distribution.
        reference_obs_size: Size of reference trajectory in observations.
        num_rollouts: Number of rollouts per evaluation.
        max_steps: Maximum steps per rollout.
        quantize_prior: Whether to quantize prior output.
        deterministic: Whether to use deterministic actions.
        eval_interval: Run evaluation every N training iterations.
        render_best_rollout: Whether to render video of best rollout.
        render_fps: FPS for rendered video.
        render_camera_name: Camera name for rendering.
        model_path: Path for saving videos.
    """

    def __init__(
        self,
        env: Any,
        latent_dim: int,
        action_size: int,
        proprioceptive_obs_size: int,
        reference_obs_size: int,
        decoder_hidden_layer_sizes: tuple[int, ...],
        prior_hidden_layer_sizes: tuple[int, ...],
        num_rollouts: int = 32,
        max_steps: int = 200,
        quantize_prior: bool = True,
        deterministic: bool = True,
        eval_interval: int = 1,
        render_best_rollout: bool = True,
        render_fps: int = 50,
        render_camera_name: str = "close_profile",
        model_path: str = "",
    ):
        """Initialize the freeloop evaluator.

        Args:
            env: MuJoCo environment (unwrapped).
            latent_dim: Dimension of latent embeddings.
            action_size: Action dimension.
            proprioceptive_obs_size: Size of proprio observations.
            reference_obs_size: Size of reference trajectory.
            decoder_hidden_layer_sizes: Decoder MLP layer sizes.
            prior_hidden_layer_sizes: Prior MLP layer sizes.
            num_rollouts: Number of rollouts per evaluation.
            max_steps: Maximum steps per rollout.
            quantize_prior: Whether to quantize prior output.
            deterministic: Whether to use deterministic actions.
            eval_interval: Run evaluation every N iterations.
            render_best_rollout: Whether to render best rollout video.
            render_fps: FPS for video rendering.
            render_camera_name: Camera for rendering.
            model_path: Path for saving videos.
        """
        self.env = env
        self.latent_dim = latent_dim
        self.action_size = action_size
        self.proprioceptive_obs_size = proprioceptive_obs_size
        self.reference_obs_size = reference_obs_size
        self.num_rollouts = num_rollouts
        self.max_steps = max_steps
        self.quantize_prior = quantize_prior
        self.deterministic = deterministic
        self.eval_interval = eval_interval
        self.render_best_rollout = render_best_rollout
        self.render_fps = render_fps
        self.render_camera_name = render_camera_name
        self.model_path = Path(model_path) if model_path else Path(".")

        # Create network modules
        self.prior_module = VQPrior(
            layer_sizes=list(prior_hidden_layer_sizes),
            latent_dim=latent_dim,
        )

        decoder_layer_sizes = list(decoder_hidden_layer_sizes) + [action_size * 2]
        self.decoder_module = Decoder(layer_sizes=decoder_layer_sizes)

        self.parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )

    def run_evaluation(
        self,
        prior_params: tuple[Any, Any],
        decoder_params: dict[str, Any],
        codebook: jnp.ndarray,
        eval_step: int,
    ) -> dict[str, Any] | None:
        """Run freeloop evaluation and log to wandb.

        Args:
            prior_params: Tuple of (normalizer_params, prior_weights).
            decoder_params: Frozen decoder parameters.
            codebook: Frozen VQ-VAE codebook.
            eval_step: Current evaluation step.

        Returns:
            Dictionary of metrics, or None if skipped due to eval_interval.
        """
        if eval_step % self.eval_interval != 0:
            return None

        t_start = time.time()
        logging.info(f"Running freeloop evaluation at step {eval_step}")

        metrics = {
            "freeloop/survival_steps": [],
            "freeloop/total_reward": [],
            "freeloop/termination_rate": 0,
            "freeloop/unique_codes_used": [],
        }

        best_rollout = None
        best_survival = 0

        all_indices = []

        for i in range(self.num_rollouts):
            result = run_freeloop_rollout(
                env=self.env,
                prior_params=prior_params,
                decoder_params=decoder_params,
                codebook=codebook,
                prior_module=self.prior_module,
                decoder_module=self.decoder_module,
                parametric_action_distribution=self.parametric_action_distribution,
                reference_obs_size=self.reference_obs_size,
                max_steps=self.max_steps,
                seed=eval_step * 1000 + i,
                quantize_prior=self.quantize_prior,
                deterministic=self.deterministic,
            )

            metrics["freeloop/survival_steps"].append(result["survival_steps"])
            metrics["freeloop/total_reward"].append(result["total_reward"])

            if self.quantize_prior:
                unique_codes = len(np.unique(np.array(result["indices"])))
                metrics["freeloop/unique_codes_used"].append(unique_codes)
                all_indices.extend(list(np.array(result["indices"])))

            if result["terminated"]:
                metrics["freeloop/termination_rate"] += 1

            if result["survival_steps"] > best_survival:
                best_survival = result["survival_steps"]
                best_rollout = result

        # Aggregate metrics
        metrics["freeloop/avg_survival_steps"] = np.mean(
            metrics["freeloop/survival_steps"]
        )
        metrics["freeloop/std_survival_steps"] = np.std(
            metrics["freeloop/survival_steps"]
        )
        metrics["freeloop/max_survival_steps"] = np.max(
            metrics["freeloop/survival_steps"]
        )
        metrics["freeloop/avg_total_reward"] = np.mean(metrics["freeloop/total_reward"])
        metrics["freeloop/termination_rate"] /= self.num_rollouts

        if self.quantize_prior and metrics["freeloop/unique_codes_used"]:
            metrics["freeloop/avg_unique_codes"] = np.mean(
                metrics["freeloop/unique_codes_used"]
            )

        # Compute codebook usage metrics
        if self.quantize_prior and all_indices:
            indices_arr = jnp.array(all_indices)
            perplexity, utilization, codes_used = compute_codebook_metrics(
                indices_arr, codebook.shape[0]
            )
            metrics["freeloop/perplexity"] = float(perplexity)
            metrics["freeloop/codebook_utilization"] = float(utilization)
            metrics["freeloop/total_codes_used"] = int(codes_used)

        # Log z_p statistics from best rollout
        if best_rollout is not None and len(best_rollout["z_p"]) > 0:
            z_p = np.array(best_rollout["z_p"])
            metrics["freeloop/z_p_mean"] = np.mean(z_p)
            metrics["freeloop/z_p_std"] = np.std(z_p)

            # Log per-dimension statistics for first few dimensions
            for i in range(min(5, z_p.shape[-1])):
                metrics[f"freeloop_latents/z_p_dim{i}_mean"] = np.mean(z_p[..., i])
                metrics[f"freeloop_latents/z_p_dim{i}_std"] = np.std(z_p[..., i])

        # Remove list metrics (already aggregated)
        metrics.pop("freeloop/survival_steps")
        metrics.pop("freeloop/total_reward")
        metrics.pop("freeloop/unique_codes_used", None)

        eval_time = time.time() - t_start
        metrics["freeloop/eval_time"] = eval_time

        logging.info(
            f"Freeloop eval: avg_survival={metrics['freeloop/avg_survival_steps']:.1f}, "
            f"termination_rate={metrics['freeloop/termination_rate']:.2%}, "
            f"time={eval_time:.1f}s"
        )

        # Render best rollout video
        if self.render_best_rollout and best_rollout is not None:
            self._render_best_rollout(best_rollout, eval_step)

        # Log to wandb
        try:
            wandb.log(metrics, commit=False)
        except Exception as e:
            logging.warning(f"Failed to log freeloop metrics to wandb: {e}")

        return metrics

    def _render_best_rollout(
        self,
        rollout: dict[str, Any],
        eval_step: int,
    ) -> None:
        """Render and log video of best freeloop rollout.

        Args:
            rollout: Rollout result dictionary.
            eval_step: Current evaluation step.
        """
        try:
            import imageio.v2 as imageio

            states = rollout["states"]
            if len(states) < 2:
                return

            # Render frames
            frames = []
            for state in states:
                self.env.step(state, jnp.zeros(self.action_size))  # Set state
                frame = self.env.render(
                    state.pipeline_state, camera=self.render_camera_name
                )
                frames.append(frame)

            # Save video
            video_path = self.model_path / f"freeloop_{eval_step}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)

            with imageio.get_writer(str(video_path), fps=self.render_fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

            # Log to wandb
            wandb.log(
                {"videos/freeloop_best": wandb.Video(str(video_path))},
                commit=False,
            )
            logging.info(f"Rendered freeloop video to {video_path}")

        except Exception as e:
            logging.warning(f"Failed to render freeloop video: {e}")


def log_freeloop_to_wandb(
    env: Any,
    prior_params: tuple[Any, Any],
    decoder_params: dict[str, Any],
    codebook: jnp.ndarray,
    prior_module: VQPrior,
    decoder_module: Decoder,
    parametric_action_distribution: Any,
    reference_obs_size: int,
    eval_step: int,
    num_rollouts: int = 32,
    max_steps: int = 200,
    quantize_prior: bool = True,
    render_best_rollout: bool = True,
    render_fps: int = 50,
    render_camera_name: str = "close_profile",
    model_path: str = "",
) -> dict[str, Any]:
    """Standalone function to run freeloop evaluation and log to wandb.

    Convenience function that creates a temporary evaluator and runs evaluation.

    Args:
        env: MuJoCo environment.
        prior_params: Prior parameters.
        decoder_params: Decoder parameters.
        codebook: VQ-VAE codebook.
        prior_module: Prior Flax module.
        decoder_module: Decoder Flax module.
        parametric_action_distribution: Action distribution.
        reference_obs_size: Reference trajectory size.
        eval_step: Current evaluation step.
        num_rollouts: Number of rollouts.
        max_steps: Max steps per rollout.
        quantize_prior: Whether to quantize prior output.
        render_best_rollout: Whether to render video.
        render_fps: Video FPS.
        render_camera_name: Camera for rendering.
        model_path: Path for saving videos.

    Returns:
        Dictionary of metrics.
    """
    evaluator = VQPriorFreelloopEvaluator(
        env=env,
        latent_dim=codebook.shape[-1],
        action_size=decoder_module.layer_sizes[-1] // 2,
        proprioceptive_obs_size=0,  # Will be inferred
        reference_obs_size=reference_obs_size,
        decoder_hidden_layer_sizes=tuple(decoder_module.layer_sizes[:-1]),
        prior_hidden_layer_sizes=tuple(prior_module.layer_sizes),
        num_rollouts=num_rollouts,
        max_steps=max_steps,
        quantize_prior=quantize_prior,
        deterministic=True,
        eval_interval=1,
        render_best_rollout=render_best_rollout,
        render_fps=render_fps,
        render_camera_name=render_camera_name,
        model_path=model_path,
    )

    return evaluator.run_evaluation(
        prior_params=prior_params,
        decoder_params=decoder_params,
        codebook=codebook,
        eval_step=eval_step,
    )
