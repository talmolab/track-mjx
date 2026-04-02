"""
Network definitions for distillation training.
Reuses the IntentionNetwork architecture from ff_ppo but with distillation-specific utilities.

Observations are expected as dictionaries with keys:
- "task_obs": Reference trajectory observations (flat array)
- "proprioception": Proprioceptive state observations (flat array)
"""

from collections.abc import Mapping
from typing import Any, Callable, Sequence, Tuple
from pathlib import Path

from brax.training import networks
from brax.training import types
from brax.training import distribution
from brax.training.types import PRNGKey
from brax.training.acme import running_statistics

import flax
import jax
from jax import numpy as jnp

from track_mjx.agent.mlp_distill import student_network
from track_mjx.agent import checkpointing
from track_mjx.agent.observation_utils import convert_flat_to_dict_normalizer


@flax.struct.dataclass
class DistillNetworks:
    """Networks used for distillation training."""

    student: student_network.StudentNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_student_inference_fn(distill_networks: DistillNetworks):
    """Creates inference function for the student network.

    Returns a function that takes (normalizer_params, policy_params) and returns
    a policy function that can be used for inference.
    """

    def make_policy(
        params: types.PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:
        student_network = distill_networks.student
        parametric_action_distribution = distill_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            normalizer_params, policy_params = params
            key_sample, key_network = jax.random.split(key_sample)

            # Get student outputs
            policy_logits, latent_mean, latent_logvar, prior_mean, prior_logvar = (
                student_network.apply(
                    normalizer_params,
                    policy_params,
                    observations,
                    key_network,
                    deterministic=deterministic,
                )
            )

            if deterministic:
                action = parametric_action_distribution.mode(policy_logits)
            else:
                action = parametric_action_distribution.sample(
                    policy_logits, key_sample
                )

            return action, {
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
                "prior_mean": prior_mean,
                "prior_logvar": prior_logvar,
            }

        return policy

    return make_policy


def make_student_networks(
    obs_sizes: Mapping[str, int],
    action_size: int,
    preprocess_observations_fn=None,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    prior_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    encoder_logvar_min: float | None = None,
    encoder_logvar_max: float | None = None,
    prior_logvar_min: float | None = None,
    prior_logvar_max: float | None = None,
    encoder_expansion_factor: int = 1,
) -> DistillNetworks:
    """Create student networks for distillation training.

    Uses the same architecture as PPO intention networks.

    Args:
        obs_sizes: Dict with "task_obs" and "proprioception" sizes.
        action_size: Size of the action space.
        preprocess_observations_fn: Function to preprocess dict observations.
        intention_latent_size: Size of the latent space.
        encoder_hidden_layer_sizes: Hidden layer sizes for encoder.
        decoder_hidden_layer_sizes: Hidden layer sizes for decoder.
        prior_hidden_layer_sizes: Hidden layer sizes for prior.
        encoder_logvar_min: Min clamp for encoder log-variance (PULSE uses -5).
        encoder_logvar_max: Max clamp for encoder log-variance (PULSE uses 2).
        prior_logvar_min: Min clamp for prior log-variance (PULSE uses -5).
        prior_logvar_max: Max clamp for prior log-variance (PULSE uses 2).
        encoder_expansion_factor: Expansion factor for encoder before mean/logvar
            heads (PULSE uses 5).
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    student = student_network.make_student_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        obs_sizes=obs_sizes,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        prior_hidden_layer_sizes=prior_hidden_layer_sizes,
        encoder_logvar_min=encoder_logvar_min,
        encoder_logvar_max=encoder_logvar_max,
        prior_logvar_min=prior_logvar_min,
        prior_logvar_max=prior_logvar_max,
        encoder_expansion_factor=encoder_expansion_factor,
    )

    return DistillNetworks(
        student=student,
        parametric_action_distribution=parametric_action_distribution,
    )


def load_teacher_policy(
    teacher_checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Callable, Any]:
    """Load a pretrained teacher policy from checkpoint.

    Args:
        teacher_checkpoint_path: Path to the teacher checkpoint directory.
        step: Optional step to load. If None, loads the latest checkpoint.

    Returns:
        Tuple of (inference_fn, policy_params) where inference_fn is a callable
        that takes (observations, key) and returns (actions, extras).
    """
    checkpoint_data = checkpointing.load_checkpoint_for_eval(
        teacher_checkpoint_path, step=step
    )
    cfg = checkpoint_data["cfg"]
    policy_params = checkpoint_data["policy"]

    # Create inference function
    inference_fn = checkpointing.load_inference_fn(
        cfg, policy_params, deterministic=True, get_activation=False
    )

    return inference_fn, policy_params


def create_teacher_inference_fn(
    teacher_checkpoint_path: str,
    step: int | None = None,
) -> Tuple[Callable, Any, Any]:
    """Create a jittable teacher inference function.

    Args:
        teacher_checkpoint_path: Path to the teacher checkpoint directory.
        step: Optional step to load. If None, loads the latest checkpoint.

    Returns:
        Tuple of (make_teacher_policy, teacher_params, teacher_cfg) where:
        - make_teacher_policy: A factory function that takes (deterministic) and returns a policy function
        - teacher_params: The teacher's policy parameters
        - teacher_cfg: The teacher's configuration
    """
    checkpoint_data = checkpointing.load_checkpoint_for_eval(
        teacher_checkpoint_path, step=step
    )
    cfg = checkpoint_data["cfg"]
    policy_params = checkpoint_data["policy"]

    # Convert legacy flat normalizer to dict normalizer if needed
    normalizer_state, network_params = policy_params
    network_config = cfg.network_config

    # Check if this is a legacy flat normalizer by looking at config format
    is_legacy = not (
        hasattr(network_config, "obs_sizes") or "obs_sizes" in network_config
    )

    if is_legacy:
        # Convert flat normalizer to dict normalizer
        reference_obs_size = network_config.reference_obs_size
        normalizer_state = convert_flat_to_dict_normalizer(
            normalizer_state, reference_obs_size
        )
        policy_params = (normalizer_state, network_params)

    # Create the ppo network from config
    ppo_network = checkpointing.make_ppo_network_from_cfg(cfg)

    def make_teacher_policy(deterministic: bool = True) -> Callable:
        """Create a teacher policy function with deterministic behavior fixed.

        Args:
            deterministic: Whether to use deterministic policy (captured in closure)

        Returns:
            A policy function that takes (params, observations, key) and returns (actions, extras)
        """

        def teacher_policy_fn(
            params: types.PolicyParams,
            observations: jnp.ndarray,
            key: PRNGKey,
        ) -> Tuple[jnp.ndarray, dict]:
            """Apply teacher network.

            Args:
                params: Tuple of (normalizer_params, policy_params)
                observations: Input observations
                key: Random key

            Returns:
                Tuple of (actions, extras)
            """
            normalizer_params, policy_network_params = params

            # Get policy outputs (deterministic captured from closure)
            policy_logits, latent_mean, latent_logvar = (
                ppo_network.policy_network.apply(
                    normalizer_params,
                    policy_network_params,
                    observations,
                    key,
                    deterministic=deterministic,
                )
            )

            if deterministic:
                action = ppo_network.parametric_action_distribution.mode(policy_logits)
            else:
                action = ppo_network.parametric_action_distribution.sample(
                    policy_logits, key
                )

            return action, {
                "policy_logits": policy_logits,
                "latent_mean": latent_mean,
                "latent_logvar": latent_logvar,
            }

        return teacher_policy_fn

    return make_teacher_policy, policy_params, cfg
