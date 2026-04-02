"""Step 2 training: Walker imitation with online reference generation.

Uses the trained Step 1 policy to generate expert data on-the-fly,
training an encoder-latent-decoder architecture for latent topology analysis.

Usage:
    cd /home/talmolab/Desktop/SalkResearch/track-mjx
    source /home/talmolab/Desktop/SalkResearch/mimic-mjx/bin/activate
    python track_mjx/train_walker_imitation.py
    # with overrides:
    python track_mjx/train_walker_imitation.py \
        network_config.intention_size=32 \
        env_config.n_transitions=2
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import functools
import logging

import hydra
import jax
import jax.numpy as jp
import orbax.checkpoint as ocp
import wandb
from brax.training.agents.ppo import networks as brax_ppo_networks
from brax.training import distribution
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf

from vnl_playground.tasks.walker.multi_behavior import (
    MultiBehaviorWalker,
    default_config as multi_behavior_config,
)
from vnl_playground.tasks.walker.online_reference import (
    OnlineReferenceGenerator,
)
from vnl_playground.tasks.walker.imitation import (
    WalkerImitation,
    default_config as imitation_default_config,
)
from track_mjx.agent.ff_ppo import ppo as ff_ppo, ppo_networks as ff_networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_step1_policy(checkpoint_path: str, step: int = None):
    """Load the trained Step 1 multi-behavior policy.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        step: Checkpoint step to load. None = latest.

    Returns:
        Restored checkpoint params.
    """
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path)
    if step is None:
        step = ckpt_mgr.latest_step()
    logger.info(
        f"Loading Step 1 checkpoint from {checkpoint_path}, step {step}"
    )

    restored = ckpt_mgr.restore(step)
    return restored


def build_step1_inference_fn(
    step1_params,
    obs_size: int,
    action_size: int,
    hidden_sizes: tuple = (256, 256, 256),
):
    """Build an inference function from Step 1 checkpoint params.

    Args:
        step1_params: Restored checkpoint params (normalizer, policy).
        obs_size: Observation size of MultiBehaviorWalker.
        action_size: Action size of MultiBehaviorWalker.
        hidden_sizes: Hidden layer sizes used during Step 1 training.

    Returns:
        Callable (obs, rng) -> action.
    """
    ppo_networks = brax_ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
    )
    parametric_action_dist = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # Extract params - structure depends on Brax checkpoint format
    normalizer_params, policy_params = step1_params[0], step1_params[1]

    def inference_fn(obs, rng):
        """Run Step 1 policy: obs -> action."""
        logits = ppo_networks.policy_network.apply(
            normalizer_params, policy_params.policy, obs
        )
        action = parametric_action_dist.sample(logits, rng)
        return action

    return inference_fn


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="walker-imitation-online",
)
def main(cfg: DictConfig) -> None:
    """Train walker imitation with online reference generation."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    try:
        n_devices = jax.device_count(backend="gpu")
        logger.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logger.info("No GPU, using CPU")

    # --- Load Step 1 policy for reference generation ---
    step1_params = load_step1_policy(
        cfg.step1_checkpoint.path,
        cfg.step1_checkpoint.get("step", None),
    )

    # Create the generator's source environment
    gen_env = MultiBehaviorWalker(config=multi_behavior_config())

    # Build inference function from Step 1
    step1_inference_fn = build_step1_inference_fn(
        step1_params,
        obs_size=gen_env.observation_size,
        action_size=gen_env.action_size,
        hidden_sizes=tuple(
            cfg.get("step1_network", {}).get(
                "hidden_sizes", [256, 256, 256]
            )
        ),
    )

    # Create online reference generator
    generator = OnlineReferenceGenerator(
        policy_fn=step1_inference_fn,
        walker_env=gen_env,
        n_frames=cfg.env_config.trajectory_length,
        deterministic=True,
    )

    # --- Create imitation environment ---
    env_cfg = imitation_default_config()
    for key in [
        "sim_dt", "ctrl_dt", "mujoco_impl", "nconmax", "njmax",
        "reference_length", "trajectory_length", "mocap_hz",
        "n_transitions",
    ]:
        if key in cfg.env_config:
            env_cfg[key] = cfg.env_config[key]
    if "reward_terms" in cfg.env_config:
        env_cfg.reward_terms = OmegaConf.to_container(
            cfg.env_config.reward_terms, resolve=True
        )
    if "termination_criteria" in cfg.env_config:
        env_cfg.termination_criteria = OmegaConf.to_container(
            cfg.env_config.termination_criteria, resolve=True
        )

    env = WalkerImitation(config=env_cfg, generator=generator)
    eval_env = WalkerImitation(config=env_cfg, generator=generator)

    logger.info(
        f"Environment created: obs_size={env.observation_size}, "
        f"action_size={env.action_size}"
    )

    # --- Setup intention network factory ---
    network_factory = functools.partial(
        ff_networks.make_intention_ppo_networks,
        intention_latent_size=cfg.network_config.intention_size,
        encoder_hidden_layer_sizes=tuple(
            cfg.network_config.encoder_layer_sizes
        ),
        decoder_hidden_layer_sizes=tuple(
            cfg.network_config.decoder_layer_sizes
        ),
        encoder_noise_std=cfg.network_config.get(
            "encoder_noise_std", 0.0
        ),
        proprioception_noise_std=cfg.network_config.get(
            "proprioception_noise_std", 0.0
        ),
        proprioception_noise_mode=cfg.network_config.get(
            "proprioception_noise_mode", "multiplicative"
        ),
        value_hidden_layer_sizes=tuple(
            cfg.network_config.critic_layer_sizes
        ),
    )

    # --- Setup checkpointing ---
    os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
    ckpt_mgr = ocp.CheckpointManager(
        cfg.checkpoint.save_dir,
        options=ocp.CheckpointManagerOptions(
            create=True, step_prefix="PPONetwork"
        ),
    )

    # --- Setup wandb ---
    if cfg.logging_config.log_to_wandb:
        wandb.init(
            project=cfg.logging_config.project_name,
            group=cfg.logging_config.group_name,
            name=cfg.logging_config.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def progress_fn(num_steps, metrics):
        logger.info(
            f"Step {num_steps}: "
            f"reward={metrics.get('eval/episode_reward', 0):.3f}"
        )
        if cfg.logging_config.log_to_wandb:
            wandb.log({"step": num_steps, **metrics})

    # --- Compute episode length ---
    episode_length = (
        cfg.env_config.trajectory_length - cfg.env_config.reference_length
    )
    logger.info(f"Episode length: {episode_length}")

    # --- Build training kwargs ---
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    train_kwargs = dict(
        **OmegaConf.to_container(
            cfg.train_setup.train_config, resolve=True
        ),
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps
            / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=(
            cfg.train_setup.eval_every // cfg.train_setup.reset_every
        ),
        episode_length=episode_length,
        latent_kl_weight=cfg.network_config.latent_kl_weight,
        latent_ar1_weight=cfg.network_config.latent_ar1_weight,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        use_kl_schedule=cfg.network_config.kl_schedule,
        eval_env_test_set=eval_env,
        wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
    )

    # --- Train ---
    make_inference_fn, params, metrics = ff_ppo.train(
        environment=env,
        progress_fn=progress_fn,
        **train_kwargs,
    )

    logger.info("Training complete!")

    if cfg.logging_config.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
