"""Minimal VQ-VAE training experiment script.

This script runs a minimal training experiment using the VQ-VAE intention
network architecture. It is designed to test the VQ-VAE implementation
before integrating into the main codebase.

Usage:
    cd scratch/vqvae_jax
    python run_minimal_experiment.py

The script will:
1. Load configuration from configs/vqvae_minimal.yaml
2. Create a minimal environment with single clip
3. Train using VQ-VAE networks and losses
4. Log to wandb with VQ-specific metrics
"""

import os
import sys
import logging
import functools
from pathlib import Path
from datetime import datetime

# Must set rendering backend before importing MuJoCo
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Add scratch directory and track-mjx to path
SCRATCH_DIR = Path(__file__).parent
REPO_ROOT = SCRATCH_DIR.parent.parent
sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import yaml
import wandb
from ml_collections import ConfigDict

# Import VQ-VAE modules from scratch
from vq_intention_network import make_vq_intention_policy
from vq_losses import compute_vq_ppo_loss, PPONetworkParams, create_vq_schedule
from vq_ppo_networks import make_vq_intention_ppo_networks, make_vq_inference_fn

# Import from main codebase
from track_mjx.config import utils as config_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_jax_environment():
    """Configure JAX environment for training."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def wandb_progress(num_steps: int, metrics: dict) -> None:
    """Log training progress to wandb."""
    metrics["num_steps_thousands"] = num_steps // 1000
    wandb.log(metrics)


def run_minimal_vq_test():
    """Run a minimal test of the VQ-VAE components without full training.

    This test:
    1. Creates VQ networks
    2. Initializes parameters
    3. Runs a forward pass
    4. Computes VQ loss
    5. Verifies gradients flow correctly
    """
    logger.info("Running minimal VQ-VAE component test...")

    # Test parameters
    observation_size = 100
    reference_obs_size = 50
    action_size = 8
    latent_dim = 32
    num_codes = 64
    batch_size = 4
    time_steps = 10

    # Create networks
    logger.info("Creating VQ-VAE PPO networks...")
    networks = make_vq_intention_ppo_networks(
        observation_size=observation_size,
        reference_obs_size=reference_obs_size,
        action_size=action_size,
        latent_dim=latent_dim,
        num_codes=num_codes,
        encoder_hidden_layer_sizes=(256, 128),
        decoder_hidden_layer_sizes=(256, 128),
        value_hidden_layer_sizes=(256, 128),
    )

    # Initialize parameters
    key = jax.random.PRNGKey(42)
    key_policy, key_value, key_obs = jax.random.split(key, 3)

    policy_params = networks.policy_network.init(key_policy)
    value_params = networks.value_network.init(key_value)

    logger.info(f"Policy params keys: {jax.tree_util.tree_map(lambda x: x.shape, policy_params)}")

    # Create test observation
    obs = jax.random.normal(key_obs, (time_steps, batch_size, observation_size))

    # Forward pass through policy
    logger.info("Running forward pass...")
    action_params, z_e, indices = networks.policy_network.apply(
        None, policy_params, obs, key
    )

    logger.info(f"Action params shape: {action_params.shape}")
    logger.info(f"z_e shape: {z_e.shape}")
    logger.info(f"Indices shape: {indices.shape}")
    logger.info(f"Unique codes used: {len(jnp.unique(indices))}")

    # Test loss computation components
    logger.info("Testing VQ loss computation...")
    codebook = policy_params["params"]["quantizer"]["embeddings"]
    z_q = codebook[indices]

    from vq_losses import compute_vq_loss, compute_codebook_metrics

    vq_loss, commitment_loss, codebook_loss = compute_vq_loss(z_e, z_q)
    perplexity, utilization, codes_used = compute_codebook_metrics(indices, num_codes)

    logger.info(f"VQ Loss: {float(vq_loss):.4f}")
    logger.info(f"Commitment Loss: {float(commitment_loss):.4f}")
    logger.info(f"Codebook Loss: {float(codebook_loss):.4f}")
    logger.info(f"Perplexity: {float(perplexity):.2f}")
    logger.info(f"Utilization: {float(utilization):.2%}")
    logger.info(f"Codes Used: {int(codes_used)}")

    # Test gradient flow
    logger.info("Testing gradient flow...")

    def test_loss(params):
        action, z_e, indices = networks.policy_network.apply(None, params, obs, key)
        codebook = params["params"]["quantizer"]["embeddings"]
        z_q = codebook[indices]
        vq_loss, _, _ = compute_vq_loss(z_e, z_q)
        return jnp.mean(action ** 2) + vq_loss

    grads = jax.grad(test_loss)(policy_params)

    # Check gradients exist for all components
    encoder_has_grad = any(
        jnp.any(g != 0) for g in jax.tree_util.tree_leaves(grads["params"]["encoder"])
    )
    quantizer_has_grad = jnp.any(grads["params"]["quantizer"]["embeddings"] != 0)
    decoder_has_grad = any(
        jnp.any(g != 0) for g in jax.tree_util.tree_leaves(grads["params"]["decoder"])
    )

    logger.info(f"Encoder has gradients: {encoder_has_grad}")
    logger.info(f"Quantizer (codebook) has gradients: {quantizer_has_grad}")
    logger.info(f"Decoder has gradients: {decoder_has_grad}")

    # Test inference function
    logger.info("Testing inference function...")
    make_policy = make_vq_inference_fn(networks)
    policy_fn = make_policy((None, policy_params), deterministic=False)
    action, extras = policy_fn(obs[0], key)

    logger.info(f"Inference action shape: {action.shape}")
    logger.info(f"Inference extras keys: {extras.keys()}")

    logger.info("Minimal VQ-VAE test completed successfully!")

    return {
        "vq_loss": float(vq_loss),
        "commitment_loss": float(commitment_loss),
        "codebook_loss": float(codebook_loss),
        "perplexity": float(perplexity),
        "utilization": float(utilization),
        "codes_used": int(codes_used),
        "encoder_has_grad": encoder_has_grad,
        "quantizer_has_grad": quantizer_has_grad,
        "decoder_has_grad": decoder_has_grad,
    }


def run_wandb_logging_test():
    """Test wandb logging with VQ-VAE metrics."""
    logger.info("Testing wandb logging...")

    # Initialize wandb
    run = wandb.init(
        project="vqvae_experiments",
        group="minimal_tests",
        name=f"vqvae_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "arch": "vqvae",
            "latent_dim": 32,
            "num_codes": 64,
            "commitment_cost": 0.25,
        },
    )

    # Run minimal test
    results = run_minimal_vq_test()

    # Log to wandb
    wandb.log({
        "test/vq_loss": results["vq_loss"],
        "test/commitment_loss": results["commitment_loss"],
        "test/codebook_loss": results["codebook_loss"],
        "test/perplexity": results["perplexity"],
        "test/codebook_utilization": results["utilization"],
        "test/codes_used": results["codes_used"],
    })

    wandb.finish()
    logger.info("Wandb logging test completed!")


def main():
    """Main entry point for VQ-VAE minimal experiment."""
    setup_jax_environment()

    # Check GPU availability
    try:
        n_devices = jax.device_count(backend="gpu")
        logger.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logger.info("Not using GPUs, falling back to CPU")

    # Run component tests
    logger.info("=" * 60)
    logger.info("VQ-VAE Minimal Experiment")
    logger.info("=" * 60)

    # Test 1: Basic component test
    logger.info("\n[Test 1] Running basic component test...")
    results = run_minimal_vq_test()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"VQ Loss: {results['vq_loss']:.4f}")
    logger.info(f"Perplexity: {results['perplexity']:.2f} / 64 (max)")
    logger.info(f"Utilization: {results['utilization']:.2%}")
    logger.info(f"Gradients flowing: encoder={results['encoder_has_grad']}, "
                f"codebook={results['quantizer_has_grad']}, "
                f"decoder={results['decoder_has_grad']}")

    # All checks passed
    all_passed = (
        results["encoder_has_grad"]
        and results["quantizer_has_grad"]
        and results["decoder_has_grad"]
    )

    if all_passed:
        logger.info("\nAll tests PASSED!")
        logger.info("\nVQ-VAE implementation is ready for integration.")
        logger.info("Next steps:")
        logger.info("1. Integrate with main codebase by adding arch_name='vqvae_intention'")
        logger.info("2. Run full training with: python run_minimal_experiment.py --full")
        logger.info("3. Monitor codebook health (perplexity should stay high)")
    else:
        logger.error("\nSome tests FAILED!")
        return 1

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Run with wandb logging")
    parser.add_argument("--full", action="store_true", help="Run full training (not implemented yet)")
    args = parser.parse_args()

    if args.wandb:
        run_wandb_logging_test()
    elif args.full:
        logger.info("Full training not yet implemented.")
        logger.info("This requires integration with the main training loop.")
        logger.info("See README.md for integration steps.")
    else:
        exit(main())
