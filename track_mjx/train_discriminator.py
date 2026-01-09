"""Entry point for discriminator training.

Load motion clip data from H5 file and train a discriminator to classify
real vs fake motion clips.

Usage:
    # Train discriminator comparing original vs prior rollouts:
    python track_mjx/train_discriminator.py \
        data_config.h5_path=/path/to/rollout_dataset.h5 \
        data_config.real_dataset=original_qpos \
        data_config.fake_dataset=prior_logvar_0_qpos

    # Train baseline (single dataset mode - should get ~50% accuracy):
    python track_mjx/train_discriminator.py \
        data_config.h5_path=/path/to/rollout_dataset.h5 \
        data_config.real_dataset=original_qpos \
        data_config.fake_dataset=original_qpos
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import jax
import orbax.checkpoint as ocp
import wandb
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent.discriminator import data_loading
from track_mjx.agent.discriminator import discriminator_train


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main(version_base=None, config_path="config", config_name="discriminator")
def main(cfg: DictConfig) -> None:
    """Main function for discriminator training using Hydra configs."""
    _setup_environment()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Validate required config
    if cfg.data_config.h5_path is None:
        raise ValueError(
            "data_config.h5_path must be specified. "
            "Provide path to H5 file from collect_rollout_dataset.py."
        )

    # Check for single dataset mode (baseline where network shouldn't learn)
    single_dataset_mode = cfg.data_config.real_dataset == cfg.data_config.fake_dataset
    if single_dataset_mode:
        logging.info(
            f"Single dataset mode: both real and fake use '{cfg.data_config.real_dataset}'. "
            "This is a baseline where the network should NOT learn meaningful discrimination "
            "(expected accuracy ~50%)."
        )

    # Log device info
    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPU(s)")
    except RuntimeError:
        n_devices = 1
        logging.info("No GPU detected, using CPU")

    # Generate run ID
    run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    # Setup checkpoint path
    if cfg.train_setup.checkpoint_path is not None:
        checkpoint_path = cfg.train_setup.checkpoint_path
    else:
        model_path = Path(cfg.logging_config.model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        checkpoint_path = str(model_path / run_id)

    logging.info(f"Run ID: {run_id}")
    logging.info(f"Checkpoint path: {checkpoint_path}")

    # Initialize checkpoint manager with Discriminator step prefix
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="Discriminator",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Load data from H5 file
    logging.info(f"Loading data from: {cfg.data_config.h5_path}")
    logging.info(f"Real dataset: {cfg.data_config.real_dataset}")
    logging.info(f"Fake dataset: {cfg.data_config.fake_dataset}")

    real_data = data_loading.load_h5_dataset(
        cfg.data_config.h5_path, cfg.data_config.real_dataset
    )
    fake_data = data_loading.load_h5_dataset(
        cfg.data_config.h5_path, cfg.data_config.fake_dataset
    )
    h5_metadata = data_loading.load_h5_metadata(cfg.data_config.h5_path)

    logging.info(f"Loaded real data shape: {real_data.shape}")
    logging.info(f"Loaded fake data shape: {fake_data.shape}")
    logging.info(f"H5 metadata: {h5_metadata}")

    # Create train/test splits
    dataset = data_loading.create_train_test_split(
        real_data=real_data,
        fake_data=fake_data,
        train_ratio=cfg.data_config.train_ratio,
        seed=cfg.data_config.split_seed,
        single_dataset_mode=single_dataset_mode,
    )

    logging.info(f"Dataset split info: {dataset.metadata}")

    # Prepare config dict for checkpointing and wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["run_id"] = run_id
    cfg_dict["checkpoint_path"] = checkpoint_path
    cfg_dict["single_dataset_mode"] = single_dataset_mode
    cfg_dict["h5_metadata"] = {k: str(v) for k, v in h5_metadata.items()}

    # Generate experiment name if not provided
    if cfg.logging_config.exp_name is None:
        exp_name = f"{cfg.data_config.real_dataset}_vs_{cfg.data_config.fake_dataset}"
    else:
        exp_name = cfg.logging_config.exp_name

    # Initialize wandb
    wandb_run_id = f"{exp_name}_{run_id}"
    wandb.init(
        project=cfg.logging_config.project_name,
        config=cfg_dict,
        id=wandb_run_id,
        group=cfg.logging_config.group_name,
        name=exp_name,
    )

    # Log dataset info to wandb (once at start)
    wandb.log(
        {
            "dataset/num_train_real": dataset.metadata["num_train_real"],
            "dataset/num_train_fake": dataset.metadata["num_train_fake"],
            "dataset/num_test_real": dataset.metadata["num_test_real"],
            "dataset/num_test_fake": dataset.metadata["num_test_fake"],
            "dataset/single_dataset_mode": int(single_dataset_mode),
            "dataset/input_size": int(real_data.shape[1] * real_data.shape[2]),
        },
        commit=False,
    )

    # Train discriminator
    logging.info("Starting discriminator training...")
    logging.info(f"Network architecture: {list(cfg.network_config.hidden_layer_sizes)}")
    logging.info(f"Training for {cfg.train_setup.num_epochs} epochs")

    final_state, final_metrics = discriminator_train.train(
        dataset=dataset,
        hidden_layer_sizes=tuple(cfg.network_config.hidden_layer_sizes),
        num_epochs=cfg.train_setup.num_epochs,
        batch_size=cfg.train_setup.batch_size,
        learning_rate=cfg.train_setup.learning_rate,
        weight_decay=cfg.train_setup.weight_decay,
        grad_clip_norm=cfg.train_setup.grad_clip_norm,
        dropout_rate=cfg.network_config.dropout_rate,
        use_layer_norm=cfg.network_config.use_layer_norm,
        seed=cfg.train_setup.seed,
        ckpt_mgr=ckpt_mgr,
        config_dict=cfg_dict,
        checkpoint_every=cfg.train_setup.checkpoint_every,
    )

    # Log final metrics
    logging.info(f"Training complete. Final metrics: {final_metrics}")
    wandb.log(
        {
            "final/train_loss": final_metrics["train_loss"],
            "final/train_accuracy": final_metrics["train_accuracy"],
            "final/test_loss": final_metrics["test_loss"],
            "final/test_accuracy": final_metrics["test_accuracy"],
            "final/best_test_accuracy": final_metrics["best_test_accuracy"],
        }
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Final train accuracy: {final_metrics['train_accuracy']:.4f}")
    print(f"Final test accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"Best test accuracy: {final_metrics['best_test_accuracy']:.4f}")
    if single_dataset_mode:
        print("\nNote: Single dataset mode was used (baseline).")
        print("Expected accuracy is ~50% (random chance).")
    print("=" * 60)

    wandb.finish()


if __name__ == "__main__":
    main()
