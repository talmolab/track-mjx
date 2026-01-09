"""Discriminator training loop.

Implements epoch-based training with binary cross-entropy loss,
checkpointing after each epoch, and comprehensive wandb logging.
"""

import logging
import time
from typing import Any, Callable, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb

from track_mjx.agent.discriminator.data_loading import MotionClipDataset, create_batches
from track_mjx.agent.discriminator.discriminator_network import make_discriminator_network


@flax.struct.dataclass
class DiscriminatorParams:
    """Container for discriminator network parameters."""

    params: Dict


@flax.struct.dataclass
class TrainingState:
    """Training state for discriminator.

    Attributes:
        optimizer_state: Optax optimizer state.
        params: Network parameters wrapped in DiscriminatorParams.
        epoch: Current epoch number (0-indexed during training, incremented after each epoch).
        step: Global step count (number of batches processed).
        best_test_accuracy: Best test accuracy seen so far.
    """

    optimizer_state: optax.OptState
    params: DiscriminatorParams
    epoch: int
    step: int
    best_test_accuracy: float


def binary_cross_entropy_loss(
    logits: jnp.ndarray, labels: jnp.ndarray
) -> jnp.ndarray:
    """Compute binary cross-entropy loss with numerical stability.

    Uses the numerically stable formulation:
        max(logits, 0) - logits * labels + log(1 + exp(-|logits|))

    Args:
        logits: Raw model outputs of shape (batch, 1) or (batch,).
        labels: Binary labels of shape (batch,) with values 0.0 or 1.0.

    Returns:
        Scalar loss value (mean over batch).
    """
    logits = logits.squeeze(-1)  # Ensure shape is (batch,)
    # Numerically stable BCE
    loss = (
        jnp.maximum(logits, 0)
        - logits * labels
        + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    )
    return jnp.mean(loss)


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute classification accuracy.

    Args:
        logits: Raw model outputs of shape (batch, 1) or (batch,).
        labels: Binary labels of shape (batch,).

    Returns:
        Scalar accuracy value (fraction of correct predictions).
    """
    predictions = (logits.squeeze(-1) > 0).astype(jnp.float32)
    return jnp.mean(predictions == labels)


def create_train_step(
    apply_fn: Callable,
    optimizer: optax.GradientTransformation,
) -> Callable:
    """Create JIT-compiled training step function.

    Args:
        apply_fn: Network apply function (params, x, training, rngs) -> logits.
        optimizer: Optax optimizer.

    Returns:
        Function (state, batch_data, batch_labels, rng) -> (new_state, metrics).
    """

    def loss_fn(
        params: Dict,
        batch_data: jnp.ndarray,
        batch_labels: jnp.ndarray,
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logits = apply_fn(params, batch_data, training=True, rngs={"dropout": rng})
        loss = binary_cross_entropy_loss(logits, batch_labels)
        accuracy = compute_accuracy(logits, batch_labels)
        return loss, {"loss": loss, "accuracy": accuracy}

    @jax.jit
    def train_step(
        state: TrainingState,
        batch_data: jnp.ndarray,
        batch_labels: jnp.ndarray,
        rng: jax.Array,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params.params, batch_data, batch_labels, rng
        )

        updates, new_optimizer_state = optimizer.update(
            grads, state.optimizer_state, state.params.params
        )
        new_params = optax.apply_updates(state.params.params, updates)

        new_state = TrainingState(
            optimizer_state=new_optimizer_state,
            params=DiscriminatorParams(params=new_params),
            epoch=state.epoch,
            step=state.step + 1,
            best_test_accuracy=state.best_test_accuracy,
        )

        return new_state, metrics

    return train_step


def create_eval_step(apply_fn: Callable) -> Callable:
    """Create JIT-compiled evaluation step function.

    Args:
        apply_fn: Network apply function.

    Returns:
        Function (params, batch_data, batch_labels) -> metrics.
    """

    @jax.jit
    def eval_step(
        params: Dict,
        batch_data: jnp.ndarray,
        batch_labels: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        logits = apply_fn(params, batch_data, training=False)
        loss = binary_cross_entropy_loss(logits, batch_labels)
        accuracy = compute_accuracy(logits, batch_labels)
        return {"loss": loss, "accuracy": accuracy}

    return eval_step


def evaluate(
    params: Dict,
    eval_step_fn: Callable,
    real_data: np.ndarray,
    fake_data: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        params: Model parameters.
        eval_step_fn: JIT-compiled evaluation step function.
        real_data: Real samples array.
        fake_data: Fake samples array.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with mean loss and accuracy over all batches.
    """
    rng = np.random.default_rng(0)  # Fixed seed for reproducible evaluation

    total_loss = 0.0
    total_accuracy = 0.0
    n_batches = 0

    for batch_data, batch_labels in create_batches(
        real_data, fake_data, batch_size, rng, shuffle=False
    ):
        batch_data_jax = jnp.array(batch_data)
        batch_labels_jax = jnp.array(batch_labels)

        metrics = eval_step_fn(params, batch_data_jax, batch_labels_jax)
        total_loss += float(metrics["loss"])
        total_accuracy += float(metrics["accuracy"])
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_accuracy / max(n_batches, 1),
    }


def train(
    dataset: MotionClipDataset,
    hidden_layer_sizes: Tuple[int, ...],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    dropout_rate: float,
    use_layer_norm: bool,
    seed: int,
    ckpt_mgr: ocp.CheckpointManager,
    config_dict: Dict,
    checkpoint_callback: Callable[[int], None] | None = None,
) -> Tuple[TrainingState, Dict[str, Any]]:
    """Train discriminator network.

    Args:
        dataset: MotionClipDataset with train/test splits.
        hidden_layer_sizes: Hidden layer sizes for MLP.
        num_epochs: Number of training epochs.
        batch_size: Batch size (must be even for balanced batching).
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        grad_clip_norm: Gradient clipping threshold.
        dropout_rate: Dropout rate for training.
        use_layer_norm: Whether to use layer normalization.
        seed: Random seed for reproducibility.
        ckpt_mgr: Orbax checkpoint manager.
        config_dict: Configuration dictionary for checkpointing.
        checkpoint_callback: Optional callback called after each checkpoint save.

    Returns:
        Tuple of (final_training_state, final_metrics).
    """
    # Determine input size from data shape
    sample_shape = dataset.train_real.shape[1:]  # (num_steps, qpos_dim)
    input_size = int(np.prod(sample_shape))  # num_steps * qpos_dim

    logging.info(f"Input size: {input_size} (from shape {sample_shape})")
    logging.info(
        f"Train samples: {len(dataset.train_real)} real, {len(dataset.train_fake)} fake"
    )
    logging.info(
        f"Test samples: {len(dataset.test_real)} real, {len(dataset.test_fake)} fake"
    )

    # Create network
    _, init_fn, apply_fn = make_discriminator_network(
        input_size=input_size,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
    )

    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    init_params = init_fn(key)

    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    # Initialize training state
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=DiscriminatorParams(params=init_params),
        epoch=0,
        step=0,
        best_test_accuracy=0.0,
    )

    # Create JIT-compiled step functions
    train_step_fn = create_train_step(apply_fn, optimizer)
    eval_step_fn = create_eval_step(apply_fn)

    # Training loop
    rng = np.random.default_rng(seed)
    train_loss = 0.0
    train_accuracy = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_accuracies = []

        # Generate random keys for this epoch's dropout
        key, epoch_key = jax.random.split(key)
        # Pre-generate enough keys for all batches
        max_batches = (
            min(len(dataset.train_real), len(dataset.train_fake)) // (batch_size // 2)
        )
        batch_keys = jax.random.split(epoch_key, max_batches + 1)
        batch_idx = 0

        # Training batches
        for batch_data, batch_labels in create_batches(
            dataset.train_real, dataset.train_fake, batch_size, rng, shuffle=True
        ):
            batch_data_jax = jnp.array(batch_data)
            batch_labels_jax = jnp.array(batch_labels)

            training_state, metrics = train_step_fn(
                training_state, batch_data_jax, batch_labels_jax, batch_keys[batch_idx]
            )

            epoch_losses.append(float(metrics["loss"]))
            epoch_accuracies.append(float(metrics["accuracy"]))
            batch_idx += 1

        # Update epoch count
        training_state = training_state.replace(epoch=epoch + 1)

        # Compute epoch averages
        train_loss = np.mean(epoch_losses)
        train_accuracy = np.mean(epoch_accuracies)

        # Evaluate on test set
        test_metrics = evaluate(
            training_state.params.params,
            eval_step_fn,
            dataset.test_real,
            dataset.test_fake,
            batch_size,
        )

        # Update best accuracy
        if test_metrics["accuracy"] > training_state.best_test_accuracy:
            training_state = training_state.replace(
                best_test_accuracy=test_metrics["accuracy"]
            )

        epoch_time = time.time() - epoch_start

        # Log to wandb
        wandb.log(
            {
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/best_accuracy": training_state.best_test_accuracy,
                "epoch": epoch + 1,
                "step": training_state.step,
                "epoch_time": epoch_time,
            }
        )

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint after each epoch
        ckpt_mgr.save(
            step=epoch + 1,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(training_state.params.params),
                train_state=ocp.args.StandardSave(training_state),
                config=ocp.args.JsonSave(config_dict),
            ),
        )

        if checkpoint_callback is not None:
            checkpoint_callback(epoch + 1)

    final_metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "best_test_accuracy": float(training_state.best_test_accuracy),
    }

    return training_state, final_metrics
