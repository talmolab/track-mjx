#!/usr/bin/env python3
"""Multi-dataset discriminator window size sweep.

Trains discriminators with varying window sizes for multiple fake datasets
and generates comparison plots.

Example usage:
    python -m track_mjx.scripts.discriminator_window_sweep \
        --h5-path data/discriminator_dataset.h5 \
        --output-dir results/discriminator_sweep \
        --clip-length 240 \
        --window-sizes 10 20 40 60 80 120 240 \
        --num-epochs 200
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Set environment variables before importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from track_mjx.analysis.discriminator import (
    MotionClipDataset,
    create_train_test_split,
    list_h5_datasets,
    load_h5_dataset,
    load_h5_metadata,
    make_discriminator_network,
    make_rnn_discriminator_network,
)
from track_mjx.analysis.discriminator.data_loading import create_batches
from track_mjx.analysis.discriminator.discriminator_train import (
    DiscriminatorParams,
    TrainingState,
    create_eval_step,
    create_train_step,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train discriminators with varying window sizes for multiple datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--h5-path",
        type=str,
        required=True,
        help="Path to H5 dataset file created by collect_rollout_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results (JSON and plots)",
    )

    # Dataset configuration
    parser.add_argument(
        "--real-dataset",
        type=str,
        default="encoder_decoder_qpos",
        help="Name of the 'real' dataset in H5 file",
    )
    parser.add_argument(
        "--fake-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Names of 'fake' datasets to compare. If not provided, uses all "
        "other datasets in the H5 file except the real dataset.",
    )

    # Window/clip configuration
    parser.add_argument(
        "--clip-length",
        type=int,
        default=240,
        help="Truncate clips to this length (use value with many divisors, e.g., 240)",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240],
        help="Window sizes to sweep (must divide clip-length evenly)",
    )

    # Training configuration
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs per discriminator",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )

    # Network configuration
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="Hidden layer sizes for discriminator MLP",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for training",
    )
    parser.add_argument(
        "--no-layer-norm",
        action="store_true",
        help="Disable layer normalization",
    )

    # Network type selection
    parser.add_argument(
        "--network-type",
        type=str,
        choices=["mlp", "rnn"],
        default="mlp",
        help="Type of discriminator network architecture",
    )

    # RNN-specific arguments
    parser.add_argument(
        "--rnn-hidden-size",
        type=int,
        default=128,
        help="Hidden size for RNN (each direction if bidirectional)",
    )
    parser.add_argument(
        "--rnn-num-layers",
        type=int,
        default=2,
        help="Number of stacked RNN layers",
    )
    parser.add_argument(
        "--rnn-bidirectional",
        action="store_true",
        default=True,
        help="Use bidirectional RNN (default: True)",
    )
    parser.add_argument(
        "--rnn-unidirectional",
        action="store_true",
        help="Use unidirectional RNN (overrides --rnn-bidirectional)",
    )
    parser.add_argument(
        "--attention-hidden-size",
        type=int,
        default=64,
        help="Hidden size for attention pooling mechanism",
    )

    # Data configuration
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training",
    )
    parser.add_argument(
        "--exclude-root",
        action="store_true",
        help="Exclude root position/orientation from qpos",
    )
    parser.add_argument(
        "--exclude-zero-joints",
        action="store_true",
        help="Exclude joints that are always zero",
    )

    return parser.parse_args()


def split_clips_into_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Split clips into smaller windows along the time dimension.

    Args:
        data: Array of shape (num_clips, num_steps, qpos_dim).
        window_size: Number of time steps per window.

    Returns:
        Array of shape (num_clips * num_windows, window_size, qpos_dim).
    """
    num_clips, num_steps, qpos_dim = data.shape
    num_windows = num_steps // window_size
    reshaped = data.reshape(num_clips, num_windows, window_size, qpos_dim)
    return reshaped.reshape(-1, window_size, qpos_dim)


def train_discriminator(
    real_dataset_name: str,
    fake_dataset_name: str,
    h5_path: str,
    num_epochs: int,
    batch_size: int,
    hidden_layer_sizes: Tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    dropout_rate: float,
    use_layer_norm: bool,
    train_ratio: float,
    split_seed: int,
    seed: int,
    exclude_root: bool = False,
    exclude_zero_joints: bool = False,
    window_size: int = None,
    clip_length: int = None,
    network_type: str = "mlp",
    rnn_hidden_size: int = 128,
    rnn_num_layers: int = 2,
    rnn_bidirectional: bool = True,
    attention_hidden_size: int = 64,
) -> Dict[str, List]:
    """Train a discriminator and return the training history."""

    # Load data
    real_data = load_h5_dataset(h5_path, real_dataset_name)
    fake_data = load_h5_dataset(h5_path, fake_dataset_name)

    # Truncate clips if clip_length is specified
    if clip_length is not None:
        if clip_length > real_data.shape[1]:
            raise ValueError(
                f"clip_length ({clip_length}) cannot exceed num_steps ({real_data.shape[1]})"
            )
        real_data = real_data[:, :clip_length, :]
        fake_data = fake_data[:, :clip_length, :]

    # Validate window size if provided
    if window_size is not None:
        num_steps = real_data.shape[1]
        if num_steps % window_size != 0:
            raise ValueError(
                f"window_size ({window_size}) must evenly divide num_steps ({num_steps})"
            )

    single_dataset_mode = real_dataset_name == fake_dataset_name

    # Create train/test splits
    ds = create_train_test_split(
        real_data=real_data,
        fake_data=fake_data,
        train_ratio=train_ratio,
        seed=split_seed,
        single_dataset_mode=single_dataset_mode,
    )

    # Apply filtering if needed
    if exclude_root or exclude_zero_joints:
        ZERO_JOINT_INDICES = [
            18,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            57,
            65,
            73,
        ]
        all_indices = set(range(74))
        indices_to_exclude = set()
        if exclude_root:
            indices_to_exclude.update(range(7))
        if exclude_zero_joints:
            indices_to_exclude.update(ZERO_JOINT_INDICES)
        indices_to_keep = sorted(all_indices - indices_to_exclude)

        ds = MotionClipDataset(
            train_real=ds.train_real[:, :, indices_to_keep],
            train_fake=ds.train_fake[:, :, indices_to_keep],
            test_real=ds.test_real[:, :, indices_to_keep],
            test_fake=ds.test_fake[:, :, indices_to_keep],
            metadata=ds.metadata,
        )

    # Apply windowing if configured (AFTER split to avoid data leakage)
    if window_size is not None:
        ds = MotionClipDataset(
            train_real=split_clips_into_windows(ds.train_real, window_size),
            train_fake=split_clips_into_windows(ds.train_fake, window_size),
            test_real=split_clips_into_windows(ds.test_real, window_size),
            test_fake=split_clips_into_windows(ds.test_fake, window_size),
            metadata=ds.metadata,
        )

    # Determine input shape
    sample_shape = ds.train_real.shape[1:]  # (num_steps, qpos_dim)

    # Create network based on type
    if network_type == "mlp":
        input_size = int(np.prod(sample_shape))
        _, init_fn, apply_fn = make_discriminator_network(
            input_size=input_size,
            hidden_layer_sizes=hidden_layer_sizes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )
    elif network_type == "rnn":
        num_steps, qpos_dim = sample_shape
        _, init_fn, apply_fn = make_rnn_discriminator_network(
            num_steps=num_steps,
            qpos_dim=qpos_dim,
            rnn_hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout_rate=dropout_rate,
            bidirectional=rnn_bidirectional,
            attention_hidden_size=attention_hidden_size,
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    # Initialize
    key = jax.random.PRNGKey(seed)
    init_params = init_fn(key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=DiscriminatorParams(params=init_params),
        epoch=0,
        step=0,
        best_test_accuracy=0.0,
    )

    train_step_fn = create_train_step(apply_fn, optimizer)
    eval_step_fn = create_eval_step(apply_fn)

    # History
    hist = {"epoch": [], "test_accuracy": [], "train_accuracy": []}

    # Initial eval
    initial_metrics = evaluate(
        state.params.params, eval_step_fn, ds.test_real, ds.test_fake, batch_size
    )
    hist["epoch"].append(0)
    hist["test_accuracy"].append(float(initial_metrics["accuracy"]))
    hist["train_accuracy"].append(None)

    # Training loop
    rng = np.random.default_rng(seed)

    for epoch in range(num_epochs):
        epoch_accuracies = []
        key, epoch_key = jax.random.split(key)
        max_batches = min(len(ds.train_real), len(ds.train_fake)) // (batch_size // 2)
        batch_keys = jax.random.split(epoch_key, max(1, max_batches + 1))
        batch_idx = 0

        for batch_data, batch_labels in create_batches(
            ds.train_real, ds.train_fake, batch_size, rng, shuffle=True
        ):
            batch_data_jax = jnp.array(batch_data)
            batch_labels_jax = jnp.array(batch_labels)
            state, metrics = train_step_fn(
                state,
                batch_data_jax,
                batch_labels_jax,
                batch_keys[min(batch_idx, len(batch_keys) - 1)],
            )
            epoch_accuracies.append(float(metrics["accuracy"]))
            batch_idx += 1

        state = state.replace(epoch=epoch + 1)
        train_acc = float(np.mean(epoch_accuracies)) if epoch_accuracies else 0.0

        test_metrics = evaluate(
            state.params.params, eval_step_fn, ds.test_real, ds.test_fake, batch_size
        )

        hist["epoch"].append(epoch + 1)
        hist["test_accuracy"].append(float(test_metrics["accuracy"]))
        hist["train_accuracy"].append(train_acc)

    return hist


def save_plot_data_csv(
    results: Dict[str, Dict[int, Dict]],
    window_sizes: List[int],
    output_path: str,
) -> None:
    """Save the plot data as a CSV file.

    CSV format:
        window_size, dataset1_mean, dataset1_std, dataset2_mean, dataset2_std, ...
    """
    import csv

    window_sizes_sorted = sorted(window_sizes)
    fake_datasets = list(results.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        header = ["window_size"]
        for ds in fake_datasets:
            short_name = ds.replace("_qpos", "").replace("prior_", "")
            header.extend([f"{short_name}_mean", f"{short_name}_std"])
        writer.writerow(header)

        # Data rows
        for ws in window_sizes_sorted:
            row = [ws]
            for ds in fake_datasets:
                row.append(results[ds][ws]["last_50_mean"])
                row.append(results[ds][ws]["last_50_std"])
            writer.writerow(row)

    print(f"CSV saved to: {output_path}")


def create_plot(
    results: Dict[str, Dict[int, Dict]],
    window_sizes: List[int],
    real_dataset: str,
    clip_length: int,
    output_path: str,
) -> None:
    """Create and save the comparison plot."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map for different datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    window_sizes_arr = np.array(sorted(window_sizes))

    for idx, (fake_ds, ws_results) in enumerate(results.items()):
        means = np.array([ws_results[ws]["last_50_mean"] for ws in window_sizes_arr])
        stds = np.array([ws_results[ws]["last_50_std"] for ws in window_sizes_arr])

        # Shorten dataset name for legend
        short_name = fake_ds.replace("_qpos", "").replace("prior_", "")

        # Plot mean with shaded std region
        ax.plot(
            window_sizes_arr,
            means,
            color=colors[idx],
            linewidth=2,
            marker="o",
            markersize=5,
            label=short_name,
        )
        ax.fill_between(
            window_sizes_arr, means - stds, means + stds, color=colors[idx], alpha=0.15
        )

    # Reference line
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Random (50%)")

    # Labels and formatting
    ax.set_xlabel("Window Size (time steps)", fontsize=12)
    ax.set_ylabel("Test Accuracy (last 50 epochs)", fontsize=12)
    ax.set_title(
        f"Window Size vs Discriminator Accuracy by Dataset\n"
        f"Real: {real_dataset} (clip_length={clip_length})",
        fontsize=14,
    )
    ax.set_xlim(0, max(window_sizes_arr) + 10)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle bidirectional flag
    bidirectional = args.rnn_bidirectional and not args.rnn_unidirectional

    # Auto-discover fake datasets if not provided
    if args.fake_datasets is None:
        all_datasets = list_h5_datasets(args.h5_path)
        args.fake_datasets = [ds for ds in all_datasets if ds != args.real_dataset]
        if not args.fake_datasets:
            raise ValueError(
                f"No fake datasets found. H5 file only contains: {all_datasets}"
            )
        print(f"Auto-discovered {len(args.fake_datasets)} fake datasets from H5 file")

    # Validate window sizes
    invalid_sizes = [ws for ws in args.window_sizes if args.clip_length % ws != 0]
    if invalid_sizes:
        valid_divisors = [
            i for i in range(10, args.clip_length + 1) if args.clip_length % i == 0
        ]
        raise ValueError(
            f"Invalid window sizes for clip_length={args.clip_length}: {invalid_sizes}. "
            f"Valid sizes (>=10): {valid_divisors}"
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("MULTI-DATASET DISCRIMINATOR WINDOW SIZE SWEEP")
    print("=" * 70)
    print(f"H5 path: {args.h5_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Real dataset: {args.real_dataset}")
    print(f"Fake datasets: {args.fake_datasets}")
    print(f"Clip length: {args.clip_length}")
    print(f"Window sizes: {args.window_sizes}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Network type: {args.network_type}")
    if args.network_type == "mlp":
        print(f"  Hidden layers: {args.hidden_layers}")
        print(f"  Layer norm: {not args.no_layer_norm}")
    else:
        print(f"  RNN hidden size: {args.rnn_hidden_size}")
        print(f"  RNN layers: {args.rnn_num_layers}")
        print(f"  Bidirectional: {bidirectional}")
        print(f"  Attention hidden size: {args.attention_hidden_size}")
    print(f"Dropout rate: {args.dropout_rate}")
    total_runs = len(args.fake_datasets) * len(args.window_sizes)
    print(
        f"Total training runs: {len(args.fake_datasets)} x {len(args.window_sizes)} = {total_runs}"
    )
    print("=" * 70)

    # Initialize results
    results = {fake_ds: {} for fake_ds in args.fake_datasets}

    run_count = 0
    start_time = time.time()

    for fake_ds in args.fake_datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {fake_ds}")
        print("=" * 60)

        for ws in args.window_sizes:
            run_count += 1
            print(f"[{run_count}/{total_runs}] Window size: {ws}", end=" ", flush=True)

            run_start = time.time()

            hist = train_discriminator(
                real_dataset_name=args.real_dataset,
                fake_dataset_name=fake_ds,
                h5_path=args.h5_path,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                hidden_layer_sizes=tuple(args.hidden_layers),
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                dropout_rate=args.dropout_rate,
                use_layer_norm=not args.no_layer_norm,
                train_ratio=args.train_ratio,
                split_seed=args.split_seed,
                seed=args.seed,
                exclude_root=args.exclude_root,
                exclude_zero_joints=args.exclude_zero_joints,
                window_size=ws,
                clip_length=args.clip_length,
                network_type=args.network_type,
                rnn_hidden_size=args.rnn_hidden_size,
                rnn_num_layers=args.rnn_num_layers,
                rnn_bidirectional=bidirectional,
                attention_hidden_size=args.attention_hidden_size,
            )

            # Compute last 50 epochs statistics
            test_accs = np.array(hist["test_accuracy"])
            last_n = min(50, len(test_accs))
            last_50_mean = float(np.mean(test_accs[-last_n:]))
            last_50_std = float(np.std(test_accs[-last_n:]))

            results[fake_ds][ws] = {
                "history": hist,
                "last_50_mean": last_50_mean,
                "last_50_std": last_50_std,
            }

            run_time = time.time() - run_start
            print(f"-> {last_50_mean:.4f} +/- {last_50_std:.4f} ({run_time:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Sweep complete! Total time: {total_time / 60:.1f} minutes")
    print("=" * 60)

    # Save results to JSON (without full history to keep file size manageable)
    results_summary = {}
    for fake_ds, ws_results in results.items():
        results_summary[fake_ds] = {
            str(ws): {
                "last_50_mean": data["last_50_mean"],
                "last_50_std": data["last_50_std"],
            }
            for ws, data in ws_results.items()
        }

    # Add metadata
    metadata = {
        "h5_path": args.h5_path,
        "real_dataset": args.real_dataset,
        "fake_datasets": args.fake_datasets,
        "clip_length": args.clip_length,
        "window_sizes": args.window_sizes,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "network_type": args.network_type,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
    }
    if args.network_type == "mlp":
        metadata["hidden_layers"] = args.hidden_layers
        metadata["use_layer_norm"] = not args.no_layer_norm
    else:
        metadata["rnn_hidden_size"] = args.rnn_hidden_size
        metadata["rnn_num_layers"] = args.rnn_num_layers
        metadata["rnn_bidirectional"] = bidirectional
        metadata["attention_hidden_size"] = args.attention_hidden_size

    output_data = {
        "metadata": metadata,
        "results": results_summary,
    }

    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Create and save plot
    plot_path = output_dir / "window_sweep_comparison.png"
    create_plot(
        results=results,
        window_sizes=args.window_sizes,
        real_dataset=args.real_dataset,
        clip_length=args.clip_length,
        output_path=str(plot_path),
    )

    # Save plot data as CSV
    csv_path = output_dir / "window_sweep_data.csv"
    save_plot_data_csv(
        results=results,
        window_sizes=args.window_sizes,
        output_path=str(csv_path),
    )

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print(f"Real: {args.real_dataset} | Clip length: {args.clip_length}")
    print("=" * 100)

    # Header
    header = f"{'Window':>8}"
    for fake_ds in args.fake_datasets:
        short_name = fake_ds.replace("_qpos", "").replace("prior_", "")[:12]
        header += f" {short_name:>14}"
    print(header)
    print("-" * 100)

    # Data rows
    for ws in sorted(args.window_sizes):
        row = f"{ws:>8}"
        for fake_ds in args.fake_datasets:
            mean = results[fake_ds][ws]["last_50_mean"]
            std = results[fake_ds][ws]["last_50_std"]
            row += f" {mean:>6.4f}+/-{std:.4f}"
        print(row)
    print("=" * 100)


if __name__ == "__main__":
    main()
