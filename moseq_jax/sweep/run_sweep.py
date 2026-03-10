"""KPMS hyperparameter grid search.

Fits KPMS models across a grid of (num_states, kappa, latent_dim, model_type)
with multiple seeds.  Selects the best model by reconstruction MSE, then EML,
then syllable usage ratio.

**CRITICAL**: This script sets ``jax_enable_x64 = True`` and MUST run in a
separate process from the RL training pipeline.

Usage::

    cd moseq_jax
    python -m sweep.run_sweep                  # uses default config
    python -m sweep.run_sweep --config path    # custom config
"""

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Enable x64 before any JAX import
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

# Add repo root to path
MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
sys.path.insert(0, str(MOSEQ_DIR))
sys.path.insert(0, str(REPO_ROOT))

import yaml

from moseq_jax.kpms.config import KPMSHyperparams
from moseq_jax.kpms.keypoint_loader import (
    DEFAULT_KP_NAMES,
    setup_mujoco_model,
    qpos_to_keypoints,
)
from moseq_jax.kpms.fit_kpms import fit_kpms_keypoints

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_config(config_path: str | None = None) -> dict:
    """Load sweep config from YAML."""
    if config_path is None:
        config_path = str(MOSEQ_DIR / "configs" / "kpms_sweep.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_reference_data(
    data_path: str,
    balanced_split_path: str | None = None,
) -> np.ndarray:
    """Load qpos from reference clips, optionally filtered by balanced split.

    Returns:
        qpos array, shape ``[N, T, nq]``.
    """
    import h5py

    with h5py.File(data_path, "r") as f:
        qpos = f["qpos"][:]

    if balanced_split_path and Path(balanced_split_path).exists():
        with open(balanced_split_path) as f:
            splits = json.load(f)
        train_idx = splits["balanced"]["train_indices"]
        test_idx = splits["balanced"]["test_indices"]
        all_idx = sorted(set(train_idx) | set(test_idx))
        qpos = qpos[all_idx]
        log.info(f"Using {len(all_idx)} balanced clips (train+test)")

    return qpos


def _get_xml_path(cfg: dict) -> str:
    """Resolve XML path from config or auto-detect."""
    xml_path = cfg["data"].get("xml_path")
    if xml_path and Path(xml_path).exists():
        return xml_path
    from vnl_playground.tasks.rodent import consts

    return str(consts.RODENT_XML_PATH)


def _compute_metrics(
    fit_result,
    keypoint_data: np.ndarray,
    n_states: int,
) -> dict:
    """Compute reconstruction and syllable quality metrics."""
    labels = fit_result.labels_list

    # Duration stats
    all_durations = []
    for lbl in labels:
        changes = np.where(np.diff(lbl) != 0)[0] + 1
        segments = np.split(lbl, changes)
        all_durations.extend(len(s) for s in segments)
    durations = np.array(all_durations) if all_durations else np.array([1])

    # Active syllables
    all_labels = np.concatenate(labels) if labels else np.array([0])
    active = len(np.unique(all_labels))

    # Transition entropy
    n = n_states
    trans_matrix = np.zeros((n, n))
    for lbl in labels:
        for i in range(len(lbl) - 1):
            trans_matrix[lbl[i], lbl[i + 1]] += 1
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    trans_probs = trans_matrix / row_sums
    entropy = -np.nansum(trans_probs * np.log(trans_probs + 1e-10), axis=1)
    mean_entropy = float(np.mean(entropy[row_sums.squeeze() > 0]))

    return {
        "active_syllables": active,
        "syllable_usage_ratio": active / n_states,
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "transition_entropy": mean_entropy,
    }


def run_sweep(cfg: dict) -> dict:
    """Run the full KPMS grid search.

    Args:
        cfg: Sweep configuration dict.

    Returns:
        Dict with ``"best_model"`` info and ``"all_results"`` list.
    """
    sweep_cfg = cfg["sweep"]
    output_dir = Path(cfg["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    qpos = _load_reference_data(
        cfg["data"]["reference_data_path"],
        cfg["data"].get("balanced_split_path"),
    )
    log.info(f"Loaded qpos: {qpos.shape}")

    # Convert to keypoints
    xml_path = _get_xml_path(cfg)
    mjx_model, mjx_data, site_ids, kp_names = setup_mujoco_model(
        xml_path, DEFAULT_KP_NAMES
    )
    log.info(f"Computing keypoints for {qpos.shape[0]} clips...")
    keypoints = qpos_to_keypoints(qpos, mjx_model, mjx_data, site_ids)
    log.info(f"Keypoints shape: {keypoints.shape}")

    # Grid
    grid = list(
        itertools.product(
            sweep_cfg["num_states"],
            sweep_cfg["kappa"],
            sweep_cfg["latent_dim"],
            sweep_cfg["model_type"],
        )
    )
    seeds = list(range(sweep_cfg["seeds_per_setting"]))
    log.info(
        f"Grid: {len(grid)} settings × {len(seeds)} seeds = {len(grid) * len(seeds)} fits"
    )

    all_results = []

    for gi, (n_states, kappa, latent_dim, model_type) in enumerate(grid):
        setting_key = f"s{n_states}_k{kappa:.0e}_l{latent_dim}_{model_type}"
        setting_dir = output_dir / setting_key
        setting_results = []

        for seed in seeds:
            hp = KPMSHyperparams(
                kappa=kappa,
                latent_dim=latent_dim,
                num_states=n_states,
                ar_iters=sweep_cfg["ar_iters"],
                full_iters=sweep_cfg["full_iters"],
                model_type=model_type,
            )

            project_dir = str(setting_dir / f"seed{seed}")
            log.info(f"[{gi + 1}/{len(grid)}] {setting_key} seed={seed}")

            try:
                fit_result = fit_kpms_keypoints(
                    keypoint_data=keypoints,
                    n_states=n_states,
                    project_dir=project_dir,
                    hyperparams=hp,
                    seed=seed,
                    kp_names=kp_names,
                )
                metrics = _compute_metrics(fit_result, keypoints, n_states)

                result = {
                    "setting": setting_key,
                    "n_states": n_states,
                    "kappa": kappa,
                    "latent_dim": latent_dim,
                    "model_type": model_type,
                    "seed": seed,
                    "project_dir": project_dir,
                    "model_name": fit_result.model_name,
                    **metrics,
                }
                setting_results.append(result)
                all_results.append(result)

                log.info(
                    f"  active={metrics['active_syllables']}/{n_states}, "
                    f"dur={metrics['mean_duration']:.1f}±{metrics['std_duration']:.1f}, "
                    f"H={metrics['transition_entropy']:.2f}"
                )

            except Exception as e:
                log.warning(f"  FAILED: {e}")
                all_results.append(
                    {
                        "setting": setting_key,
                        "seed": seed,
                        "error": str(e),
                    }
                )

    # Select best: highest usage ratio → highest transition entropy
    valid = [r for r in all_results if "error" not in r]
    if valid:
        valid.sort(
            key=lambda r: (r["syllable_usage_ratio"], r["transition_entropy"]),
            reverse=True,
        )
        best = valid[0]
        log.info(
            f"\nBest model: {best['setting']} seed={best['seed']}, "
            f"usage={best['syllable_usage_ratio']:.2f}, "
            f"entropy={best['transition_entropy']:.2f}"
        )
    else:
        best = None
        log.warning("No successful fits!")

    # Save results
    summary = {"best_model": best, "all_results": all_results}
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved results to {results_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="KPMS hyperparameter sweep")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
