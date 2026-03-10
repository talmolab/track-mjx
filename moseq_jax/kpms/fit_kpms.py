"""Single KPMS fit on keypoint data.

**CRITICAL**: This module sets ``jax_enable_x64 = True`` at import time,
which is process-global and irreversible.  It MUST run in a separate process
from the RL training (which requires float32 JAX).
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms

    HAS_KPMS = True
except ImportError:
    HAS_KPMS = False

from moseq_jax.kpms.config import KPMSHyperparams
from moseq_jax.kpms.keypoint_loader import prepare_keypoints_for_kpms


@dataclass
class KPMSFitResult:
    """Result of a single KPMS fit.

    Attributes:
        model: Model checkpoint dict.
        results: Per-recording extraction results.
        data: Formatted data dict.
        metadata: Format metadata tuple.
        model_name: Name of this model.
        project_dir: Project directory path.
        labels_list: Per-recording syllable label arrays.
    """

    model: dict
    results: dict
    data: dict
    metadata: tuple
    model_name: str
    project_dir: str
    labels_list: list[np.ndarray]


def fit_kpms_keypoints(
    keypoint_data: np.ndarray,
    n_states: int,
    project_dir: str,
    hyperparams: KPMSHyperparams,
    seed: int,
    kp_names: list[str] | None = None,
    clear_project: bool = True,
) -> KPMSFitResult:
    """Fit a KPMS model on keypoint trajectories.

    Args:
        keypoint_data: Shape ``[N, T, K, 3]``.
        n_states: Number of syllable states.
        project_dir: Output directory for this fit.
        hyperparams: KPMS hyperparameters.
        seed: Random seed.
        kp_names: Keypoint names (defaults to ``kp_0``, ``kp_1``, ...).
        clear_project: Remove existing project directory before fitting.

    Returns:
        KPMSFitResult with model, results, labels.
    """
    if not HAS_KPMS:
        raise ImportError("keypoint_moseq not installed")

    coordinates, confidences = prepare_keypoints_for_kpms(keypoint_data)

    if kp_names is None:
        n_keypoints = keypoint_data.shape[2]
        kp_names = [f"kp_{i}" for i in range(n_keypoints)]

    project_path = Path(project_dir)
    if clear_project and project_path.exists():
        shutil.rmtree(project_path)
    project_path.mkdir(parents=True, exist_ok=True)

    # Setup project config
    kpms.setup_project(
        project_dir,
        fix_heading=True,
        bodyparts=kp_names,
        skeleton=[],
        fps=50,
        overwrite=True,
    )

    kpms.update_config(
        project_dir,
        fix_heading=True,
        bodyparts=kp_names,
        use_bodyparts=kp_names,
        anterior_bodyparts=[kp_names[0]],
        posterior_bodyparts=[kp_names[-1]],
        verbose=False,
        latent_dim=hyperparams.latent_dim,
        num_states=n_states,
        kappa=hyperparams.kappa,
    )

    config = kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    model_name = f"keypoint_{hyperparams.model_type}_states{n_states}_seed{seed}"

    model = kpms.init_model(
        data=data,
        metadata=metadata,
        states=None,
        **config,
        seed=jax.random.PRNGKey(seed + n_states * 1000),
    )
    model = kpms.update_hypparams(model, kappa=hyperparams.kappa)

    # AR-only fitting
    model = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=project_dir,
        model_name=model_name,
        ar_only=True,
        num_iters=hyperparams.ar_iters,
        num_states=n_states,
        generate_progress_plots=False,
    )[0]

    # Full SLDS fitting (if requested)
    if hyperparams.model_type == "slds":
        model = kpms.update_hypparams(model, kappa=hyperparams.kappa)
        kpms.fit_model(
            model=model,
            data=data,
            metadata=metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=False,
            start_iter=hyperparams.ar_iters,
            num_iters=hyperparams.full_iters,
            generate_progress_plots=False,
        )

    # Post-process
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
    model_ckpt["project_dir"] = project_dir
    model_ckpt["model_name"] = model_name

    results = kpms.extract_results(model_ckpt, metadata, project_dir, model_name)

    rec_names = sorted(results.keys())
    labels_list = [results[rn]["syllable"] for rn in rec_names]

    return KPMSFitResult(
        model=model_ckpt,
        results=results,
        data=data,
        metadata=metadata,
        model_name=model_name,
        project_dir=project_dir,
        labels_list=labels_list,
    )
