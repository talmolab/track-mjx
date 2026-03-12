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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
from moseq_jax.kpms.fit_kpms import fit_kpms_keypoints

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_config(config_path: str | None = None) -> dict:
    """Load sweep config from YAML."""
    if config_path is None:
        config_path = str(MOSEQ_DIR / "configs" / "kpms_sweep.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _setup_stac_model(
    h5_path: str,
    xml_path: str,
) -> tuple["mujoco.MjModel", "mujoco.MjData", list[int], list[str]]:
    """Set up MuJoCo model with keypoint sites from stac-mjx config.

    Replicates what ``stac_mjx.Stac._build_body_spec()`` does:

    1. Load the base XML and add keypoint sites to the correct bodies.
    2. Override site positions with optimized offsets from the H5 file.

    This avoids a dependency on ``stac_mjx`` while producing identical
    FK output.

    Args:
        h5_path: Path to reference clips H5 (must contain ``config``,
            ``offsets``, ``kp_names``).
        xml_path: Path to the stac-mjx rodent XML model.

    Returns:
        ``(mj_model, mj_data, site_ids, kp_names)`` where ``site_ids``
        and ``kp_names`` are both in the H5's alphabetical order.
    """
    import h5py
    import mujoco
    import yaml

    with h5py.File(h5_path, "r") as f:
        cfg_yaml = f["config"][()].decode()
        offsets = f["offsets"][:]  # [K, 3] in kp_names order
        kp_names = [n.decode() for n in f["kp_names"][:]]

    cfg = yaml.safe_load(cfg_yaml)
    kmp = cfg["model"]["KEYPOINT_MODEL_PAIRS"]

    # Build model with keypoint sites using MjSpec
    spec = mujoco.MjSpec.from_file(xml_path)
    name_to_offset = {name: offsets[i] for i, name in enumerate(kp_names)}

    for kp_name, body_name in kmp.items():
        parent = spec.body(body_name)
        pos = name_to_offset[kp_name].tolist()
        parent.add_site(
            name=kp_name,
            size=[0.005, 0.005, 0.005],
            pos=pos,
            group=3,
        )

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)

    # Get site IDs in kp_names (alphabetical) order
    site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in kp_names
    ]

    return mj_model, mj_data, site_ids, kp_names


def _qpos_to_keypoints_fk(
    qpos: np.ndarray,
    mj_model: "mujoco.MjModel",
    mj_data: "mujoco.MjData",
    site_ids: list[int],
) -> np.ndarray:
    """Compute keypoints from qpos via MuJoCo forward kinematics.

    Uses batched JAX-vmapped FK for speed, matching the reference
    implementation in ``TopoVNL/topo_mimic/benchmark/hyperparam_sweep/
    keypoint_loader.py``.

    Args:
        qpos: Joint positions ``[N_total_frames, nq]``.
        mj_model: Compiled MuJoCo model with keypoint sites.
        mj_data: MuJoCo data template.
        site_ids: Site indices for keypoints.

    Returns:
        Keypoint positions ``[N_total_frames, K, 3]``.
    """
    import jax.numpy as jnp
    import mujoco.mjx as mjx

    site_ids_array = jnp.array(site_ids)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    @jax.jit
    def forward_kinematics(qpos_single):
        data = mjx_data.replace(qpos=qpos_single)
        data = mjx.forward(mjx_model, data)
        return data.site_xpos[site_ids_array]

    # Process in batches to avoid OOM
    batch_fk = jax.vmap(forward_kinematics)
    n_total = qpos.shape[0]
    batch_size = 1000
    all_kps = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_qpos = jnp.array(qpos[start:end])
        batch_kps = batch_fk(batch_qpos)
        all_kps.append(np.array(batch_kps))

    return np.concatenate(all_kps, axis=0)


def _load_keypoints(
    data_path: str,
    stac_xml_path: str,
    balanced_split_path: str | None = None,
    n_frames_per_clip: int = 250,
) -> tuple[np.ndarray, list[str]]:
    """Load qpos from H5 and convert to keypoints via FK.

    Matches the reference implementation in ``TopoVNL/topo_mimic/benchmark/
    hyperparam_sweep/kpms/kpms_sweep.py::load_qpos_and_convert_keypoints``:

    1. Load ``qpos`` from the reference clips H5.
    2. Set up a MuJoCo model with keypoint sites (replicating stac-mjx).
    3. Run forward kinematics to get keypoint positions ``[N, T, K, 3]``.
    4. Optionally filter to balanced clips.

    Returns:
        ``(keypoints, kp_names)`` where keypoints has shape ``[N, T, K, 3]``
        with columns in the H5's kp_names order.
    """
    import h5py

    with h5py.File(data_path, "r") as f:
        qpos = f["qpos"][:]  # [N_total_frames, nq]

    log.info(f"Loaded qpos: {qpos.shape}")

    # Set up model with keypoint sites
    mj_model, mj_data, site_ids, kp_names = _setup_stac_model(data_path, stac_xml_path)
    log.info(f"Set up model with {len(site_ids)} keypoint sites")

    # FK: qpos → keypoints
    log.info("Running forward kinematics...")
    kp_3d = _qpos_to_keypoints_fk(qpos, mj_model, mj_data, site_ids)
    log.info(f"FK output: {kp_3d.shape}")

    # Reshape from [N_total_frames, K, 3] to [N_clips, T, K, 3]
    n_total = kp_3d.shape[0]
    n_clips = n_total // n_frames_per_clip
    keypoints = kp_3d[: n_clips * n_frames_per_clip].reshape(
        n_clips, n_frames_per_clip, *kp_3d.shape[1:]
    )

    if balanced_split_path and Path(balanced_split_path).exists():
        with open(balanced_split_path) as f:
            splits = json.load(f)
        train_idx = splits["balanced"]["train_indices"]
        test_idx = splits["balanced"]["test_indices"]
        all_idx = sorted(set(train_idx) | set(test_idx))
        keypoints = keypoints[all_idx]
        log.info(f"Using {len(all_idx)} balanced clips (train+test)")

    return keypoints, kp_names


def _reconstruct_keypoints(
    fit_result,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract original and reconstructed keypoints from KPMS fit.

    Uses ``data["Y"]`` from ``format_data`` as original and
    ``estimate_coordinates`` + ``unbatch`` for reconstruction, matching
    the reference implementation in ``TopoVNL/topo_mimic/benchmark/
    hyperparam_sweep/kpms/kpms_sweep.py``.

    Both outputs are flattened to ``[N, T, K*D]`` for per-dimension
    comparison.

    Args:
        fit_result: KPMSFitResult from fitting.

    Returns:
        ``(original, reconstructed)`` both as ``[N, T, K*D]``.
    """
    import jax.numpy as jnp
    from jax_moseq.models.keypoint_slds.alignment import estimate_coordinates
    from jax_moseq.utils import unbatch

    # Original: data["Y"] from format_data [n_segs, seg_length, K, D]
    Y_data = fit_result.data["Y"]
    if isinstance(Y_data, dict):
        keys = sorted(Y_data.keys())
        original = np.stack([np.array(Y_data[k]) for k in keys], axis=0)
    else:
        original = np.array(Y_data)

    if original.ndim == 4:
        n, t, k, d = original.shape
        original = original.reshape(n, t, k * d)

    # Reconstruction: estimate_coordinates + unbatch
    model = fit_result.model
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"]),
    )
    coords_dict = unbatch(np.array(Y_est), *fit_result.metadata)
    rec_keys = sorted(coords_dict.keys())
    recon_list = []
    for rk in rec_keys:
        arr = np.array(coords_dict[rk])
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.ndim == 4:
            arr = arr.reshape(arr.shape[0], -1)
        recon_list.append(arr)
    reconstructed = np.stack(recon_list, axis=0)
    if reconstructed.ndim == 4:
        n, t, k, d = reconstructed.shape
        reconstructed = reconstructed.reshape(n, t, k * d)

    return original, reconstructed


def _plot_keypoint_sample(
    Y: np.ndarray,
    kp_names: list[str],
    clip_idx: int = 0,
    n_keypoints: int = 6,
) -> plt.Figure:
    """Plot raw keypoint trajectories (X/Y/Z) for one clip.

    Args:
        Y: Keypoints ``[T, K, 3]``.
        kp_names: Keypoint names.
        clip_idx: Clip index (for title).
        n_keypoints: How many keypoints to show.
    """
    kp_indices = np.linspace(0, len(kp_names) - 1, n_keypoints, dtype=int)
    dims = ["X", "Y", "Z"]

    fig, axes = plt.subplots(n_keypoints, 3, figsize=(14, 2.5 * n_keypoints))
    if n_keypoints == 1:
        axes = axes[np.newaxis, :]

    for row, ki in enumerate(kp_indices):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            ax.plot(Y[:, ki, col], linewidth=1)
            if row == 0:
                ax.set_title(dim)
            if col == 0:
                ax.set_ylabel(kp_names[ki], fontsize=8)
            if row == n_keypoints - 1:
                ax.set_xlabel("Frame")
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Clip {clip_idx}: Input Keypoints", fontsize=11)
    fig.tight_layout()
    return fig


def _plot_reconstruction(
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
    clip_idx: int = 0,
    n_dims: int = 5,
    title_prefix: str = "Reconstruction",
    max_frames: int = 500,
) -> plt.Figure:
    """Plot original vs reconstructed timeseries for flattened dimensions.

    Matches the reference ``plot_reconstruction_timeseries`` from
    ``TopoVNL/topo_mimic/benchmark/hyperparam_sweep/visualization.py``.

    Args:
        original_data: Original data ``[N, T, K*D]`` (flattened dims).
        reconstructed_data: Reconstructed data ``[N, T, K*D]``.
        clip_idx: Which clip to plot.
        n_dims: Number of flattened dimensions to show.
        title_prefix: Title prefix string.
        max_frames: Maximum frames to show.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    n_frames = min(original_data.shape[1], reconstructed_data.shape[1], max_frames)
    t = np.arange(n_frames)

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]

        orig = original_data[clip_idx, :n_frames, dim_idx]
        recon = reconstructed_data[clip_idx, :n_frames, dim_idx]

        ax.plot(t, orig, label="Original", color="blue", alpha=0.7, linewidth=1)
        ax.plot(t, recon, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

        mse = np.mean((orig - recon) ** 2)
        ax.set_ylabel(f"Dim {dim_idx}")
        ax.set_title(f"Dimension {dim_idx} (MSE: {mse:.4f})", fontsize=10)
        if dim_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"{title_prefix} - Clip {clip_idx}", fontsize=12)
    plt.tight_layout()
    return fig


RODENT_SKELETON = [
    ("Snout", "SpineF"),
    ("SpineF", "SpineM"),
    ("SpineM", "SpineL"),
    ("SpineL", "TailBase"),
    ("SpineF", "ShoulderL"),
    ("ShoulderL", "ElbowL"),
    ("ElbowL", "WristL"),
    ("WristL", "HandL"),
    ("SpineF", "ShoulderR"),
    ("ShoulderR", "ElbowR"),
    ("ElbowR", "WristR"),
    ("WristR", "HandR"),
    ("SpineL", "HipL"),
    ("HipL", "KneeL"),
    ("KneeL", "AnkleL"),
    ("AnkleL", "FootL"),
    ("SpineL", "HipR"),
    ("HipR", "KneeR"),
    ("KneeR", "AnkleR"),
    ("AnkleR", "FootR"),
    ("Snout", "EarL"),
    ("Snout", "EarR"),
]


def _render_keypoint_video(
    Y: np.ndarray,
    output_path: str,
    kp_names: list[str],
    Y_bar: np.ndarray | None = None,
    codes: np.ndarray | None = None,
    fps: int = 30,
    title: str = "Keypoint 3D",
    elev: float = 20,
    azim: float = -60,
) -> str:
    """Render 3D keypoint skeleton animation as mp4.

    Draws keypoints as dots connected by skeleton edges (``RODENT_SKELETON``).
    Optionally overlays reconstructed keypoints in red with syllable labels.

    Adapted from the reference implementation in
    ``TopoVNL/topo_mimic/benchmark/hyperparam_sweep/visualization.py``.

    Args:
        Y: Original keypoints ``[T, K, 3]``.
        output_path: Path to save mp4 video.
        kp_names: Keypoint names (used to resolve skeleton edges).
        Y_bar: Reconstructed keypoints ``[T, K, 3]`` (optional overlay).
        codes: Syllable labels ``[T]`` (shown in frame text).
        fps: Frames per second for output video.
        title: Video title.
        elev: 3D view elevation angle.
        azim: 3D view azimuth angle.

    Returns:
        The output path.
    """
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Point matplotlib to imageio_ffmpeg's bundled ffmpeg binary
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    T, K, _ = Y.shape

    # Build skeleton edge index pairs from keypoint names
    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    edges = []
    for a, b in RODENT_SKELETON:
        if a in name_to_idx and b in name_to_idx:
            edges.append((name_to_idx[a], name_to_idx[b]))

    # Compute axis limits (equal range, centered)
    all_pts = Y if Y_bar is None else np.concatenate([Y, Y_bar], axis=0)
    mins = all_pts.reshape(-1, 3).min(0)
    maxs = all_pts.reshape(-1, 3).max(0)
    rng = float(np.max(maxs - mins))
    ctr = (mins + maxs) / 2
    pad = 0.05 * rng if rng > 0 else 1e-3

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    # Initial frame: original keypoints + skeleton
    (pts,) = ax.plot(Y[0, :, 0], Y[0, :, 1], Y[0, :, 2], "o", ms=4, color="blue")
    lines = [
        ax.plot(
            [Y[0, i, 0], Y[0, j, 0]],
            [Y[0, i, 1], Y[0, j, 1]],
            [Y[0, i, 2], Y[0, j, 2]],
            "-",
            color="gray",
            linewidth=2,
        )[0]
        for i, j in edges
    ]

    # Optional: reconstructed overlay
    pts_recon = None
    lines_recon = []
    if Y_bar is not None:
        (pts_recon,) = ax.plot(
            Y_bar[0, :, 0],
            Y_bar[0, :, 1],
            Y_bar[0, :, 2],
            "o",
            ms=3,
            color="red",
        )
        lines_recon = [
            ax.plot(
                [Y_bar[0, i, 0], Y_bar[0, j, 0]],
                [Y_bar[0, i, 1], Y_bar[0, j, 1]],
                [Y_bar[0, i, 2], Y_bar[0, j, 2]],
                "-",
                color="salmon",
                linewidth=1,
                alpha=0.6,
            )[0]
            for i, j in edges
        ]

    ax.set_xlim(ctr[0] - rng / 2 - pad, ctr[0] + rng / 2 + pad)
    ax.set_ylim(ctr[1] - rng / 2 - pad, ctr[1] + rng / 2 + pad)
    ax.set_zlim(ctr[2] - rng / 2 - pad, ctr[2] + rng / 2 + pad)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    frame_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=10)

    def update(t):
        pts.set_data(Y[t, :, 0], Y[t, :, 1])
        pts.set_3d_properties(Y[t, :, 2])
        for ln, (i, j) in zip(lines, edges):
            ln.set_data([Y[t, i, 0], Y[t, j, 0]], [Y[t, i, 1], Y[t, j, 1]])
            ln.set_3d_properties([Y[t, i, 2], Y[t, j, 2]])

        if pts_recon is not None:
            pts_recon.set_data(Y_bar[t, :, 0], Y_bar[t, :, 1])
            pts_recon.set_3d_properties(Y_bar[t, :, 2])
            for ln, (i, j) in zip(lines_recon, edges):
                ln.set_data(
                    [Y_bar[t, i, 0], Y_bar[t, j, 0]],
                    [Y_bar[t, i, 1], Y_bar[t, j, 1]],
                )
                ln.set_3d_properties([Y_bar[t, i, 2], Y_bar[t, j, 2]])

        label = f"Frame {t}/{T}"
        if codes is not None and t < len(codes):
            label += f"  code={codes[t]}"
        frame_text.set_text(label)
        return [pts, *lines, frame_text]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    log.info(f"Saved 3D keypoint video: {output_path}")
    return output_path


def _compute_metrics(
    fit_result,
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_states: int,
) -> dict:
    """Compute reconstruction and syllable quality metrics.

    Args:
        fit_result: KPMSFitResult from fitting.
        original: Flattened original data ``[N, T, K*D]``.
        reconstructed: Flattened reconstructed data ``[N, T, K*D]``.
        n_states: Number of syllable states.

    Returns:
        Dict with reconstruction_mse, syllable stats, and transition entropy.
    """
    labels = fit_result.labels_list

    min_len = min(original.shape[1], reconstructed.shape[1])
    mse = float(np.mean((original[:, :min_len] - reconstructed[:, :min_len]) ** 2))

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
        "reconstruction_mse": mse,
        "active_syllables": active,
        "syllable_usage_ratio": active / n_states,
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "transition_entropy": mean_entropy,
    }


def _log_sweep_summary_plots(
    valid: list[dict], setting_seeds: dict[str, list[dict]]
) -> None:
    """Create and log summary plots to WandB."""
    import wandb

    # Aggregate per-setting means
    settings = []
    usage_means = []
    entropy_means = []
    for setting_key, seeds_results in setting_seeds.items():
        settings.append(setting_key)
        usage_means.append(np.mean([r["syllable_usage_ratio"] for r in seeds_results]))
        entropy_means.append(np.mean([r["transition_entropy"] for r in seeds_results]))

    # Bar chart: usage ratio per setting
    fig, ax = plt.subplots(figsize=(max(8, len(settings) * 0.5), 5))
    ax.bar(range(len(settings)), usage_means)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, rotation=90, fontsize=6)
    ax.set_ylabel("Usage Ratio")
    ax.set_title("Syllable Usage Ratio per Setting")
    fig.tight_layout()
    wandb.log({"summary/usage_ratio_bar": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # Bar chart: transition entropy per setting
    fig, ax = plt.subplots(figsize=(max(8, len(settings) * 0.5), 5))
    ax.bar(range(len(settings)), entropy_means)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, rotation=90, fontsize=6)
    ax.set_ylabel("Transition Entropy")
    ax.set_title("Transition Entropy per Setting")
    fig.tight_layout()
    wandb.log({"summary/transition_entropy_bar": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # Scatter: usage ratio vs transition entropy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(usage_means, entropy_means, s=40)
    for i, s in enumerate(settings):
        ax.annotate(s, (usage_means[i], entropy_means[i]), fontsize=5)
    ax.set_xlabel("Usage Ratio")
    ax.set_ylabel("Transition Entropy")
    ax.set_title("Usage Ratio vs Transition Entropy")
    fig.tight_layout()
    wandb.log({"summary/usage_vs_entropy": wandb.Image(fig)}, commit=False)
    plt.close(fig)


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

    # Load qpos and convert to keypoints via FK (matching the reference)
    keypoints, kp_names = _load_keypoints(
        cfg["data"]["reference_data_path"],
        cfg["data"]["stac_xml_path"],
        cfg["data"].get("balanced_split_path"),
    )
    log.info(f"Loaded keypoints: {keypoints.shape}, {len(kp_names)} keypoints")

    # WandB init
    wandb_enabled = False
    if cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb

            wandb.init(
                project=cfg["wandb"].get("project", "moseq_experiments"),
                entity=cfg["wandb"].get("entity"),
                group=cfg["wandb"].get("group", "kpms_sweep"),
                name=f"kpms_sweep_{datetime.now().strftime('%y%m%d_%H%M%S')}",
                config=cfg["sweep"],
            )
            wandb_enabled = True
        except Exception as e:
            log.warning(f"Failed to init WandB: {e}")

    # Log input keypoint sample — first thing in wandb
    if wandb_enabled:
        import wandb

        fig = _plot_keypoint_sample(keypoints[0], kp_names, clip_idx=0)
        input_log = {"input/keypoint_sample": wandb.Image(fig)}
        plt.close(fig)

        # Input keypoint video (3D skeleton animation, clip 0)
        try:
            input_video_path = str(output_dir / "input_keypoint_clip0.mp4")
            _render_keypoint_video(
                keypoints[0],
                input_video_path,
                kp_names=kp_names,
                title="Input clip 0",
            )
            input_log["input/keypoint_video"] = wandb.Video(
                input_video_path, fps=30, format="mp4"
            )
        except Exception as ve:
            log.warning(f"Input video rendering failed: {ve}")

        wandb.log(input_log)

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
                original, reconstructed = _reconstruct_keypoints(fit_result)
                metrics = _compute_metrics(
                    fit_result, original, reconstructed, n_states
                )

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

                if wandb_enabled:
                    import wandb

                    # Reconstruction example (clip 0)
                    fig = _plot_reconstruction(
                        original,
                        reconstructed,
                        clip_idx=0,
                        title_prefix=f"{setting_key} seed={seed}",
                    )

                    log_dict = {
                        "sweep/n_states": n_states,
                        "sweep/kappa": kappa,
                        "sweep/latent_dim": latent_dim,
                        "sweep/model_type": model_type,
                        "sweep/seed": seed,
                        "sweep/reconstruction_mse": metrics["reconstruction_mse"],
                        "sweep/active_syllables": metrics["active_syllables"],
                        "sweep/usage_ratio": metrics["syllable_usage_ratio"],
                        "sweep/mean_duration": metrics["mean_duration"],
                        "sweep/std_duration": metrics["std_duration"],
                        "sweep/transition_entropy": metrics["transition_entropy"],
                        "sweep/reconstruction_example": wandb.Image(fig),
                    }

                    # 3D keypoint skeleton video (input only, with syllable codes)
                    # NOTE: reconstruction overlay removed — KPMS states are
                    # Gibbs samples (jittery), not smooth MAP estimates.
                    # Reconstruction quality is shown via the 2D timeseries
                    # plot above (sweep/reconstruction_example).
                    try:
                        video_path = str(
                            setting_dir / f"seed{seed}" / f"keypoint_video_clip0.mp4"
                        )
                        _render_keypoint_video(
                            keypoints[0],
                            video_path,
                            kp_names=kp_names,
                            codes=fit_result.labels_list[0],
                            title=f"{setting_key} s{seed}",
                        )
                        log_dict["sweep/keypoint_video"] = wandb.Video(
                            video_path, fps=30, format="mp4"
                        )
                    except Exception as ve:
                        log.warning(f"  Video rendering failed: {ve}")

                    wandb.log(log_dict)
                    plt.close(fig)

                log.info(
                    f"  MSE={metrics['reconstruction_mse']:.4f}, "
                    f"active={metrics['active_syllables']}/{n_states}, "
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

    # Compute EML per setting (requires ≥2 seeds), then select best
    valid = [r for r in all_results if "error" not in r]

    setting_seeds = defaultdict(list)
    for r in valid:
        setting_seeds[r["setting"]].append(r)

    for setting_key, seeds_results in setting_seeds.items():
        if len(seeds_results) >= 2:
            checkpoint_paths = [
                os.path.join(r["project_dir"], r["model_name"], "checkpoint.h5")
                for r in seeds_results
            ]
            try:
                from keypoint_moseq.fitting import expected_marginal_likelihoods

                scores, std_errors = expected_marginal_likelihoods(
                    checkpoint_paths=checkpoint_paths
                )
                for i, r in enumerate(seeds_results):
                    r["eml_score"] = float(scores[i])
                    r["eml_stderr"] = float(std_errors[i])
                if wandb_enabled:
                    import wandb

                    for i, r in enumerate(seeds_results):
                        wandb.log(
                            {
                                "eml/setting": setting_key,
                                "eml/seed": r["seed"],
                                "eml/score": r["eml_score"],
                                "eml/stderr": r["eml_stderr"],
                            }
                        )
                log.info(
                    f"  EML for {setting_key}: "
                    f"scores={[f'{s:.2f}' for s in scores]}"
                )
            except Exception as e:
                log.warning(f"  EML computation failed for {setting_key}: {e}")

    # Select best: lowest MSE → highest EML → highest usage ratio
    # Filter to reasonable duration range (3-100 frames) when possible
    MIN_DUR, MAX_DUR = 3.0, 100.0
    reasonable = [r for r in valid if MIN_DUR <= r["mean_duration"] <= MAX_DUR]
    candidates = reasonable if reasonable else valid

    if candidates:
        candidates.sort(
            key=lambda r: (
                r["reconstruction_mse"],  # lower is better
                -r.get("eml_score", float("-inf")),  # higher is better
                -r["syllable_usage_ratio"],  # higher is better
            ),
        )
        best = candidates[0]
        log.info(
            f"\nBest model: {best['setting']} seed={best['seed']}, "
            f"MSE={best['reconstruction_mse']:.4f}, "
            f"EML={best.get('eml_score', 'N/A')}, "
            f"usage={best['syllable_usage_ratio']:.2f}, "
            f"dur={best['mean_duration']:.1f}"
        )
        if reasonable:
            log.info(
                f"  ({len(reasonable)}/{len(valid)} models had "
                f"mean duration in [{MIN_DUR}, {MAX_DUR}] frames)"
            )
    else:
        best = None
        log.warning("No successful fits!")

    # Summary plots
    if wandb_enabled and valid:
        import wandb

        try:
            _log_sweep_summary_plots(valid, setting_seeds)
        except Exception as e:
            log.warning(f"Failed to create summary plots: {e}")

    # Best model summary
    if wandb_enabled and best:
        import wandb

        wandb.run.summary.update(
            {
                "best/setting": best["setting"],
                "best/n_states": best["n_states"],
                "best/kappa": best["kappa"],
                "best/latent_dim": best["latent_dim"],
                "best/model_type": best["model_type"],
                "best/reconstruction_mse": best["reconstruction_mse"],
                "best/usage_ratio": best["syllable_usage_ratio"],
                "best/transition_entropy": best["transition_entropy"],
            }
        )
        if "eml_score" in best:
            wandb.run.summary["best/eml_score"] = best["eml_score"]
        wandb.finish()

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
