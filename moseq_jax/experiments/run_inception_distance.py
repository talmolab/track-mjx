"""Experiment 9: Inception Distance (FID/KID) for generative code models.

Trains a VAE on real mocap qpos, then computes FID/KID between real
distribution and decoder rollouts driven by generative code models
(ARHMM, HMM, TM, etc.).

Usage:
    cd moseq_jax
    python -m experiments.run_inception_distance
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import flax
import flax.serialization
import hydra
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import scipy.linalg
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

MOSEQ_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MOSEQ_DIR.parent
for _p in (str(MOSEQ_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from moseq_env_wrapper import MoSeqImitation

from experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    load_mimic_checkpoint,
    make_inference_fn,
    make_mimic_inference_fn,
    run_rollout,
)
from experiments.shared.vae_network import make_vae_network
from experiments.shared.plotting import set_nature_style, NATURE_COLORS

# Import code generation functions
from experiments.run_code_generation import (
    generate_arhmm_level1_sequences,
    generate_arhmm_level2_sequences,
    generate_empirical_tm_sequences,
    generate_hmm_sequences,
    load_arhmm_params,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 1: Preprocessing functions (adapted from SCAMPER)
# ---------------------------------------------------------------------------


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.stack(
        [r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1
    ).reshape(q.shape[:-1] + (3, 3))


def handle_rotation(data: np.ndarray) -> np.ndarray:
    """Replace root quaternion with yaw-removed 6D rotation representation.

    Indices 3-6 (MuJoCo w,x,y,z quaternion) -> 6D rotation (74 -> 76 dims).
    """
    quat = data[..., 3:7]
    R = quat_to_rotmat(quat)
    heading = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    R_yaw_inv = np.zeros(R.shape[:-2] + (3, 3), dtype=data.dtype)
    R_yaw_inv[..., 0, 0] = cos_h
    R_yaw_inv[..., 0, 1] = -sin_h
    R_yaw_inv[..., 1, 0] = sin_h
    R_yaw_inv[..., 1, 1] = cos_h
    R_yaw_inv[..., 2, 2] = 1.0
    R_local = np.einsum("...ij,...jk->...ik", R_yaw_inv, R)
    col1 = R_local[..., :, 0]
    col2 = R_local[..., :, 1]
    rot_6d = np.concatenate([col1, col2], axis=-1)
    return np.concatenate([data[..., :3], rot_6d, data[..., 7:]], axis=-1)


def preprocess_data(
    data: np.ndarray,
    exclude_xy: bool,
    do_handle_rotation: bool,
) -> np.ndarray:
    """Preprocess qpos data: handle_rotation then exclude_xy."""
    if do_handle_rotation:
        data = handle_rotation(data)
    if exclude_xy:
        data = data[..., 2:]
    return data


def get_joint_start_index(exclude_xy: bool, handle_rotation: bool) -> int:
    """Index where joint dimensions begin in preprocessed data."""
    pos_dims = 1 if exclude_xy else 3
    rot_dims = 6 if handle_rotation else 4
    return pos_dims + rot_dims


def compute_joint_normalization(
    data: np.ndarray,
    joint_start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute z-score stats for joint dimensions from real data."""
    joints = data[..., joint_start:]
    flat = joints.reshape(-1, joints.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_joints(
    data: np.ndarray,
    joint_start: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization to joint dimensions."""
    result = data.copy()
    result[..., joint_start:] = (data[..., joint_start:] - mean) / std
    return result


# ---------------------------------------------------------------------------
# Section 2: VAE training (adapted from SCAMPER)
# ---------------------------------------------------------------------------


@flax.struct.dataclass
class VAETrainingState:
    """Training state for VAE."""

    optimizer_state: optax.OptState
    params: Dict
    epoch: int
    step: int


def vae_loss(
    params: dict,
    apply_fn,
    batch: jnp.ndarray,
    rng: jax.Array,
    beta: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute VAE loss: MSE reconstruction + beta * KL divergence."""
    dropout_rng, sample_rng = jax.random.split(rng)
    reconstruction, mu, logvar = apply_fn(
        params, batch, rng=sample_rng, training=True, rngs={"dropout": dropout_rng}
    )
    flat_batch = batch.reshape(batch.shape[0], -1)
    recon_loss = jnp.mean(jnp.sum((reconstruction - flat_batch) ** 2, axis=-1))
    kl_loss = -0.5 * jnp.mean(
        jnp.sum(1.0 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
    )
    total_loss = recon_loss + beta * kl_loss
    metrics = {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "beta": jnp.array(beta),
    }
    return total_loss, metrics


def create_vae_train_step(apply_fn, optimizer: optax.GradientTransformation):
    """Create JIT-compiled VAE training step."""

    @jax.jit
    def train_step(
        state: VAETrainingState,
        batch: jnp.ndarray,
        rng: jax.Array,
        beta: float,
    ) -> Tuple[VAETrainingState, Dict[str, jnp.ndarray]]:
        (loss, metrics), grads = jax.value_and_grad(vae_loss, has_aux=True)(
            state.params, apply_fn, batch, rng, beta
        )
        updates, new_optimizer_state = optimizer.update(
            grads, state.optimizer_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)
        new_state = VAETrainingState(
            optimizer_state=new_optimizer_state,
            params=new_params,
            epoch=state.epoch,
            step=state.step + 1,
        )
        return new_state, metrics

    return train_step


def train_vae(
    data: np.ndarray,
    input_size: int,
    latent_dim: int,
    encoder_hidden_layer_sizes: Tuple[int, ...],
    decoder_hidden_layer_sizes: Tuple[int, ...] | None,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    dropout_rate: float,
    use_layer_norm: bool,
    target_beta: float,
    beta_warmup_epochs: int,
    seed: int,
) -> Tuple[dict, tuple, Dict, list]:
    """Train VAE. Returns (params, network_fns, final_metrics, epoch_losses)."""
    vae, init_fn, apply_fn, encode_fn = make_vae_network(
        input_size=input_size,
        latent_dim=latent_dim,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
    )

    key = jax.random.PRNGKey(seed)
    init_params = init_fn(key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    state = VAETrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        epoch=0,
        step=0,
    )

    train_step_fn = create_vae_train_step(apply_fn, optimizer)

    flat_data = data.reshape(data.shape[0], -1).astype(np.float32)
    n_samples = len(flat_data)

    rng = np.random.default_rng(seed)
    final_metrics = {}
    epoch_losses = []

    for epoch in tqdm(range(num_epochs), desc=f"VAE training (seed={seed})"):
        if beta_warmup_epochs > 0:
            beta = min(target_beta, target_beta * (epoch + 1) / beta_warmup_epochs)
        else:
            beta = target_beta

        indices = rng.permutation(n_samples)
        epoch_metrics = []
        key, epoch_key = jax.random.split(key)
        n_batches = n_samples // batch_size
        batch_keys = jax.random.split(epoch_key, max(1, n_batches))

        for i in range(n_batches):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size]
            batch = jnp.array(flat_data[batch_idx])
            state, metrics = train_step_fn(state, batch, batch_keys[i], beta)
            epoch_metrics.append({k: float(v) for k, v in metrics.items()})

        state = state.replace(epoch=epoch + 1)

        if epoch_metrics:
            final_metrics = {
                k: float(np.mean([m[k] for m in epoch_metrics]))
                for k in epoch_metrics[0]
            }
            epoch_losses.append({
                "epoch": epoch + 1,
                "recon_loss": final_metrics["recon_loss"],
                "kl_loss": final_metrics["kl_loss"],
                "total_loss": final_metrics["total_loss"],
            })

    return state.params, (vae, init_fn, apply_fn, encode_fn), final_metrics, epoch_losses


def _plot_vae_loss(epoch_losses: list[dict], output_path: str) -> None:
    """Save VAE training loss curve (2-panel: recon+KL, total)."""
    set_nature_style()
    epochs = [e["epoch"] for e in epoch_losses]
    recon = [e["recon_loss"] for e in epoch_losses]
    kl = [e["kl_loss"] for e in epoch_losses]
    total = [e["total_loss"] for e in epoch_losses]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))

    ax1.plot(epochs, recon, color="#0072B2", linewidth=1, label="Recon (MSE)")
    ax1.plot(epochs, kl, color="#D55E00", linewidth=1, label="KL")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Component Losses")
    ax1.legend(frameon=False, fontsize=6)

    ax2.plot(epochs, total, color="#333333", linewidth=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Total Loss")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"VAE loss plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Section 3: Feature extraction + metric computation (from SCAMPER)
# ---------------------------------------------------------------------------


def extract_features(
    params: dict,
    encode_fn,
    data: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Extract mu vectors from data using the trained VAE encoder."""
    flat_data = data.reshape(data.shape[0], -1).astype(np.float32)
    n_samples = len(flat_data)
    mu_batches = []
    for i in range(0, n_samples, batch_size):
        batch = jnp.array(flat_data[i : i + batch_size])
        mu = encode_fn(params, batch)
        mu_batches.append(np.array(mu))
    return np.concatenate(mu_batches, axis=0)


def compute_fid(mu_real: np.ndarray, mu_fake: np.ndarray) -> float:
    """Compute Frechet Inception Distance between two sets of features."""
    mu1 = np.mean(mu_real, axis=0)
    mu2 = np.mean(mu_fake, axis=0)
    sigma1 = np.cov(mu_real, rowvar=False)
    sigma2 = np.cov(mu_fake, rowvar=False)

    diff = mu1 - mu2
    mean_diff_sq = np.dot(diff, diff)

    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_diff_sq + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_kid(
    mu_real: np.ndarray,
    mu_fake: np.ndarray,
    degree: int = 3,
    num_subsets: int = 100,
    subset_size: int | None = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute KID using polynomial kernel with unbiased MMD^2 estimator."""
    n_real = len(mu_real)
    n_fake = len(mu_fake)
    d = mu_real.shape[1]

    if subset_size is None:
        subset_size = min(n_real, n_fake, 1000)

    rng = np.random.default_rng(seed)
    kid_values = []

    for _ in range(num_subsets):
        real_idx = rng.choice(n_real, size=subset_size, replace=(subset_size > n_real))
        fake_idx = rng.choice(n_fake, size=subset_size, replace=(subset_size > n_fake))

        x = mu_real[real_idx]
        y = mu_fake[fake_idx]

        kxx = (x @ x.T / d + 1.0) ** degree
        kyy = (y @ y.T / d + 1.0) ** degree
        kxy = (x @ y.T / d + 1.0) ** degree

        m = subset_size
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)

        mmd2 = (
            np.sum(kxx) / (m * (m - 1))
            + np.sum(kyy) / (m * (m - 1))
            - 2.0 * np.sum(kxy) / (m * m)
        )
        kid_values.append(float(mmd2))

    return float(np.mean(kid_values)), float(np.std(kid_values))


# ---------------------------------------------------------------------------
# Section 4: Plotting
# ---------------------------------------------------------------------------


def create_barplot(
    data: Dict[str, Dict[str, float]],
    metric_name: str,
    reference_value: float | None,
    reference_label: str,
    output_path: str,
) -> None:
    """Create horizontal bar chart for FID or KID results."""
    set_nature_style()
    sorted_items = sorted(data.items(), key=lambda x: x[1]["mean"])
    names = [item[0].replace("_qpos", "") for item in sorted_items]
    means = [item[1]["mean"] for item in sorted_items]
    stds = [item[1]["std"] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5 + 1)))
    y_pos = np.arange(len(names))

    colors = list(NATURE_COLORS.values())
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    ax.barh(
        y_pos, means, xerr=stds, align="center", alpha=0.8, capsize=3, color=bar_colors
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(metric_name)
    ax.set_title(f"{metric_name} by Generative Method")

    if reference_value is not None:
        ax.axvline(
            x=reference_value,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=reference_label,
        )
        ax.legend()

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Section 5: VAE caching
# ---------------------------------------------------------------------------


def _compute_vae_cache_key(
    real_data_shape: tuple,
    latent_dim: int,
    hidden_layers: list[int],
    num_epochs: int,
    beta: float,
    exclude_xy: bool,
    handle_rotation_flag: bool,
    do_normalize_joints: bool,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
    use_layer_norm: bool,
) -> str:
    """Compute deterministic hash key for VAE caching."""
    key_dict = {
        "data_shape": list(real_data_shape),
        "latent_dim": latent_dim,
        "hidden_layers": hidden_layers,
        "num_epochs": num_epochs,
        "beta": beta,
        "exclude_xy": exclude_xy,
        "handle_rotation": handle_rotation_flag,
        "normalize_joints": do_normalize_joints,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "use_layer_norm": use_layer_norm,
    }
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _load_cached_vae(
    cache_dir: Path,
    cache_key: str,
    seed: int,
    input_size: int,
    latent_dim: int,
    hidden_layers: tuple[int, ...],
    dropout_rate: float,
    use_layer_norm: bool,
) -> tuple[dict, tuple] | None:
    """Load cached VAE params. Returns None if not found."""
    cache_path = cache_dir / f"{cache_key}_seed{seed}.msgpack"
    if not cache_path.exists():
        return None

    vae, init_fn, apply_fn, encode_fn = make_vae_network(
        input_size=input_size,
        latent_dim=latent_dim,
        encoder_hidden_layer_sizes=hidden_layers,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
    )

    dummy_params = init_fn(jax.random.PRNGKey(0))
    with open(cache_path, "rb") as f:
        params = flax.serialization.from_bytes(dummy_params, f.read())

    return params, (vae, init_fn, apply_fn, encode_fn)


def _save_vae_cache(
    cache_dir: Path,
    cache_key: str,
    seed: int,
    params: dict,
) -> None:
    """Save VAE params to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}_seed{seed}.msgpack"
    with open(cache_path, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    log.info(f"  VAE cached to: {cache_path}")


# ---------------------------------------------------------------------------
# Section 6: Code generation + rollout collection
# ---------------------------------------------------------------------------


def compute_steps_per_frame(ctrl_dt: float, mocap_hz: float) -> int:
    """Number of control steps per mocap frame."""
    return int(round(1.0 / (mocap_hz * ctrl_dt)))


def _make_generation_cfg(
    cfg: DictConfig,
    clip_length: int,
    n_seqs: int,
) -> DictConfig:
    """Build a config compatible with run_code_generation's generation functions."""
    arhmm_cfg = OmegaConf.to_container(
        cfg.inception_distance.arhmm, resolve=True
    )
    gen_seed = int(cfg.inception_distance.generation.seed)
    tm_cfg = OmegaConf.to_container(
        cfg.inception_distance.generation.get(
            "transition_matrix", {"temperatures": [1.0]}
        ),
        resolve=True,
    )
    hmm_cfg = OmegaConf.to_container(
        cfg.inception_distance.generation.get(
            "hmm_dynamax",
            {
                "num_states_sweep": [4, 8, 16, 32],
                "num_em_iters": 100,
                "train_ratio": 0.8,
                "seed": 0,
            },
        ),
        resolve=True,
    )
    return OmegaConf.create(
        {
            "generation": {
                "sequence_length": clip_length,
                "num_sequences": n_seqs,
                "seed": gen_seed,
                "arhmm": arhmm_cfg,
                "transition_matrix": tm_cfg,
                "hmm_dynamax": hmm_cfg,
            }
        }
    )


def generate_method_codes(
    method_name: str,
    gen_cfg: DictConfig,
    all_codes: np.ndarray,
    num_codes: int,
    n_seqs: int,
    arhmm_params: dict | None = None,
) -> list[np.ndarray]:
    """Generate code sequences at mocap rate for a given method.

    Returns list of N arrays, each shape (clip_length,).
    """
    if method_name == "arhmm_level2":
        assert arhmm_params is not None
        return generate_arhmm_level2_sequences(arhmm_params, gen_cfg)[:n_seqs]
    elif method_name == "arhmm_level1":
        assert arhmm_params is not None
        return generate_arhmm_level1_sequences(arhmm_params, gen_cfg)[:n_seqs]
    elif method_name == "hmm_dynamax":
        return generate_hmm_sequences(all_codes, num_codes, gen_cfg)[:n_seqs]
    elif method_name == "transition_matrix":
        tm_results = generate_empirical_tm_sequences(all_codes, num_codes, gen_cfg)
        first_key = next(iter(tm_results))
        return tm_results[first_key][:n_seqs]
    elif method_name == "uniform_random":
        rng = np.random.RandomState(int(gen_cfg.generation.seed) + 9000)
        clip_len = int(gen_cfg.generation.sequence_length)
        return [rng.randint(0, num_codes, size=clip_len) for _ in range(n_seqs)]
    else:
        raise ValueError(f"Unknown generative method: {method_name}")


def collect_code_driven_rollouts(
    code_sequences_mocap: list[np.ndarray],
    env,
    inf_fn,
    params: tuple,
    ppo_networks,
    use_rnn: bool,
    reference_clips,
    clip_length: int,
    steps_per_frame: int,
    seed: int,
    per_clip_indices: np.ndarray | None = None,
    jit_reset=None,
    jit_step=None,
) -> list[np.ndarray]:
    """Run decoder rollouts with code sequences. Returns list of raw qpos arrays.

    Each returned array has shape (T_i, 74) where T_i is the number of
    control steps before termination (or max_steps if survived).

    Args:
        per_clip_indices: If provided, use clip i's initial qpos from
            reference_clips[per_clip_indices[i]]. Otherwise sample randomly.
    """
    n_clips_ref = reference_clips.qpos.shape[0]
    rng = np.random.RandomState(seed)
    max_control_steps = clip_length * steps_per_frame

    all_qpos = []
    for i, codes_mocap in enumerate(
        tqdm(code_sequences_mocap, desc="Collecting rollouts")
    ):
        if per_clip_indices is not None:
            ref_idx = per_clip_indices[i]
        else:
            ref_idx = rng.randint(0, n_clips_ref)
        initial_qpos = np.array(reference_clips.qpos[ref_idx, 0])

        codes_ctrl = np.repeat(codes_mocap, steps_per_frame)

        key = jax.random.PRNGKey(seed + i)
        result = run_rollout(
            env, inf_fn, params, ppo_networks, use_rnn, key,
            max_steps=max_control_steps,
            code_override=codes_ctrl,
            initial_qpos=initial_qpos,
            reset_clip_idx=0,
            jit_reset=jit_reset,
            jit_step=jit_step,
            ignore_done=True,
        )
        all_qpos.append(result["qpos"])  # (T+1, 74), variable length

    return all_qpos


def collect_mimic_rollouts(
    env,
    inf_fn,
    params: tuple,
    ppo_networks,
    n_clips: int,
    clip_length: int,
    steps_per_frame: int,
    seed: int,
    jit_reset=None,
    jit_step=None,
) -> list[np.ndarray]:
    """Run oracle Mimic-MJX rollouts per clip. Returns list of raw qpos arrays."""
    max_control_steps = clip_length * steps_per_frame

    all_qpos = []
    for i in tqdm(range(n_clips), desc="Mimic-MJX rollouts"):
        key = jax.random.PRNGKey(seed + i)
        result = run_rollout(
            env, inf_fn, params, ppo_networks,
            use_rnn=False, key=key,
            max_steps=max_control_steps,
            reset_clip_idx=i,
            jit_reset=jit_reset,
            jit_step=jit_step,
            model_type="mimic_mjx",
            ignore_done=True,
        )
        all_qpos.append(result["qpos"])

    return all_qpos


def filter_and_truncate(
    raw_qpos_list: list[np.ndarray],
    survival_threshold: int,
    steps_per_frame: int,
) -> tuple[np.ndarray, int, int]:
    """Filter clips by survival and truncate to threshold length.

    Args:
        raw_qpos_list: List of (T_i, 74) arrays at control rate.
        survival_threshold: Minimum control steps to include a clip.
        steps_per_frame: Subsampling factor for mocap rate.

    Returns:
        (qpos_array, n_kept, n_total) where qpos_array is
        (n_kept, mocap_frames, 74) at mocap rate.
    """
    mocap_frames = survival_threshold // steps_per_frame
    kept = []
    for q in raw_qpos_list:
        if len(q) >= survival_threshold:
            q_trunc = q[:survival_threshold]
            q_mocap = q_trunc[::steps_per_frame][:mocap_frames]
            kept.append(q_mocap)
    if len(kept) == 0:
        return np.zeros((0, mocap_frames, 74)), 0, len(raw_qpos_list)
    return np.stack(kept, axis=0), len(kept), len(raw_qpos_list)


# ---------------------------------------------------------------------------
# Section 7: Aggregation + results saving
# ---------------------------------------------------------------------------


def _aggregate_results(
    all_seed_results: list[dict],
    methods: list[str],
) -> dict:
    """Aggregate FID/KID across VAE seeds."""
    aggregated = {}

    # Self metrics
    self_fids = [r["self_fid"] for r in all_seed_results]
    self_kid_means = [r["self_kid_mean"] for r in all_seed_results]
    aggregated["self_comparison"] = {
        "fid_mean": float(np.mean(self_fids)),
        "fid_std": float(np.std(self_fids)),
        "kid_mean": float(np.mean(self_kid_means)),
        "kid_std": float(np.std(self_kid_means)),
    }

    # Split metrics
    split_fids = [r["split_fid"] for r in all_seed_results]
    split_kid_means = [r["split_kid_mean"] for r in all_seed_results]
    aggregated["split_baseline"] = {
        "fid_mean": float(np.mean(split_fids)),
        "fid_std": float(np.std(split_fids)),
        "kid_mean": float(np.mean(split_kid_means)),
        "kid_std": float(np.std(split_kid_means)),
    }

    # Per-method metrics
    for method in methods:
        fids = [r["datasets"][method]["fid"] for r in all_seed_results]
        kid_means = [r["datasets"][method]["kid_mean"] for r in all_seed_results]
        aggregated[method] = {
            "fid_mean": float(np.mean(fids)),
            "fid_std": float(np.std(fids)),
            "kid_mean": float(np.mean(kid_means)),
            "kid_std": float(np.std(kid_means)),
        }

    return aggregated


def _save_csv(csv_path: Path, aggregated: dict) -> None:
    """Save aggregated results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "fid_mean", "fid_std", "kid_mean", "kid_std"])
        for name, vals in aggregated.items():
            writer.writerow(
                [
                    name,
                    f"{vals['fid_mean']:.6f}",
                    f"{vals['fid_std']:.6f}",
                    f"{vals['kid_mean']:.6f}",
                    f"{vals['kid_std']:.6f}",
                ]
            )
    log.info(f"CSV saved to: {csv_path}")


def _create_plots(output_dir: Path, aggregated: dict, methods: list[str]) -> None:
    """Create FID and KID bar plots."""
    plot_data = {
        name: vals
        for name, vals in aggregated.items()
        if name not in ("self_comparison", "split_baseline")
    }

    if not plot_data:
        return

    # FID barplot
    fid_data = {
        name: {"mean": vals["fid_mean"], "std": vals["fid_std"]}
        for name, vals in plot_data.items()
    }
    create_barplot(
        fid_data, "FID", None, "", str(output_dir / "fid_barplot.png"),
    )

    # KID barplot
    kid_data = {
        name: {"mean": vals["kid_mean"], "std": vals["kid_std"]}
        for name, vals in plot_data.items()
    }
    create_barplot(
        kid_data, "KID", None, "", str(output_dir / "kid_barplot.png"),
    )


# ---------------------------------------------------------------------------
# Section 8: Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="inception_distance_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Inception Distance (FID/KID) Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Stage 1: Load checkpoint, codes, reference clips
    # ==================================================================
    log.info("\n--- Loading checkpoint ---")
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        cfg.checkpoint.path, step=cfg.checkpoint.get("step"),
    )
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    c2a_params = (norm_state, policy_params)

    codes_data = np.load(cfg.data.codes_path)
    all_codes = codes_data["all_codes"]  # (N_balanced, 250)

    with open(cfg.data.balanced_split_path) as f:
        splits = json.load(f)
    balanced_indices = np.array(
        splits["balanced"]["train_indices"] + splits["balanced"]["test_indices"]
    )

    clip_length = int(ckpt_cfg.env_config.clip_length)
    ctrl_dt = float(ckpt_cfg.env_config.ctrl_dt)
    mocap_hz = float(ckpt_cfg.env_config.mocap_hz)
    steps_per_frame = compute_steps_per_frame(ctrl_dt, mocap_hz)
    survival_threshold = int(cfg.inception_distance.survival_threshold)
    mocap_eval_frames = survival_threshold // steps_per_frame

    log.info(
        f"Timing: ctrl_dt={ctrl_dt}, mocap_hz={mocap_hz}, "
        f"steps_per_frame={steps_per_frame}"
    )
    log.info(
        f"Survival threshold: {survival_threshold} control steps "
        f"= {mocap_eval_frames} mocap frames"
    )

    # Load balanced clips for rollouts (decoder needs codes)
    balanced_clips = ReferenceClips(
        data_path=cfg.data.reference_data_path,
        n_frames_per_clip=clip_length,
        keep_clips_idx=balanced_indices,
    )

    n_rollouts_cfg = cfg.inception_distance.get("num_rollout_clips", None)
    n_rollouts = int(n_rollouts_cfg) if n_rollouts_cfg else len(all_codes)
    log.info(f"Rollouts per method: {n_rollouts}")

    # ==================================================================
    # Stage 2: Collect ALL rollouts (mimic-mjx first — used as VAE data)
    # ==================================================================
    methods = list(cfg.inception_distance.methods)
    # Ensure mimic_mjx is first so its rollouts are ready for VAE training
    if "mimic_mjx" in methods:
        methods.remove("mimic_mjx")
        methods.insert(0, "mimic_mjx")

    gen_cfg = _make_generation_cfg(cfg, clip_length, n_rollouts)

    # Load ARHMM params if needed
    arhmm_params = None
    if any(m.startswith("arhmm") for m in methods):
        arhmm_params = load_arhmm_params(gen_cfg)

    # Load mimic checkpoint if needed
    mimic_inf_fn = mimic_params = mimic_ppo = None
    if "mimic_mjx" in methods:
        log.info("Loading mimic-mjx checkpoint...")
        mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
            cfg.mimic_checkpoint.path, step=cfg.mimic_checkpoint.get("step"),
        )
        mimic_params = (mimic_norm, mimic_policy)
        mimic_inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)

    # Create env with balanced clips for code-driven rollouts
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg,
        clips=balanced_clips,
        kpms_codes=all_codes,
        code_stack_size=code_stack_size,
    )
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    c2a_inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)

    seed = int(cfg.rollout.seed)
    raw_rollouts = {}  # method -> list of (T_i, 74) arrays
    fake_datasets = {}  # method -> (N_kept, mocap_eval_frames, 74)
    survival_rates = {}  # method -> (n_kept, n_total)

    for method in methods:
        rollout_path = output_dir / f"rollouts_{method}.npz"
        if rollout_path.exists():
            log.info(f"\n--- Loading cached rollouts: {method} ---")
            cached = np.load(rollout_path, allow_pickle=True)
            if "n_clips" not in cached:
                log.warning(f"  Stale cache format for {method}, re-collecting")
                rollout_path.unlink()
            else:
                raw_list = [cached[f"raw_{i}"] for i in range(int(cached["n_clips"]))]
                qpos_arr, n_kept, n_total = filter_and_truncate(
                    raw_list, survival_threshold, steps_per_frame,
                )
                fake_datasets[method] = qpos_arr
                survival_rates[method] = (n_kept, n_total)
                log.info(
                    f"  Loaded from cache: {qpos_arr.shape} "
                    f"({n_kept}/{n_total} survived)"
                )
                continue

        log.info(f"\n--- Collecting: {method} ---")

        if method == "mimic_mjx":
            raw_list = collect_mimic_rollouts(
                env, mimic_inf_fn, mimic_params, mimic_ppo,
                n_clips=n_rollouts,
                clip_length=clip_length,
                steps_per_frame=steps_per_frame,
                seed=seed,
                jit_reset=jit_reset,
                jit_step=jit_step,
            )
        elif method == "decoder_original_codes":
            # Use each balanced clip's real KPMS codes
            codes_mocap = [
                all_codes[i, :clip_length].astype(np.int32)
                for i in range(n_rollouts)
            ]
            per_clip_idx = np.arange(n_rollouts)
            raw_list = collect_code_driven_rollouts(
                codes_mocap, env, c2a_inf_fn, c2a_params, ppo_networks,
                use_rnn, balanced_clips, clip_length, steps_per_frame,
                seed=seed, per_clip_indices=per_clip_idx,
                jit_reset=jit_reset, jit_step=jit_step,
            )
        else:
            # Generative methods (arhmm, hmm, tm, uniform_random)
            codes_mocap = generate_method_codes(
                method, gen_cfg, all_codes, num_codes, n_rollouts,
                arhmm_params=arhmm_params,
            )
            log.info(f"  Generated {len(codes_mocap)} code sequences")
            raw_list = collect_code_driven_rollouts(
                codes_mocap, env, c2a_inf_fn, c2a_params, ppo_networks,
                use_rnn, balanced_clips, clip_length, steps_per_frame,
                seed=seed,
                jit_reset=jit_reset, jit_step=jit_step,
            )

        # Save raw rollouts (variable length, as separate arrays)
        save_dict = {"n_clips": len(raw_list)}
        for i, q in enumerate(raw_list):
            save_dict[f"raw_{i}"] = q
        np.savez_compressed(rollout_path, **save_dict)

        # Filter and truncate
        qpos_arr, n_kept, n_total = filter_and_truncate(
            raw_list, survival_threshold, steps_per_frame,
        )
        fake_datasets[method] = qpos_arr
        survival_rates[method] = (n_kept, n_total)
        log.info(
            f"  {method}: {qpos_arr.shape} "
            f"({n_kept}/{n_total} survived >= {survival_threshold} steps, "
            f"rate={100*n_kept/n_total:.1f}%)"
        )

    # Log survival summary + warnings
    log.info("\n--- Survival Summary ---")
    for method in methods:
        if method in survival_rates:
            n_k, n_t = survival_rates[method]
            rate = 100 * n_k / n_t if n_t > 0 else 0
            msg = f"  {method}: {n_k}/{n_t} ({rate:.1f}%)"
            if n_k < n_t * 0.5:
                log.warning(f"{msg}  ** LOW SURVIVAL — KID may be unreliable **")
            else:
                log.info(msg)

    # ==================================================================
    # Stage 3: Build "real" data from mimic-mjx rollouts
    # ==================================================================
    # The VAE is trained on mimic-mjx rollouts (oracle behavior), not raw
    # mocap.  This gives cleaner training data and makes KID measure
    # "distance from oracle behavior" rather than "distance from noisy mocap."
    if "mimic_mjx" not in fake_datasets:
        raise RuntimeError(
            "mimic_mjx must be in methods — its rollouts are used as VAE training data"
        )
    real_qpos = fake_datasets["mimic_mjx"]  # (N_survived, mocap_eval_frames, 74)
    n_real = len(real_qpos)
    log.info(f"VAE training data: mimic-mjx rollouts, shape {real_qpos.shape}")

    # ==================================================================
    # Stage 4: Preprocess all datasets
    # ==================================================================
    log.info("\n--- Preprocessing ---")
    pp = cfg.inception_distance.preprocessing
    exclude_xy = bool(pp.exclude_xy)
    do_rotation = bool(pp.handle_rotation)
    do_normalize = bool(pp.normalize_joints)

    real_data = preprocess_data(real_qpos, exclude_xy, do_rotation)

    joint_norm_params = None
    if do_normalize:
        joint_start = get_joint_start_index(exclude_xy, do_rotation)
        joint_mean, joint_std = compute_joint_normalization(real_data, joint_start)
        joint_norm_params = (joint_start, joint_mean, joint_std)
        real_data = normalize_joints(real_data, joint_start, joint_mean, joint_std)
        log.info(
            f"  Normalized {real_data.shape[-1] - joint_start} joint dims "
            f"(starting at index {joint_start})"
        )

    processed_fakes = {}
    for method, raw in fake_datasets.items():
        processed = preprocess_data(raw, exclude_xy, do_rotation)
        if joint_norm_params is not None:
            js, jm, jstd = joint_norm_params
            processed = normalize_joints(processed, js, jm, jstd)
        processed_fakes[method] = processed

    input_size = int(np.prod(real_data.shape[1:]))
    log.info(f"Preprocessed real shape: {real_data.shape}, input_size={input_size}")

    # ==================================================================
    # Stage 5: VAE training + feature extraction + metrics
    # ==================================================================
    vae_cfg = cfg.inception_distance.vae
    kid_cfg = cfg.inception_distance.kid

    cache_dir = (
        Path(vae_cfg.cache_dir) if vae_cfg.cache_dir else output_dir / "vae_cache"
    )
    cache_key = _compute_vae_cache_key(
        real_data.shape,
        int(vae_cfg.latent_dim),
        list(vae_cfg.hidden_layers),
        int(vae_cfg.num_epochs),
        float(vae_cfg.beta),
        exclude_xy,
        do_rotation,
        do_normalize,
        float(vae_cfg.learning_rate),
        float(vae_cfg.weight_decay),
        float(vae_cfg.dropout_rate),
        bool(vae_cfg.use_layer_norm),
    )
    log.info(f"VAE cache key: {cache_key}")

    all_seed_results = []

    for seed in vae_cfg.seeds:
        log.info(f"\n{'='*60}")
        log.info(f"SEED {seed}")
        log.info(f"{'='*60}")

        # Try cache
        cached = _load_cached_vae(
            cache_dir,
            cache_key,
            int(seed),
            input_size,
            int(vae_cfg.latent_dim),
            tuple(vae_cfg.hidden_layers),
            float(vae_cfg.dropout_rate),
            bool(vae_cfg.use_layer_norm),
        )

        if cached is not None:
            trained_params, network_fns = cached
            log.info("  Loaded VAE from cache")
            train_metrics = {"cached": True}
        else:
            warmup = vae_cfg.beta_warmup_epochs
            if warmup is None:
                warmup = int(vae_cfg.num_epochs) // 2

            t0 = time.time()
            trained_params, network_fns, train_metrics, epoch_losses = train_vae(
                data=real_data,
                input_size=input_size,
                latent_dim=int(vae_cfg.latent_dim),
                encoder_hidden_layer_sizes=tuple(vae_cfg.hidden_layers),
                decoder_hidden_layer_sizes=None,
                num_epochs=int(vae_cfg.num_epochs),
                batch_size=int(vae_cfg.batch_size),
                learning_rate=float(vae_cfg.learning_rate),
                weight_decay=float(vae_cfg.weight_decay),
                grad_clip_norm=float(vae_cfg.grad_clip_norm),
                dropout_rate=float(vae_cfg.dropout_rate),
                use_layer_norm=bool(vae_cfg.use_layer_norm),
                target_beta=float(vae_cfg.beta),
                beta_warmup_epochs=int(warmup),
                seed=int(seed),
            )
            _save_vae_cache(cache_dir, cache_key, int(seed), trained_params)
            _plot_vae_loss(
                epoch_losses, str(output_dir / f"vae_loss_seed{seed}.png")
            )
            log.info(f"  VAE trained in {time.time() - t0:.1f}s: {train_metrics}")

        _, _, _, encode_fn = network_fns

        # Extract real features
        mu_real = extract_features(trained_params, encode_fn, real_data)
        log.info(f"  Real mu shape: {mu_real.shape}")

        # Self-FID/KID
        self_fid = compute_fid(mu_real, mu_real)
        self_kid_mean, self_kid_std = compute_kid(mu_real, mu_real)
        log.info(f"  Self-FID: {self_fid:.6f}")
        log.info(f"  Self-KID: {self_kid_mean:.6f} +/- {self_kid_std:.6f}")

        # Split-FID/KID (noise floor)
        split_seed = int(kid_cfg.split_seed)
        split_rng = np.random.default_rng(split_seed)
        split_indices = split_rng.permutation(len(mu_real))
        mid = len(mu_real) // 2
        mu_real_a = mu_real[split_indices[:mid]]
        mu_real_b = mu_real[split_indices[mid : 2 * mid]]

        split_fid = compute_fid(mu_real_a, mu_real_b)
        split_kid_mean, split_kid_std = compute_kid(mu_real_a, mu_real_b, seed=split_seed)
        log.info(f"  Split-FID: {split_fid:.6f}")
        log.info(f"  Split-KID: {split_kid_mean:.6f} +/- {split_kid_std:.6f}")

        # Per-method metrics
        seed_results = {
            "seed": int(seed),
            "train_metrics": train_metrics,
            "self_fid": self_fid,
            "self_kid_mean": self_kid_mean,
            "self_kid_std": self_kid_std,
            "split_fid": split_fid,
            "split_kid_mean": split_kid_mean,
            "split_kid_std": split_kid_std,
            "datasets": {},
        }

        for method, processed_fake in processed_fakes.items():
            mu_fake = extract_features(trained_params, encode_fn, processed_fake)

            # Compare real half vs full fake (sizes may differ)
            fid = compute_fid(mu_real_a, mu_fake)
            kid_mean, kid_std = compute_kid(
                mu_real_a,
                mu_fake,
                degree=int(kid_cfg.degree),
                num_subsets=int(kid_cfg.num_subsets),
                subset_size=kid_cfg.get("subset_size", None),
                seed=split_seed,
            )
            seed_results["datasets"][method] = {
                "fid": fid,
                "kid_mean": kid_mean,
                "kid_std": kid_std,
            }
            log.info(
                f"  {method}: FID={fid:.4f}, KID={kid_mean:.6f} +/- {kid_std:.6f}"
            )

        all_seed_results.append(seed_results)

    # ==================================================================
    # Stage 6: Aggregate + save results
    # ==================================================================
    log.info(f"\n{'='*60}")
    log.info("AGGREGATING ACROSS SEEDS")
    log.info(f"{'='*60}")

    all_dataset_names = list(processed_fakes.keys())
    aggregated = _aggregate_results(all_seed_results, all_dataset_names)

    # Save JSON
    output_data = {
        "metadata": {
            "checkpoint_path": str(cfg.checkpoint.path),
            "codes_path": str(cfg.data.codes_path),
            "methods": methods,
            "num_real_clips": n_real,
            "num_rollout_clips": n_rollouts,
            "clip_length": clip_length,
            "latent_dim": int(vae_cfg.latent_dim),
            "hidden_layers": list(vae_cfg.hidden_layers),
            "num_epochs": int(vae_cfg.num_epochs),
            "beta": float(vae_cfg.beta),
            "seeds": list(vae_cfg.seeds),
            "exclude_xy": exclude_xy,
            "handle_rotation": do_rotation,
            "normalize_joints": do_normalize,
            "steps_per_frame": steps_per_frame,
            "timestamp": datetime.now().isoformat(),
        },
        "per_seed_results": all_seed_results,
        "aggregated": aggregated,
    }

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to: {json_path}")

    _save_csv(output_dir / "results.csv", aggregated)
    _create_plots(output_dir, aggregated, all_dataset_names)

    # Print summary
    log.info(f"\n{'='*80}")
    log.info("SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"{'Dataset':<30} {'FID':>14} {'KID':>18}")
    log.info("-" * 80)
    for name, vals in aggregated.items():
        fid_str = f"{vals['fid_mean']:.4f} +/- {vals['fid_std']:.4f}"
        kid_str = f"{vals['kid_mean']:.6f} +/- {vals['kid_std']:.6f}"
        log.info(f"{name:<30} {fid_str:>14} {kid_str:>18}")
    log.info("=" * 80)

    log.info("\n=== Inception Distance Experiment Complete ===")


if __name__ == "__main__":
    main()
