"""Experiment 3: Generative models of KPMS codes + free-loop rollouts.

Four methods: empirical transition matrix, dynamax HMM, ARHMM L1, ARHMM L2.

Usage:
    cd moseq_jax
    python -m experiments.run_code_generation
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
from omegaconf import DictConfig
from scipy.special import softmax

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
from experiments.shared.metrics import compute_transition_matrix, plot_transition_matrix
from experiments.shared.plotting import (
    set_nature_style,
    fig_to_image,
    get_code_colormap,
    NATURE_COLORS,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Method A: Empirical Transition Matrix
# ---------------------------------------------------------------------------


def sample_from_transition_matrix(
    T: np.ndarray,
    length: int,
    temperature: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Sample a code sequence from the empirical transition matrix.

    Args:
        T: Row-normalized transition matrix ``[K, K]``.
        length: Sequence length.
        temperature: Sampling temperature (1.0 = original, <1 = greedy).
        seed: Random seed.

    Returns:
        ``[length]`` int array of code indices.
    """
    rng = np.random.RandomState(seed)
    K = T.shape[0]

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi) / np.abs(pi).sum()

    # Temperature-scaled transition
    log_T = np.log(T + 1e-12)
    T_temp = softmax(log_T / max(temperature, 1e-6), axis=1)

    seq = np.zeros(length, dtype=np.int32)
    seq[0] = rng.choice(K, p=pi)
    for t in range(1, length):
        seq[t] = rng.choice(K, p=T_temp[seq[t - 1]])
    return seq


def generate_empirical_tm_sequences(
    all_codes: np.ndarray,
    num_codes: int,
    cfg: DictConfig,
) -> dict[str, list[np.ndarray]]:
    """Generate sequences via empirical transition matrix."""
    log.info("  Method A: Empirical Transition Matrix")
    T = compute_transition_matrix(list(all_codes), num_codes).astype(float)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-12)

    results = {}
    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)

    for temp in cfg.generation.transition_matrix.temperatures:
        key = f"tm_T{temp}"
        seqs = [
            sample_from_transition_matrix(T_norm, length, temperature=float(temp), seed=seed + i)
            for i in range(n_seq)
        ]
        results[key] = seqs
        log.info(f"    T={temp}: generated {n_seq} sequences")

    return results


# ---------------------------------------------------------------------------
# Method B: HMM via dynamax
# ---------------------------------------------------------------------------


def generate_hmm_sequences(
    all_codes: np.ndarray,
    num_codes: int,
    cfg: DictConfig,
) -> list[np.ndarray]:
    """Fit discrete HMM with dynamax and sample sequences."""
    log.info("  Method B: HMM via dynamax")
    from dynamax.hidden_markov_model import CategoricalHMM

    hmm_cfg = cfg.generation.hmm_dynamax
    seed = int(hmm_cfg.seed)
    n_seq = int(cfg.generation.num_sequences)
    length = int(cfg.generation.sequence_length)

    # Prepare emissions: [n_clips, n_frames, 1]
    emissions = all_codes[:, :, None].astype(np.int32)

    # Train/test split
    n_clips = len(all_codes)
    n_train = int(n_clips * float(hmm_cfg.train_ratio))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_clips)
    train_emissions = emissions[perm[:n_train]]
    test_emissions = emissions[perm[n_train:]]

    best_ll = -np.inf
    best_params = None
    best_K = None

    for K in hmm_cfg.num_states_sweep:
        log.info(f"    Fitting K={K}...")
        hmm = CategoricalHMM(num_states=int(K), emission_dim=1, num_classes=num_codes)
        key = jax.random.PRNGKey(seed)
        params, props = hmm.initialize(key)

        # Fit on concatenated train sequences
        for ep in range(len(train_emissions)):
            try:
                params, lls = hmm.fit_em(
                    params, props, train_emissions[ep],
                    num_iters=int(hmm_cfg.num_em_iters),
                )
            except Exception as e:
                log.warning(f"      EM failed for clip {ep}: {e}")
                continue

        # Evaluate on test set
        test_ll = 0.0
        for ep in range(len(test_emissions)):
            try:
                ll = hmm.marginal_log_prob(params, test_emissions[ep])
                test_ll += float(ll)
            except Exception:
                pass

        log.info(f"      K={K}: test_ll={test_ll:.1f}")
        if test_ll > best_ll:
            best_ll = test_ll
            best_params = params
            best_K = K

    log.info(f"    Best K={best_K} (test_ll={best_ll:.1f})")

    # Sample from best model
    hmm = CategoricalHMM(num_states=best_K, emission_dim=1, num_classes=num_codes)
    seqs = []
    for i in range(n_seq):
        key = jax.random.PRNGKey(seed + 1000 + i)
        states, emissions = hmm.sample(best_params, key, num_timesteps=length)
        seqs.append(np.array(emissions[:, 0], dtype=np.int32))

    return seqs


# ---------------------------------------------------------------------------
# Method C: ARHMM Level 1 (discrete only)
# ---------------------------------------------------------------------------


def load_arhmm_params(cfg: DictConfig) -> dict[str, np.ndarray]:
    """Load ARHMM parameters from keypoint-moseq checkpoint."""
    model_path = cfg.generation.arhmm.model_path
    snapshot = int(cfg.generation.arhmm.snapshot_idx)

    with h5py.File(model_path, "r") as f:
        params = f[f"model_snapshots/{snapshot}/params"]
        result = {
            "pi": params["pi"][()],
            "betas": params["betas"][()],
            "Ab": params["Ab"][()],
            "Q": params["Q"][()],
        }
    log.info(f"  Loaded ARHMM params: pi={result['pi'].shape}, Ab={result['Ab'].shape}")
    return result


def generate_arhmm_level1_sequences(
    arhmm_params: dict[str, np.ndarray],
    cfg: DictConfig,
) -> list[np.ndarray]:
    """Sample discrete-only sequences from ARHMM transition matrix."""
    log.info("  Method C: ARHMM Level 1 (discrete only)")
    pi = arhmm_params["pi"]
    betas = arhmm_params["betas"]
    pi0 = softmax(betas)

    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)

    rng = np.random.RandomState(seed)
    seqs = []
    for i in range(n_seq):
        seq = np.zeros(length, dtype=np.int32)
        seq[0] = rng.choice(len(pi0), p=pi0)
        for t in range(1, length):
            seq[t] = rng.choice(pi.shape[1], p=pi[seq[t - 1]])
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Method D: ARHMM Level 2 (discrete + AR latent)
# ---------------------------------------------------------------------------


def generate_arhmm_level2_sequences(
    arhmm_params: dict[str, np.ndarray],
    cfg: DictConfig,
) -> list[np.ndarray]:
    """Forward-simulate ARHMM with AR latent dynamics."""
    log.info("  Method D: ARHMM Level 2 (discrete + AR latent)")
    pi = arhmm_params["pi"]
    betas = arhmm_params["betas"]
    Ab = arhmm_params["Ab"]  # [K, latent_dim, nlags*latent_dim+1]
    Q = arhmm_params["Q"]  # [K, latent_dim, latent_dim]

    pi0 = softmax(betas)
    K, latent_dim, ar_input_dim = Ab.shape
    nlags = (ar_input_dim - 1) // latent_dim

    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)

    rng = np.random.RandomState(seed + 5000)
    seqs = []

    for si in range(n_seq):
        seq = np.zeros(length, dtype=np.int32)
        x_history = [np.zeros(latent_dim) for _ in range(nlags)]

        # Initial state
        seq[0] = rng.choice(K, p=pi0)
        x_t = rng.multivariate_normal(np.zeros(latent_dim), Q[seq[0]])
        x_history.append(x_t)

        for t in range(1, length):
            # Build AR input: [x_{t-1}, ..., x_{t-nlags}, 1]
            x_lags = np.concatenate(
                [x_history[-(l + 1)] for l in range(nlags)] + [np.ones(1)]
            )

            # Compute log-posterior for each candidate state
            log_probs = np.log(pi[seq[t - 1]] + 1e-30)
            for k in range(K):
                mu_k = Ab[k] @ x_lags
                try:
                    Q_reg = Q[k] + 1e-6 * np.eye(latent_dim)
                    diff = x_t - mu_k
                    Q_inv = np.linalg.solve(Q_reg, diff)
                    sign, logdet = np.linalg.slogdet(Q_reg)
                    log_lik = -0.5 * (diff @ Q_inv + logdet)
                    log_probs[k] += log_lik
                except np.linalg.LinAlgError:
                    pass

            # Sample state from posterior
            probs = softmax(log_probs)
            seq[t] = rng.choice(K, p=probs)

            # Sample latent
            mu = Ab[seq[t]] @ x_lags
            x_t = rng.multivariate_normal(mu, Q[seq[t]] + 1e-6 * np.eye(latent_dim))
            x_history.append(x_t)

        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Control sequences
# ---------------------------------------------------------------------------


def generate_control_sequences(
    all_codes: np.ndarray,
    num_codes: int,
    cfg: DictConfig,
) -> dict[str, list[np.ndarray]]:
    """Generate control code sequences."""
    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)
    rng = np.random.RandomState(seed + 9000)
    controls = {}

    if cfg.controls.get("uniform_random", True):
        controls["uniform_random"] = [
            rng.randint(0, num_codes, size=length) for _ in range(n_seq)
        ]
    if cfg.controls.get("single_code", True):
        codes_to_use = rng.choice(num_codes, size=n_seq, replace=True)
        controls["single_code"] = [
            np.full(length, c, dtype=np.int32) for c in codes_to_use
        ]
    if cfg.controls.get("reversed", True):
        real_seqs = all_codes[rng.choice(len(all_codes), size=n_seq)]
        controls["reversed"] = [seq[::-1].copy() for seq in real_seqs]
        # Pad to length
        controls["reversed"] = [
            np.pad(s, (0, max(0, length - len(s))), mode="edge")[:length]
            for s in controls["reversed"]
        ]

    return controls


# ---------------------------------------------------------------------------
# Free-loop rollout with generated codes
# ---------------------------------------------------------------------------


def run_free_loop_rollouts(
    method_name: str,
    code_sequences: list[np.ndarray],
    env,
    inf_fn,
    params: tuple,
    ppo_networks,
    use_rnn: bool,
    max_steps: int,
    seed: int,
    num_codes: int,
    output_dir: Path,
    wandb_enabled: bool,
    jit_reset=None,
    jit_step=None,
) -> dict:
    """Run free-loop rollouts with generated code sequences."""
    log.info(f"  Free-loop rollouts for {method_name} ({len(code_sequences)} sequences)")

    code_colors = get_code_colormap(num_codes)
    survivals = []
    all_qpos = []

    for si, seq in enumerate(code_sequences):
        key = jax.random.PRNGKey(seed + si)
        result = run_rollout(
            env, inf_fn, params, ppo_networks, use_rnn, key,
            max_steps=min(max_steps, len(seq)),
            code_override=seq,
            jit_reset=jit_reset, jit_step=jit_step,
        )
        survivals.append(result["survival"])
        all_qpos.append(result["qpos"][:-1])

        if si < 3:  # Render first 3 as solo videos
            try:
                from experiments.shared.ghost_rendering import render_solo_video
                vid_path = output_dir / f"{method_name}_solo_{si}.mp4"
                render_solo_video(
                    env, result["qpos"][:-1], result["code_indices"], vid_path,
                    fps=50, num_codes=num_codes, title=f"{method_name} #{si}",
                )
                if wandb_enabled:
                    wandb.log(
                        {f"code_gen/{method_name}/solo_{si}": wandb.Video(str(vid_path), format="mp4")},
                        commit=False,
                    )
            except Exception as e:
                log.warning(f"    Solo video {si} failed: {e}")

    mean_surv = np.mean(survivals)
    log.info(f"    Mean survival: {mean_surv:.1f} steps")

    if wandb_enabled:
        wandb.log({f"code_gen/{method_name}/mean_survival": mean_surv}, commit=False)

    return {"survivals": survivals, "qpos": all_qpos}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="code_generation_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Code Generation Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = cfg.wandb.get("enabled", False)
    if wandb_enabled:
        run_name = f"moseq_code_gen_{datetime.now():%y%m%d_%H%M%S}"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.get("entity"), name=run_name, config=dict(cfg))

    # Load code2act (MoSeq decoder) checkpoint
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(cfg.checkpoint.path)
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    code2act_params = (norm_state, policy_params)

    # Load mimic-mjx (oracle VAE) checkpoint
    mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
        cfg.mimic_checkpoint.path, step=cfg.mimic_checkpoint.get("step"),
    )
    mimic_params = (mimic_norm, mimic_policy)

    # Load KPMS codes
    codes_data = np.load(cfg.data.codes_path)
    all_codes = codes_data["all_codes"]

    # Create env for free-loop
    splits_json = cfg.data.balanced_split_path
    ref_path = cfg.data.reference_data_path
    import json
    with open(splits_json) as f:
        splits = json.load(f)
    test_indices = splits["balanced"]["test_indices"]
    test_codes = codes_data["test_codes"]
    test_clips = ReferenceClips(
        data_path=ref_path,
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes,
                         code_stack_size=code_stack_size)

    # Pre-compile JIT functions ONCE (critical for performance)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Code2Act inference fn for free-loop
    inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)

    max_steps = int(cfg.free_loop.max_steps)
    seed = int(cfg.free_loop.seed)

    # -------------------------------------------------------------------
    # Generate sequences
    # -------------------------------------------------------------------
    all_generated: dict[str, list[np.ndarray]] = {}

    methods = list(cfg.generation.methods)

    if "transition_matrix" in methods:
        tm_results = generate_empirical_tm_sequences(all_codes, num_codes, cfg)
        all_generated.update(tm_results)

    if "hmm_dynamax" in methods:
        try:
            hmm_seqs = generate_hmm_sequences(all_codes, num_codes, cfg)
            all_generated["hmm_dynamax"] = hmm_seqs
        except Exception as e:
            log.warning(f"  HMM fitting failed: {e}")

    arhmm_params = None
    if "arhmm_level1" in methods or "arhmm_level2" in methods:
        arhmm_params = load_arhmm_params(cfg)

    if "arhmm_level1" in methods and arhmm_params is not None:
        l1_seqs = generate_arhmm_level1_sequences(arhmm_params, cfg)
        all_generated["arhmm_level1"] = l1_seqs

    if "arhmm_level2" in methods and arhmm_params is not None:
        l2_seqs = generate_arhmm_level2_sequences(arhmm_params, cfg)
        all_generated["arhmm_level2"] = l2_seqs

    # Controls
    ctrl_results = generate_control_sequences(all_codes, num_codes, cfg)
    all_generated.update(ctrl_results)

    # Save all generated sequences for post-hoc analysis
    for method_name, seqs in all_generated.items():
        np.savez_compressed(
            output_dir / f"generated_{method_name}.npz",
            sequences=np.array(seqs),
        )
    log.info(f"  Saved generated sequences for {len(all_generated)} methods")

    # -------------------------------------------------------------------
    # Diagnostics: transition matrices per method
    # -------------------------------------------------------------------
    log.info("\n--- Transition matrix diagnostics ---")
    for method_name, seqs in all_generated.items():
        T = compute_transition_matrix(seqs, num_codes)
        fig = plot_transition_matrix(T, title=f"TM: {method_name}")
        fig.savefig(output_dir / f"tm_{method_name}.png", dpi=300)
        np.save(output_dir / f"tm_{method_name}.npy", T)
        if wandb_enabled:
            wandb.log({f"code_gen/{method_name}/transition_matrix": fig_to_image(fig)}, commit=False)
        plt.close(fig)

    # -------------------------------------------------------------------
    # Free-loop rollouts per method
    # -------------------------------------------------------------------
    log.info("\n--- Free-loop rollouts ---")
    survival_summary = {}
    all_rollout_data = {}
    for method_name, seqs in all_generated.items():
        result = run_free_loop_rollouts(
            method_name, seqs[:int(cfg.generation.num_sequences)],
            env, inf_fn, code2act_params, ppo_networks, use_rnn,
            max_steps, seed, num_codes, output_dir, wandb_enabled,
            jit_reset=jit_reset, jit_step=jit_step,
        )
        survival_summary[method_name] = np.mean(result["survivals"])
        all_rollout_data[method_name] = result
        # Save per-method rollout data
        np.savez_compressed(
            output_dir / f"rollouts_{method_name}.npz",
            survivals=np.array(result["survivals"]),
            qpos=np.array(result["qpos"], dtype=object),
        )

    # --- Mimic-MJX oracle baseline (survival upper bound) ---
    log.info("\n--- Mimic-MJX oracle baseline ---")
    mimic_inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)
    n_seq = int(cfg.generation.num_sequences)
    mimic_survivals = []
    for si in range(n_seq):
        key = jax.random.PRNGKey(seed + si)
        result = run_rollout(
            env, mimic_inf_fn, mimic_params, mimic_ppo, False, key,
            max_steps=max_steps,
            jit_reset=jit_reset, jit_step=jit_step,
            model_type="mimic_mjx",
        )
        mimic_survivals.append(result["survival"])
    survival_summary["mimic_mjx (oracle)"] = np.mean(mimic_survivals)
    log.info(f"  Mimic-MJX mean survival: {np.mean(mimic_survivals):.1f}")

    # Save survival summary for all methods
    import json as _json
    with open(output_dir / "survival_summary.json", "w") as f:
        _json.dump(survival_summary, f, indent=2)
    np.savez_compressed(
        output_dir / "rollouts_mimic_mjx.npz",
        survivals=np.array(mimic_survivals),
    )

    # -------------------------------------------------------------------
    # Survival comparison plot
    # -------------------------------------------------------------------
    set_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    methods_sorted = sorted(survival_summary.keys(), key=lambda x: survival_summary[x], reverse=True)
    bar_colors = [
        NATURE_COLORS["orange"] if "mimic" in m.lower() else NATURE_COLORS["blue"]
        for m in methods_sorted
    ]
    bars = ax.barh(
        range(len(methods_sorted)),
        [survival_summary[m] for m in methods_sorted],
        color=bar_colors,
        edgecolor="none",
    )
    ax.set_yticks(range(len(methods_sorted)))
    ax.set_yticklabels(methods_sorted)
    ax.set_xlabel("Mean survival (steps)")
    ax.set_title("Free-loop survival by generation method")
    plt.tight_layout()
    if wandb_enabled:
        wandb.log({"code_gen/survival_comparison": fig_to_image(fig)}, commit=False)
    fig.savefig(output_dir / "survival_comparison.png", dpi=300)
    plt.close(fig)

    if wandb_enabled:
        wandb.log({}, commit=True)
        wandb.finish()

    log.info("=== Code Generation Experiment Complete ===")


if __name__ == "__main__":
    main()
