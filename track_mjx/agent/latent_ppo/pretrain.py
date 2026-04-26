"""Phase 1 pre-training entry point for the Latent Prior Module."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from omegaconf import DictConfig, OmegaConf

from track_mjx.agent.latent_ppo.data.frame_features import (
    MOTION_FRAME_DIM,
    extract_motion_frames,
)
from track_mjx.agent.latent_ppo.data.normalizer import fit_normalizer
from track_mjx.agent.latent_ppo.data.window_dataset import make_windows
from track_mjx.agent.latent_ppo.losses.pretrain_losses import pretrain_loss
from track_mjx.agent.latent_ppo.networks.decoder import MotionDecoder
from track_mjx.agent.latent_ppo.networks.encoder import MotionEncoder
from track_mjx.agent.latent_ppo.networks.predictor import MotionPredictor
from track_mjx.agent.latent_ppo.wandb_log import WandbLogger


@dataclass
class PretrainState:
    params: dict
    opt_state: Any
    rng: jax.Array
    losses: list = field(default_factory=list)
    best_val_total: float = float("inf")
    best_val_step: int = -1


def _load_clips(cfg: DictConfig):
    """Default loader: ReferenceClips from HDF5. Tests monkey-patch this."""
    from vnl_playground.tasks.reference_clips import ReferenceClips
    return ReferenceClips(
        cfg.reference_data_path,
        n_frames_per_clip=cfg.clip_length,
        keep_clips_idx=cfg.keep_clips_idx,
    )


def _split_clips(motion: np.ndarray, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    n_clips = motion.shape[0]
    perm = rng.permutation(n_clips)
    split = int(round(n_clips * train_ratio))
    train_idx, val_idx = perm[:split], perm[split:]
    return motion[train_idx], motion[val_idx]


def _save_msgpack(path: Path, params):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))


def run(cfg: DictConfig) -> PretrainState:
    rng = jax.random.PRNGKey(cfg.seed)

    # ---- data ----
    clips = _load_clips(cfg)
    motion = extract_motion_frames(clips, n_joints=cfg.n_joints)
    train_motion, val_motion = _split_clips(motion, cfg.train_ratio, cfg.seed)
    normalizer = fit_normalizer(train_motion.reshape(-1, motion.shape[-1]))

    train_in, train_tgt = make_windows(train_motion, cfg.window_len, cfg.horizon)
    val_in, val_tgt = make_windows(val_motion, cfg.window_len, cfg.horizon)
    train_in = jnp.asarray(normalizer.apply(jnp.asarray(train_in)))
    train_tgt = jnp.asarray(normalizer.apply(jnp.asarray(train_tgt)))
    val_in = jnp.asarray(normalizer.apply(jnp.asarray(val_in)))
    val_tgt = jnp.asarray(normalizer.apply(jnp.asarray(val_tgt)))

    feat_dim = MOTION_FRAME_DIM(cfg.n_joints)

    # ---- nets ----
    enc = MotionEncoder(layer_sizes=tuple(cfg.encoder_layer_sizes),
                        latent_dim=cfg.latent_dim)
    dec = MotionDecoder(layer_sizes=tuple(cfg.decoder_layer_sizes),
                        window_len=cfg.window_len, feat_dim=feat_dim)
    pred = MotionPredictor(layer_sizes=tuple(cfg.predictor_layer_sizes),
                           horizon=cfg.horizon, feat_dim=feat_dim)

    rng, k_e, k_d, k_p = jax.random.split(rng, 4)
    dummy_in = jnp.zeros((1, cfg.window_len, feat_dim))
    dummy_z = jnp.zeros((1, cfg.latent_dim))
    params = {
        "enc": enc.init(k_e, dummy_in),
        "dec": dec.init(k_d, dummy_z),
        "pred": pred.init(k_p, dummy_z),
    }

    # ---- optim ----
    schedule = optax.linear_schedule(
        init_value=0.0, end_value=cfg.beta_kl,
        transition_steps=cfg.beta_kl_anneal_steps,
    )

    # Build LR schedule: linear warmup -> hold at peak -> cosine decay to lr_end_value.
    # When lr_schedule="constant", just use the flat learning_rate.
    lr_schedule_kind = cfg.get("lr_schedule", "constant")
    if lr_schedule_kind == "constant":
        lr_fn: Any = cfg.learning_rate
    else:
        warmup_steps = max(1, int(cfg.num_steps * float(cfg.get("lr_warmup_frac", 0.0))))
        hold_steps = max(0, int(cfg.num_steps * float(cfg.get("lr_hold_frac", 0.0))))
        decay_steps = max(1, cfg.num_steps - warmup_steps - hold_steps)
        end_value = float(cfg.get("lr_end_value", 0.0))
        peak = float(cfg.learning_rate)
        # alpha is the floor as a fraction of peak (optax convention).
        alpha = end_value / peak if peak > 0 else 0.0
        lr_fn = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0, end_value=peak, transition_steps=warmup_steps
                ),
                optax.constant_schedule(peak),
                optax.cosine_decay_schedule(
                    init_value=peak, decay_steps=decay_steps, alpha=alpha
                ),
            ],
            boundaries=[warmup_steps, warmup_steps + hold_steps],
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(lr_fn, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(params)

    # ---- step ----
    def loss_fn(params, rng, inputs, targets, beta):
        return pretrain_loss(
            enc, dec, pred, params["enc"], params["dec"], params["pred"],
            inputs, targets, rng=rng, beta_kl=beta, w_pred=cfg.w_pred,
        )

    @jax.jit
    def step(params, opt_state, rng, inputs, targets, beta):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, rng, inputs, targets, beta
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    state = PretrainState(params=params, opt_state=opt_state, rng=rng)
    n_train = train_in.shape[0]
    n_val = val_in.shape[0]

    # Pinned slices for figures (first batch, deterministic) — we show the SAME
    # windows across training so curves are directly comparable across steps.
    val_fig_in = val_in[: min(8, n_val)]
    val_fig_tgt = val_tgt[: min(8, n_val)]
    train_fig_in = train_in[: min(8, n_train)]
    train_fig_tgt = train_tgt[: min(8, n_train)]

    logger = WandbLogger(cfg)

    @jax.jit
    def eval_loss(params, rng, inputs, targets, beta):
        return loss_fn(params, rng, inputs, targets, beta)

    @jax.jit
    def encode_for_viz(params, inputs):
        mean, logvar = enc.apply(params["enc"], inputs)
        return mean, logvar

    @jax.jit
    def reconstruct_for_viz(params, inputs):
        mean, _ = enc.apply(params["enc"], inputs)
        return dec.apply(params["dec"], mean)

    @jax.jit
    def predict_for_viz(params, inputs):
        mean, _ = enc.apply(params["enc"], inputs)
        return pred.apply(params["pred"], mean)

    try:
        for i in range(cfg.num_steps):
            state.rng, k_b, k_step = jax.random.split(state.rng, 3)
            idx = jax.random.choice(k_b, n_train,
                                    shape=(cfg.batch_size,), replace=False)
            beta = schedule(i)
            state.params, state.opt_state, loss, aux = step(
                state.params, state.opt_state, k_step,
                train_in[idx], train_tgt[idx], beta,
            )
            state.losses.append(float(loss))

            if (i + 1) % cfg.log_every == 0 or i == 0:
                current_lr = float(lr_fn(i)) if callable(lr_fn) else float(lr_fn)
                logger.log_scalars(i, {
                    "train/total": loss,
                    "train/recon": aux["recon"],
                    "train/kl": aux["kl"],
                    "train/pred": aux["pred"],
                    "train/beta_kl": float(beta),
                    "train/lr": current_lr,
                })

            if (i + 1) % cfg.eval_every == 0 and n_val > 0:
                state.rng, k_v = jax.random.split(state.rng)
                val_idx = jax.random.choice(
                    k_v, n_val,
                    shape=(min(cfg.batch_size, n_val),), replace=False,
                )
                v_loss, v_aux = eval_loss(
                    state.params, k_v,
                    val_in[val_idx], val_tgt[val_idx], beta,
                )
                logger.log_scalars(i, {
                    "val/total": v_loss,
                    "val/recon": v_aux["recon"],
                    "val/kl": v_aux["kl"],
                    "val/pred": v_aux["pred"],
                    "val/best_total": state.best_val_total,
                    "val/best_step": state.best_val_step,
                })
                # latent diagnostics
                v_mean, v_logvar = encode_for_viz(state.params, val_fig_in)
                logger.log_histogram(i, "val/z_mean", v_mean)
                logger.log_histogram(i, "val/z_std", jnp.exp(0.5 * v_logvar))

                # Save best-val checkpoint if val_total improved.
                v_total_f = float(v_loss)
                if v_total_f < state.best_val_total:
                    state.best_val_total = v_total_f
                    state.best_val_step = i + 1
                    best_dir = Path(cfg.ckpt_dir) / "best"
                    _save_msgpack(best_dir / "encoder.msgpack", state.params["enc"])
                    _save_msgpack(best_dir / "decoder.msgpack", state.params["dec"])
                    _save_msgpack(best_dir / "predictor.msgpack", state.params["pred"])
                    np.savez(best_dir / "normalizer.npz",
                             mean=np.asarray(normalizer.mean),
                             std=np.asarray(normalizer.std))
                    OmegaConf.save(cfg, best_dir / "config.yaml")
                    np.savez(best_dir / "meta.npz",
                             best_val_total=v_total_f,
                             best_val_step=i + 1,
                             best_val_recon=float(v_aux["recon"]),
                             best_val_pred=float(v_aux["pred"]),
                             best_val_kl=float(v_aux["kl"]))

            if (i + 1) % cfg.viz_every == 0:
                if n_val > 0:
                    v_recon = reconstruct_for_viz(state.params, val_fig_in)
                    v_pred = predict_for_viz(state.params, val_fig_in)
                    logger.log_reconstruction_figure(
                        i, val_fig_in[0], v_recon[0],
                        name="val/reconstruction",
                        n_dims_to_show=cfg.viz_n_dims,
                    )
                    logger.log_reconstruction_figure(
                        i, val_fig_tgt[0], v_pred[0],
                        name="val/prediction",
                        n_dims_to_show=cfg.viz_n_dims,
                    )
                # Training-side figures using the same pinned-slice convention,
                # so train vs val alignment is directly comparable.
                t_recon = reconstruct_for_viz(state.params, train_fig_in)
                t_pred = predict_for_viz(state.params, train_fig_in)
                logger.log_reconstruction_figure(
                    i, train_fig_in[0], t_recon[0],
                    name="train/reconstruction",
                    n_dims_to_show=cfg.viz_n_dims,
                )
                logger.log_reconstruction_figure(
                    i, train_fig_tgt[0], t_pred[0],
                    name="train/prediction",
                    n_dims_to_show=cfg.viz_n_dims,
                )

            if (i + 1) % cfg.ckpt_every == 0 or (i + 1) == cfg.num_steps:
                ckpt_dir = Path(cfg.ckpt_dir)
                _save_msgpack(ckpt_dir / "encoder.msgpack", state.params["enc"])
                _save_msgpack(ckpt_dir / "decoder.msgpack", state.params["dec"])
                _save_msgpack(ckpt_dir / "predictor.msgpack", state.params["pred"])
                np.savez(ckpt_dir / "normalizer.npz",
                         mean=np.asarray(normalizer.mean),
                         std=np.asarray(normalizer.std))
                OmegaConf.save(cfg, ckpt_dir / "config.yaml")
    finally:
        logger.finish()

    return state


@hydra.main(version_base=None, config_path="../../config",
            config_name="latent_mimic_pretrain")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
