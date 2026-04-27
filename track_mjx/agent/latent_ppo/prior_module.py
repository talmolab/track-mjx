"""Frozen Latent Prior Module: encoder + predictor + normalizer loaded from a Phase 1 ckpt dir."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from track_mjx.agent.latent_ppo.data.frame_features import MOTION_FRAME_DIM
from track_mjx.agent.latent_ppo.data.normalizer import FeatureNormalizer
from track_mjx.agent.latent_ppo.networks.encoder import (
    MotionEncoder,
    MotionEncoderConv1D,
)
from track_mjx.agent.latent_ppo.networks.predictor import MotionPredictor


@dataclass
class FrozenLatentPrior:
    """Container holding a frozen Phase 1 encoder + predictor + normalizer.

    Loaded from a Phase 1 checkpoint directory containing:
        - config.yaml         : the Hydra config used to train
        - encoder.msgpack     : flax params for MotionEncoder / MotionEncoderConv1D
        - predictor.msgpack   : flax params for MotionPredictor
        - normalizer.npz      : {mean, std} per-feature stats

    Predictor outputs are in NORMALIZED space (the predictor was trained on
    normalized targets). Callers that need raw frames must denormalize.
    """

    encoder: Any
    predictor: MotionPredictor
    enc_params: Any
    pred_params: Any
    normalizer: FeatureNormalizer
    window_len: int
    horizon: int
    latent_dim: int
    feat_dim: int
    use_qvel: bool
    n_joints: int

    @classmethod
    def from_dir(cls, ckpt_dir: str | Path) -> "FrozenLatentPrior":
        ckpt_dir = Path(ckpt_dir)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        use_qvel = bool(cfg.get("use_qvel", True))  # legacy default
        n_joints = int(cfg.n_joints)
        feat_dim = MOTION_FRAME_DIM(n_joints, use_qvel=use_qvel)

        encoder_type = str(cfg.get("encoder_type", "mlp")).lower()
        if encoder_type == "conv1d":
            encoder = MotionEncoderConv1D(
                conv_channels=tuple(cfg.get("conv_channels", (64, 128, 256))),
                kernel_size=int(cfg.get("conv_kernel_size", 3)),
                head_layer_sizes=tuple(cfg.get("conv_head_layers", (256,))),
                latent_dim=int(cfg.latent_dim),
            )
        elif encoder_type == "mlp":
            encoder = MotionEncoder(
                layer_sizes=tuple(cfg.encoder_layer_sizes),
                latent_dim=int(cfg.latent_dim),
            )
        else:
            raise ValueError(f"unknown encoder_type {encoder_type!r}")

        predictor = MotionPredictor(
            layer_sizes=tuple(cfg.predictor_layer_sizes),
            horizon=int(cfg.horizon),
            feat_dim=feat_dim,
        )

        rng = jax.random.PRNGKey(0)
        dummy_in = jnp.zeros((1, int(cfg.window_len), feat_dim))
        dummy_z = jnp.zeros((1, int(cfg.latent_dim)))
        enc_init = encoder.init(rng, dummy_in)
        pred_init = predictor.init(rng, dummy_z)

        with open(ckpt_dir / "encoder.msgpack", "rb") as f:
            enc_params = serialization.from_bytes(enc_init, f.read())
        with open(ckpt_dir / "predictor.msgpack", "rb") as f:
            pred_params = serialization.from_bytes(pred_init, f.read())

        norm_npz = np.load(ckpt_dir / "normalizer.npz")
        normalizer = FeatureNormalizer(
            mean=jnp.asarray(norm_npz["mean"], dtype=jnp.float32),
            std=jnp.asarray(norm_npz["std"], dtype=jnp.float32),
        )
        return cls(
            encoder=encoder,
            predictor=predictor,
            enc_params=enc_params,
            pred_params=pred_params,
            normalizer=normalizer,
            window_len=int(cfg.window_len),
            horizon=int(cfg.horizon),
            latent_dim=int(cfg.latent_dim),
            feat_dim=feat_dim,
            use_qvel=use_qvel,
            n_joints=n_joints,
        )

    def normalize(self, window: jnp.ndarray) -> jnp.ndarray:
        """window: (..., feat_dim) raw -> (..., feat_dim) normalized."""
        return self.normalizer.apply(window)

    def encode(self, window: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """window: (B, w, feat_dim) RAW -> (mean, logvar) each (B, latent_dim).

        Applies the saved normalizer before the encoder.
        """
        return self.encoder.apply(self.enc_params, self.normalize(window))

    def encode_normalized(
        self, window_n: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """For pre-normalized windows (e.g. predictor output), skip normalization."""
        return self.encoder.apply(self.enc_params, window_n)

    def predict(self, z: jnp.ndarray) -> jnp.ndarray:
        """z: (B, latent_dim) -> (B, horizon, feat_dim) in NORMALIZED space.

        Predictor was trained on normalized targets; we leave outputs in that
        space so they can be re-encoded without a normalize/denormalize round
        trip.
        """
        return self.predictor.apply(self.pred_params, z)
