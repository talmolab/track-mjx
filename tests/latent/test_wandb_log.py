import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from track_mjx.agent.latent_ppo.wandb_log import WandbLogger


def test_disabled_logger_is_a_silent_noop():
    cfg = OmegaConf.create({"wandb_enabled": False, "wandb_project": "x"})
    logger = WandbLogger(cfg)
    # All methods must work even if wandb itself is not installed.
    logger.log_scalars(0, {"foo": 1.0})
    logger.log_histogram(0, "z", jnp.zeros((4,)))
    logger.log_reconstruction_figure(0, jnp.zeros((3, 4)), jnp.zeros((3, 4)),
                                     name="recon")
    logger.finish()


def test_disabled_logger_does_not_import_wandb(monkeypatch):
    """If wandb is missing in the env, the disabled path must still work."""
    import sys
    # Hide wandb from import for the duration of this test.
    saved = sys.modules.get("wandb")
    sys.modules["wandb"] = None  # type: ignore[assignment]
    try:
        cfg = OmegaConf.create({"wandb_enabled": False, "wandb_project": "x"})
        logger = WandbLogger(cfg)
        logger.log_scalars(0, {"foo": 1.0})
    finally:
        if saved is not None:
            sys.modules["wandb"] = saved
        else:
            del sys.modules["wandb"]
