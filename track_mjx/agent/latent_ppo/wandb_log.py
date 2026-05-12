"""Optional wandb logger for Phase 1 pre-training.

When `cfg.wandb_enabled` is False, every method is a no-op and wandb is never
imported. This keeps unit tests offline-clean.
"""
from typing import Any, Mapping

import numpy as np
from omegaconf import DictConfig, OmegaConf


class WandbLogger:
    def __init__(self, cfg: DictConfig):
        self.enabled = bool(cfg.get("wandb_enabled", False))
        self._wandb = None
        if not self.enabled:
            return
        import wandb  # imported lazily so tests without wandb still pass
        self._wandb = wandb
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.get("wandb_run_name", None),
            tags=list(cfg.get("wandb_tags", []) or []),
            group=cfg.get("wandb_group", None),
        )

    def log_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        if not self.enabled:
            return
        # Cast jax/np scalars to plain floats so wandb is happy.
        flat = {k: float(np.asarray(v)) for k, v in scalars.items()}
        self._wandb.log(flat, step=step)

    def log_histogram(self, step: int, name: str, values: Any) -> None:
        if not self.enabled:
            return
        arr = np.asarray(values).reshape(-1)
        self._wandb.log({name: self._wandb.Histogram(arr)}, step=step)

    def log_reconstruction_figure(self, step: int, true_window: Any,
                                  recon_window: Any, name: str,
                                  n_dims_to_show: int = 6) -> None:
        if not self.enabled:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        true = np.asarray(true_window)   # (w_or_n, feat)
        recon = np.asarray(recon_window)
        n_show = min(n_dims_to_show, true.shape[-1])
        fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.5 * n_show), sharex=True)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            axes[i].plot(true[:, i], label="true", linewidth=1.5)
            axes[i].plot(recon[:, i], label="model", linestyle="--", linewidth=1.5)
            axes[i].set_ylabel(f"dim {i}", fontsize=8)
            if i == 0:
                axes[i].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("frame")
        fig.tight_layout()
        self._wandb.log({name: self._wandb.Image(fig)}, step=step)
        plt.close(fig)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()
