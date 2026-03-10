"""KPMS hyperparameter configuration."""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class KPMSHyperparams:
    """Hyperparameters for a single KPMS fit.

    Attributes:
        kappa: Concentration parameter for AR-HMM prior.
        latent_dim: PCA latent dimension for keypoints.
        num_states: Number of syllable states.
        ar_iters: Number of AR-only fitting iterations.
        full_iters: Total iterations (AR + SLDS if model_type is slds).
        model_type: Either ``"arhmm"`` or ``"slds"``.
    """

    kappa: float = 1e4
    latent_dim: int = 10
    num_states: int = 20
    ar_iters: int = 50
    full_iters: int = 300
    model_type: str = "arhmm"


def hyperparams_to_dict(hp: KPMSHyperparams) -> dict[str, Any]:
    """Convert hyperparams dataclass to a plain dict."""
    return asdict(hp)
