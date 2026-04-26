"""Linear-ramp joint-error termination threshold (paper Eq. 10)."""
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class JointErrorCurriculum:
    start: float
    end: float
    total_steps: int

    def __call__(self, step: int) -> float:
        frac = min(max(step / max(1, self.total_steps), 0.0), 1.0)
        return self.start + frac * (self.end - self.start)


def joint_error_terminated(sim_q: jnp.ndarray, target_q: jnp.ndarray,
                           threshold: float) -> jnp.ndarray:
    """Returns 1.0 if max |sim_q - target_q| > threshold, else 0.0."""
    err = jnp.max(jnp.abs(sim_q - target_q))
    return (err > threshold).astype(jnp.float32)
