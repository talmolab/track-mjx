import jax.numpy as jnp
import numpy as np

from track_mjx.agent.latent_ppo.term_curriculum import (
    JointErrorCurriculum,
    joint_error_terminated,
)


def test_curriculum_linear_schedule():
    cur = JointErrorCurriculum(start=0.5, end=2 * 3.14159265, total_steps=1000)
    assert np.isclose(cur(0), 0.5)
    assert np.isclose(cur(500), (0.5 + 2 * np.pi) / 2, atol=1e-2)
    assert np.isclose(cur(1000), 2 * np.pi)
    assert np.isclose(cur(2000), 2 * np.pi)


def test_termination_max_per_joint_error():
    sim = jnp.array([0.0, 0.4, 0.0])
    target = jnp.array([0.0, 0.0, 0.0])
    assert not bool(joint_error_terminated(sim, target, threshold=0.5))
    assert bool(joint_error_terminated(sim, target, threshold=0.3))
