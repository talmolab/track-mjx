# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Physics domain randomization for the rodent walker.

Randomizes sliding friction, static (joint) friction, armature, center of mass,
link and torso mass, and the initial joint configuration on training environments
only. Sliding friction is applied as one per-environment scale across every geom;
the walker's collision geoms outrank the floor plane in contact priority, so
scaling their friction is what varies the effective floor-contact friction.
"""

import jax
from mujoco import mjx

TORSO_BODY_ID = 3

FREE_JOINT_DOFS = 6
FREE_JOINT_QPOS = 7


def domain_randomization_maker(
    friction_scale: tuple[float, float] = (0.6, 1.4),
    static_friction_scale: tuple[float, float] = (0.9, 1.1),
    armature_scale: tuple[float, float] = (1.0, 1.05),
    com_jitter: tuple[float, float] = (-0.05, 0.05),
    link_mass_scale: tuple[float, float] = (0.9, 1.1),
    torso_mass_jitter: tuple[float, float] = (-0.02, 0.02),
    qpos0_jitter: tuple[float, float] = (-0.05, 0.05),
):
    """Create a domain randomization function for MJX environments.

    Returns a function compatible with Brax's ``randomization_fn`` interface
    that randomizes physics parameters per-environment using ``jax.vmap``.

    Args:
        floor_friction: (min, max) range for uniform floor friction sampling.
        static_friction_scale: (min, max) multiplicative scale for DOF frictionloss.
        armature_scale: (min, max) multiplicative scale for DOF armature.
        com_jitter: (min, max) additive jitter for torso center of mass position.
        link_mass_scale: (min, max) multiplicative scale for all body masses.
        torso_mass_jitter: (min, max) additive jitter for torso mass.
        qpos0_jitter: (min, max) additive jitter for initial joint positions.

    Returns:
        A function ``(model, rng) -> (model, in_axes)`` that applies randomized
        dynamics and returns the per-axis vmap specification.
    """

    def domain_randomize(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand_dynamics(rng):
            n_actuated = model.nv - FREE_JOINT_DOFS

            # Sliding friction: one per-env scale across every geom. The walker's
            # collision geoms outrank the floor in contact priority, so scaling
            # their friction is what varies the effective floor-contact friction.
            rng, key = jax.random.split(rng)
            fscale = jax.random.uniform(
                key, minval=friction_scale[0], maxval=friction_scale[1]
            )
            geom_friction = model.geom_friction.at[:, 0].set(
                model.geom_friction[:, 0] * fscale
            )

            # Scale static friction: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[
                FREE_JOINT_DOFS:
            ] * jax.random.uniform(
                key,
                shape=(n_actuated,),
                minval=static_friction_scale[0],
                maxval=static_friction_scale[1],
            )
            dof_frictionloss = model.dof_frictionloss.at[FREE_JOINT_DOFS:].set(
                frictionloss
            )

            # Scale armature: *U(1.0, 1.05).
            rng, key = jax.random.split(rng)
            armature = model.dof_armature[FREE_JOINT_DOFS:] * jax.random.uniform(
                key,
                shape=(n_actuated,),
                minval=armature_scale[0],
                maxval=armature_scale[1],
            )
            dof_armature = model.dof_armature.at[FREE_JOINT_DOFS:].set(armature)

            # Jitter center of mass positiion: +U(-0.05, 0.05).
            rng, key = jax.random.split(rng)
            dpos = jax.random.uniform(
                key, (3,), minval=com_jitter[0], maxval=com_jitter[1]
            )
            body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
                model.body_ipos[TORSO_BODY_ID] + dpos
            )

            # Scale all link masses: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key,
                shape=(model.nbody,),
                minval=link_mass_scale[0],
                maxval=link_mass_scale[1],
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

            # Add mass to torso: +U(-0.02, 0.02).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, minval=torso_mass_jitter[0], maxval=torso_mass_jitter[1]
            )
            body_mass = body_mass.at[TORSO_BODY_ID].set(
                body_mass[TORSO_BODY_ID] + dmass
            )

            # Jitter qpos0: +U(-0.05, 0.05).
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[FREE_JOINT_QPOS:].set(
                qpos0[FREE_JOINT_QPOS:]
                + jax.random.uniform(
                    key,
                    shape=(n_actuated,),
                    minval=qpos0_jitter[0],
                    maxval=qpos0_jitter[1],
                )
            )

            return (
                geom_friction,
                body_ipos,
                body_mass,
                qpos0,
                dof_frictionloss,
                dof_armature,
            )

        (
            friction,
            body_ipos,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        ) = rand_dynamics(rng)

        in_axes = jax.tree.map(lambda x: None, model)
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "body_ipos": 0,
                "body_mass": 0,
                "qpos0": 0,
                "dof_frictionloss": 0,
                "dof_armature": 0,
            }
        )

        model = model.tree_replace(
            {
                "geom_friction": friction,
                "body_ipos": body_ipos,
                "body_mass": body_mass,
                "qpos0": qpos0,
                "dof_frictionloss": dof_frictionloss,
                "dof_armature": dof_armature,
            }
        )

        return model, in_axes

    return domain_randomize
