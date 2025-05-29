"""Utility functions for scaling and recoloring geometries in a MuJoCo model."""

import numpy as np


def _scale_vec(vec: list[float] | np.ndarray, s: float) -> None:
    """Scale a vector in-place by a scalar.

    Args:
        vec (list[float] | np.ndarray): The vector to scale.
        s (float): The scalar multiplier.

    Returns:
        None
    """
    for i in range(len(vec)):
        vec[i] *= s


def dm_scale_spec(spec, scale):
    scaled_spec = spec.copy()

    # Traverse the kinematic tree, scaling all geoms
    def scale_bodies(parent, scale=1.0):
        body = parent.first_body()
        while body:
            if body.pos is not None:
                body.pos = body.pos * scale
            for geom in body.geoms:
                geom.fromto = geom.fromto * scale
                geom.size = geom.size * scale
                if geom.pos is not None:
                    geom.pos = geom.pos * scale
            scale_bodies(body, scale)
            body = parent.next_body(body)

    # if scale_actuators:
    # # scale gear
    for actuator in scaled_spec.actuators:
        # scale the actuator gear by (scale ** 2),
        # this is because muscle force-generating capacity
        # scales with the cross-sectional area of the muscle
        actuator.gear = actuator.gear * scale * scale

    # scale the z-position for all keypoints
    for keypoint in scaled_spec.keys:
        qpos = keypoint.qpos
        qpos[2] = qpos[2] * scale
        keypoint.qpos = qpos
        keypoint.qpos[2] = keypoint.qpos[2] * scale

    scale_bodies(scaled_spec.body("walker"), scale)
    return scaled_spec


def _scale_body_tree(body, s: float) -> None:
    """Recursively scale position, size, and fromto attributes on a body and its descendants.

    Args:
        body (Any): The body object to scale.
        s (float): The scalar multiplier.

    Returns:
        None
    """
    if hasattr(body, "pos"):
        _scale_vec(body.pos, s)

    for geom in body.geoms:
        if hasattr(geom, "pos"):
            _scale_vec(geom.pos, s)
        if hasattr(geom, "size"):
            _scale_vec(geom.size, s)
        if hasattr(geom, "fromto"):
            _scale_vec(geom.fromto, s)

    for site in body.sites:
        if hasattr(site, "pos"):
            _scale_vec(site.pos, s)
        if hasattr(site, "size"):
            _scale_vec(site.size, s)

    for joint in body.joints:
        if hasattr(joint, "pos"):
            _scale_vec(joint.pos, s)

    for child in body.bodies:
        _scale_body_tree(child, s)


def _recolour_geom(geom, rgba: list[float]) -> None:
    """Modify the color and collision group of a geometry.

    Args:
        geom (Any): The geometry object to recolor.
        rgba (list[float]): The RGBA color values.

    Returns:
        None
    """
    geom.rgba = list(rgba)
    geom.group = 2  # separate collision group
    # Route via ghost <default> if you want inheritance; otherwise this is fine.


def _recolour_tree(body, rgba: list[float]) -> None:
    """Recursively recolor all geometries in a body and its descendants.

    Args:
        body (Any): The body object whose geometries are to be recolored.
        rgba (list[float]): The RGBA color values.

    Returns:
        None
    """
    for geom in body.geoms:
        _recolour_geom(geom, rgba)
    for child in body.bodies:
        _recolour_tree(child, rgba)
