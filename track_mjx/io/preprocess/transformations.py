"""Transformations (adapted from dmcontrol) for jax"""

import jax
from jax import numpy as jp

# Constants used to determine when a rotation is close to a pole.
_POLE_LIMIT = 1.0 - 1e-6
_TOL = 1e-10


def _get_qmat_indices_and_signs():
    """Precomputes index and sign arrays for constructing `qmat` in `quat_mul`."""
    w, x, y, z = range(4)
    qmat_idx_and_sign = jp.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )
    indices = jp.abs(qmat_idx_and_sign)
    signs = 2 * (qmat_idx_and_sign >= 0) - 1
    return indices, signs


_qmat_idx, _qmat_sign = _get_qmat_indices_and_signs()


def quat_mul(quat1, quat2):
    """Computes the Hamilton product of two quaternions.

    Any number of leading batch dimensions is supported.

    Args:
      quat1: A quaternion [w, i, j, k].
      quat2: A quaternion [w, i, j, k].

    Returns:
      The quaternion product quat1 * quat2.
    """
    # Construct a (..., 4, 4) matrix to multiply with quat2 as shown below.
    qmat = quat1[..., _qmat_idx] * _qmat_sign

    # Compute the batched Hamilton product:
    # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
    # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
    # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
    # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
    return (qmat @ quat2[..., None])[..., 0]


def _clip_within_precision(number, low, high, precision=_TOL):
    """Clips input to provided range, checking precision.

    Args:
      number: (float) number to be clipped.
      low: (float) lower bound.
      high: (float) upper bound.
      precision: (float) tolerance.

    Returns:
      Input clipped to given range.

    Raises:
      ValueError: If number is outside given range by more than given precision.
    """

    # This is raising an error when jitted
    # def _raise_if_not_in_precision():
    #     if (number < low - precision).any() or (number > high + precision).any():
    #         raise ValueError(
    #             "Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}".format(
    #                 number, low, high, precision
    #             )
    #         )

    # jax.debug.callback(_raise_if_not_in_precision)

    return jp.clip(number, low, high)


def quat_conj(quat):
    """Return conjugate of quaternion.

    This function supports inputs with or without leading batch dimensions.

    Args:
      quat: A quaternion [w, i, j, k].

    Returns:
      A quaternion [w, -i, -j, -k] representing the inverse of the rotation
      defined by `quat` (not assuming normalization).
    """
    # Ensure quat is an np.array in case a tuple or a list is passed
    quat = jp.asarray(quat)
    return jp.stack(
        [quat[..., 0], -quat[..., 1], -quat[..., 2], -quat[..., 3]], axis=-1
    )


def quat_diff(source, target):
    """Computes quaternion difference between source and target quaternions.

    This function supports inputs with or without leading batch dimensions.

    Args:
      source: A quaternion [w, i, j, k].
      target: A quaternion [w, i, j, k].

    Returns:
      A quaternion representing the rotation from source to target.
    """
    return quat_mul(quat_conj(source), target)


def quat_to_axisangle(quat):
    """Returns the axis-angle corresponding to the provided quaternion.

    Args:
      quat: A quaternion [w, i, j, k].

    Returns:
      axisangle: A 3x1 numpy array describing the axis of rotation, with angle
          encoded by its length.
    """
    angle = 2 * jp.arccos(_clip_within_precision(quat[0], -1.0, 1.0))

    def true_fn(angle):
        return jp.zeros(3)

    def false_fn(angle):
        qn = jp.sin(angle / 2)
        angle = (angle + jp.pi) % (2 * jp.pi) - jp.pi
        axis = quat[1:4] / qn
        out = axis * angle
        return out

    return jax.lax.cond(angle < _TOL, true_fn, false_fn, angle)
