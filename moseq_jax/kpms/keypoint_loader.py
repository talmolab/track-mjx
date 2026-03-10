"""Convert qpos trajectories to keypoints using MuJoCo forward kinematics.

Provides utilities for:

1. Setting up a MuJoCo/MJX model and identifying site IDs for keypoints.
2. Batched JAX-vectorized FK: ``qpos_to_keypoints``.
3. Formatting keypoint arrays into the dict format expected by
   ``keypoint_moseq``.
"""

from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jp
    import mujoco
    import mujoco.mjx as mjx

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

DEFAULT_KP_NAMES = [
    "Snout",
    "EarL",
    "EarR",
    "SpineF",
    "SpineM",
    "SpineL",
    "TailBase",
    "ShoulderL",
    "ElbowL",
    "WristL",
    "HandL",
    "ShoulderR",
    "ElbowR",
    "WristR",
    "HandR",
    "HipL",
    "KneeL",
    "AnkleL",
    "FootL",
    "HipR",
    "KneeR",
    "AnkleR",
    "FootR",
]


def setup_mujoco_model(
    xml_path: str,
    kp_names: list[str] | None = None,
) -> tuple[Any, Any, list[int], list[str]]:
    """Load MuJoCo model and identify keypoint site IDs.

    Args:
        xml_path: Path to MuJoCo XML.
        kp_names: Keypoint site names.  Defaults to ``DEFAULT_KP_NAMES``.

    Returns:
        ``(mjx_model, mjx_data, site_ids, valid_kp_names)``
    """
    if not HAS_MUJOCO:
        raise ImportError("mujoco not installed")

    if kp_names is None:
        kp_names = DEFAULT_KP_NAMES

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    site_ids: list[int] = []
    valid_kp_names: list[str] = []
    for name in kp_names:
        try:
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                site_ids.append(sid)
                valid_kp_names.append(name)
        except Exception:
            pass

    if not site_ids:
        print("Warning: No sites found matching keypoint names. Using all sites.")
        site_ids = list(range(mj_model.nsite))
        valid_kp_names = [mj_model.site(i).name for i in site_ids]

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    return mjx_model, mjx_data, site_ids, valid_kp_names


def qpos_to_keypoints(
    qpos_data: np.ndarray,
    mjx_model: Any,
    mjx_data: Any,
    site_ids: list[int],
) -> np.ndarray:
    """Compute keypoint positions via batched FK.

    Args:
        qpos_data: Joint positions, shape ``[N, T, nq]``.
        mjx_model: MJX model.
        mjx_data: MJX data template.
        site_ids: Site indices for keypoints.

    Returns:
        Keypoint positions, shape ``[N, T, K, 3]``.
    """
    if not HAS_MUJOCO:
        raise ImportError("mujoco not installed")

    site_ids_array = jp.array(site_ids)

    @jax.jit
    def forward_kinematics(qpos_single):
        data = mjx_data.replace(qpos=qpos_single)
        data = mjx.forward(mjx_model, data)
        return data.site_xpos[site_ids_array]

    process_sequence = jax.vmap(forward_kinematics)
    process_all = jax.vmap(process_sequence)

    keypoints = process_all(jp.array(qpos_data))
    return np.array(keypoints)


def prepare_keypoints_for_kpms(
    keypoint_data: np.ndarray,
    prefix: str = "recording",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Format keypoint array into dicts for ``keypoint_moseq``.

    Args:
        keypoint_data: Shape ``[N, T, K, 3]``.
        prefix: Key prefix for recording names.

    Returns:
        ``(coordinates, confidences)`` dicts keyed by ``"prefix_NNNN"``.
    """
    n_clips, n_frames, n_keypoints, _ = keypoint_data.shape

    coordinates: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}

    for i in range(n_clips):
        key = f"{prefix}_{i:04d}"
        coordinates[key] = np.array(keypoint_data[i]).astype(np.float64)
        confidences[key] = np.ones((n_frames, n_keypoints), dtype=np.float64)

    return coordinates, confidences
