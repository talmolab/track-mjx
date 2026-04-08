"""Forward kinematics: qpos → keypoints via MuJoCo site positions.

Extracted from ``sweep/run_sweep.py`` for reuse in experiments.
Requires ``JAX_ENABLE_X64=1`` set before import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Default paths
_STAC_XML = "/home/jovyan/vast/kaiwen/TopoVNL/stac-mjx/models/rodent.xml"


def setup_stac_model(
    h5_path: str | Path,
    xml_path: str | Path = _STAC_XML,
) -> tuple:
    """Set up MuJoCo model with keypoint sites from stac-mjx config.

    Replicates ``stac_mjx.Stac._build_body_spec()`` without the stac_mjx
    dependency.  Loads optimized offsets from the H5 file and adds sites
    to the correct bodies in the rodent XML.

    Args:
        h5_path: Path to reference clips H5 (must contain ``config``,
            ``offsets``, ``kp_names``).
        xml_path: Path to the stac-mjx rodent XML model.

    Returns:
        ``(mj_model, mj_data, site_ids, kp_names)`` where ``site_ids``
        and ``kp_names`` are both in the H5's alphabetical order.
    """
    import h5py
    import mujoco
    import yaml

    with h5py.File(str(h5_path), "r") as f:
        cfg_yaml = f["config"][()].decode()
        offsets = f["offsets"][:]  # [K, 3] in kp_names order
        kp_names = [n.decode() for n in f["kp_names"][:]]

    cfg = yaml.safe_load(cfg_yaml)
    kmp = cfg["model"]["KEYPOINT_MODEL_PAIRS"]

    spec = mujoco.MjSpec.from_file(str(xml_path))
    name_to_offset = {name: offsets[i] for i, name in enumerate(kp_names)}

    for kp_name, body_name in kmp.items():
        parent = spec.body(body_name)
        pos = name_to_offset[kp_name].tolist()
        parent.add_site(
            name=kp_name,
            size=[0.005, 0.005, 0.005],
            pos=pos,
            group=3,
        )

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)

    site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in kp_names
    ]

    return mj_model, mj_data, site_ids, kp_names


def qpos_to_keypoints_fk(
    qpos: np.ndarray,
    mj_model,
    mj_data,
    site_ids: list[int],
    batch_size: int = 1000,
) -> np.ndarray:
    """Compute keypoints from qpos via batched JAX-vmapped MuJoCo FK.

    Args:
        qpos: Joint positions ``[N, nq]``.
        mj_model: Compiled MuJoCo model with keypoint sites.
        mj_data: MuJoCo data template.
        site_ids: Site indices for keypoints.
        batch_size: Frames per GPU batch (reduce if OOM).

    Returns:
        Keypoint positions ``[N, K, 3]``.
    """
    import jax
    import jax.numpy as jnp
    import mujoco.mjx as mjx

    site_ids_array = jnp.array(site_ids)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    @jax.jit
    def forward_kinematics(qpos_single):
        data = mjx_data.replace(qpos=qpos_single)
        data = mjx.forward(mjx_model, data)
        return data.site_xpos[site_ids_array]

    batch_fk = jax.vmap(forward_kinematics)
    n_total = qpos.shape[0]
    all_kps = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_qpos = jnp.array(qpos[start:end])
        batch_kps = batch_fk(batch_qpos)
        all_kps.append(np.array(batch_kps))
        if (start // batch_size) % 10 == 0:
            log.info(f"  FK batch {start // batch_size}: {start}/{n_total}")

    return np.concatenate(all_kps, axis=0)
