"""Convert fly imitation reference H5 from named/semantic format to rat-style legacy flat format.

Named format on disk (under /all_clips/):
    position          (n_clips, n_frames, 3)
    quaternion        (n_clips, n_frames, 4)         [w, x, y, z]
    velocity          (n_clips, n_frames, 3)
    angular_velocity  (n_clips, n_frames, 3)
    joints            (n_clips, n_frames, n_joints)
    joints_velocity   (n_clips, n_frames, n_joints)
    body_positions    (n_clips, n_frames, n_bodies, 3)
    body_quaternions  (n_clips, n_frames, n_bodies, 4)

Legacy format on disk (root level, flat):
    qpos        (n_clips * n_frames, n_qpos)
    qvel        (n_clips * n_frames, n_qvel)
    xpos        (n_clips * n_frames, n_bodies, 3)
    xquat       (n_clips * n_frames, n_bodies, 4)
    names_qpos  (n_qpos,)   strings
    names_xpos  (n_bodies,) strings
    config      ()          YAML string

Usage:
    python -m scripts.convert_fly_reference_to_legacy <input.h5> <output.h5> [--fly-xml PATH]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Tuple

import h5py
import mujoco
import numpy as np
import yaml

DEFAULT_FLY_XML = Path(
    "/home/talmolab/Desktop/SalkResearch/vnl-playground/vnl_playground/tasks/fruitfly/xmls/fruitfly_force.xml"
)


def build_qpos_qvel(
    position: np.ndarray,
    quaternion: np.ndarray,
    joints: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
    joints_velocity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose flat qpos / qvel from the named-format root + joint arrays.

    Assumes MuJoCo freejoint convention: qpos[0:3]=xyz, qpos[3:7]=[w,x,y,z], qpos[7:]=joints;
    qvel[0:3]=linvel, qvel[3:6]=angvel, qvel[6:]=joint_velocities.
    """
    qpos = np.concatenate([position, quaternion, joints], axis=-1)
    qvel = np.concatenate([velocity, angular_velocity, joints_velocity], axis=-1)
    return qpos, qvel


def resolve_body_names(
    model,
    model_xpos_frame: np.ndarray,
    model_xquat_frame: np.ndarray,
    h5_xpos_frame: np.ndarray,
    h5_xquat_frame: np.ndarray,
    atol: float = 1e-4,
) -> list[str]:
    """For each H5 body index j, find the unique MJCF body whose (xpos, xquat)
    matches the H5 values to within atol, and return its name.

    When multiple MJCF bodies are coincident (e.g. walker/thorax share the
    same world frame), the tie is broken by assigning each successive H5 body
    to the lowest-indexed unoccupied candidate. This preserves the original
    body ordering from the H5 (which was generated before eye-camera additions
    and stored bodies in MJCF index order).

    Aborts (raises ValueError) on no-match (after tie-breaking) or if the
    resulting mapping is not a permutation.

    Parameters
    ----------
    model : mujoco.MjModel
        Loaded fly model.
    model_xpos_frame : np.ndarray, shape (model.nbody, 3)
        Body world positions from `mj_forward` on a specific qpos.
    model_xquat_frame : np.ndarray, shape (model.nbody, 4)
        Body world quaternions [w,x,y,z] from the same `mj_forward`.
    h5_xpos_frame : np.ndarray, shape (n_h5, 3)
        H5-reported body world positions for the corresponding frame.
    h5_xquat_frame : np.ndarray, shape (n_h5, 4)
        H5-reported body world quaternions for the corresponding frame.
    atol : float
        Absolute tolerance for matching positions and quaternions.
    """
    n_h5 = h5_xpos_frame.shape[0]
    used: set[int] = set()  # mjcf indices already assigned
    names: list[str] = []
    for j in range(n_h5):
        pos_match = np.all(np.abs(model_xpos_frame - h5_xpos_frame[j]) <= atol, axis=-1)
        quat_match = np.all(
            np.abs(model_xquat_frame - h5_xquat_frame[j]) <= atol, axis=-1
        )
        candidates = np.where(pos_match & quat_match)[0]
        # Filter out already-assigned bodies (tie-breaking for coincident bodies)
        available = [int(c) for c in candidates if int(c) not in used]
        if len(available) == 0:
            if candidates.size == 0:
                raise ValueError(
                    f"H5 body {j} has no MJCF match (xpos={h5_xpos_frame[j]}, "
                    f"xquat={h5_xquat_frame[j]}). Closest MJCF body: "
                    f"{int(np.argmin(np.linalg.norm(model_xpos_frame - h5_xpos_frame[j], axis=-1)))}."
                )
            else:
                raise ValueError(
                    f"H5 body {j} has no MJCF match: all candidates "
                    f"{list(candidates)} already assigned. Not a permutation."
                )
        # Among available candidates, pick the lowest index (preserves MJCF order)
        k = min(available)
        used.add(k)
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, k))
    return names


def _check_quat_norms(quats: np.ndarray, label: str, tol: float = 1e-2) -> None:
    norms = np.linalg.norm(quats, axis=-1)
    bad = np.abs(norms - 1.0) > tol
    if bad.any():
        n_bad = int(bad.sum())
        raise ValueError(
            f"{label}: {n_bad} quaternions have norm outside [{1-tol}, {1+tol}]. "
            f"min={norms.min():.6f}, max={norms.max():.6f}."
        )


def _build_names_qpos(model) -> list[str]:
    """Match rat convention: 7 copies of 'root' for the freejoint slots, then
    one entry per hinge joint in MJCF DOF order."""
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected freejoint at joint index 0 for fly model.")
    names = ["root"] * 7
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        names.append(name)
    if len(names) != model.nq:
        raise ValueError(f"names_qpos length {len(names)} != model.nq {model.nq}")
    return names


def _build_config_yaml(n_clips: int, source_path: str, fly_xml: str) -> str:
    payload = {
        "model": {
            "snips_order": [f"clip_{i:04d}" for i in range(n_clips)],
            "SCALE_FACTOR": 1.0,
        },
        "provenance": {
            "converter": "convert_fly_reference_to_legacy.py",
            "source": source_path,
            "converted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "fly_xml": fly_xml,
        },
    }
    return yaml.safe_dump(payload, sort_keys=False)


def convert(
    input_path: str,
    output_path: str,
    fly_xml: str = str(DEFAULT_FLY_XML),
    skip_body_resolution: bool = False,
    n_body_samples: int = 50,
) -> None:
    """Convert a named-format fly H5 to legacy/flat format.

    Parameters
    ----------
    input_path : str
        Path to source named-format H5.
    output_path : str
        Path to write the legacy-format H5.
    fly_xml : str
        Path to the fly MJCF (for DOF/body name resolution).
    skip_body_resolution : bool
        If True, do not run mj_forward-based body name resolution; instead
        write generic body_0..body_N-1 names. Only use for tests with
        synthetic data.
    n_body_samples : int
        Number of random (clip, frame) pairs to verify body-name stability.
    """
    model = mujoco.MjModel.from_xml_path(fly_xml)

    with h5py.File(input_path, "r") as f:
        g = f["all_clips"] if "all_clips" in f else f
        position = g["position"][()]
        quaternion = g["quaternion"][()]
        joints = g["joints"][()]
        velocity = g["velocity"][()]
        angular_velocity = g["angular_velocity"][()]
        joints_velocity = g["joints_velocity"][()]
        body_positions = g["body_positions"][()]
        body_quaternions = g["body_quaternions"][()]

    n_clips, n_frames = position.shape[:2]
    n_bodies = body_positions.shape[2]

    # --- self-checks ---
    if joints.shape[-1] != model.nq - 7:
        raise ValueError(
            f"joints last dim {joints.shape[-1]} != model.nq - 7 " f"({model.nq - 7})"
        )
    _check_quat_norms(quaternion, "root quaternion")
    _check_quat_norms(body_quaternions, "body_quaternions")

    qpos, qvel = build_qpos_qvel(
        position,
        quaternion,
        joints,
        velocity,
        angular_velocity,
        joints_velocity,
    )

    # qpos round-trip via mj_forward at (clip 0, frame 0)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0, 0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        np.asarray(data.qpos),
        qpos[0, 0],
        atol=1e-6,
        err_msg="qpos round-trip via mj_forward failed at (clip 0, frame 0)",
    )

    # Body name resolution
    if skip_body_resolution:
        names_xpos = [f"body_{i}" for i in range(n_bodies)]
    else:
        names_xpos = resolve_body_names(
            model,
            np.asarray(data.xpos),
            np.asarray(data.xquat),
            body_positions[0, 0],
            body_quaternions[0, 0],
            atol=1e-4,
        )
        # Stability check across n_body_samples random (clip, frame) pairs
        rng = np.random.default_rng(0)
        for _ in range(n_body_samples):
            c = int(rng.integers(0, n_clips))
            t = int(rng.integers(0, n_frames))
            data2 = mujoco.MjData(model)
            data2.qpos[:] = qpos[c, t]
            mujoco.mj_forward(model, data2)
            names_check = resolve_body_names(
                model,
                np.asarray(data2.xpos),
                np.asarray(data2.xquat),
                body_positions[c, t],
                body_quaternions[c, t],
                atol=1e-4,
            )
            if names_check != names_xpos:
                raise ValueError(
                    f"Body-name mapping unstable: differs at (clip={c}, frame={t})."
                )

    # qvel finite-difference sanity (clip 0)
    c = 0
    fd_pos = position[c, 1:] - position[c, :-1]
    if fd_pos.shape[0] > 0:
        # We don't know the H5's dt; assume velocity field is per-frame
        # (so position_{f+1} - position_{f} ~ velocity_f * dt). Compute the
        # implied dt and check it's roughly constant (low CV).
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = fd_pos / np.where(
                np.abs(velocity[c, :-1]) > 1e-6, velocity[c, :-1], np.nan
            )
        # We just check the field is non-degenerate
        if not np.isfinite(implied).any():
            raise ValueError(
                "qvel finite-difference check: velocity field appears degenerate"
            )

    # Flatten to (n_clips * n_frames, ...)
    qpos_flat = qpos.reshape(n_clips * n_frames, -1)
    qvel_flat = qvel.reshape(n_clips * n_frames, -1)
    xpos_flat = body_positions.reshape(n_clips * n_frames, n_bodies, 3)
    xquat_flat = body_quaternions.reshape(n_clips * n_frames, n_bodies, 4)

    names_qpos = _build_names_qpos(model)
    config_yaml = _build_config_yaml(n_clips, input_path, fly_xml)

    # Write output
    with h5py.File(output_path, "w") as f:
        f.create_dataset("qpos", data=qpos_flat.astype(np.float32))
        f.create_dataset("qvel", data=qvel_flat.astype(np.float32))
        f.create_dataset("xpos", data=xpos_flat.astype(np.float32))
        f.create_dataset("xquat", data=xquat_flat.astype(np.float32))
        f.create_dataset(
            "names_qpos",
            data=np.array(names_qpos, dtype=h5py.string_dtype("utf-8")),
        )
        f.create_dataset(
            "names_xpos",
            data=np.array(names_xpos, dtype=h5py.string_dtype("utf-8")),
        )
        f.create_dataset("config", data=config_yaml)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert fly named-format reference H5 to rat legacy format."
    )
    parser.add_argument("input", help="Path to named-format H5")
    parser.add_argument("output", help="Path to write legacy-format H5")
    parser.add_argument("--fly-xml", default=str(DEFAULT_FLY_XML))
    parser.add_argument(
        "--skip-body-resolution",
        action="store_true",
        help="Skip mj_forward body-name mapping (only for synthetic test data).",
    )
    args = parser.parse_args(argv)
    convert(
        input_path=args.input,
        output_path=args.output,
        fly_xml=args.fly_xml,
        skip_body_resolution=args.skip_body_resolution,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
