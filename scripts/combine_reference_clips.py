"""Combine Monsees gap-crossing + original rodent reference clips into one H5.

Applies init-only floor lift to Monsees data (15 frame lift + 20 frame fade),
pads Monsees clips from 100→250 frames, concatenates with original rodent clips,
and stores per-clip `clip_lengths` so the imitation env can terminate correctly.

Usage:
    python scripts/combine_reference_clips.py
"""

from __future__ import annotations

import numpy as np
import h5py
import mujoco
import yaml
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


# --- Paths ---
MONSEES_H5 = Path("/home/talmolab/Desktop/SalkResearch/monsees-retarget/output/monsees_gap_reference_clips.h5")
ORIGINAL_H5 = Path("/home/talmolab/Desktop/SalkResearch/track-mjx/data/rodent/rodent_reference_clips.h5")
OUTPUT_H5 = Path("/home/talmolab/Desktop/SalkResearch/track-mjx/data/rodent/rodent_combined_reference_clips.h5")
XML_PATH = Path("/home/talmolab/Desktop/SalkResearch/stac-mjx/models/rodent_relaxed.xml")

TARGET_CLIP_LENGTH = 250  # Pad Monsees clips to match original
LIFT_FRAMES = 15          # Full floor lift for first N frames
FADE_FRAMES = 20          # Smooth transition from lift to raw


def compute_floor_lift(qpos_clip: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Compute per-frame floor penetration for a single clip.

    Returns the needed lift (positive = needs lifting) for each frame.
    """
    data = mujoco.MjData(model)
    contact_geom_ids = [i for i in range(model.ngeom) if model.geom_contype[i] > 0]
    n_frames = qpos_clip.shape[0]
    min_geom_z = np.zeros(n_frames)

    for f in range(n_frames):
        data.qpos[:] = qpos_clip[f]
        mujoco.mj_forward(model, data)
        frame_min = float("inf")
        for gid in contact_geom_ids:
            gz = data.geom_xpos[gid, 2]
            gtype = model.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = model.geom_size[gid, 0]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r = model.geom_size[gid, 0]
            elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                r = model.geom_size[gid, 2]
            else:
                r = 0
            frame_min = min(frame_min, gz - r)
        min_geom_z[f] = frame_min

    clearance = 0.002  # 2mm above floor
    raw_lift = np.maximum(0.0, -min_geom_z + clearance)
    return raw_lift


def apply_init_only_lift(
    qpos_clips: np.ndarray,
    xpos_clips: np.ndarray,
    model: mujoco.MjModel,
    lift_frames: int = 15,
    fade_frames: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply floor lift only to initial frames, then fade to raw data.

    For each clip:
      - Frames [0, lift_frames): full smoothed floor lift
      - Frames [lift_frames, lift_frames + fade_frames): linear fade to zero
      - Frames [lift_frames + fade_frames, ...): raw data (no lift)
    """
    n_clips, clip_len = qpos_clips.shape[:2]
    qpos_out = qpos_clips.copy()
    xpos_out = xpos_clips.copy()

    for c in range(n_clips):
        raw_lift = compute_floor_lift(qpos_clips[c], model)

        # Smooth the lift signal within the init region
        sigma = 3
        smoothed = gaussian_filter1d(raw_lift, sigma=sigma, mode="nearest")

        # Build the fade mask: 1.0 for init, linear fade, 0.0 for raw
        mask = np.zeros(clip_len)
        mask[:lift_frames] = 1.0
        if fade_frames > 0:
            fade_end = min(lift_frames + fade_frames, clip_len)
            fade_len = fade_end - lift_frames
            mask[lift_frames:fade_end] = np.linspace(1.0, 0.0, fade_len)

        lift = smoothed * mask

        # Apply to root Z and all body Z positions
        qpos_out[c, :, 2] += lift
        xpos_out[c, :, :, 2] += lift[:, None]

    return qpos_out, xpos_out


def pad_clips(
    data: dict[str, np.ndarray],
    real_length: int,
    target_length: int,
) -> dict[str, np.ndarray]:
    """Pad clips from real_length to target_length by repeating last frame."""
    if real_length >= target_length:
        return data

    padded = {}
    for key, arr in data.items():
        # arr shape: (n_clips, real_length, ...)
        n_clips = arr.shape[0]
        pad_len = target_length - real_length
        last_frame = arr[:, -1:, ...]  # (n_clips, 1, ...)
        padding = np.repeat(last_frame, pad_len, axis=1)
        padded[key] = np.concatenate([arr, padding], axis=1)

    return padded


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    # --- Load Monsees gap data ---
    print(f"Loading Monsees data: {MONSEES_H5}")
    with h5py.File(MONSEES_H5, "r") as f:
        m_qpos = f["qpos"][:]
        m_qvel = f["qvel"][:]
        m_xpos = f["xpos"][:]
        m_xquat = f["xquat"][:]
        m_config = yaml.safe_load(
            f["config"][()].decode() if isinstance(f["config"][()], bytes) else f["config"][()]
        )

    monsees_clip_len = 100
    n_monsees = m_qpos.shape[0] // monsees_clip_len
    print(f"  {n_monsees} clips x {monsees_clip_len} frames")

    # Reshape to clips
    m_qpos = m_qpos.reshape(n_monsees, monsees_clip_len, -1)
    m_qvel = m_qvel.reshape(n_monsees, monsees_clip_len, -1)
    m_xpos = m_xpos.reshape(n_monsees, monsees_clip_len, *m_xpos.shape[1:])
    m_xquat = m_xquat.reshape(n_monsees, monsees_clip_len, *m_xquat.shape[1:])

    # Apply init-only floor lift
    print("  Applying init-only floor lift...")
    m_qpos, m_xpos = apply_init_only_lift(
        m_qpos, m_xpos, model, lift_frames=LIFT_FRAMES, fade_frames=FADE_FRAMES
    )

    # Pad to target length
    print(f"  Padding {monsees_clip_len} → {TARGET_CLIP_LENGTH} frames...")
    monsees_data = pad_clips(
        {"qpos": m_qpos, "qvel": m_qvel, "xpos": m_xpos, "xquat": m_xquat},
        real_length=monsees_clip_len,
        target_length=TARGET_CLIP_LENGTH,
    )

    monsees_clip_lengths = np.full(n_monsees, monsees_clip_len, dtype=np.int32)

    # Extract Monsees clip names
    monsees_clip_names = []
    if "model" in m_config and "snips_order" in m_config["model"]:
        monsees_clip_names = m_config["model"]["snips_order"]
    else:
        monsees_clip_names = [f"MonseeGap_{i:04d}" for i in range(n_monsees)]

    # --- Load original rodent data ---
    print(f"\nLoading original data: {ORIGINAL_H5}")
    with h5py.File(ORIGINAL_H5, "r") as f:
        o_qpos = f["qpos"][:]
        o_qvel = f["qvel"][:]
        o_xpos = f["xpos"][:]
        o_xquat = f["xquat"][:]
        o_names_qpos = f["names_qpos"][()].astype(str)
        o_names_xpos = f["names_xpos"][()].astype(str)
        o_config = yaml.safe_load(
            f["config"][()].decode() if isinstance(f["config"][()], bytes) else f["config"][()]
        )

    n_original = o_qpos.shape[0] // TARGET_CLIP_LENGTH
    print(f"  {n_original} clips x {TARGET_CLIP_LENGTH} frames")

    # Reshape to clips
    o_qpos = o_qpos.reshape(n_original, TARGET_CLIP_LENGTH, -1)
    o_qvel = o_qvel.reshape(n_original, TARGET_CLIP_LENGTH, -1)
    o_xpos = o_xpos.reshape(n_original, TARGET_CLIP_LENGTH, *o_xpos.shape[1:])
    o_xquat = o_xquat.reshape(n_original, TARGET_CLIP_LENGTH, *o_xquat.shape[1:])

    original_clip_lengths = np.full(n_original, TARGET_CLIP_LENGTH, dtype=np.int32)

    # Extract original clip names
    original_clip_names = []
    if "model" in o_config and "snips_order" in o_config["model"]:
        original_clip_names = o_config["model"]["snips_order"]
    else:
        original_clip_names = [f"Original_{i:04d}" for i in range(n_original)]

    # --- Combine ---
    print(f"\nCombining: {n_original} original + {n_monsees} Monsees = {n_original + n_monsees} clips")

    # Concatenate along clip dimension, then flatten back to (total_frames, ...)
    combined = {}
    for key in ["qpos", "qvel", "xpos", "xquat"]:
        m_arr = monsees_data[key]
        o_arr = locals()[f"o_{key}"]
        cat = np.concatenate([o_arr, m_arr], axis=0)  # original first
        # Flatten back: (n_clips, clip_len, ...) -> (n_clips * clip_len, ...)
        combined[key] = cat.reshape(-1, *cat.shape[2:]).astype(np.float32)

    clip_lengths = np.concatenate([original_clip_lengths, monsees_clip_lengths])
    all_clip_names = original_clip_names + monsees_clip_names
    n_total = n_original + n_monsees

    # Build combined config
    combined_config = {
        "model": {
            "snips_order": all_clip_names,
            "SCALE_FACTOR": 0.9,
        },
        "source": {
            "original": str(ORIGINAL_H5),
            "monsees": str(MONSEES_H5),
            "n_original_clips": int(n_original),
            "n_monsees_clips": int(n_monsees),
            "monsees_real_clip_length": int(monsees_clip_len),
            "padded_clip_length": int(TARGET_CLIP_LENGTH),
        },
    }

    # --- Save ---
    print(f"\nSaving to: {OUTPUT_H5}")
    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_H5, "w") as f:
        for key, arr in combined.items():
            f.create_dataset(key, data=arr)
            print(f"  {key}: {arr.shape} {arr.dtype}")

        f.create_dataset("names_qpos", data=np.array(o_names_qpos, dtype="S"))
        f.create_dataset("names_xpos", data=np.array(o_names_xpos, dtype="S"))
        f.create_dataset("config", data=yaml.dump(combined_config))
        f.create_dataset("clip_lengths", data=clip_lengths)
        print(f"  clip_lengths: {clip_lengths.shape} (original={TARGET_CLIP_LENGTH}, monsees={monsees_clip_len})")

    print(f"\nDone! {n_total} total clips ({n_total * TARGET_CLIP_LENGTH} frames)")


if __name__ == "__main__":
    main()
