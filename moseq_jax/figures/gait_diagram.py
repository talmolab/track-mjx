"""Gait diagram rendering utilities.

Computes foot contacts from qpos via MuJoCo forward kinematics and renders
ethogram-style gait diagrams (HR, HL, FR, FL rows) as numpy images.

The rodent model foot geom names (with ``-rodent`` suffix):
  - HR: ``foot_R_collision-rodent``
  - HL: ``foot_L_collision-rodent``
  - FR: ``hand_R_collision-rodent``
  - FL: ``hand_L_collision-rodent``
"""

from __future__ import annotations

import mujoco
import numpy as np

# Limb order (top to bottom in diagram) and geom names
LIMB_ORDER = ["HR", "HL", "FR", "FL"]
LIMB_GEOM_NAMES = {
    "HR": "foot_R_collision-rodent",
    "HL": "foot_L_collision-rodent",
    "FR": "hand_R_collision-rodent",
    "FL": "hand_L_collision-rodent",
}

# Default contact threshold: foot geom z < this = on ground
DEFAULT_Z_THRESH = 0.008


def compute_foot_contacts(
    mj_model: mujoco.MjModel,
    qpos_trajectory: np.ndarray,
    z_thresh: float = DEFAULT_Z_THRESH,
    base_nq: int | None = None,
) -> np.ndarray:
    """Compute binary foot contact array from a qpos trajectory.

    Args:
        mj_model: Compiled MuJoCo model (base model, not ghost).
        qpos_trajectory: ``[T, nq]`` array.
        z_thresh: Foot geom z below this → contact (1).
        base_nq: If given, only set ``qpos[:base_nq]``.

    Returns:
        ``[T, 4]`` bool array, columns in ``LIMB_ORDER`` (HR, HL, FR, FL).
        True = foot on ground (stance), False = foot in air (swing).
    """
    T = len(qpos_trajectory)
    nq = base_nq or mj_model.nq
    data = mujoco.MjData(mj_model)

    geom_ids = {}
    for limb, gname in LIMB_GEOM_NAMES.items():
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            # Try without suffix
            gid = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname.replace("-rodent", "")
            )
        geom_ids[limb] = gid

    contacts = np.zeros((T, 4), dtype=bool)

    for t in range(T):
        data.qpos[:nq] = qpos_trajectory[t, :nq]
        mujoco.mj_forward(mj_model, data)
        for li, limb in enumerate(LIMB_ORDER):
            gid = geom_ids[limb]
            if gid >= 0:
                contacts[t, li] = data.geom_xpos[gid, 2] < z_thresh

    return contacts


def render_gait_bar(
    contacts: np.ndarray,
    current_frame: int,
    width: int,
    bar_height: int = 48,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    contact_color: tuple[int, int, int] = (30, 30, 30),
    swing_color: tuple[int, int, int] = (220, 220, 220),
    playhead_color: tuple[int, int, int] = (220, 50, 50),
    label_color: tuple[int, int, int] = (80, 80, 80),
) -> np.ndarray:
    """Render a gait diagram bar image for the current frame.

    Shows the full contact timeline with a red playhead at ``current_frame``.

    Args:
        contacts: ``[T, 4]`` bool array from ``compute_foot_contacts``.
        current_frame: Current timestep (red vertical line).
        width: Output image width in pixels.
        bar_height: Total height of the gait bar.
        bg_color: Background colour.
        contact_color: Colour for stance (foot on ground).
        swing_color: Colour for swing (foot in air).
        playhead_color: Colour for the current-frame indicator.
        label_color: Colour for limb labels.

    Returns:
        ``[bar_height, width, 3]`` uint8 RGB image.
    """
    T, n_limbs = contacts.shape
    row_h = bar_height // n_limbs
    label_w = 30  # pixels reserved for limb labels
    timeline_w = width - label_w

    img = np.full((bar_height, width, 3), bg_color, dtype=np.uint8)

    for li, limb in enumerate(LIMB_ORDER):
        y0 = li * row_h
        y1 = y0 + row_h - 1  # 1px gap between rows

        # Draw contact/swing blocks
        for px in range(timeline_w):
            t_idx = int(px * T / timeline_w)
            t_idx = min(t_idx, T - 1)
            color = contact_color if contacts[t_idx, li] else swing_color
            img[y0:y1, label_w + px] = color

        # Limb label
        _draw_text_simple(img, limb, x=2, y=y0 + 2, color=label_color, scale=0.8)

    # Playhead
    px = label_w + int(current_frame * timeline_w / max(T, 1))
    px = min(px, width - 1)
    img[:, max(px - 1, 0) : px + 2] = playhead_color

    return img


def render_gait_bar_colored(
    contacts: np.ndarray,
    current_frame: int,
    width: int,
    behavior_boundaries: list[tuple[str, int, int]],
    behavior_colors: dict[str, tuple[int, int, int]],
    bar_height: int = 48,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    swing_color: tuple[int, int, int] = (235, 235, 235),
    playhead_color: tuple[int, int, int] = (220, 50, 50),
    label_color: tuple[int, int, int] = (80, 80, 80),
) -> np.ndarray:
    """Like ``render_gait_bar`` but contact blocks use behavior-phase colours."""
    T, n_limbs = contacts.shape
    row_h = bar_height // n_limbs
    label_w = 30
    timeline_w = width - label_w

    img = np.full((bar_height, width, 3), bg_color, dtype=np.uint8)

    for li, limb in enumerate(LIMB_ORDER):
        y0 = li * row_h
        y1 = y0 + row_h - 1

        for px in range(timeline_w):
            t_idx = int(px * T / timeline_w)
            t_idx = min(t_idx, T - 1)

            if contacts[t_idx, li]:
                # Find behavior at this timestep
                color = (30, 30, 30)  # default dark
                for beh_name, beh_start, beh_end in behavior_boundaries:
                    if beh_start <= t_idx < beh_end:
                        color = behavior_colors.get(beh_name, color)
                        break
            else:
                color = swing_color

            img[y0:y1, label_w + px] = color

        _draw_text_simple(img, limb, x=2, y=y0 + 2, color=label_color, scale=0.8)

    # Playhead
    px = label_w + int(current_frame * timeline_w / max(T, 1))
    px = min(px, width - 1)
    img[:, max(px - 1, 0) : px + 2] = playhead_color

    return img


def _draw_text_simple(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.35,
) -> None:
    """Draw text on image using cv2 (thin, small)."""
    import cv2

    cv2.putText(
        img,
        text,
        (x, y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )
