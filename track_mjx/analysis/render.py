# imports
import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation

from sklearn.decomposition import PCA
from PIL import Image
from IPython.display import HTML


from track_mjx.environment.task.multi_clip_tracking import MultiClipTracking
from track_mjx.environment.walker.rodent import Rodent

import mujoco
from pathlib import Path
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale
import imageio
import numpy as np

import multiprocessing as mp
import functools


def agg_backend_context(func):
    """
    Decorator to switch to a headless backend during function execution.
    """

    def wrapper(*args, **kwargs):
        orig_backend = matplotlib.get_backend()
        matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
        # Code to execute BEFORE the original function
        result = func(*args, **kwargs)
        # Code to execute AFTER the original function
        plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
        matplotlib.use(orig_backend)
        return result

    return wrapper


def render_from_saved_rollout(
    rollout: dict,
    walker_type: str = "rodent",
) -> list:
    """
    Render a rollout from saved qposes.

    Args:
        rollout (dict): A dictionary containing the qposes of the reference and rollout trajectories.

    Returns:
        list: list of frames of the rendering
    """
    qposes_ref, qposes_rollout = rollout["qposes_ref"], rollout["qposes_rollout"]
    if walker_type == "rodent":
        pair_render_xml_path = str(
            (
                Path(__file__).parent
                / ".."
                / "environment"
                / "walker"
                / "assets"
                / "rodent"
                / "rodent_ghostpair_scale080.xml"
            ).resolve()
        )
    elif walker_type == "fly":
        pair_render_xml_path = str(
            (
                Path(__file__).parent
                / ".."
                / "environment"
                / "walker"
                / "assets"
                / "fruitfly"
                / "fruitfly_force_pair.xml"
            ).resolve()
        )
    else:
        raise ValueError(f"Unsupported walker type: {walker_type}")
    # TODO: Make this ghost rendering walker agonist
    root = mjcf_dm.from_path(pair_render_xml_path)
    rescale.rescale_subtree(
        root,
        0.9 / 0.8,
        0.9 / 0.8,
    )

    mj_model = mjcf_dm.Physics.from_mjcf_model(root).model.ptr
    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    print("MuJoCo Rendering...")
    for qpos1, qpos2 in tqdm(zip(qposes_rollout, qposes_ref), total=len(qposes_rollout)):
        mj_data.qpos = np.append(qpos1, qpos2)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(
            mj_data,
            camera="close_profile",
        )
        pixels = renderer.render()
        frames.append(pixels)
    return frames


def plot_pca_intention(
    idx,
    episode_start,
    pca: PCA,
    pca_projections: np.ndarray,
    clip_idx: int,
    feature_name: str,
    n_components: int = 4,
    terminated: bool = False,
    window_size: int = 530,
) -> np.ndarray:
    """
    plot pca intention progression of the episode
    Args:
        idx: the current timestep
        episode_start: the start timestep of the episode
        pca: the pca object fitted on the dataset
        pca_projections: the pca projection of the episode, shape (timestep, n_components)
        clip_idx: the clip index
        feature_name: the feature name
        n_components: the number of pca components to plot
        ylim: the y-axis limit
        terminated: whether the episode is terminated
        window_size: the window size of the plot
    Returns:
        np.ndarray: the image array of the plot
    """
    max_y = np.max(list(pca_projections[:, :n_components]))
    min_y = np.min(list(pca_projections[:, :n_components]))
    y_lim = (min_y - 0.2, max_y + 0.2)
    idx_in_this_episode = idx - episode_start  # the current timestep in this episode
    plt.figure(figsize=(9.6, 4.8))
    for pc_ind in range(n_components):
        # Plot the PCA projection of the episode
        plt.plot(
            pca_projections[episode_start:idx, pc_ind],
            label=f"PC {pc_ind} ({pca.explained_variance_ratio_[pc_ind]*100:.1f}%)",
        )
        plt.scatter(idx - episode_start, pca_projections[idx - 1, pc_ind])
    if terminated:
        # Mark the episode termination
        plt.axvline(x=idx - episode_start, color="r", linestyle="-")
        plt.text(
            idx - episode_start - 8,  # Adjust the x-offset as needed
            sum(y_lim) / 2,  # Adjust the y-position as needed
            "Episode Terminated",
            color="r",
            rotation=90,
        )  # Rotate the text vertically
    if idx_in_this_episode <= window_size:
        plt.xlim(0, window_size)
    else:
        plt.xlim(idx_in_this_episode - window_size, idx_in_this_episode)  # dynamically move xlim as time progress
    plt.ylim(*y_lim)
    plt.legend(loc="upper right")
    plt.xlabel("Timestep")
    plt.title(f"PCA {feature_name} Progression for Clip {clip_idx}")  # TODO make it configurable
    # Get the current figure
    fig = plt.gcf()
    # Create a canvas for rendering
    canvas = FigureCanvasAgg(fig)
    # Render the canvas to a buffer
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    # Convert the buffer to a PIL Image
    image = Image.frombytes("RGBA", (width, height), s)
    rgb_array = np.array(image.convert("RGB"))
    return rgb_array


def render_with_pca_progression(
    rollout: dict,
    pca_projections: np.ndarray,
    n_components: int = 4,
    feature_name: str = "ctrl",
) -> list:
    """
    render with the rewards progression graph concat alongside with the rendering.

    Args:
        rollout (dict): the rollout dictionary
        pca_projections (np.ndarray): the pca projections of the rollout
        n_components (int): the number of pca components to plot
        feature_name (str): the feature name

    Returns:
        list: list of frames of the rendering
    """
    frames_mujoco = render_from_saved_rollout(rollout)[1:]
    # skip the first frame, since we don't have intention for the first frame
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    clip_idx = int(rollout["info"][0]["clip_idx"])
    worker = functools.partial(
        plot_pca_intention,
        episode_start=0,
        clip_idx=clip_idx,
        pca_projections=pca_embedded,
        n_components=n_components,
        feature_name=feature_name,
    )
    print("Rendering with PCA progression...")
    # Use multiprocessing to parallelize the rendering of the reward graph
    with mp.Pool(processes=mp.cpu_count()) as pool:
        frames_pca = pool.map(worker, range(len(rollout["qposes_rollout"])))
    concat_frames = []
    episode_start = 0
    # implement reset logics of the reward graph too.
    print("Concatenating frames...")
    for idx, frame in tqdm(enumerate(frames_mujoco)):
        concat_frames.append(np.hstack([frame, frames_pca[idx]]))
    reward_plot = plot_pca_intention(
        len(frames_mujoco) - 1, episode_start, pca_projections, clip_idx, feature_name, n_components, terminated=True
    )
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)
    for _ in range(50):
        concat_frames.append(np.hstack([frames_mujoco[-1], reward_plot]))  # create stoppage when episode terminates
    return concat_frames


def display_video(frames, framerate=30):
    """
    Args:
        frames (array): (n_frames, height, width, 3)
        framerate (int): the framerate of the video
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())
