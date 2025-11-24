# imports
import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

from typing import List, Any, Dict
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation

from sklearn.decomposition import PCA
from PIL import Image
from IPython.display import HTML

# TODO: Add other walker consts
from vnl_playground.tasks.rodent import consts as rodent_consts
from vnl_playground.tasks.fruitfly import consts as fruitfly_consts
from vnl_playground.tasks.mouse import consts as mouse_consts

import numpy as np

import multiprocessing as mp
import functools


def plot_pca_intention(
    idx: int,
    episode_start: int,
    pca: PCA,
    pca_projections: np.ndarray,
    clip_idx: int,
    feature_name: str,
    n_components: int = 4,
    terminated: bool = False,
    window_size: int = 530,
) -> np.ndarray:
    """Plot PCA intention progression of the episode.

    Args:
        idx (int): The current timestep.
        episode_start (int): The start timestep of the episode.
        pca (PCA): The PCA object fitted on the dataset.
        pca_projections (np.ndarray): The PCA projection of the episode, shape (timestep, n_components).
        clip_idx (int): The clip index.
        feature_name (str): The feature name.
        n_components (int, optional): The number of PCA components to plot. Defaults to 4.
        terminated (bool, optional): Whether the episode is terminated. Defaults to False.
        window_size (int, optional): The window size of the plot. Defaults to 530.

    Returns:
        np.ndarray: The image array of the plot.
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
        plt.xlim(
            idx_in_this_episode - window_size, idx_in_this_episode
        )  # dynamically move xlim as time progress
    plt.ylim(*y_lim)
    plt.legend(loc="upper right")
    plt.xlabel("Timestep")
    plt.title(
        f"PCA {feature_name} Progression for Clip {clip_idx}"
    )  # TODO make it configurable
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
    rollout: Dict[str, Any],
    pca_projections: np.ndarray,
    n_components: int = 4,
    feature_name: str = "ctrl",
) -> List[np.ndarray]:
    """Render rollout frames concatenated with PCA progression plots.

    Args:
        rollout (Dict[str, Any]): The rollout dictionary.
        pca_projections (np.ndarray): The PCA projections of the rollout.
        n_components (int, optional): The number of PCA components to plot. Defaults to 4.
        feature_name (str, optional): The feature name. Defaults to "ctrl".

    Returns:
        List[np.ndarray]: List of frames of the rendering concatenated with PCA plots.
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
        len(frames_mujoco) - 1,
        episode_start,
        pca_projections,
        clip_idx,
        feature_name,
        n_components,
        terminated=True,
    )
    plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)
    for _ in range(50):
        concat_frames.append(
            np.hstack([frames_mujoco[-1], reward_plot])
        )  # create stoppage when episode terminates
    return concat_frames


def display_video(frames: List[np.ndarray], framerate: int = 30) -> HTML:
    """Display a video from a list of frames.

    Args:
        frames (List[np.ndarray]): List of frames with shape (height, width, 3).
        framerate (int, optional): The framerate of the video. Defaults to 30.

    Returns:
        HTML: HTML video object for display in Jupyter notebooks.
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

    def update(frame: np.ndarray) -> List[Any]:
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return HTML(anim.to_html5_video())
