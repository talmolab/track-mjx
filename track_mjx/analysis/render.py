# imports
import os

os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "egl")

from typing import List, Tuple, Callable, Any, Dict
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
from track_mjx.environment.walker.utils import _scale_body_tree, _recolour_tree

import mujoco
from pathlib import Path
from dm_control import mjcf as mjcf_dm
from dm_control.locomotion.walkers import rescale
import imageio
import numpy as np

import multiprocessing as mp
import functools


_GHOST_RENDER_XML_PATHS = {
    # "rodent": "assets/rodent/rodent_ghostpair_scale080.xml",
    "fly": "assets/fruitfly/fruitfly_force_pair.xml",
}


def agg_backend_context(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to switch to a headless backend during function execution.

    Args:
        func (Callable[..., Any]): The function to decorate.

    Returns:
        Callable[..., Any]: The wrapped function using headless backend.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        orig_backend = matplotlib.get_backend()
        matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
        # Code to execute BEFORE the original function
        result = func(*args, **kwargs)
        # Code to execute AFTER the original function
        plt.close("all")  # Figure auto-closing upon backend switching is deprecated.
        matplotlib.use(orig_backend)
        return result

    return wrapper


def make_ghost_pair(
    input_xml_str: str,
    *,
    scale: float = 0.80,
    rgba: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.2),
) -> Tuple[mujoco.MjSpec, mujoco.MjModel, str]:
    """Build output XML containing the original model plus a ghost copy.

    Args:
        input_xml_str (str): The XML string of the original model.
        scale (float, optional): Scale factor for the ghost model. Defaults to 0.80.
        rgba (Tuple[float, float, float, float], optional): Color and transparency for ghost model. Defaults to (0.8, 0.8, 0.8, 0.2).

    Returns:
        Tuple[mujoco.MjSpec, mujoco.MjModel, str]: The modified MjSpec, compiled MjModel, and XML string.
    """

    # ---------------------------------------------------------------------
    # A) Load the original as a spec
    # ---------------------------------------------------------------------
    base = mujoco.MjSpec.from_string(input_xml_str)

    for top in base.worldbody.bodies:
        _scale_body_tree(top, scale)

    # ---------------------------------------------------------------------
    # B) Deep‑copy the spec to obtain the second (ghost) rodent
    # ---------------------------------------------------------------------
    ghost = base.copy()

    # the first body is the floor
    floor = ghost.worldbody.bodies[0]
    ghost.detach_body(floor)

    for top in ghost.worldbody.bodies:
        _recolour_tree(top, rgba)

    # ---------------------------------------------------------------------
    # C) Shrink & recolour the copy
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # D) Attach the ghost back into the parent spec
    #    – prefix ensures unique names
    # ---------------------------------------------------------------------
    anchor = base.worldbody.add_site(name="ghost_anchor", pos=[0, 0, 0])
    base.attach(ghost, prefix="ghost_", site=anchor)

    # ---------------------------------------------------------------------
    # E) Compile & write out
    # ---------------------------------------------------------------------
    model = base.compile()
    xml = base.to_xml()
    return base, model, xml


def make_rollout_renderer(
    cfg: Any,
) -> Tuple[mujoco.Renderer, mujoco.MjModel, mujoco.MjData, mujoco.MjvOption]:
    """Create a renderer and related MuJoCo model and data for rollout visualization.

    Args:
        cfg (Any): Configuration object with environment and walker settings.

    Returns:
        Tuple[mujoco.Renderer, mujoco.MjModel, mujoco.MjData, mujoco.MjvOption]: Renderer, model, data, and scene options.
    """

    walker_config = cfg["walker_config"]

    if cfg.env_config.walker_name == "rodent":
        # Programmatically build ghost-pair model from base rodent.xml
        base_xml_path = (
            Path(__file__).resolve().parent.parent
            / "environment"
            / "walker"
            / "assets"
            / "rodent"
            / "rodent.xml"
        )
        xml_str = base_xml_path.read_text()
        # produce (spec, model, xml) with correct scale
        _, mj_model, _ = make_ghost_pair(
            xml_str, scale=walker_config.rescale_factor, rgba=(0.8, 0.8, 0.8, 0.2)
        )
    elif cfg.env_config.walker_name == "fly":
        _XML_PATH = (
            Path(__file__).resolve().parent.parent
            / "environment"
            / "walker"
            / _GHOST_RENDER_XML_PATHS[cfg.env_config.walker_name]
        )
        spec = mujoco.MjSpec()
        spec = spec.from_file(str(_XML_PATH))

        # in training scaled by this amount as well
        for geom in spec.geoms:
            if geom.size is not None:
                geom.size *= walker_config.rescale_factor
            if geom.pos is not None:
                geom.pos *= walker_config.rescale_factor

        mj_model = spec.compile()
    else:
        raise ValueError(f"Unknown walker_name: {cfg.env_config.walker_name}")

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]

    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)

    site_id = [
        mj_model.site(i).id
        for i in range(mj_model.nsite)
        if "-0" in mj_model.site(i).name
    ]
    for id in site_id:
        mj_model.site(id).rgba = [1, 0, 0, 1]

    # visual mujoco rendering
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    return renderer, mj_model, mj_data, scene_option


def render_rollout(
    cfg: Any,
    rollout: Dict[str, Any],
    height: int = 480,
    width: int = 640,
) -> Tuple[List[np.ndarray], float]:
    """Render a rollout from saved qposes.

    Args:
        cfg (Any): Configuration object with environment settings.
        rollout (Dict[str, Any]): A dictionary containing the qposes of the reference and rollout trajectories.
        height (int, optional): Height of the rendered frames. Defaults to 480.
        width (int, optional): Width of the rendered frames. Defaults to 640.

    Returns:
        Tuple[List[np.ndarray], float]: List of frames of the rendering and the rendering FPS.
    """
    qposes_ref, qposes_rollout = rollout["qposes_ref"], rollout["qposes_rollout"]
    renderer, mj_model, mj_data, scene_option = make_rollout_renderer(cfg)

    # Calulate realtime rendering fps
    render_fps = (
        1.0 / mj_model.opt.timestep
    ) / cfg.env_config.env_args.physics_steps_per_control_step

    # save rendering and log to wandb
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames = []
    print("MuJoCo Rendering...")
    for qpos1, qpos2 in tqdm(
        zip(qposes_rollout, qposes_ref), total=len(qposes_rollout)
    ):
        mj_data.qpos = np.append(qpos1, qpos2)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(
            mj_data, camera=cfg.env_config.render_camera_name, scene_option=scene_option
        )
        pixels = renderer.render()
        frames.append(pixels)
    return frames, render_fps


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
