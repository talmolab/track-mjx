"""Install script for setuptools."""

from setuptools import setup, find_packages

# The flybody package can be installed in three modes:
#
# 1. Core installation: light-weight installation for experimenting with the
#    fly model in MuJoCo or with dm_control task environments. ML components
#    such as Tensorflow and Acme are not installed and policy rollouts and
#    training are not supported.
#    To install, use: pip install -e .
#
# 2. Add ML components: same as (1), plus Tensorflow, Acme to allow bringing
#    policy networks into play (e.g. for inference), but without training them.
#    To install, use: pip install -e .[tf]
#
# 3. Add training components: Same as (1) and (2), plus Ray to also allow
#    distributed policy training in the dm_control task environments.
#    To install, use: pip install -e .[ray]

core_requirements = [
    "mujoco",
    "dm_control",
    "mujoco-mjx",
    "mediapy",
]


dev_requirements = [
    "yapf",
    "ruff",
    "jupyterlab",
    "tqdm",
]

setup(
    name="track_mjx",
    version="0.1",
    packages=find_packages(),
    package_data={
        "track_mjx": ["track_mjx/assets/*.obj", "track_mjx/assets/*.xml"],
    },
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
    },
    author="Charles Zhang, Scott Yang, Kevin Bian, etc.",
    description="track-mjx, imitation learning and inverse kinematics for complex biomechanical models via reinforcement learning",
    url="https://github.com/talmolab/track-mjx",
)
