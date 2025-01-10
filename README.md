# track-mjx

This is a package for training control policies through motion capture tracking using deep reinforcement learning.

## Quick start (development)

We recommend using our Docker image for development. This image handles setting up the NVIDIA drivers with CUDA 12.6 support, EGL, miniforge and an SSH server for remote development.

### Docker (local)

> Make sure you have a NVIDIA GPU enabled Linux environment setup for this repo.

<!-- Need to re-test the docker system locally in Linux/Windows -->

Pull and run the docker image from the DockerHub registry:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -p 8888:22 scottyang17/track-mjx:vscode
```

The `8888` is the local port that you want to forward to. Choose one that's unoccupied as you'll use this later to connect to the Docker container from VSCode.

[See
here](https://github.com/talmolab/internal-dockerfiles/tree/3245903ec48b633ae205eeab0583d6413c32530b/remote-dev)
for more info on our Remote Dev Docker image.

This will soon be defined through the [`Dockerfile`](Dockerfile) in this repo (see [#1](https://github.com/talmolab/track-mjx/issues/1)).


### Run:AI (Salk)

1. Create a **new Job (Legacy)** → **Interactive** → set **Project** to `talmo-lab`.
2. Load the `remote-dev-track-mjx` template, or set this configuration:
    - **Image:** `scottyang17/track-mjx:vscode`
    - **Environment Variables:** `NVIDIA_DRIVER_CAPABILITIES` → `all`
    - **Port:** `External Port (Auto-generate)` → `22`
    - **Storage:** (see internal docs)
3. Submit the job. Once it starts running, you'll be able to see an internal IP and port to connect to.


### Setup VS Code Remote Dev

First, install the `Remote Development` (with id: ms-vscode-remote.vscode-remote-extensionpack) extension on vscode. Bring up the command palette, search and choose `Remote-SSH: Connect to Host` -> `Configure SSH Hosts` -> `<your ssh config path>`, and put following config:

```
Host local-testing
    HostName <ip>
    Port <port>
    User root
```

The `<ip>` will be `localhost` if running on the same machine, or the IP of the remote machine if running on a cluster.

Bring up your command palette, choose `Remote-SSH: Connect to Host` -> `track-mjx-remote-dev`, type in the password `root`, you are now connecting to the image.


### Installation

1. Clone the repository with the following command:
    ```bash
    git clone https://github.com/talmolab/track-mjx.git && cd track-mjx
    ```
2. Create a new development environment via `conda`:
    ```bash
    conda env create -f environment.yml
    ```
    This will install the necessary dependencies and install the package in editable mode.
3. Test the environment.
    Active the conda environment that was just installed:
    ```bash
    conda activate track_mjx
    ```
    Then run `jupyter lab` and execute the tests in [`notebooks/test_setup.ipynb`](notebooks/test_setup.ipynb). This will check if MuJoCo, GPU support and Jax appear to be working.


## Usage

### Training

The main training entrypoint is defined in [`track_mjx/train.py`](track_mjx/train.py) and relies on the config in [`track_mjx/config/rodent-mc-intention.yaml`](track_mjx/config/rodent-mc-intention.yaml).

After running `conda activate track_mjx`, you can run training with:

```bash
python -m track_mjx.train data_path="data/FlyReferenceClip.p" hydra.config_name="fly-mc-intention"
```

The `data_path` will need to point to a Pickle file with the outputs of [`stac-mjx`](https://github.com/talmolab/stac-mjx) (see [#23](https://github.com/talmolab/track-mjx/issues/23)).


### `screen` based terminal

This enables you to use persistent sessions even if you get disconnected from the Docker image. See [this issue](https://github.com/talmolab/track-mjx/issues/8#issuecomment-2469376476) for a workflow description.


## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](https://github.com/talmolab/track-mjx/blob/main/LICENSE) for details.