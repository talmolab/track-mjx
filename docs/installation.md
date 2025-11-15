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

#### Option 1: uv (Recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager that handles dependencies and virtual environments. **This is the recommended method for CPU-only development on macOS and other non-Linux platforms**, as well as GPU-enabled Linux systems.

**Tested on:**
- **Linux**: Ubuntu 24.04 (headless) with CUDA 13.0, EGL rendering (no X11 required)
- **macOS**: Apple Silicon (M2 Pro) with CPU-only JAX, GLFW rendering

1. Install `uv` if you haven't already:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/talmolab/track-mjx.git && cd track-mjx
    ```

3. Install all dependencies (creates `.venv` automatically):
    ```bash
    uv sync --all-extras
    ```

4. Run the example notebook (overwrites with outputs):
    ```bash
    uv run jupyter nbconvert --execute --to notebook --inplace notebooks/download_and_run_rodent.ipynb
    ```

5. Verify GPU/CPU detection:
    ```bash
    uv run python -c "import jax; print('Devices:', jax.devices())"
    ```
    - **Linux**: You should see your CUDA devices (e.g., `[CudaDevice(id=0), CudaDevice(id=1)]`)
    - **macOS**: You will see `[CpuDevice(id=0)]` (GPU acceleration not supported with JAX 0.6.2)

**Platform-specific notes:**
- **Linux**: The `pyproject.toml` includes all necessary NVIDIA CUDA libraries for JAX GPU support. No additional system packages are required beyond CUDA drivers.
- **macOS**: Uses CPU-only JAX as `jax-metal` 0.1.1 is incompatible with JAX 0.6.2. Platform markers ensure CUDA dependencies are not installed on macOS.

**Troubleshooting:**

If you encounter EGL/OpenGL errors when rendering (e.g., `AttributeError: 'NoneType' object has no attribute 'glGetError'`), you may need to install system EGL libraries:

```bash
sudo apt update && sudo apt-get install libglapi-mesa libegl-mesa0 libegl1 libopengl0
```

After installing, try running your script again with `uv run`.