# track-mjx

This is a package for training control policies through motion capture tracking using deep reinforcement learning.

## Installation

### Option 1: `uv`

#### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- CUDA 12.x or 13.x (for GPU support, optional)

#### Installing `uv`

If you don't have uv installed:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/talmolab/track-mjx.git
cd track-mjx
```
2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install the package with your CUDA version:
```bash
# For CUDA 12.x:
uv pip install -e ".[cuda12]"
```
```bash
# For CUDA 13.x:
uv pip install -e ".[cuda13]"
```
```bash
# For CPU-only:
uv pip install -e .
```
```bash
# For development (includes testing and documentation tools):
uv pip install -e ".[cuda13,dev]"
```
4. Verify the installation:
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Available devices: {jax.devices()}')"
```
5. Test the environment:
    Execute the tests in [`notebooks/test_setup.ipynb`](notebooks/test_setup.ipynb). This will check if MuJoCo, GPU support and Jax appear to be working.

#### Alternative: Using `pip`

If you prefer using pip instead of uv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[cuda13]"  # or cuda12
```

### Troubleshooting

**CUDA version mismatch:**
- Check your CUDA version: `nvcc --version` or `nvidia-smi`
- Ensure you install the matching JAX CUDA version (cuda12 or cuda13)

**Import errors:**
- Verify the virtual environment is activated
- Try reinstalling: `uv pip install --force-reinstall -e ".[cuda13]"`

**GPU not detected:**
- Verify CUDA installation: `nvidia-smi`
- Check that JAX can see GPUs: `python -c "import jax; print(jax.devices())"`

Expected output:
- GPU: Should show `cuda` or `gpu` devices
- CPU: Should show `cpu` device

## Option 2: conda

1. Clone the repository:
    ```bash
    git clone https://github.com/talmolab/track-mjx.git && cd track-mjx
    ```
2. Create a new development environment via `conda` (this will create the necessary base environment):
    ```bash
    conda env create -f environment.yml
    ```
3. Activate the environment:
    ```bash
    conda activate track-mjx
    ```
4. Install the package with desired CUDA version:
    If your machine supports up to CUDA 13:
        ```bash
        pip install -e ".[cuda12]"
        ```
    If your machiine supports up to CUDA 12:
        ```bash
        pip install -e ".[cuda13]"
        ```
    This will install the package with the desired CUDA version.
5. Test the environment:
    Execute the tests in [`notebooks/test_setup.ipynb`](notebooks/test_setup.ipynb). This will check if MuJoCo, GPU support and Jax appear to be working.


## Usage

### Training

The main training entrypoint is defined in [`track_mjx/train.py`](track_mjx/train.py) and relies on the config in [`track_mjx/config/rodent-mc-intention.yaml`](track_mjx/config/rodent-mc-intention.yaml).

To download data, you can call:

```bash
gdown --id <google file id> -O data.h5
```

Run training with:

**Using uv:**
```bash
uv run python -m track_mjx.train data_path="data/RodentReferenceClip.h5"
```

**Using conda:**
```bash
conda activate track_mjx
python -m track_mjx.train data_path="data/RodentReferenceClip.h5"
```

The `data_path` will need to point to a Pickle file with the outputs of [`stac-mjx`](https://github.com/talmolab/stac-mjx) (see [#23](https://github.com/talmolab/track-mjx/issues/23)).


### `screen` based terminal

This enables you to use persistent sessions even if you get disconnected from the Docker image. See [this issue](https://github.com/talmolab/track-mjx/issues/8#issuecomment-2469376476) for a workflow description.


## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](https://github.com/talmolab/track-mjx/blob/main/LICENSE) for details.