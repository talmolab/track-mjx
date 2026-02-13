# track-mjx

This is a package for training control policies through motion imitation using deep reinforcement learning. Part of [MIMIC-MJX](https://mimic-mjx.talmolab.org/), along with [STAC-MJX](https://github.com/talmolab/stac-mjx) (a tool for performing inverse kinematics on markerless motion tracking data).

## IMPORTANT (For reviewers and new users):
**Please use the latest stable version of track-mjx (v0.0.1) for notebook demos and running rodent training example.**

## Prerelease (track-mjx >= v1.0.0)
**track-mjx v1 will soon include all body models, related notebooks and training logic. track-mjx v1 and on will rely on [vnl-playground](https://github.com/talmolab/vnl-playground) for the environment and task logic. vnl-playground will be installed during the following installation steps along with other needed libraries. For more information regarding the environment and task logic, please visit [vnl-playground](https://github.com/talmolab/vnl-playground).**

## Installation

### Option 1: `uv` (fastest)

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
3. Install the package with optional dependencies based on your hardware. CUDA 12, CUDA 13, and CPU-only configurations are supported. This will take a few minutes:

For CUDA 12.x:
```bash
uv pip install -e ".[cuda12]"
```

For CUDA 13.x:
```bash
uv pip install -e ".[cuda13]"
```

For CPU-only:
```bash
uv pip install -e .
```

For development, include the `[dev]` extras in addition to the hardware optional dependencies:
```bash
uv pip install -e ".[cuda13,dev]"
```
4. Verify the installation:
```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Available devices: {jax.devices()}')"
```
5. Register the environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name=track-mjx --display-name="Python (track-mjx)"
```
6. Test the environment:
    Execute the tests in [`notebooks/test_setup.ipynb`](notebooks/test_setup.ipynb). This will check if MuJoCo, GPU support and Jax appear to be working.

#### Alternative: Using `pip`

If you prefer using pip instead of uv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[cuda13]"  # or cuda12/no optional deps
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

### Option 2: `conda`

#### Installation steps

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
    If your machine supports up to CUDA 12:
    ```bash
    pip install -e ".[cuda13]"
    ```
    If your machine only has a CPU:
    ```bash
    pip install -e .
    ```
5. Test the environment:
    Execute the tests in [`notebooks/test_setup.ipynb`](notebooks/test_setup.ipynb). This will check if MuJoCo, GPU support and Jax appear to be working.


## Training

### Rodent

The main training entrypoint is defined in [`scripts/train.py`](scripts/train.py) and relies on the config in [`track_mjx/config/rodent-full-clips.yaml`](track_mjx/config/rodent-full-clips.yaml).

#### Download the data

To download data, run `notebooks/rodent_demo.ipynb`

##### OR

Execute the following command in terminal
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='talmolab/MIMIC-MJX', repo_type='dataset', filename='data/rodent/rodent_reference_clips.h5', local_dir='.')"
```

#### Run training:

**Using uv:**
```bash
uv run python -m track_mjx.train --config-name rodent-full-clips.yaml
```

**Using conda:**
```bash
conda activate track_mjx
python -m track_mjx.train --config-name rodent-full-clips.yaml
```


## Task Training

We provide generic scripts to train policies on any task registered in [vnl-playground](https://github.com/talmolab/vnl-playground).

### Standard PPO (`train_task.py`)

Trains an end-to-end MLP policy using Brax PPO. Supports both the default JAX/MJX backend and the Warp backend (for full-collision body models).

```bash
# Any registered task
python scripts/train_task.py --task RodentBowlEscape

# With PPO overrides
python scripts/train_task.py --task RodentRearing --num_timesteps 1e8 --entropy_cost 0.1

# With env config overrides
python scripts/train_task.py --task RodentBowlEscape --env "target_speed=1.5"

# Warp backend (full-collision body model)
python scripts/train_task.py --task RodentBowlEscape --env "mujoco_impl=warp"
```

### High-Level Transfer (`train_highlvl.py`)

Trains a high-level policy that outputs latent intentions to a frozen pretrained mimic decoder. The decoder converts intentions into naturalistic motor commands.

```bash
# Any registered task
python scripts/train_highlvl.py --task RodentBowlEscape --mimic_checkpoint <checkpoint_id>

# With PPO overrides
python scripts/train_highlvl.py --task RodentRearing \
    --mimic_checkpoint <checkpoint_id> --num_timesteps 1e8 --entropy_cost 0.1
```

Both scripts support `--policy_hidden_sizes`, `--value_hidden_sizes`, `--env` (for env config overrides), and standard PPO hyperparameter flags. Run with `--help` for full usage.


## Citation

If you use track-mjx in your research, please cite our paper:

```bibtex
@misc{zhang2025mimicmjxneuromechanicalemulationanimal,
      title={MIMIC-MJX: Neuromechanical Emulation of Animal Behavior}, 
      author={Charles Y. Zhang and Yuanjia Yang and Aidan Sirbu and Elliott T. T. Abe and Emil Wärnberg and Eric J. Leonardis and Diego E. Aldarondo and Adam Lee and Aaditya Prasad and Jason Foat and Kaiwen Bian and Joshua Park and Rusham Bhatt and Hutton Saunders and Akira Nagamori and Ayesha R. Thanawalla and Kee Wui Huang and Fabian Plum and Hendrik K. Beck and Steven W. Flavell and David Labonte and Blake A. Richards and Bingni W. Brunton and Eiman Azim and Bence P. Ölveczky and Talmo D. Pereira},
      year={2025},
      eprint={2511.20532},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC},
      url={https://arxiv.org/abs/2511.20532}, 
}
```




## License
This package is distributed under a BSD 3-Clause License and can be used without
restrictions. See [`LICENSE`](https://github.com/talmolab/track-mjx/blob/main/LICENSE) for details.
