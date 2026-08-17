# CVDM-SMLM

This repository contains code and experiments for applying the conditional variational diffusion model (CVDM) in single molecule localization microscopy (SMLM). It includes model definitions, training and evaluation scripts, dataset generators, and plotting utilities used for experiments and figures.

## Repository Layout

- `cvdm/` – Core Python package
  - `architectures/` – UNet/SR3 blocks and components
  - `models/` – Model definitions
  - `data/` – Dataset loaders and data utilities
  - `generators/` – Synthetic data generators
  - `utils/` – Training/inference/metrics utilities
  - `psf/` – PSF modeling utilities
  - `mains/` – Entry points for training, testing, and plotting
  - `configs/` – YAML configuration files
  - `make/` – Image simulation methods
- `requirements.txt` – Python dependencies
- `setup.py` – Package setup

## Installation and basic usage

Create an environment with Anaconda/Miniconda:

```bash
conda env create -f environment.yml
conda activate cvdm-smlm
pip install -e .
```

For Linux GPU setups with CUDA 11, use:

```bash
conda env create -f environment.gpu.yml
conda activate cvdm-smlm-gpu
pip install -e .
```

Notes:

- `environment.yml` is the default CPU-safe environment (works on macOS).
- `environment.gpu.yml` is intended for Linux + NVIDIA CUDA and will not work on macOS.

## HPC

This section covers using Apptainer/Singularity to run `cvdm-smlm` with a reproducible Ubuntu 24 + Python 3.10 container.

### 1) Build the `.sif` from the `.def`

Load Apptainer on your cluster, then build from `cvdm_ubuntu24_py310.def`:

```bash
module load apptainer
cd /homes/seitzcx/git/cvdm-smlm

# If your cluster allows unprivileged builds:
apptainer build --fakeroot cvdm-smlm.sif cvdm_ubuntu24_py310.def

# If --fakeroot is not available, ask your admin for the supported build method.
```

### 2) Enter the container interactively

Bind your project directory so files (including the venv) persist on the host:

```bash
module load apptainer
apptainer shell --nv \
  --bind /homes/seitzcx/git/cvdm-smlm:/workspace \
  /homes/seitzcx/git/cvdm-smlm/cvdm-smlm.sif
```

### 3) Create a venv using container Python

Inside the container shell:

```bash
python --version
python -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r /workspace/requirements.txt
python -m pip install -e /workspace
```
Job submission scripts should then use this `venv` python and dependencies. 

