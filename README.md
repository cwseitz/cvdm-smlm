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

## Plotting Workflow

Plotting is now separated from test-time inference.

1. Run `cvdm.mains.test` to generate stack artifacts (`x_stack.tif`, `y_stack.tif`, `z_stack.tif`).
2. Run `cvdm.mains.plot` to render figures from saved output directories.

Direct plotting without test config:

- For `mode: "test"`, set `plot.output_dir` or `plot.output_dirs` to folders containing stack artifacts. Do not include `test_config`.
- For `mode: "probe"`, set `plot.probe_output_dir` to cached probe outputs and provide inline plot settings in the `plot` block. Do not include `probe_config` or `probe_template_config`.

Example commands:

```bash
python -m cvdm.mains.test --config-path cvdm/configs/test/test_nanoruler.yaml
python -m cvdm.mains.plot --config cvdm/configs/plot/plot_test_nanoruler.yaml
python -m cvdm.mains.plot --config cvdm/configs/plot/plot_probe_nanoruler.yaml
```
