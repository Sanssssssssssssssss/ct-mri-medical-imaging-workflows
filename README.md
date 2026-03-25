# A2MI Coursework Package

`a2mi` is a small Python package and experiment repository for the CT and MRI
coursework exercises in *Data Science Applications to Medical Imaging*. The
repository is organised so that reusable algorithmic code lives in `src/a2mi`,
while notebooks remain thin orchestration layers for loading data, running
experiments, and exporting figures and metrics.

## Repository Layout

```text
src/
  a2mi/
    common/        Shared filesystem and export helpers
    ct/            CT simulation, reconstruction, and comparison utilities
    mri/           MRI visualisation and denoising utilities
notebooks/         Workflow notebooks for coursework experiments
tests/             Automated unit and smoke tests
results/           Generated figures and metrics organised by exercise
report/            Coursework report assets and exported PDF
```

## Package Overview

- `a2mi.ct`
  - reference CT image loading and preprocessing
  - sinogram simulation with Gaussian and Poisson noise
  - FBP, gradient-descent, limited-angle, and OS-SART comparisons
- `a2mi.mri`
  - multi-coil k-space loading and image-space reconstruction
  - rSoS combination
  - image-space and k-space denoising utilities
- `a2mi.common`
  - project-relative path resolution
  - figure saving and CSV export helpers

## Installation

This project is designed around a local virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

If you only want the runtime dependencies, use:

```bash
pip install -e .
```

To install the documentation toolchain as well, use:

```bash
pip install -e .[docs,dev]
```

## Running the Coursework Workflows

The repository includes command-line entry points that reproduce the notebook
export workflows without interactive plotting:

```bash
python -m a2mi.ct.main
python -m a2mi.mri.main
```

By default, they read from `data/` and write outputs into:

- `results/ct/...`
- `results/mri/...`

The original notebooks remain available as exploratory companions:

- `notebooks/CT_workflow.ipynb`
- `notebooks/MRI_workflow.ipynb`

## Building the Documentation

The project includes a Sphinx documentation site under `docs/`.

```bash
python -m sphinx -b html docs docs/_build/html
```

To treat documentation warnings as build failures:

```bash
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

To preview the built site locally:

```bash
python -m http.server 8000 -d docs/_build/html
```

Then open [http://localhost:8000](http://localhost:8000).

## Docker Reproduction

The main user-facing entry points are the shell scripts in `scripts/`. This is
the most important reproduction path for a reviewer.

Open a terminal in the repository root and run:

```bash
sh scripts/run_docs.sh
```

This starts the documentation site at [http://localhost:8000](http://localhost:8000).
After that, from a second terminal in the same repository, run:

```bash
sh scripts/run_ct.sh
sh scripts/run_mri.sh
```

What each script does:

- `scripts/run_docs.sh`
  - builds the Sphinx site inside Docker and serves it at [http://localhost:8000](http://localhost:8000)
- `scripts/run_ct.sh`
  - runs the full CT export workflow and writes into `results/ct`
- `scripts/run_mri.sh`
  - runs the full MRI export workflow and writes into `results/mri`

These `sh` scripts are intended for POSIX-style terminals. They have been
checked in Windows WSL and are written in a macOS-compatible `sh` style, but
they have not yet been hardware-verified on a real macOS machine. The workflow
containers mount `results/` from the host so exported figures and CSV files
remain available in the repository after the container exits.

The lower-level Docker launchers remain available under `docker/`, but reviewers
should use `scripts/` first.

## Results Conventions

Generated outputs are organised by module and exercise:

- `results/ct/figures/...`
- `results/ct/metrics/...`
- `results/mri/figures/ex2_1/...`
- `results/mri/figures/ex2_2/part1|part2|part3/...`
- `results/mri/metrics/ex2_2/...`

These are experiment artefacts rather than package source files.

## Testing and Linting

Run linting:

```bash
python -m ruff check src tests
```

Run the test suite:

```bash
python -m pytest -q
```

Run the test suite with coverage:

```bash
python -m pytest --cov=src/a2mi --cov-report=term-missing
```

## Minimal API Example

```python
from pathlib import Path

import a2mi

ct_ref = a2mi.ct.load_reference_ct_image("data/CT_exercise_1.png")
theta = a2mi.ct.make_theta(90)

mri_data = a2mi.mri.prepare_exercise_2_1_data(Path("data") / "knee.npy")
rsos = a2mi.mri.combine_rsos(mri_data["img_coils"])
```

## Reproducibility Notes

- The project uses a standard `src/` layout and editable installation through
  `pyproject.toml`.
- Public package entry points are exposed through `a2mi`, `a2mi.ct`, and
  `a2mi.mri`.
- Automated tests are focused on the reusable library layer rather than on full
  notebook execution.
- The Sphinx documentation site automatically renders package docstrings and
  source links from the `src/` tree.
- Docker entry points are included for documentation serving and for both
  workflow exports, in addition to the local virtual environment workflow.
- The shell-script entry points have been validated in Windows/WSL; macOS is
  expected to work with Docker installed, but has not yet been directly
  tested on Apple hardware.
