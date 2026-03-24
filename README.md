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

## Running the Coursework Workflows

The notebooks are intended to be run from the repository workspace after the
package has been installed in editable mode.

- `notebooks/CT_workflow.ipynb`
- `notebooks/MRI_workflow.ipynb`

The notebooks load input data from `data/` and write outputs into `results/`.

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
- No container setup is included in the repository; reproducibility is based on
  the local virtual environment workflow documented above.
