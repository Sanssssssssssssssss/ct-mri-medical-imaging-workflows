# Medical Imaging Coursework A2

This repository contains the submission for the *Data Science Applications to Medical Imaging* coursework described in `Medical_Imaging_Coursework_2026.pdf`. The project covers the practical workflow for Module 1 (CT tomographic reconstruction) and Module 2 (MRI image denoising), together with the written coursework report and the reproducibility material needed to rerun the experiments. The reusable Python package in `src/a2mi` supports the workflows, but the repository should be read first as a coursework project rather than as a standalone library.

The coursework report is included directly in the repository at [`report/Data_Science_Applications_to_Medical_ImagingCoursework_Minor_A2_Report.pdf`](report/Data_Science_Applications_to_Medical_ImagingCoursework_Minor_A2_Report.pdf). If you want the final written submission first, start there.

## Fastest Reproduction Path

The main reviewer-facing entry points are the shell scripts in `scripts/`. Open a terminal in the repository root and run:

```bash
sh scripts/run_docs.sh
```

This starts the documentation site at [http://localhost:8000](http://localhost:8000). Then, from a second terminal in the same repository, run:

```bash
sh scripts/run_ct.sh
sh scripts/run_mri.sh
```

What these scripts do:

- `scripts/run_docs.sh`
  - builds the Sphinx documentation site inside Docker and serves it locally
- `scripts/run_ct.sh`
  - runs the full CT workflow and exports figures and metrics to `results/ct`
- `scripts/run_mri.sh`
  - runs the full MRI workflow and exports figures and metrics to `results/mri`

These scripts have been checked in Windows WSL and are written in portable POSIX `sh`. They are expected to work in macOS terminals with Docker installed, but they have not yet been directly hardware-verified on a real Apple machine.

## Manual Run Path

If you prefer to run the workflows without Docker, create a local virtual environment and install the project:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[docs,dev]
```

Then run the two coursework workflows directly:

```bash
python -m a2mi.ct.main
python -m a2mi.mri.main
```

Build the documentation locally with:

```bash
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

The rendered documentation entry point is `docs/_build/html/index.html`.

## Where Outputs Go

The workflows write their exported artefacts into `results/` using exercise-based folders:

- `results/ct/figures/exercise_1_1`, `exercise_1_2`, `exercise_1_3`
- `results/ct/metrics/exercise_1_1`, `exercise_1_2`, `exercise_1_3`
- `results/mri/figures/ex2_1`
- `results/mri/figures/ex2_2/part1`, `part2`, `part3`
- `results/mri/metrics/ex2_2`

The report artefact is stored separately under `report/`.

## Project Structure

```text
src/
  a2mi/
    common/        Shared path, figure, and CSV helpers
    ct/            CT reconstruction logic and CLI workflow entry point
    mri/           MRI denoising logic and CLI workflow entry point
notebooks/         Original exploratory notebooks used to develop the workflows
scripts/           Main shell entry points for reviewers
docker/            Dockerfiles and lower-level container launchers
docs/              Sphinx documentation source
tests/             Automated validation for the package layer
results/           Generated figures and metrics
report/            Final coursework report PDF
```

## Documentation

The repository includes a Sphinx documentation site with narrative pages and auto-generated API documentation from package docstrings. The quickest way to view it is still:

```bash
sh scripts/run_docs.sh
```

The lower-level Docker launcher remains available at `docker/docs/run.sh`, but `scripts/run_docs.sh` is the intended top-level entry point.

## Brief Package Note

The internal package is named `a2mi`. It exists to keep the coursework implementation structured and testable:

- `a2mi.ct`
  - CT image loading, sinogram simulation, FBP, gradient-descent, limited-angle, and OS-SART helpers
- `a2mi.mri`
  - multi-coil k-space loading, Fourier reconstruction, rSoS combination, and denoising helpers
- `a2mi.common`
  - shared project-relative path resolution and export helpers

The package is intentionally lightweight. The primary deliverables in this repository are still the report, the reproducible workflows, and the generated results.

## Validation

Useful validation commands:

```bash
python -m ruff check src tests
python -m pytest -q
python -m pytest --cov=src/a2mi --cov-report=term-missing
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

## Coursework Requirements Check

Against the wording in the coursework PDF, the repository now aligns well with the main software expectations:

- Python implementation: yes
- sensible folder structure with `README.md`, `LICENSE`, report folder, tests, docs: yes
- Sphinx-compatible documentation: yes
- automated unit tests with high coverage: yes
- containerised reproduction path: yes
- version-controlled history with multiple commits: yes

The main caveat is that branch protection and commit hooks are not things that can be fully demonstrated only from repository contents. Branch protection is a remote Git hosting setting, and commit hooks are local developer tooling. So the repository is strong on structure and reproducibility, but that single requirement is only partially evidenced from the codebase itself.
