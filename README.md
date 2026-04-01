# Medical Imaging Coursework A2

This repository contains the submission for the *Data Science Applications to Medical Imaging* coursework. It includes the final report, the reproducible CT and MRI workflows for the practical modules, and the supporting documentation, tests, and Docker scripts needed to rerun the work. The internal `a2mi` package is only a lightweight implementation layer for the coursework workflows.

The latest report can be found here:

- [`report/Chang_Xu_A2_report.pdf`](report/Chang_Xu_A2_report.pdf)

## Quick Start

From the repository root, start the documentation site with:

```bash
sh scripts/run_docs.sh
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

From a second terminal, run the workflows:

```bash
sh scripts/run_ct.sh
sh scripts/run_mri.sh
```

These scripts have been checked in Windows WSL and are written in portable POSIX `sh`. They should also work on macOS with Docker installed, but they have not yet been directly verified on Apple hardware.

## Where Outputs Go

The workflow exports are written to:

- `results/ct/figures/...`
- `results/ct/metrics/...`
- `results/mri/figures/ex2_1/...`
- `results/mri/figures/ex2_2/part1|part2|part3/...`
- `results/mri/metrics/ex2_2/...`

The final report is stored separately in `report/`.

## Manual Run

If you want to run everything without Docker:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[docs,dev]
python -m a2mi.ct.main
python -m a2mi.mri.main
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

## Project Structure

```text
src/            Coursework implementation package
notebooks/      Original exploratory notebooks
scripts/        Main reviewer-facing shell entry points
docker/         Dockerfiles and lower-level launchers
docs/           Sphinx documentation source
tests/          Automated validation
results/        Generated figures and metrics
report/         Final coursework report PDF
```

## Licence

This repository is released under the MIT licence. See [`LICENSE`](LICENSE).
