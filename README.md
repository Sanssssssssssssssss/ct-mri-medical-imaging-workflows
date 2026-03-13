## A2 Coursework

Medical imaging coursework repository covering CT and MRI exercises.

### What is included

- `src/a2mi/ct`: CT simulation, noise modelling, reconstruction, and comparison utilities
- `src/a2mi/mri`: MRI visualisation and denoising utilities
- `notebooks/CT_workflow.ipynb`: thin workflow notebook for CT experiments
- `notebooks/MRI_workflow.ipynb`: thin workflow notebook for MRI experiments

### Project structure

```text
src/
  a2mi/
    ct/
    mri/
notebooks/
report/
tests/
```

### Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

For editable package installation:

```bash
pip install -e .
```

### Notes

- Local datasets under `data/` are intentionally not tracked.
- Generated experiment outputs under `results/` are not required for the source repository.
- The practical scratch notebook is excluded from version control.
