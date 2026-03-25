Quick Start
===========

This page gives the shortest path from a fresh clone to a working development
environment with notebooks, tests, Sphinx documentation, and Docker-based
reproduction entry points.

Recommended local setup
-----------------------

From the repository root, create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate

Install the package together with development and documentation extras:

.. code-block:: bash

   pip install -e .[docs,dev]

Build the Sphinx site:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

Treat documentation warnings as errors:

.. code-block:: bash

   python -m sphinx -b html -W --keep-going docs docs/_build/html

The rendered site entry point will be ``docs/_build/html/index.html``.

Docker-first reproduction
-------------------------

The main reviewer-facing entry points are the shell scripts in ``scripts/``.
Open a terminal in the repository root and start with:

.. code-block:: bash

   sh scripts/run_docs.sh

This serves the documentation site at ``http://localhost:8000``. From a second
terminal, run the workflow containers:

.. code-block:: bash

   sh scripts/run_ct.sh
   sh scripts/run_mri.sh

The workflow containers mount the repository ``results/`` directory so exported
figures and CSV files remain on the host after the containers exit. These shell
scripts have been checked in Windows WSL and are written in a POSIX ``sh``
style that should work on macOS as well, but they have not yet been directly
verified on Apple hardware.

Script summary:

- ``scripts/run_docs.sh`` builds and serves the Sphinx documentation site
- ``scripts/run_ct.sh`` runs the full CT export workflow
- ``scripts/run_mri.sh`` runs the full MRI export workflow

Running the coursework workflows
--------------------------------

The repository includes package entry points that reproduce the notebook export
pipelines without interactive plotting:

.. code-block:: bash

   python -m a2mi.ct.main
   python -m a2mi.mri.main

The notebooks remain available for exploratory work and visual inspection:

.. code-block:: text

   notebooks/CT_workflow.ipynb
   notebooks/MRI_workflow.ipynb

Both the package entry points and the notebooks read input arrays from
``data/`` and write outputs into ``results/``.

Validation commands
-------------------

Run linting:

.. code-block:: bash

   python -m ruff check src tests

Run the test suite:

.. code-block:: bash

   python -m pytest -q

Run tests with coverage:

.. code-block:: bash

   python -m pytest --cov=src/a2mi --cov-report=term-missing

Minimal import example
----------------------

.. code-block:: python

   from pathlib import Path

   import a2mi

   ct_ref = a2mi.ct.load_reference_ct_image("data/CT_exercise_1.png")
   theta = a2mi.ct.make_theta(90)

   mri = a2mi.mri.prepare_exercise_2_1_data(Path("data") / "knee.npy")
   rsos = a2mi.mri.combine_rsos(mri["img_coils"])
