Installation and Environment
============================

Project structure
-----------------

The repository uses a standard ``src/`` layout so that imports resolve through
the installed package rather than the current working directory.

.. code-block:: text

   src/a2mi/        Reusable package code
   notebooks/       Experiment orchestration notebooks
   tests/           Automated validation
   results/         Generated figures and metrics
   report/          Report assets
   docs/            Sphinx documentation source

Python and package configuration
--------------------------------

The package metadata is defined in ``pyproject.toml``. The current project is
configured for Python ``>=3.11`` and installs through editable mode using
Setuptools. Runtime dependencies cover numerical computing, plotting, image
processing, and progress reporting.

The documentation toolchain is exposed via the ``docs`` extra and includes:

* ``sphinx``
* ``pydata-sphinx-theme``
* ``myst-parser``
* ``sphinx-design``
* ``sphinx-copybutton``

Editable install workflow
-------------------------

The recommended install keeps the repository importable while you are editing
the source tree:

.. code-block:: bash

   pip install -e .[docs,dev]

This makes ``a2mi`` importable from notebooks, tests, and documentation builds
without requiring repeated reinstall steps after each source change.

Repository conventions
----------------------

The package layer is the primary source of truth for reusable logic. Notebooks
remain important, but they are intentionally thin wrappers around the package.
This structure keeps the project compatible with automated documentation and
enables targeted unit tests against the underlying library functions.

The documentation site follows the same principle: narrative pages explain how
the workflows are organised, while the API reference is generated directly from
the package docstrings.
