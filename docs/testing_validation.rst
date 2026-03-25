Testing and Validation
======================

The repository validates the reusable library layer through automated checks.
The goal is to keep algorithm helpers, plotting/export utilities, and public
package entry points stable without relying on full notebook execution for
every regression check.

Validation workflow
-------------------

The standard checks are:

.. code-block:: bash

   python -m ruff check src tests
   python -m pytest -q
   python -m pytest --cov=src/a2mi --cov-report=term-missing
   python -m sphinx -b html -W --keep-going docs docs/_build/html

Test coverage focus
-------------------

The tests exercise both CT and MRI package layers, including:

* angle generation and validation helpers
* noise simulation and reconstruction wrappers
* k-space loading and coil-axis handling
* rSoS combination and denoising helpers
* CSV/figure export utilities
* import-level smoke checks for the public package API

Coverage targets are enforced through the pytest configuration in
``pyproject.toml``. This keeps the project aligned with the coursework
requirement for robust automatic validation.

Why the docs build matters
--------------------------

The documentation build is treated as part of validation rather than as a
purely cosmetic step. A passing Sphinx build confirms that:

* public modules import correctly
* docstrings remain structurally valid
* the API reference stays in sync with the package
* navigation pages and generated reference pages do not accumulate warnings
