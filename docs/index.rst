a2mi Documentation
==================

.. container:: hero-intro

   **Medical imaging coursework package for CT reconstruction and MRI denoising**

   The ``a2mi`` repository combines a reusable Python package, notebook-driven
   experiment workflows, automated validation, and coursework report assets in
   one place. This documentation site is designed to make the codebase easy to
   explore from two directions: a quick narrative route for understanding the
   workflow, and an API route that renders package docstrings and source links
   automatically.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Quick Start
      :link: quickstart.html
      :class-card: nav-card

      Install the package, build the docs, run the notebooks, and execute the
      validation suite from a clean local environment.

   .. grid-item-card:: Installation and Environment
      :link: installation_environment.html
      :class-card: nav-card

      Review the repository layout, editable-install workflow, Python version,
      and the dependencies used by the package and documentation site.

   .. grid-item-card:: Workflow Guide
      :link: workflow_guide.html
      :class-card: nav-card

      Understand how the narrative notebooks and the reusable ``src/a2mi``
      package fit together, including the expected data and results locations.

   .. grid-item-card:: CT Guide
      :link: ct_guide.html
      :class-card: nav-card

      Follow the main helper functions that support dose reduction, limited-angle
      reconstruction, and reconstruction-method comparisons.

   .. grid-item-card:: MRI Guide
      :link: mri_guide.html
      :class-card: nav-card

      See how k-space is loaded, transformed to image space, denoised, and
      combined into final rSoS comparisons.

   .. grid-item-card:: Testing and Validation
      :link: testing_validation.html
      :class-card: nav-card

      Check the automated validation workflow, including linting, pytest, and
      coverage-based quality gates for the library layer.

   .. grid-item-card:: API Reference
      :link: api_reference.html
      :class-card: nav-card

      Browse auto-generated package documentation for ``a2mi``, ``a2mi.ct``,
      ``a2mi.mri``, and ``a2mi.common`` with source-code links.

Coursework Overview
-------------------

The repository is structured around a lightweight ``src/`` package with
notebooks acting as orchestration layers rather than the primary implementation
surface. This keeps the experiment logic importable, testable, and compatible
with tools such as Sphinx. In practice, CT and MRI experiments are run from the
notebooks, which call into the package to load data, execute algorithms,
generate figures, and save metric tables.

The documentation is therefore organised around three questions:

1. How do I install and run the project?
2. How are the CT and MRI workflows implemented?
3. Which public functions and classes are available in the package?

.. note::

   If you only need the shortest setup path, start with :doc:`quickstart`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   Quick <quickstart>
   Install <installation_environment>
   Workflow <workflow_guide>
   CT <ct_guide>
   MRI <mri_guide>
   Tests <testing_validation>
   API <api_reference>
