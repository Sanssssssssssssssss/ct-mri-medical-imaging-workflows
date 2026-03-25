Workflow Guide
==============

The repository combines two layers:

* notebooks for experiment orchestration and reporting artefact generation
* the ``a2mi`` package for reusable logic, plotting helpers, and metric export

This separation is important for maintainability. It means the notebooks can
stay focused on experiment sequencing, while the package remains importable,
testable, and fully documentable by Sphinx.

Shared data flow
----------------

Across both modules, the workflow follows a similar pattern:

1. load source data from ``data/``
2. prepare a standard in-memory representation
3. run experiment helpers from ``a2mi.ct`` or ``a2mi.mri``
4. save figures and metric tables under ``results/``
5. use the exported artefacts in the report

CT workflow
-----------

The CT path focuses on simulation and reconstruction:

* load and normalise the reference CT image
* generate projection angles and clean sinograms
* inject Gaussian and Poisson noise
* reconstruct with filtered back-projection, gradient descent, or OS-SART
* save comparison figures and CSV metrics

MRI workflow
------------

The MRI path focuses on multi-coil visualisation and denoising:

* load the complex ``knee.npy`` k-space array
* infer the coil axis and move it to a coil-first layout
* reconstruct image-space data with an inverse Fourier transform
* combine coils with root-sum-of-squares
* apply per-coil and combined-image denoising strategies
* export comparison figures and metric tables

Notebook-package relationship
-----------------------------

The notebooks are designed to call the package rather than re-implement core
logic inline. This provides three benefits:

* public experiment helpers are easier to test
* docstrings become the single source of API documentation
* report artefacts can be regenerated without duplicating implementation logic

For module-specific entry points, continue with :doc:`ct_guide` and
:doc:`mri_guide`.
