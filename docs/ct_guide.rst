CT Guide
========

The CT module is exposed through :mod:`a2mi.ct` and covers three main exercise
families: reducing dose, limited-angle tomography, and reconstruction-method
comparison.

Exercise 1.1: Reducing dose
---------------------------

Typical helpers for this workflow include:

* ``load_reference_ct_image`` for reading and normalising the reference image
* ``make_theta`` for constructing projection angles
* ``simulate_noisy_sinograms`` for Gaussian and Poisson corruption
* ``reconstruct_fbp`` and ``reconstruct_gradient_descent`` for baseline and
  iterative reconstruction
* ``run_exercise_1_1_*`` wrappers for notebook-level experiment execution

Example usage:

.. code-block:: python

   import a2mi

   reference = a2mi.ct.load_reference_ct_image("data/CT_exercise_1.png")
   theta = a2mi.ct.make_theta(90)
   results = a2mi.ct.simulate_noisy_sinograms(reference.image, theta)

Exercise 1.2: Limited-angle tomography
--------------------------------------

The limited-angle workflow reuses the same general simulation pattern but
changes the acquisition range through ``make_theta_limited`` and the
corresponding limited-angle simulation and plotting helpers.

Typical outputs include:

* restricted-angle sinograms
* FBP versus gradient-descent reconstructions
* metric summaries across angular ranges

Exercise 1.3: Reconstruction comparisons
----------------------------------------

The final CT exercise compares FBP filters and iterative strategies. The key
entry points are:

* ``available_fbp_filters``
* ``run_fbp_filter_comparison``
* ``run_os_sart_comparison``
* summary helpers for notebook-ready tables and descriptions

Output conventions
------------------

CT figures and metrics are written below:

.. code-block:: text

   results/ct/figures/
   results/ct/metrics/

The package provides helper functions for writing figures and CSV files in a
consistent, project-relative manner.
