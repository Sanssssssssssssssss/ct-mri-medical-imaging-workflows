MRI Guide
=========

The MRI module is exposed through :mod:`a2mi.mri` and supports visualisation of
multi-coil k-space, image-space reconstruction, and denoising comparisons.

Exercise 2.1: Visualisation and coil combination
------------------------------------------------

The first MRI workflow focuses on understanding the data layout and visual
properties of the acquisition. The main helpers are:

* ``load_kspace``
* ``infer_coil_axis``
* ``move_coils_first``
* ``kspace_to_image_space``
* ``combine_rsos``
* plotting helpers for k-space, magnitude, phase, and rSoS views

Example usage:

.. code-block:: python

   from pathlib import Path

   import a2mi

   prepared = a2mi.mri.prepare_exercise_2_1_data(Path("data") / "knee.npy")
   rsos = prepared["rsos"]

Exercise 2.2: Denoising
-----------------------

The second MRI workflow compares multiple denoising methods in image space and
k-space. Key helpers include:

* ``denoise_gaussian``
* ``denoise_median``
* ``denoise_bilateral_img``
* ``butterworth_lowpass_filter``
* ``denoise_all_coils``
* ``denoise_all_coils_rsos``
* metric and plotting helpers for side-by-side comparisons

Notebook-friendly wrappers such as ``run_exercise_2_2_part1`` and
``run_exercise_2_2_part3`` bundle the full experiment flow for reuse.

Output conventions
------------------

MRI outputs are organised by exercise and part:

.. code-block:: text

   results/mri/figures/ex2_1/
   results/mri/figures/ex2_2/part1/
   results/mri/figures/ex2_2/part2/
   results/mri/figures/ex2_2/part3/
   results/mri/metrics/ex2_2/

This keeps report figures and metric tables grouped by the experiment stage
that produced them.
