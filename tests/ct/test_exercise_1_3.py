"""Tests for CT Exercise 1.3 helpers."""

from __future__ import annotations

import numpy as np
import pytest

from a2mi.ct.exercise_1_1 import simulate_noisy_sinograms
from a2mi.ct.exercise_1_3 import (
    available_fbp_filters,
    reconstruct_os_sart,
    run_fbp_filter_comparison,
    run_os_sart_comparison,
    summarize_filter_comparison,
    summarize_filter_options,
    summarize_iterative_comparison,
)


def test_filter_lists_and_filter_comparison(tmp_path, synthetic_ct_image: np.ndarray) -> None:
    """Filter comparison helpers should save figures and metrics for valid filters."""
    sinogram_sets = simulate_noisy_sinograms(
        synthetic_ct_image,
        angles_list=(12,),
        poisson_i0_levels=(1e3, 1e2),
        seed=0,
    )
    filters = available_fbp_filters()
    assert ("ramp", "Ram-Lak") in filters
    assert any("Ram-Lak" in label for label in summarize_filter_options())

    result = run_fbp_filter_comparison(
        synthetic_ct_image,
        sinogram_sets,
        n_angles=12,
        i0_level=1e2,
        filter_names=("ramp", "hann"),
        out_dir=tmp_path / "filters",
        metrics_out_path=tmp_path / "filter_metrics.csv",
        show=False,
    )
    assert result.metrics_path.exists()
    assert len(result.reconstructions) == 2
    assert len(summarize_filter_comparison(result)) == 2


def test_iterative_comparison_and_invalid_filter(tmp_path, synthetic_ct_image: np.ndarray) -> None:
    """Iterative helpers should run and invalid filters should fail clearly."""
    sinogram_sets = simulate_noisy_sinograms(
        synthetic_ct_image,
        angles_list=(10,),
        poisson_i0_levels=(1e3, 1e2),
        seed=0,
    )

    sinogram = sinogram_sets[0].gaussian_poisson[1e2]
    theta = sinogram_sets[0].theta
    os_sart = reconstruct_os_sart(
        sinogram,
        theta,
        output_size=synthetic_ct_image.shape[0],
        n_iters=2,
        n_subsets=3,
        step_size=0.001,
        positivity=True,
    )
    assert os_sart.shape == synthetic_ct_image.shape

    result = run_os_sart_comparison(
        synthetic_ct_image,
        sinogram_sets,
        n_angles=10,
        i0_level=1e2,
        sirt_iters=2,
        sirt_step_size=0.001,
        os_sart_iters=2,
        os_sart_step_size=0.001,
        n_subsets=2,
        out_dir=tmp_path / "iterative",
        metrics_out_path=tmp_path / "iterative_metrics.csv",
        show=False,
    )
    assert result.figure_path.exists()
    assert result.metrics_path.exists()
    assert len(summarize_iterative_comparison(result)) == 3

    with pytest.raises(ValueError):
        run_fbp_filter_comparison(
            synthetic_ct_image,
            sinogram_sets,
            n_angles=10,
            i0_level=1e2,
            filter_names=("unknown",),
            out_dir=tmp_path / "bad",
            metrics_out_path=tmp_path / "bad.csv",
            show=False,
        )

    with pytest.raises(KeyError):
        run_os_sart_comparison(
            synthetic_ct_image,
            sinogram_sets,
            n_angles=999,
            i0_level=1e2,
            out_dir=tmp_path / "missing",
            metrics_out_path=tmp_path / "missing.csv",
            show=False,
        )
