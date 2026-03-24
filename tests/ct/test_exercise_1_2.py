"""Tests for CT Exercise 1.2 helpers."""

from __future__ import annotations

import numpy as np
import pytest

from a2mi.ct.exercise_1_2 import (
    forward_project_limited,
    make_theta_limited,
    run_limited_angle_reconstruction_experiment,
    run_limited_angle_sinogram_experiment,
    save_limited_angle_sinogram_panel,
    simulate_noisy_sinograms_limited,
    summarize_limited_angle_sinograms,
    to_reconstruction_sinogram_sets,
)


def test_theta_limited_validation_and_projection(synthetic_ct_image: np.ndarray) -> None:
    """Limited-angle theta generation should validate bounds and project correctly."""
    theta = make_theta_limited(90, step_deg=10.0)
    np.testing.assert_allclose(theta, np.arange(0.0, 90.0, 10.0, dtype=np.float32))

    with pytest.raises(ValueError):
        make_theta_limited(0)
    with pytest.raises(ValueError):
        make_theta_limited(181)
    with pytest.raises(ValueError):
        make_theta_limited(90, step_deg=0)

    sino, theta_out = forward_project_limited(synthetic_ct_image, angle_range=40, step_deg=5.0)
    assert sino.shape[1] == len(theta_out)

    with pytest.raises(ValueError):
        forward_project_limited(np.zeros((2, 2, 2), dtype=np.float32), angle_range=40)


def test_limited_angle_workflow_wrappers(tmp_path, synthetic_ct_image: np.ndarray) -> None:
    """Exercise 1.2 workflows should save panels, figures, and metrics."""
    sinogram_sets = simulate_noisy_sinograms_limited(
        synthetic_ct_image,
        angle_ranges=(180, 40),
        step_deg=20.0,
        poisson_i0_levels=(1e4, 1e3, 1e2),
        seed=0,
    )
    assert len(sinogram_sets) == 2
    assert "range=" in summarize_limited_angle_sinograms(sinogram_sets)[0]

    recon_sets = to_reconstruction_sinogram_sets(sinogram_sets)
    assert recon_sets[0].clean.shape[0] == sinogram_sets[0].clean.shape[0]

    panel_path = save_limited_angle_sinogram_panel(
        sinogram_sets,
        poisson_i0_levels=(1e4, 1e3, 1e2),
        gaussian_mu=0.0,
        gaussian_sigma=0.05,
        out_path=tmp_path / "limited_panel.png",
        show=False,
    )
    assert panel_path.exists()

    result_a = run_limited_angle_sinogram_experiment(
        synthetic_ct_image,
        angle_ranges=(120,),
        step_deg=30.0,
        poisson_i0_levels=(1e4, 1e3, 1e2),
        panel_out_path=tmp_path / "panel_a.png",
        show_panel=False,
    )
    assert result_a.panel_path is not None and result_a.panel_path.exists()

    result_b = run_limited_angle_reconstruction_experiment(
        synthetic_ct_image,
        sinogram_sets,
        poisson_i0_levels=(1e4, 1e3, 1e2),
        figures_out_dir=tmp_path / "figures",
        metrics_out_path=tmp_path / "metrics.csv",
        gd_iters=2,
        gd_step_size=0.001,
        show=False,
        hardest_case=(2, "gaussian+poisson", "I0=1e+04"),
    )
    assert result_b.metrics_path.exists()
    assert result_b.figure_paths
