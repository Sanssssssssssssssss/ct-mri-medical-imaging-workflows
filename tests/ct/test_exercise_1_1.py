"""Tests for CT Exercise 1.1 helpers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from a2mi.ct.exercise_1_1 import (
    add_gaussian_noise,
    add_gaussian_poisson_noise_practical,
    add_poisson_noise,
    forward_project,
    load_metrics_csv,
    load_reference_ct_image,
    make_theta,
    prepare_reference_ct_image,
    reconstruct_fbp,
    reconstruct_gradient_descent,
    run_exercise_1_1_reconstruction_experiment,
    run_exercise_1_1_sinogram_experiment,
    run_reconstruction_comparison,
    save_metrics_csv,
    save_reconstruction_panels,
    scale_image_for_practical,
    simulate_noisy_sinograms,
    summarize_metrics,
    summarize_sinogram_sets,
)


def test_prepare_reference_ct_image_and_loader(tmp_path, synthetic_ct_image: np.ndarray) -> None:
    """Reference image helpers should normalize and load RGB-like images."""
    rgb = np.dstack([synthetic_ct_image * 255] * 3).astype(np.uint8)
    prepared = prepare_reference_ct_image(rgb)
    assert prepared.dtype == np.float32
    assert prepared.shape == synthetic_ct_image.shape
    assert prepared.max() <= 1.0

    image_path = tmp_path / "ct.png"
    Image.fromarray(rgb).save(image_path)
    loaded = load_reference_ct_image(image_path)
    assert loaded.image_path == image_path
    assert loaded.image.shape == synthetic_ct_image.shape
    assert loaded.source_shape == rgb.shape


def test_theta_projection_and_noise_models(synthetic_ct_image: np.ndarray) -> None:
    """Projection and noise helpers should produce finite arrays with expected shapes."""
    theta = make_theta(8)
    np.testing.assert_allclose(theta, np.linspace(0.0, 360.0, 8, endpoint=False, dtype=np.float32))

    scaled = scale_image_for_practical(synthetic_ct_image, attenuation_scale=1000.0)
    sinogram, theta_out = forward_project(scaled, n_angles=8)
    assert sinogram.shape[1] == 8
    assert theta_out.shape == (8,)

    rng = np.random.default_rng(0)
    noisy_g = add_gaussian_noise(sinogram, sigma=0.01, rng=rng)
    noisy_p = add_poisson_noise(sinogram, i0=1e4, rng=rng)
    noisy_gp = add_gaussian_poisson_noise_practical(sinogram, i0=1e4, gaussian_sigma=0.01, rng=rng)
    assert noisy_g.shape == sinogram.shape
    assert noisy_p.shape == sinogram.shape
    assert noisy_gp.shape == sinogram.shape
    assert np.isfinite(noisy_p).all()
    assert np.isfinite(noisy_gp).all()


def test_simulation_workflow_and_metric_helpers(tmp_path, synthetic_ct_image: np.ndarray) -> None:
    """Exercise 1.1 workflow helpers should save artefacts and reload metrics."""
    sinogram_sets = simulate_noisy_sinograms(
        synthetic_ct_image,
        angles_list=(12, 6),
        poisson_i0_levels=(1e4, 1e3, 1e2),
        seed=0,
    )
    assert len(sinogram_sets) == 2
    summary = summarize_sinogram_sets(sinogram_sets)
    assert "angles=" in summary[0]

    comparison = run_reconstruction_comparison(
        reference_image=scale_image_for_practical(synthetic_ct_image),
        sinogram_sets=sinogram_sets,
        poisson_i0_levels=(1e4, 1e3, 1e2),
        gd_iters=2,
        gd_step_size=0.001,
        metric_mode="practical",
    )
    rows = summarize_metrics(comparison)
    metrics_path = save_metrics_csv(rows, tmp_path / "metrics.csv")
    reloaded = load_metrics_csv(metrics_path)
    assert len(reloaded) == len(rows)

    figure_paths = save_reconstruction_panels(
        scale_image_for_practical(synthetic_ct_image),
        comparison,
        out_dir=tmp_path / "figures",
        hardest_case=(6, "gaussian+poisson", "I0=1e+04"),
        show=False,
    )
    assert figure_paths
    assert all(path.exists() for path in figure_paths)


def test_exercise_1_1_wrappers_save_expected_outputs(tmp_path, synthetic_ct_image: np.ndarray) -> None:
    """Top-level Exercise 1.1 wrappers should save panels, figures, and CSVs."""
    simulation = run_exercise_1_1_sinogram_experiment(
        synthetic_ct_image,
        angles_list=(8,),
        poisson_i0_levels=(1e4, 1e3, 1e2),
        panel_out_path=tmp_path / "sinograms.png",
        show_panel=False,
        seed=1,
    )
    assert simulation.panel_path is not None and simulation.panel_path.exists()

    reconstruction = run_exercise_1_1_reconstruction_experiment(
        reference_image=synthetic_ct_image,
        sinogram_sets=simulation.sinogram_sets,
        poisson_i0_levels=(1e4, 1e3, 1e2),
        figures_out_dir=tmp_path / "reconstruction_figures",
        metrics_out_path=tmp_path / "reconstruction_metrics.csv",
        gd_iters=2,
        gd_step_size=0.001,
        show_figures=False,
    )
    assert reconstruction.metrics_path.exists()
    assert reconstruction.figure_paths


def test_reconstructors_and_invalid_inputs(synthetic_ct_image: np.ndarray) -> None:
    """Reconstruction helpers should handle valid and invalid input branches."""
    scaled = scale_image_for_practical(synthetic_ct_image)
    sinogram, theta = forward_project(scaled, n_angles=10)

    fbp = reconstruct_fbp(sinogram, theta, output_size=scaled.shape[0])
    gd = reconstruct_gradient_descent(
        sinogram,
        theta,
        output_size=scaled.shape[0],
        n_iters=2,
        step_size=0.001,
        init_mode="zeros",
        positivity=True,
        clip_range=(0.0, 1.0),
    )
    assert fbp.shape == scaled.shape
    assert gd.shape == scaled.shape
    assert np.isfinite(gd).all()

    with pytest.raises(ValueError):
        forward_project(np.zeros((4, 4, 2), dtype=np.float32), n_angles=4)
    with pytest.raises(ValueError):
        reconstruct_gradient_descent(sinogram, theta, init_mode="bad-mode")
    with pytest.raises(ValueError):
        reconstruct_gradient_descent(sinogram, theta, init_image=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        run_reconstruction_comparison(
            reference_image=scaled,
            sinogram_sets=simulate_noisy_sinograms(synthetic_ct_image, angles_list=(4,), poisson_i0_levels=(1e3,)),
            poisson_i0_levels=(1e3,),
            metric_mode="unknown",
            gd_iters=1,
        )
