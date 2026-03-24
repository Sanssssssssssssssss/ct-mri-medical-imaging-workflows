"""Tests for MRI Exercise 2.2 helpers."""

from __future__ import annotations

import numpy as np
import pytest

from a2mi.mri.exercise_2_1 import combine_rsos
from a2mi.mri.exercise_2_2 import (
    butterworth_lowpass_filter,
    compute_metrics,
    denoise_all_coils,
    denoise_all_coils_rsos,
    denoise_bilateral_img,
    denoise_combined_image,
    denoise_gaussian,
    denoise_median,
    metrics_part2,
    metrics_part3,
    plot_denoise_all_coils,
    plot_denoise_per_coil,
    plot_part2_compare,
    plot_part2_mag_phase,
    plot_part3_combined_compare,
    print_metrics_rows,
    run_butterworth_all_coils_rsos,
    run_exercise_2_2_part1,
    run_exercise_2_2_part2,
    run_exercise_2_2_part3,
    run_part2_first_coil,
    save_metrics_rows,
)


def test_denoisers_metrics_and_csv(tmp_path, capsys, synthetic_mri_image_coils: np.ndarray, synthetic_kspace_coils: np.ndarray) -> None:
    """Image-space denoisers, metrics, and CSV helpers should behave consistently."""
    magnitude = np.abs(synthetic_mri_image_coils[0]).astype(np.float32)
    gaussian = denoise_gaussian(magnitude, sigma=1.0)
    median = denoise_median(magnitude, size=3)
    bilateral = denoise_bilateral_img(magnitude, sigma_color=0.1, sigma_spatial=2.0)
    assert gaussian.shape == magnitude.shape
    assert median.shape == magnitude.shape
    assert bilateral.shape == magnitude.shape

    denoised = denoise_all_coils(synthetic_mri_image_coils, sigma=1.0, median_size=3, sigma_color=0.1, sigma_spatial=2.0)
    assert set(denoised) == {"original", "gaussian", "median", "bilateral"}

    mse, psnr, ssim = compute_metrics(magnitude, magnitude)
    assert mse == 0.0
    assert psnr == float("inf")
    assert ssim == 1.0

    rows = [{"part": "test", "coil": 0, "method": "gaussian", "mse": 1.0, "psnr": 2.0, "ssim": 3.0}]
    csv_path = save_metrics_rows(rows, tmp_path / "metrics.csv")
    assert csv_path.exists()
    print_metrics_rows(rows, title="Rows")
    captured = capsys.readouterr()
    assert "method=gaussian" in captured.out

    filt = butterworth_lowpass_filter(synthetic_kspace_coils.shape[-2:], D0=5.0, n=2)
    assert filt.shape == synthetic_kspace_coils.shape[-2:]
    part2 = run_part2_first_coil(synthetic_kspace_coils, coil_id=0, D0=5.0, n=2)
    assert "img_filtered" in part2
    rsos_butter = run_butterworth_all_coils_rsos(synthetic_kspace_coils, D0=5.0, n=2)
    assert rsos_butter.shape == magnitude.shape


def test_plotting_and_wrapper_workflows(tmp_path, synthetic_mri_image_coils: np.ndarray, synthetic_kspace_coils: np.ndarray) -> None:
    """Exercise 2.2 plotting helpers and wrappers should save their outputs."""
    rsos = combine_rsos(synthetic_mri_image_coils)
    denoised = denoise_all_coils(synthetic_mri_image_coils, sigma=1.0, median_size=3, sigma_color=0.1, sigma_spatial=2.0)
    part2 = run_part2_first_coil(synthetic_kspace_coils, coil_id=0, D0=5.0, n=2)

    plot_denoise_per_coil(denoised, coil_id=0, out_path=tmp_path / "coil.png", show=False)
    saved = plot_denoise_all_coils(denoised, out_dir=tmp_path / "all_coils", show=False)
    plot_part2_mag_phase(part2, out_path=tmp_path / "mag_phase.png", show=False)
    plot_part2_compare(part2, denoised, coil_id=0, out_path=tmp_path / "compare.png", show=False)
    plot_part3_combined_compare(rsos, rsos, method_name="identity", out_path=tmp_path / "part3.png", show=False)
    assert (tmp_path / "coil.png").exists()
    assert saved and all(path.exists() for path in saved)
    assert (tmp_path / "mag_phase.png").exists()
    assert (tmp_path / "compare.png").exists()
    assert (tmp_path / "part3.png").exists()

    part1 = run_exercise_2_2_part1(
        synthetic_mri_image_coils,
        sigma=1.0,
        median_size=3,
        sigma_color=0.1,
        sigma_spatial=2.0,
        out_dir=tmp_path / "part1",
        show=False,
    )
    assert part1["figure_paths"]

    part2_result = run_exercise_2_2_part2(
        synthetic_kspace_coils,
        denoised,
        coil_id=0,
        D0=5.0,
        n=2,
        reference=np.abs(synthetic_mri_image_coils[0]).astype(np.float32),
        mag_phase_out_path=tmp_path / "wrapper_mag_phase.png",
        compare_out_path=tmp_path / "wrapper_compare.png",
        metrics_out_path=tmp_path / "wrapper_metrics.csv",
        show=False,
    )
    assert part2_result["metrics_path"].exists()

    part3_result = run_exercise_2_2_part3(
        synthetic_mri_image_coils,
        synthetic_kspace_coils,
        rsos,
        reference=np.abs(synthetic_mri_image_coils[0]).astype(np.float32),
        bilateral_sigma_color=0.1,
        bilateral_sigma_spatial=2.0,
        butter_D0=5.0,
        butter_n=2,
        bilateral_out_path=tmp_path / "bilateral.png",
        butterworth_out_path=tmp_path / "butter.png",
        metrics_out_path=tmp_path / "part3_metrics.csv",
        show=False,
    )
    assert part3_result["metrics_path"].exists()


def test_metrics_helpers_and_error_paths(synthetic_mri_image_coils: np.ndarray, synthetic_kspace_coils: np.ndarray) -> None:
    """Metric helpers and failure branches should behave as expected."""
    denoised = denoise_all_coils(synthetic_mri_image_coils, sigma=1.0, median_size=3, sigma_color=0.1, sigma_spatial=2.0)
    part2 = run_part2_first_coil(synthetic_kspace_coils, coil_id=0, D0=5.0, n=2)
    rows_part2 = metrics_part2(denoised, part2, coil_id=0)
    assert any(row["method"] == "kspace_butterworth" for row in rows_part2)

    rsos = combine_rsos(synthetic_mri_image_coils)
    rows_part3 = metrics_part3(rsos, rsos, method_name="same")
    assert rows_part3[0]["method"] == "original"

    denoised_combined = denoise_combined_image(rsos, method="gaussian", sigma=1.0)
    assert denoised_combined.shape == rsos.shape

    rsos_bilateral = denoise_all_coils_rsos(synthetic_mri_image_coils, method="bilateral", sigma_color=0.1, sigma_spatial=2.0)
    assert rsos_bilateral.shape == rsos.shape

    with pytest.raises(ValueError):
        denoise_combined_image(rsos, method="unknown")
    with pytest.raises(ValueError):
        denoise_all_coils_rsos(synthetic_mri_image_coils, method="unknown")
    with pytest.raises(ValueError):
        run_butterworth_all_coils_rsos(np.zeros((4, 4), dtype=np.float32))
