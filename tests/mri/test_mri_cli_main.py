"""Tests for the MRI workflow CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from a2mi.mri import main as mri_main


def test_mri_main_runs_all_stages(monkeypatch, tmp_path, capsys) -> None:
    """The MRI main entry point should wire all notebook stages together."""

    img_coils = np.ones((6, 8, 8), dtype=np.complex64)
    kspace = np.ones((6, 8, 8), dtype=np.complex64)
    rsos = np.ones((8, 8), dtype=np.float32)
    calls: dict[str, object] = {}

    def fake_prepare(path: str | Path) -> dict[str, object]:
        calls["prepare"] = Path(path)
        return {
            "k_raw": np.ones((6, 8, 8), dtype=np.complex64),
            "coil_axis": 0,
            "kspace_coils_first": kspace,
            "img_coils": img_coils,
            "rsos": rsos,
            "shape": (6, 8, 8),
        }

    def fake_plot(*args, **kwargs) -> None:
        calls.setdefault("plots", []).append(Path(kwargs["out_path"]))

    def fake_part1(**kwargs) -> dict[str, object]:
        calls["part1"] = kwargs
        return {"denoised": {"gaussian": np.ones((6, 8, 8), dtype=np.float32)}}

    def fake_part2(**kwargs) -> dict[str, object]:
        calls["part2"] = kwargs
        return {"metrics_path": Path(kwargs["metrics_out_path"])}

    def fake_part3(**kwargs) -> dict[str, object]:
        calls["part3"] = kwargs
        return {
            "rsos_bilateral": np.ones((8, 8), dtype=np.float32),
            "metrics_path": Path(kwargs["metrics_out_path"]),
        }

    def fake_mixed(**kwargs) -> dict[str, object]:
        calls["mixed"] = kwargs
        return {
            "rsos_mixed": np.ones((8, 8), dtype=np.float32),
            "figure_result": {
                "paths": {
                    "mixed_vs_reference": tmp_path / "a.png",
                    "mixed_vs_bilateral": tmp_path / "b.png",
                    "mixed_zoom": tmp_path / "c.png",
                },
                "metrics_vs_reference": {"mse": 1.0, "psnr": 2.0, "ssim": 3.0},
                "metrics_vs_bilateral": {"mse": 4.0, "psnr": 5.0, "ssim": 6.0},
            },
        }

    monkeypatch.setattr(mri_main, "prepare_exercise_2_1_data", fake_prepare)
    monkeypatch.setattr(mri_main, "plot_kspace_magnitude_all_coils", fake_plot)
    monkeypatch.setattr(mri_main, "plot_single_coil_magnitude_phase", fake_plot)
    monkeypatch.setattr(mri_main, "plot_image_magnitude_all_coils", fake_plot)
    monkeypatch.setattr(mri_main, "plot_image_magnitude_all_coils_adaptive", fake_plot)
    monkeypatch.setattr(mri_main, "plot_rsos_image", fake_plot)
    monkeypatch.setattr(mri_main, "run_exercise_2_2_part1", fake_part1)
    monkeypatch.setattr(mri_main, "run_exercise_2_2_part2", fake_part2)
    monkeypatch.setattr(mri_main, "run_exercise_2_2_part3", fake_part3)
    monkeypatch.setattr(mri_main, "_run_mixed_filter_pipeline", fake_mixed)

    exit_code = mri_main.main(["--results-root", str(tmp_path / "mri-results")])

    assert exit_code == 0
    assert calls["prepare"].name == "knee.npy"
    assert calls["part1"]["sigma"] == 1.0
    assert calls["part2"]["D0"] == 70
    assert calls["part3"]["butter_n"] == 8
    assert calls["mixed"]["gaussian_sigma"] == 1.0
    assert len(calls["plots"]) == 5

    output = capsys.readouterr().out
    assert "MRI workflow completed" in output
    assert "Mixed-filter pipeline" in output


def test_save_mixed_pipeline_figures_writes_expected_outputs(tmp_path) -> None:
    """The mixed MRI helper should export all three notebook comparison figures."""

    reference = np.zeros((10, 10), dtype=np.float32)
    bilateral = np.ones((10, 10), dtype=np.float32)
    mixed = np.full((10, 10), 0.5, dtype=np.float32)

    result = mri_main._save_mixed_pipeline_figures(reference, bilateral, mixed, tmp_path)

    assert result["paths"]["mixed_vs_reference"].exists()
    assert result["paths"]["mixed_vs_bilateral"].exists()
    assert result["paths"]["mixed_zoom"].exists()
    assert {"mse", "psnr", "ssim"} == set(result["metrics_vs_reference"])


def test_run_mixed_filter_pipeline_applies_expected_methods(monkeypatch, tmp_path) -> None:
    """The mixed MRI helper should route each coil through the intended filter."""

    img_coils = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4).astype(np.complex64)
    kspace = np.ones((6, 4, 4), dtype=np.complex64)
    bilateral_calls: list[int] = []

    monkeypatch.setattr(mri_main, "denoise_gaussian", lambda img, sigma: img + 1.0)

    def fake_bilateral(img: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
        bilateral_calls.append(int(img[0, 0]))
        return img + 2.0

    monkeypatch.setattr(mri_main, "denoise_bilateral_img", fake_bilateral)
    monkeypatch.setattr(
        mri_main,
        "run_part2_first_coil",
        lambda *_args, **_kwargs: {"img_filtered": np.full((4, 4), 5.0, dtype=np.float32)},
    )
    monkeypatch.setattr(
        mri_main,
        "_save_mixed_pipeline_figures",
        lambda **kwargs: {
            "paths": {
                "mixed_vs_reference": tmp_path / "a.png",
                "mixed_vs_bilateral": tmp_path / "b.png",
                "mixed_zoom": tmp_path / "c.png",
            },
            "metrics_vs_reference": {"mse": 1.0, "psnr": 2.0, "ssim": 3.0},
            "metrics_vs_bilateral": {"mse": 4.0, "psnr": 5.0, "ssim": 6.0},
        },
    )

    result = mri_main._run_mixed_filter_pipeline(
        img_coils=img_coils,
        kspace_coils_first=kspace,
        reference_image=np.zeros((4, 4), dtype=np.float32),
        rsos_bilateral=np.ones((4, 4), dtype=np.float32),
        figures_dir=tmp_path,
        gaussian_sigma=1.0,
        bilateral_sigma_color=0.05,
        bilateral_sigma_spatial=15.0,
        butter_D0=87.0,
        butter_n=8,
    )

    assert result["rsos_mixed"].shape == (4, 4)
    assert len(bilateral_calls) == 3
