"""Tests for the CT workflow CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from a2mi.ct import main as ct_main


class _ReferenceResult:
    def __init__(self, output_path: Path) -> None:
        self.image = np.ones((8, 8), dtype=np.float32)
        self.image_path = output_path
        self.output_path = output_path


class _SinogramSimulationResult:
    def __init__(self, panel_path: Path) -> None:
        self.sinogram_sets = ["dummy"]
        self.summary_lines = ["angles=360 | gaussian+poisson generated"]
        self.panel_path = panel_path


class _ComparisonResult:
    def __init__(self, metrics_path: Path, figure_dir: Path) -> None:
        self.figure_paths = [figure_dir / "figure.png"]
        self.metrics_path = metrics_path


class _FilterResult:
    def __init__(self, output_dir: Path, metrics_path: Path) -> None:
        self.output_dir = output_dir
        self.metrics_path = metrics_path


class _IterativeResult:
    def __init__(self, figure_path: Path, metrics_path: Path) -> None:
        self.figure_path = figure_path
        self.metrics_path = metrics_path


def test_ct_main_runs_and_passes_progress_flag(monkeypatch, tmp_path, capsys) -> None:
    """The CT main entry point should call every workflow stage with notebook settings."""

    calls: dict[str, dict[str, object]] = {}

    def fake_load_reference_ct_image(path: str | Path) -> _ReferenceResult:
        calls["load_reference"] = {"path": Path(path)}
        return _ReferenceResult(tmp_path / "reference.png")

    def fake_run_ex11_sinograms(**kwargs) -> _SinogramSimulationResult:
        calls["ex11_sinograms"] = kwargs
        return _SinogramSimulationResult(Path(kwargs["panel_out_path"]))

    def fake_run_ex11_recon(**kwargs) -> _ComparisonResult:
        calls["ex11_recon"] = kwargs
        return _ComparisonResult(Path(kwargs["metrics_out_path"]), Path(kwargs["figures_out_dir"]))

    def fake_run_ex12_sinograms(**kwargs) -> _SinogramSimulationResult:
        calls["ex12_sinograms"] = kwargs
        return _SinogramSimulationResult(Path(kwargs["panel_out_path"]))

    def fake_run_ex12_recon(**kwargs) -> _ComparisonResult:
        calls["ex12_recon"] = kwargs
        return _ComparisonResult(Path(kwargs["metrics_out_path"]), Path(kwargs["figures_out_dir"]))

    def fake_run_filters(**kwargs) -> _FilterResult:
        calls["filters"] = kwargs
        return _FilterResult(Path(kwargs["out_dir"]), Path(kwargs["metrics_out_path"]))

    def fake_run_iterative(**kwargs) -> _IterativeResult:
        calls["iterative"] = kwargs
        return _IterativeResult(Path(kwargs["out_dir"]) / "iterative.png", Path(kwargs["metrics_out_path"]))

    monkeypatch.setattr(ct_main, "load_reference_ct_image", fake_load_reference_ct_image)
    monkeypatch.setattr(ct_main, "run_exercise_1_1_sinogram_experiment", fake_run_ex11_sinograms)
    monkeypatch.setattr(ct_main, "run_exercise_1_1_reconstruction_experiment", fake_run_ex11_recon)
    monkeypatch.setattr(ct_main, "run_limited_angle_sinogram_experiment", fake_run_ex12_sinograms)
    monkeypatch.setattr(ct_main, "run_limited_angle_reconstruction_experiment", fake_run_ex12_recon)
    monkeypatch.setattr(ct_main, "run_fbp_filter_comparison", fake_run_filters)
    monkeypatch.setattr(ct_main, "run_os_sart_comparison", fake_run_iterative)

    exit_code = ct_main.main(
        [
            "--data-path",
            "data/CT_exercise_1.png",
            "--results-root",
            str(tmp_path / "ct-results"),
        ],
    )

    assert exit_code == 0
    assert calls["load_reference"]["path"].name == "CT_exercise_1.png"
    assert calls["ex11_sinograms"]["angles_list"] == [360, 90, 20]
    assert calls["ex11_recon"]["show_progress"] is True
    assert calls["ex11_recon"]["gd_step_size"] == 0.0001
    assert calls["ex12_sinograms"]["angle_ranges"] == [180, 120, 40]
    assert calls["filters"]["filter_names"] == ("ramp", "shepp-logan", "hann")
    assert calls["iterative"]["n_subsets"] == 6
    assert calls["iterative"]["show_progress"] is True

    output = capsys.readouterr().out
    assert "CT workflow completed" in output
    assert "exercise 1.3 iterative metrics" in output


def test_ct_main_can_disable_progress(monkeypatch, tmp_path) -> None:
    """The CT CLI should turn off iterative progress bars when requested."""

    monkeypatch.setattr(ct_main, "load_reference_ct_image", lambda path: _ReferenceResult(tmp_path / "ref.png"))
    monkeypatch.setattr(
        ct_main,
        "run_exercise_1_1_sinogram_experiment",
        lambda **kwargs: _SinogramSimulationResult(Path(kwargs["panel_out_path"])),
    )
    monkeypatch.setattr(
        ct_main,
        "run_limited_angle_sinogram_experiment",
        lambda **kwargs: _SinogramSimulationResult(Path(kwargs["panel_out_path"])),
    )
    monkeypatch.setattr(
        ct_main,
        "run_fbp_filter_comparison",
        lambda **kwargs: _FilterResult(Path(kwargs["out_dir"]), Path(kwargs["metrics_out_path"])),
    )

    ex11_progress: dict[str, object] = {}
    ex12_progress: dict[str, object] = {}
    ex13_progress: dict[str, object] = {}

    def fake_ex11_recon(**kwargs) -> _ComparisonResult:
        ex11_progress.update(kwargs)
        return _ComparisonResult(Path(kwargs["metrics_out_path"]), Path(kwargs["figures_out_dir"]))

    def fake_ex12_recon(**kwargs) -> _ComparisonResult:
        ex12_progress.update(kwargs)
        return _ComparisonResult(Path(kwargs["metrics_out_path"]), Path(kwargs["figures_out_dir"]))

    def fake_iterative(**kwargs) -> _IterativeResult:
        ex13_progress.update(kwargs)
        return _IterativeResult(Path(kwargs["out_dir"]) / "iterative.png", Path(kwargs["metrics_out_path"]))

    monkeypatch.setattr(ct_main, "run_exercise_1_1_reconstruction_experiment", fake_ex11_recon)
    monkeypatch.setattr(ct_main, "run_limited_angle_reconstruction_experiment", fake_ex12_recon)
    monkeypatch.setattr(ct_main, "run_os_sart_comparison", fake_iterative)

    exit_code = ct_main.main(["--results-root", str(tmp_path / "ct-results"), "--no-progress"])

    assert exit_code == 0
    assert ex11_progress["show_progress"] is False
    assert ex12_progress["show_progress"] is False
    assert ex13_progress["show_progress"] is False
