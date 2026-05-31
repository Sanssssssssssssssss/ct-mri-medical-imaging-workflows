"""Command-line entry point for the full CT reconstruction workflow."""

from __future__ import annotations

import argparse
from time import strftime
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

from a2mi.common.io import resolve_project_path
from a2mi.ct import (
    load_reference_ct_image,
    run_exercise_1_1_reconstruction_experiment,
    run_exercise_1_1_sinogram_experiment,
    run_fbp_filter_comparison,
    run_limited_angle_reconstruction_experiment,
    run_limited_angle_sinogram_experiment,
    run_os_sart_comparison,
)


def _timestamp() -> str:
    """Return a short wall-clock timestamp for progress logs."""

    return strftime("%H:%M:%S")


def _print_heading(title: str) -> None:
    """Print a top-level workflow heading."""

    print(f"\n[{_timestamp()}] {title}")


def _print_stage(name: str, detail: str | None = None) -> None:
    """Print a stage marker with an optional detail line."""

    print(f"\n[{_timestamp()}] Stage: {name}")
    if detail:
        print(f"  {detail}")


def _print_mapping(title: str, mapping: dict[str, object]) -> None:
    """Print a small key-value block."""

    print(title)
    for key, value in mapping.items():
        print(f"  - {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CT workflow command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the full CT reconstruction workflow and export results.",
    )
    parser.add_argument(
        "--data-path",
        default="data/CT_exercise_1.png",
        help="Path to the reference CT image.",
    )
    parser.add_argument(
        "--results-root",
        default="results/ct",
        help="Root directory for CT figures and metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for noisy sinogram simulation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable iterative progress bars for GD and OS-SART.",
    )
    return parser


def run_workflow(args: argparse.Namespace) -> int:
    """Run the full CT workflow with notebook-matched parameters."""

    data_path = resolve_project_path(args.data_path)
    results_root = resolve_project_path(args.results_root)
    figures_root = results_root / "figures"
    metrics_root = results_root / "metrics"

    ex11_fig_dir = figures_root / "exercise_1_1"
    ex11_metric_path = metrics_root / "exercise_1_1" / "exercise_1_1c_metrics.csv"
    ex12_fig_dir = figures_root / "exercise_1_2"
    ex12_metric_path = metrics_root / "exercise_1_2" / "exercise_1_2_metrics.csv"
    ex13_filter_fig_dir = figures_root / "exercise_1_3" / "filters"
    ex13_filter_metric_path = metrics_root / "exercise_1_3" / "exercise_1_3_filter_metrics.csv"
    ex13_iter_fig_dir = figures_root / "exercise_1_3" / "iterative"
    ex13_iter_metric_path = metrics_root / "exercise_1_3" / "exercise_1_3_iterative_metrics.csv"

    show_progress = not args.no_progress

    _print_heading("CT workflow")
    _print_mapping(
        "Resolved paths",
        {
            "reference image": data_path,
            "results root": results_root,
            "figures root": figures_root,
            "metrics root": metrics_root,
        },
    )
    _print_mapping(
        "Notebook-matched parameters",
        {
            "exercise 1.1 angles": [360, 90, 20],
            "exercise 1.1 noise": "Gaussian(mu=0.0, sigma=0.05) + Poisson(I0=[1e5, 1e3, 1e2])",
            "exercise 1.1/1.2 attenuation_scale": 1000.0,
            "exercise 1.1 GD": "iters=50, step_size=1e-4, init=fbp, normalize=True, clip=True",
            "exercise 1.2 angle ranges": [180, 120, 40],
            "exercise 1.3 filters": ("ramp", "shepp-logan", "hann"),
            "exercise 1.3 iterative": "SIRT iters=50, OS-SART iters=50, step_size=1e-4, subsets=6",
            "iterative progress bars": show_progress,
            "seed": args.seed,
        },
    )

    _print_stage("Load reference image", "Reading and normalizing the CT reference image.")
    reference_result = load_reference_ct_image(data_path)
    reference_image = reference_result.image
    print(f"  Loaded image shape: {reference_image.shape}")
    print(f"  Source image path: {reference_result.image_path}")

    _print_stage("Exercise 1.1(a-b)", "Simulating noisy sinograms for dose reduction.")
    simulation_11 = run_exercise_1_1_sinogram_experiment(
        image=reference_image,
        angles_list=[360, 90, 20],
        gaussian_mu=0.0,
        gaussian_sigma=0.05,
        poisson_i0_levels=[1e5, 1e3, 1e2],
        attenuation_scale=1000.0,
        seed=args.seed,
        panel_out_path=ex11_fig_dir / "exercise_1_1b_noisy_sinograms.png",
        show_panel=False,
    )
    print(f"  Generated {len(simulation_11.sinogram_sets)} sinogram groups.")
    for line in simulation_11.summary_lines:
        print(f"  {line}")
    print(f"  Saved figure: {simulation_11.panel_path}")

    _print_stage("Exercise 1.1(c)", "Running FBP and gradient-descent reconstructions.")
    comparison_11 = run_exercise_1_1_reconstruction_experiment(
        reference_image=reference_image,
        sinogram_sets=simulation_11.sinogram_sets,
        poisson_i0_levels=[1e5, 1e3, 1e2],
        attenuation_scale=1000.0,
        fbp_filter="ramp",
        gd_iters=50,
        gd_step_size=0.0001,
        gd_init_mode="fbp",
        gd_normalize_gradient=True,
        gd_clip_to_reference_range=True,
        gd_mask_each_iter=True,
        gd_positivity=False,
        figures_out_dir=ex11_fig_dir,
        metrics_out_path=ex11_metric_path,
        show_figures=False,
        show_progress=show_progress,
    )
    print(f"  Saved {len(comparison_11.figure_paths)} reconstruction panels.")
    print(f"  Saved metrics: {comparison_11.metrics_path}")

    _print_stage("Exercise 1.2(a)", "Simulating limited-angle noisy sinograms.")
    simulation_12 = run_limited_angle_sinogram_experiment(
        image=reference_image,
        angle_ranges=[180, 120, 40],
        step_deg=1.0,
        poisson_i0_levels=[1e5, 1e3, 1e2],
        gaussian_mu=0.0,
        gaussian_sigma=0.05,
        attenuation_scale=1000.0,
        seed=args.seed,
        panel_out_path=ex12_fig_dir / "exercise_1_2_limited_angle_noisy_sinograms.png",
        show_panel=False,
    )
    print(f"  Generated {len(simulation_12.sinogram_sets)} limited-angle groups.")
    for line in simulation_12.summary_lines:
        print(f"  {line}")
    print(f"  Saved figure: {simulation_12.panel_path}")

    _print_stage("Exercise 1.2(b)", "Running limited-angle FBP and gradient-descent reconstructions.")
    comparison_12 = run_limited_angle_reconstruction_experiment(
        reference_image=reference_image,
        sinogram_sets=simulation_12.sinogram_sets,
        poisson_i0_levels=[1e5, 1e3, 1e2],
        attenuation_scale=1000.0,
        figures_out_dir=ex12_fig_dir,
        metrics_out_path=ex12_metric_path,
        show=False,
        show_progress=show_progress,
        fbp_filter="ramp",
        gd_iters=50,
        gd_step_size=0.0001,
        gd_init_mode="fbp",
        gd_normalize_gradient=True,
        gd_clip_to_reference_range=True,
        gd_mask_each_iter=True,
        gd_positivity=False,
        metric_mode="reporting",
    )
    print(f"  Saved {len(comparison_12.figure_paths)} reconstruction panels.")
    print(f"  Saved metrics: {comparison_12.metrics_path}")

    _print_stage("Exercise 1.3(a-b)", "Comparing FBP filters on the selected low-dose case.")
    filter_result = run_fbp_filter_comparison(
        reference_image=reference_image,
        sinogram_sets=simulation_11.sinogram_sets,
        n_angles=90,
        i0_level=1e3,
        attenuation_scale=1000.0,
        filter_names=("ramp", "shepp-logan", "hann"),
        metric_mode="reporting",
        out_dir=ex13_filter_fig_dir,
        metrics_out_path=ex13_filter_metric_path,
        show=False,
    )
    print(f"  Saved filter comparison figures under: {filter_result.output_dir}")
    print(f"  Saved metrics: {filter_result.metrics_path}")

    _print_stage("Exercise 1.3(c)", "Comparing SIRT-style GD and OS-SART on the hardest low-dose case.")
    iterative_result = run_os_sart_comparison(
        reference_image=reference_image,
        sinogram_sets=simulation_11.sinogram_sets,
        n_angles=360,
        i0_level=1e2,
        attenuation_scale=1000.0,
        sirt_iters=50,
        sirt_step_size=0.0001,
        sirt_init_mode="fbp",
        sirt_normalize_gradient=True,
        sirt_clip_to_reference_range=True,
        sirt_mask_each_iter=True,
        os_sart_iters=50,
        os_sart_step_size=0.0001,
        n_subsets=6,
        metric_mode="reporting",
        positivity=False,
        out_dir=ex13_iter_fig_dir,
        metrics_out_path=ex13_iter_metric_path,
        show=False,
        show_progress=show_progress,
    )
    print(f"  Saved figure: {iterative_result.figure_path}")
    print(f"  Saved metrics: {iterative_result.metrics_path}")

    _print_heading("CT workflow completed")
    _print_mapping(
        "Export summary",
        {
            "exercise 1.1 figures": ex11_fig_dir,
            "exercise 1.1 metrics": comparison_11.metrics_path,
            "exercise 1.2 figures": ex12_fig_dir,
            "exercise 1.2 metrics": comparison_12.metrics_path,
            "exercise 1.3 filter figures": filter_result.output_dir,
            "exercise 1.3 filter metrics": filter_result.metrics_path,
            "exercise 1.3 iterative figures": iterative_result.figure_path.parent,
            "exercise 1.3 iterative metrics": iterative_result.metrics_path,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the CT workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
