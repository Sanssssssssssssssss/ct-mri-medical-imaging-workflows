"""Command-line entry point for the full MRI denoising workflow."""

from __future__ import annotations

import argparse
from time import strftime
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from a2mi.common.io import resolve_project_path
from a2mi.mri import (
    compute_metrics,
    denoise_bilateral_img,
    denoise_gaussian,
    plot_image_magnitude_all_coils,
    plot_image_magnitude_all_coils_adaptive,
    plot_kspace_magnitude_all_coils,
    plot_rsos_image,
    plot_single_coil_magnitude_phase,
    prepare_exercise_2_1_data,
    run_exercise_2_2_part1,
    run_exercise_2_2_part2,
    run_exercise_2_2_part3,
    run_part2_first_coil,
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


def _save_mixed_pipeline_figures(
    reference_image: np.ndarray,
    rsos_bilateral: np.ndarray,
    rsos_mixed: np.ndarray,
    figures_dir,
) -> dict[str, object]:
    """Save the mixed-filter comparison figures used in the MRI notebook tail."""

    ref_img = np.asarray(reference_image, dtype=np.float32)
    bilat_img = np.asarray(rsos_bilateral, dtype=np.float32)
    mixed_img = np.asarray(rsos_mixed, dtype=np.float32)

    mse_ref, psnr_ref, ssim_ref = compute_metrics(ref_img, mixed_img)
    mse_bi, psnr_bi, ssim_bi = compute_metrics(bilat_img, mixed_img)
    shared_vmax = float(max(np.max(bilat_img), np.max(mixed_img)))

    mixed_vs_ref_path = figures_dir / "ex2_2_part3_rsos_compare_mixed_reference.png"
    fig1, axes1 = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes1[0].imshow(ref_img, cmap="gray", vmin=0.0, vmax=shared_vmax)
    axes1[0].set_title("Reference (coil 0 magnitude)")
    axes1[0].axis("off")
    fig1.colorbar(im0, ax=axes1[0], fraction=0.046, pad=0.04)
    im1 = axes1[1].imshow(mixed_img, cmap="gray", vmin=0.0, vmax=shared_vmax)
    axes1[1].set_title("Mixed filters -> rSoS")
    axes1[1].axis("off")
    fig1.colorbar(im1, ax=axes1[1], fraction=0.046, pad=0.04)
    fig1.savefig(mixed_vs_ref_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    mixed_vs_bilateral_path = figures_dir / "ex2_2_part3_rsos_compare_mixed_vs_bilateral.png"
    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im2 = axes2[0].imshow(bilat_img, cmap="gray", vmin=0.0, vmax=shared_vmax)
    axes2[0].set_title("All bilateral -> rSoS")
    axes2[0].axis("off")
    fig2.colorbar(im2, ax=axes2[0], fraction=0.046, pad=0.04)
    im3 = axes2[1].imshow(mixed_img, cmap="gray", vmin=0.0, vmax=shared_vmax)
    axes2[1].set_title("Mixed filters -> rSoS")
    axes2[1].axis("off")
    fig2.colorbar(im3, ax=axes2[1], fraction=0.046, pad=0.04)
    fig2.savefig(mixed_vs_bilateral_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    h, w = mixed_img.shape
    roi_h = int(0.28 * h)
    roi_w = int(0.28 * w)
    y0 = (h - roi_h) // 2
    x0 = (w - roi_w) // 2
    y1 = y0 + roi_h
    x1 = x0 + roi_w

    mixed_zoom_path = figures_dir / "ex2_2_part3_rsos_compare_mixed_vs_bilateral_zoom.png"
    fig3, axes3 = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    for ax, img, title in (
        (axes3[0, 0], bilat_img, "All bilateral -> rSoS"),
        (axes3[0, 1], mixed_img, "Mixed filters -> rSoS"),
    ):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=shared_vmax)
        ax.add_patch(
            Rectangle((x0, y0), roi_w, roi_h, fill=False, edgecolor="red", linewidth=2.0),
        )
        ax.set_title(title)
        ax.axis("off")
    axes3[1, 0].imshow(bilat_img[y0:y1, x0:x1], cmap="gray", vmin=0.0, vmax=shared_vmax)
    axes3[1, 0].set_title("Zoomed bilateral ROI")
    axes3[1, 0].axis("off")
    axes3[1, 1].imshow(mixed_img[y0:y1, x0:x1], cmap="gray", vmin=0.0, vmax=shared_vmax)
    axes3[1, 1].set_title("Zoomed mixed ROI")
    axes3[1, 1].axis("off")
    fig3.savefig(mixed_zoom_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    return {
        "paths": {
            "mixed_vs_reference": mixed_vs_ref_path,
            "mixed_vs_bilateral": mixed_vs_bilateral_path,
            "mixed_zoom": mixed_zoom_path,
        },
        "metrics_vs_reference": {
            "mse": mse_ref,
            "psnr": psnr_ref,
            "ssim": ssim_ref,
        },
        "metrics_vs_bilateral": {
            "mse": mse_bi,
            "psnr": psnr_bi,
            "ssim": ssim_bi,
        },
    }


def _run_mixed_filter_pipeline(
    img_coils: np.ndarray,
    kspace_coils_first: np.ndarray,
    reference_image: np.ndarray,
    rsos_bilateral: np.ndarray,
    figures_dir,
    gaussian_sigma: float,
    bilateral_sigma_color: float,
    bilateral_sigma_spatial: float,
    butter_D0: float,
    butter_n: int,
) -> dict[str, object]:
    """Run the notebook's mixed per-coil filtering experiment."""

    mixed_coils = np.abs(img_coils).astype(np.float32).copy()
    mixed_coils[0] = np.abs(img_coils[0]).astype(np.float32)
    mixed_coils[1] = denoise_gaussian(np.abs(img_coils[1]).astype(np.float32), sigma=gaussian_sigma)
    for coil_id in (2, 3, 4):
        mixed_coils[coil_id] = denoise_bilateral_img(
            np.abs(img_coils[coil_id]).astype(np.float32),
            sigma_color=bilateral_sigma_color,
            sigma_spatial=bilateral_sigma_spatial,
        )
    part3_mixed_butter = run_part2_first_coil(kspace_coils_first, coil_id=5, D0=butter_D0, n=butter_n)
    mixed_coils[5] = np.abs(part3_mixed_butter["img_filtered"]).astype(np.float32)
    rsos_mixed = np.sqrt(np.sum(mixed_coils**2, axis=0)).astype(np.float32)

    figure_result = _save_mixed_pipeline_figures(
        reference_image=reference_image,
        rsos_bilateral=rsos_bilateral,
        rsos_mixed=rsos_mixed,
        figures_dir=figures_dir,
    )
    return {
        "rsos_mixed": rsos_mixed,
        "figure_result": figure_result,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the MRI workflow command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the full MRI denoising workflow and export results.",
    )
    parser.add_argument(
        "--data-path",
        default="data/knee.npy",
        help="Path to the MRI k-space array.",
    )
    parser.add_argument(
        "--results-root",
        default="results/mri",
        help="Root directory for MRI figures and metrics.",
    )
    return parser


def run_workflow(args: argparse.Namespace) -> int:
    """Run the full MRI workflow with notebook-matched parameters."""

    data_path = resolve_project_path(args.data_path)
    results_root = resolve_project_path(args.results_root)
    figures_root = results_root / "figures"
    metrics_root = results_root / "metrics"

    ex21_fig_dir = figures_root / "ex2_1"
    ex22_part1_fig_dir = figures_root / "ex2_2" / "part1"
    ex22_part2_fig_dir = figures_root / "ex2_2" / "part2"
    ex22_part3_fig_dir = figures_root / "ex2_2" / "part3"
    ex22_metric_dir = metrics_root / "ex2_2"

    for out_dir in (
        ex21_fig_dir,
        ex22_part1_fig_dir,
        ex22_part2_fig_dir,
        ex22_part3_fig_dir,
        ex22_metric_dir,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)

    _print_heading("MRI workflow")
    _print_mapping(
        "Resolved paths",
        {
            "k-space array": data_path,
            "results root": results_root,
            "figures root": figures_root,
            "metrics root": metrics_root,
        },
    )
    _print_mapping(
        "Notebook-matched parameters",
        {
            "part 1 gaussian sigma": 1.0,
            "part 1 median size": 3,
            "part 1 bilateral sigma_color": 0.05,
            "part 1 bilateral sigma_spatial": 20.0,
            "part 2 coil_id": 0,
            "part 2 Butterworth": "D0=70, n=2",
            "part 3 bilateral": "sigma_color=0.05, sigma_spatial=15.0",
            "part 3 Butterworth": "D0=87.0, n=8",
            "mixed pipeline": "coil0 original, coil1 Gaussian, coil2-4 bilateral, coil5 Butterworth",
        },
    )

    _print_stage("Exercise 2.1", "Loading k-space data and preparing coil-first arrays.")
    mri_data = prepare_exercise_2_1_data(data_path)
    k_raw = mri_data["k_raw"]
    coil_axis = mri_data["coil_axis"]
    kspace_coils_first = mri_data["kspace_coils_first"]
    img_coils = mri_data["img_coils"]
    rsos = mri_data["rsos"]
    print(f"  Raw shape: {tuple(k_raw.shape)}")
    print(f"  Inferred coil axis: {coil_axis}")
    print(f"  Coil-first shape: {tuple(kspace_coils_first.shape)}")

    _print_stage("Exercise 2.1 exports", "Saving k-space, image-space, adaptive, and rSoS figures.")
    plot_kspace_magnitude_all_coils(
        kspace_coils_first,
        out_path=ex21_fig_dir / "ex2_1_kspace_all_coils.png",
        show=False,
    )
    plot_single_coil_magnitude_phase(
        img_coils,
        coil_id=0,
        out_path=ex21_fig_dir / "ex2_1_single_coil_mag_phase.png",
        show=False,
    )
    plot_image_magnitude_all_coils(
        img_coils,
        out_path=ex21_fig_dir / "ex2_1_image_all_coils.png",
        show=False,
    )
    plot_image_magnitude_all_coils_adaptive(
        img_coils,
        out_path=ex21_fig_dir / "ex2_1_image_all_coils_adaptive.png",
        show=False,
    )
    plot_rsos_image(
        rsos,
        out_path=ex21_fig_dir / "ex2_1_rsos.png",
        show=False,
    )
    print(f"  Saved figures under: {ex21_fig_dir}")

    _print_stage("Exercise 2.2 Part 1", "Running per-coil Gaussian, median, and bilateral denoising.")
    part1_result = run_exercise_2_2_part1(
        img_coils=img_coils,
        sigma=1.0,
        median_size=3,
        sigma_color=0.05,
        sigma_spatial=20.0,
        out_dir=ex22_part1_fig_dir,
        show=False,
    )
    denoised = part1_result["denoised"]
    print(f"  Saved figures under: {ex22_part1_fig_dir}")

    _print_stage("Exercise 2.2 Part 2", "Comparing first-coil image-space denoising and Butterworth filtering.")
    part2_reference = np.abs(img_coils[0]).astype(np.float32)
    part2_result = run_exercise_2_2_part2(
        kspace_coils_first=kspace_coils_first,
        denoised=denoised,
        coil_id=0,
        D0=70,
        n=2,
        reference=part2_reference,
        mag_phase_out_path=ex22_part2_fig_dir / "ex2_2_part2_coil0_mag_phase.png",
        compare_out_path=ex22_part2_fig_dir / "ex2_2_part2_compare_with_image_methods.png",
        metrics_out_path=ex22_metric_dir / "ex2_2_part2_metrics.csv",
        show=False,
    )
    print(f"  Saved figures under: {ex22_part2_fig_dir}")
    print(f"  Saved metrics: {part2_result['metrics_path']}")

    _print_stage("Exercise 2.2 Part 3", "Comparing per-coil bilateral and Butterworth filtering before rSoS.")
    part3_reference = np.abs(img_coils[0]).astype(np.float32)
    part3_result = run_exercise_2_2_part3(
        img_coils=img_coils,
        kspace_coils_first=kspace_coils_first,
        rsos=rsos,
        reference=part3_reference,
        bilateral_sigma_color=0.05,
        bilateral_sigma_spatial=15.0,
        butter_D0=87.0,
        butter_n=8,
        bilateral_out_path=ex22_part3_fig_dir / "ex2_2_part3_rsos_compare_bilateral.png",
        butterworth_out_path=ex22_part3_fig_dir / "ex2_2_part3_rsos_compare_butterworth.png",
        metrics_out_path=ex22_metric_dir / "ex2_2_part3_metrics.csv",
        show=False,
    )
    rsos_bilateral = np.asarray(part3_result["rsos_bilateral"], dtype=np.float32)
    print(f"  Saved figures under: {ex22_part3_fig_dir}")
    print(f"  Saved metrics: {part3_result['metrics_path']}")

    _print_stage("Mixed-filter experiment", "Running the notebook tail experiment and ROI zoom comparison.")
    mixed_result = _run_mixed_filter_pipeline(
        img_coils=img_coils,
        kspace_coils_first=kspace_coils_first,
        reference_image=part3_reference,
        rsos_bilateral=rsos_bilateral,
        figures_dir=ex22_part3_fig_dir,
        gaussian_sigma=1.0,
        bilateral_sigma_color=0.05,
        bilateral_sigma_spatial=15.0,
        butter_D0=87.0,
        butter_n=8,
    )
    figure_result = mixed_result["figure_result"]
    print("  Mixed-filter pipeline: coil 0 = original, coil 1 = Gaussian, coils 2-4 = bilateral, coil 5 = Butterworth")
    print(
        "  vs reference | "
        f"mse={figure_result['metrics_vs_reference']['mse']:.6f} | "
        f"psnr={figure_result['metrics_vs_reference']['psnr']:.4f} | "
        f"ssim={figure_result['metrics_vs_reference']['ssim']:.4f}",
    )
    print(
        "  vs all-bilateral rSoS | "
        f"mse={figure_result['metrics_vs_bilateral']['mse']:.6f} | "
        f"psnr={figure_result['metrics_vs_bilateral']['psnr']:.4f} | "
        f"ssim={figure_result['metrics_vs_bilateral']['ssim']:.4f}",
    )
    for name, path in figure_result["paths"].items():
        print(f"  Saved {name}: {path}")

    _print_heading("MRI workflow completed")
    _print_mapping(
        "Export summary",
        {
            "exercise 2.1 figures": ex21_fig_dir,
            "exercise 2.2 part 1 figures": ex22_part1_fig_dir,
            "exercise 2.2 part 2 figures": ex22_part2_fig_dir,
            "exercise 2.2 part 2 metrics": part2_result["metrics_path"],
            "exercise 2.2 part 3 figures": ex22_part3_fig_dir,
            "exercise 2.2 part 3 metrics": part3_result["metrics_path"],
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the MRI workflow."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
