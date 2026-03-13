"""Exercise 1.3 utilities: filter and iterative reconstruction comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import iradon, radon

from .exercise_1_1 import (
    SinogramSet,
    _image_display_limits,
    _progress,
    reconstruct_fbp,
    reconstruct_gradient_descent,
)


def _resolve_out_path(out_path: str | Path) -> Path:
    p = Path(out_path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[3]
    return project_root / p


def available_fbp_filters() -> list[tuple[str, str]]:
    """Return standard FBP filters commonly cited in CT literature."""
    return [
        ("ramp", "Ram-Lak"),
        ("shepp-logan", "Shepp-Logan"),
        ("cosine", "Cosine"),
        ("hamming", "Hamming"),
        ("hann", "Hann"),
    ]


@dataclass
class FilterReconstruction:
    filter_name: str
    display_name: str
    reconstruction: np.ndarray
    metrics: dict[str, float]
    out_path: Path


@dataclass
class FilterComparisonResult:
    n_angles: int
    noise_level: str
    output_dir: Path
    reconstructions: list[FilterReconstruction]


@dataclass
class IterativeComparisonResult:
    n_angles: int
    noise_level: str
    n_subsets: int
    sirt_reconstruction: np.ndarray
    os_sart_reconstruction: np.ndarray
    sirt_metrics: dict[str, float]
    os_sart_metrics: dict[str, float]
    figure_path: Path


def _compute_metrics(reference: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float32)
    rec = np.asarray(recon, dtype=np.float32)
    mse = float(np.mean((ref - rec) ** 2))
    mae = float(np.mean(np.abs(ref - rec)))
    data_range = float(ref.max() - ref.min())
    if data_range <= 0:
        data_range = 1.0
    psnr = float(20.0 * np.log10(data_range) - 10.0 * np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(ref, rec, data_range=data_range))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def _find_low_dose_case(
    sinogram_sets: list[SinogramSet],
    n_angles: int,
    i0_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    level = float(i0_level)
    for sinogram_set in sinogram_sets:
        if int(sinogram_set.n_angles) != int(n_angles):
            continue
        for key, value in sinogram_set.gaussian_poisson.items():
            if float(key) == level:
                return np.asarray(value, dtype=np.float32), np.asarray(sinogram_set.theta, dtype=np.float32)
    raise KeyError(f"Low-dose case not found for angles={n_angles}, I0={i0_level}")


def summarize_filter_options() -> list[str]:
    """Return notebook-friendly descriptions of standard FBP filters."""
    return [f"{display} (`{name}`)" for name, display in available_fbp_filters()]


def run_fbp_filter_comparison(
    reference_image: np.ndarray,
    sinogram_sets: list[SinogramSet],
    n_angles: int = 360,
    i0_level: float = 1e2,
    filter_names: Iterable[str] = ("ramp", "shepp-logan", "hann"),
    out_dir: str | Path = "results/ct/figures/exercise_1_3",
    show: bool = False,
) -> FilterComparisonResult:
    """Reconstruct one low-dose case with multiple FBP filters and save each image."""
    filter_lookup = dict(available_fbp_filters())
    selected_filters = list(filter_names)
    sinogram, theta = _find_low_dose_case(sinogram_sets=sinogram_sets, n_angles=n_angles, i0_level=i0_level)
    ref = np.asarray(reference_image, dtype=np.float32)

    output_dir = _resolve_out_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_vmin, image_vmax = _image_display_limits(ref)

    reconstructions: list[FilterReconstruction] = []
    for filter_name in selected_filters:
        if filter_name not in filter_lookup:
            raise ValueError(f"Unknown filter '{filter_name}'. Available filters: {sorted(filter_lookup)}")
        reconstruction = reconstruct_fbp(
            sinogram=sinogram,
            theta=theta,
            filter_name=filter_name,
            output_size=ref.shape[0],
        )
        metrics = _compute_metrics(ref, reconstruction)
        out_path = output_dir / f"exercise_1_3_filter_{filter_name}_angles_{int(n_angles)}_I0_{float(i0_level):.0e}.png"

        fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6), constrained_layout=True)
        ax.imshow(reconstruction, cmap="gray", vmin=image_vmin, vmax=image_vmax)
        ax.set_title(
            f"{filter_lookup[filter_name]} | angles={int(n_angles)} | I0={float(i0_level):.0e}",
            fontsize=10,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        reconstructions.append(
            FilterReconstruction(
                filter_name=filter_name,
                display_name=filter_lookup[filter_name],
                reconstruction=reconstruction,
                metrics=metrics,
                out_path=out_path,
            )
        )

    return FilterComparisonResult(
        n_angles=int(n_angles),
        noise_level=f"I0={float(i0_level):.0e}",
        output_dir=output_dir,
        reconstructions=reconstructions,
    )


def reconstruct_os_sart(
    sinogram: np.ndarray,
    theta: np.ndarray,
    output_size: int | None = None,
    n_iters: int = 50,
    n_subsets: int = 6,
    step_size: float = 0.001,
    positivity: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Simple ordered-subsets reconstruction: one update per subset inside each epoch."""
    sino = np.asarray(sinogram, dtype=np.float32)
    if output_size is None:
        output_size = int(sino.shape[0])
    n = int(output_size)
    x = np.zeros((n, n), dtype=np.float32)

    subset_count = max(1, min(int(n_subsets), int(len(theta))))
    iterator = _progress(range(int(n_iters)), enabled=show_progress, desc=progress_desc or "OS-SART")
    for _ in iterator:
        for subset_idx in range(subset_count):
            cols = np.arange(subset_idx, int(len(theta)), subset_count)
            if len(cols) == 0:
                continue
            theta_subset = np.asarray(theta[cols], dtype=np.float32)
            sino_subset = np.asarray(sino[:, cols], dtype=np.float32)
            ax = radon(x, theta=theta_subset, circle=True).astype(np.float32)
            residual = ax - sino_subset
            grad = iradon(
                residual,
                theta=theta_subset,
                circle=True,
                filter_name=None,
                output_size=n,
            ).astype(np.float32)
            x = x - step_size * grad
            if positivity:
                x = np.maximum(x, 0.0)
    return x


def run_os_sart_comparison(
    reference_image: np.ndarray,
    sinogram_sets: list[SinogramSet],
    n_angles: int = 360,
    i0_level: float = 1e2,
    sirt_iters: int = 50,
    sirt_step_size: float = 0.001,
    os_sart_iters: int = 50,
    os_sart_step_size: float = 0.001,
    n_subsets: int = 6,
    positivity: bool = False,
    out_dir: str | Path = "results/ct/figures/exercise_1_3",
    show: bool = False,
    show_progress: bool = False,
) -> IterativeComparisonResult:
    """Compare SIRT-like GD and simple OS-SART on one selected low-dose case."""
    sinogram, theta = _find_low_dose_case(sinogram_sets=sinogram_sets, n_angles=n_angles, i0_level=i0_level)
    ref = np.asarray(reference_image, dtype=np.float32)

    sirt = reconstruct_gradient_descent(
        sinogram=sinogram,
        theta=theta,
        output_size=ref.shape[0],
        n_iters=sirt_iters,
        step_size=sirt_step_size,
        positivity=positivity,
        show_progress=show_progress,
        progress_desc=f"SIRT {int(n_angles)} views | I0={float(i0_level):.0e}",
    )
    os_sart = reconstruct_os_sart(
        sinogram=sinogram,
        theta=theta,
        output_size=ref.shape[0],
        n_iters=os_sart_iters,
        n_subsets=n_subsets,
        step_size=os_sart_step_size,
        positivity=positivity,
        show_progress=show_progress,
        progress_desc=f"OS-SART {int(n_angles)} views | I0={float(i0_level):.0e}",
    )

    sirt_metrics = _compute_metrics(ref, sirt)
    os_sart_metrics = _compute_metrics(ref, os_sart)

    output_dir = _resolve_out_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"exercise_1_3_iterative_compare_angles_{int(n_angles)}_I0_{float(i0_level):.0e}.png"
    image_vmin, image_vmax = _image_display_limits(ref)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
    items = [
        ("Reference", ref),
        ("SIRT-like GD", sirt),
        (f"OS-SART ({int(n_subsets)} subsets)", os_sart),
    ]
    for ax, (title, arr) in zip(axes, items):
        ax.imshow(arr, cmap="gray", vmin=image_vmin, vmax=image_vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return IterativeComparisonResult(
        n_angles=int(n_angles),
        noise_level=f"I0={float(i0_level):.0e}",
        n_subsets=int(n_subsets),
        sirt_reconstruction=sirt,
        os_sart_reconstruction=os_sart,
        sirt_metrics=sirt_metrics,
        os_sart_metrics=os_sart_metrics,
        figure_path=figure_path,
    )


