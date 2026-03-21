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
    scale_image_for_practical,
)


def _resolve_out_path(out_path: str | Path) -> Path:
    """Resolve a project-relative output path to an absolute path."""
    p = Path(out_path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[3]
    return project_root / p


def available_fbp_filters() -> list[tuple[str, str]]:
    """Return standard FBP filters commonly cited in CT literature.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of skimage filter names and display labels.
    """
    return [
        ("ramp", "Ram-Lak"),
        ("shepp-logan", "Shepp-Logan"),
        ("cosine", "Cosine"),
        ("hamming", "Hamming"),
        ("hann", "Hann"),
    ]


@dataclass
class FilterReconstruction:
    """Reconstruction and metrics for one FBP filter choice."""

    filter_name: str
    display_name: str
    reconstruction: np.ndarray
    metrics: dict[str, float]
    out_path: Path


@dataclass
class FilterComparisonResult:
    """Outputs of the Exercise 1.3(a)/(b) filter comparison workflow."""

    n_angles: int
    noise_level: str
    output_dir: Path
    reconstructions: list[FilterReconstruction]


@dataclass
class IterativeComparisonResult:
    """Outputs of the Exercise 1.3(c) iterative comparison workflow."""

    n_angles: int
    noise_level: str
    n_subsets: int
    sirt_reconstruction: np.ndarray
    os_sart_reconstruction: np.ndarray
    sirt_metrics: dict[str, float]
    os_sart_metrics: dict[str, float]
    figure_path: Path


def _compute_metrics(reference: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    """Compute scalar comparison metrics between a reference and reconstruction."""
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
    """Locate the sinogram corresponding to a selected low-dose case."""
    level = float(i0_level)
    for sinogram_set in sinogram_sets:
        if int(sinogram_set.n_angles) != int(n_angles):
            continue
        for key, value in sinogram_set.gaussian_poisson.items():
            if float(key) == level:
                return np.asarray(value, dtype=np.float32), np.asarray(sinogram_set.theta, dtype=np.float32)
    raise KeyError(f"Low-dose case not found for angles={n_angles}, I0={i0_level}")


def summarize_filter_options() -> list[str]:
    """Create notebook-friendly labels for the supported FBP filters.

    Returns
    -------
    list[str]
        Human-readable filter descriptions.
    """
    return [f"{display} (`{name}`)" for name, display in available_fbp_filters()]


def run_fbp_filter_comparison(
    reference_image: np.ndarray,
    sinogram_sets: list[SinogramSet],
    n_angles: int = 360,
    i0_level: float = 1e2,
    attenuation_scale: float = 1000.0,
    filter_names: Iterable[str] = ("ramp", "shepp-logan", "hann"),
    out_dir: str | Path = "results/ct/figures/exercise_1_3",
    show: bool = False,
) -> FilterComparisonResult:
    """Run the Exercise 1.3(a)/(b) filter comparison workflow.

    Parameters
    ----------
    reference_image:
        Ground-truth CT image in normalized intensity space. It is converted to
        practical attenuation units internally before comparison.
    sinogram_sets:
        Simulated sinogram groups from Exercise 1.1.
    n_angles:
        Selected view count for the comparison.
    i0_level:
        Selected low-dose I0 level.
    attenuation_scale:
        Scaling factor used to convert the reference image to practical attenuation units.
    filter_names:
        Filters to compare against the standard Ram-Lak filter.
    out_dir:
        Output directory for saved figures.
    show:
        Whether to display the generated figures.

    Returns
    -------
    FilterComparisonResult
        Saved reconstructions, metrics, and output metadata.
    """
    filter_lookup = dict(available_fbp_filters())
    selected_filters = list(filter_names)
    sinogram, theta = _find_low_dose_case(sinogram_sets=sinogram_sets, n_angles=n_angles, i0_level=i0_level)
    ref = scale_image_for_practical(reference_image, attenuation_scale=attenuation_scale)

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
    """Reconstruct an image with a simple OS-SART style ordered-subsets scheme.

    Parameters
    ----------
    sinogram:
        Input sinogram to reconstruct.
    theta:
        Projection angles associated with ``sinogram``.
    output_size:
        Optional reconstructed image size.
    n_iters:
        Number of outer iterations.
    n_subsets:
        Number of ordered subsets used per outer iteration.
    step_size:
        Update step size.
    positivity:
        Whether to clip negative voxels after each subset update.
    show_progress:
        Whether to show a progress bar.
    progress_desc:
        Optional label for the progress bar.

    Returns
    -------
    np.ndarray
        Float32 reconstructed image.
    """
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
    attenuation_scale: float = 1000.0,
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
    """Run the Exercise 1.3(c) iterative reconstruction comparison.

    Parameters
    ----------
    reference_image:
        Ground-truth CT image in normalized intensity space. It is converted to
        practical attenuation units internally before comparison.
    sinogram_sets:
        Simulated sinogram groups from Exercise 1.1.
    n_angles:
        Selected view count for the comparison.
    i0_level:
        Selected low-dose I0 level.
    attenuation_scale:
        Scaling factor used to convert the reference image to practical attenuation units.
    sirt_iters:
        Iteration count for the SIRT-like baseline.
    sirt_step_size:
        Step size for the SIRT-like baseline.
    os_sart_iters:
        Iteration count for OS-SART.
    os_sart_step_size:
        Step size for OS-SART.
    n_subsets:
        Number of subsets used by OS-SART.
    positivity:
        Whether to enforce positivity in the iterative methods.
    out_dir:
        Output directory for the comparison figure.
    show:
        Whether to display the comparison figure.
    show_progress:
        Whether to show progress bars.

    Returns
    -------
    IterativeComparisonResult
        Reconstructions, metrics, and saved figure path.
    """
    sinogram, theta = _find_low_dose_case(sinogram_sets=sinogram_sets, n_angles=n_angles, i0_level=i0_level)
    ref = scale_image_for_practical(reference_image, attenuation_scale=attenuation_scale)

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


def summarize_filter_comparison(result: FilterComparisonResult) -> list[str]:
    """Create notebook-friendly summary lines for filter comparisons.

    Parameters
    ----------
    result:
        Filter comparison output to summarize.

    Returns
    -------
    list[str]
        Human-readable lines summarizing saved figures and metrics.
    """
    lines: list[str] = []
    for recon in result.reconstructions:
        lines.append(
            f"{recon.display_name}: {recon.out_path} | "
            f"mse={recon.metrics['mse']:.6f} | psnr={recon.metrics['psnr']:.4f} | ssim={recon.metrics['ssim']:.4f}"
        )
    return lines


def summarize_iterative_comparison(result: IterativeComparisonResult) -> list[str]:
    """Create notebook-friendly summary lines for the iterative comparison.

    Parameters
    ----------
    result:
        Iterative comparison output to summarize.

    Returns
    -------
    list[str]
        Human-readable lines describing the saved figure and metrics.
    """
    return [
        f"figure: {result.figure_path}",
        f"SIRT-like GD metrics: {result.sirt_metrics}",
        f"OS-SART metrics: {result.os_sart_metrics}",
    ]


