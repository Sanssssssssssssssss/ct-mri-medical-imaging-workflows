"""Exercise 1.1 utilities: sinogram simulation, reconstruction, and reporting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity
from skimage.transform import iradon, radon

from ..common import resolve_project_path as _resolve_out_path
from ..common import write_csv_rows

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _progress(iterable, enabled: bool = False, desc: str | None = None):
    """Wrap an iterable with a progress bar when tqdm is available."""
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def _format_i0(level: str) -> str:
    """Normalize an I0 label for figure titles."""
    return level if level.startswith("I0=") else level


def prepare_reference_ct_image(raw_image: np.ndarray) -> np.ndarray:
    """Convert a raw CT image array to a normalized float image.

    Parameters
    ----------
    raw_image:
        Raw 2D or RGB-like CT image array.
    Returns
    -------
    np.ndarray
        Float32 image clipped to ``[0, 1]``.
    """
    image = np.asarray(raw_image)
    if image.ndim == 3:
        image = image[..., 0]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _image_display_limits(image: np.ndarray) -> tuple[float, float]:
    """Return a stable grayscale display range for an image."""
    vmax = float(np.max(np.asarray(image, dtype=np.float32)))
    return 0.0, max(vmax, 1e-6)


def _mask_to_reconstruction_circle(image: np.ndarray) -> np.ndarray:
    """Set pixels outside the reconstruction circle to zero."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img.shape}")
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0 - 1.0
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    return np.where(circle_mask, img, 0.0).astype(np.float32)


def scale_image_for_practical(image: np.ndarray, attenuation_scale: float = 1000.0) -> np.ndarray:
    """Convert a normalized CT image to the attenuation scale used in the practical.

    Parameters
    ----------
    image:
        Normalized 2D CT image in ``[0, 1]``.
    attenuation_scale:
        Scaling factor used to move the image into attenuation space.

    Returns
    -------
    np.ndarray
        Float32 image scaled to ``image / attenuation_scale``.
    """
    return np.asarray(image, dtype=np.float32) / float(attenuation_scale)


@dataclass
class ReferenceImageResult:
    """Container for a prepared CT reference image and its metadata."""

    image: np.ndarray
    image_path: Path
    source_shape: tuple[int, ...]
    source_dtype: str


def make_theta(n_angles: int) -> np.ndarray:
    """Create evenly spaced projection angles in ``[0, 360)`` degrees.

    Parameters
    ----------
    n_angles:
        Number of projection angles to generate.

    Returns
    -------
    np.ndarray
        Float32 array of projection angles.
    """
    return np.linspace(0.0, 360.0, int(n_angles), endpoint=False, dtype=np.float32)


def forward_project(image: np.ndarray, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute a clean sinogram for an attenuation-space reference image.

    Parameters
    ----------
    image:
        Input 2D CT image in attenuation space.
    n_angles:
        Number of projection views.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Clean sinogram and the associated projection angles.
    """
    theta = make_theta(n_angles)
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image for forward projection, got shape={img.shape}")
    img = _mask_to_reconstruction_circle(img)
    sinogram = radon(img, theta=theta, circle=True)
    return sinogram.astype(np.float32), theta


def add_gaussian_noise(
    sinogram: np.ndarray,
    mu: float = 0.0,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise directly in sinogram space.

    Parameters
    ----------
    sinogram:
        Clean sinogram array.
    mu:
        Mean of the Gaussian noise.
    sigma:
        Standard deviation of the Gaussian noise.
    rng:
        Optional random number generator.

    Returns
    -------
    np.ndarray
        Noisy sinogram.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=mu, scale=sigma, size=sinogram.shape)
    return (sinogram + noise).astype(np.float32)


def add_poisson_noise(
    sinogram: np.ndarray,
    i0: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Poisson counting noise using a Beer-Lambert transmission model.

    Parameters
    ----------
    sinogram:
        Clean line-integral sinogram.
    i0:
        Incident photon count.
    rng:
        Optional random number generator.
    Returns
    -------
    np.ndarray
        Noisy sinogram reconstructed from simulated counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    sino = np.asarray(sinogram, dtype=np.float32)
    transmission = np.exp(-sino)
    transmission = np.clip(transmission, 0.0, 1.0)

    expected_counts = float(i0) * transmission
    measured_counts = rng.poisson(expected_counts).astype(np.float32)

    ratio = measured_counts / float(i0)
    ratio = np.maximum(ratio, np.finfo(np.float32).eps)
    noisy_sino = -np.log(ratio)
    return noisy_sino.astype(np.float32)


def add_gaussian_poisson_noise_practical(
    sinogram: np.ndarray,
    i0: float,
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the practical notebook noise model in the transmission domain.

    Parameters
    ----------
    sinogram:
        Clean line-integral sinogram already expressed in attenuation units.
    i0:
        Incident photon count.
    gaussian_mu:
        Mean of the additive Gaussian count noise.
    gaussian_sigma:
        Standard deviation of the additive Gaussian count noise.
    rng:
        Optional random number generator.
    Returns
    -------
    np.ndarray
        Noisy sinogram in the same attenuation scale as the input sinogram.
    """
    if rng is None:
        rng = np.random.default_rng()

    sino = np.asarray(sinogram, dtype=np.float32)
    transmission = np.exp(-sino)
    expected_counts = float(i0) * transmission
    poisson_counts = rng.poisson(expected_counts).astype(np.float32)

    gaussian_counts = rng.normal(loc=gaussian_mu, scale=gaussian_sigma, size=sino.shape).astype(np.float32)
    noisy_counts = poisson_counts + gaussian_counts
    noisy_counts = np.maximum(noisy_counts, np.finfo(np.float32).eps)

    noisy_sino = -np.log(noisy_counts / float(i0))
    return noisy_sino.astype(np.float32)


@dataclass
class SinogramSet:
    """Sinograms generated for one projection-view configuration."""

    n_angles: int
    theta: np.ndarray
    clean: np.ndarray
    gaussian_poisson: dict[float, np.ndarray]


@dataclass
class ReconstructionCase:
    """Reconstruction outputs and metrics for one noise/view configuration."""

    n_angles: int
    noise_kind: str
    noise_level: str
    sinogram: np.ndarray
    recon_fbp: np.ndarray
    recon_gd: np.ndarray
    metrics_fbp: dict[str, float]
    metrics_gd: dict[str, float]


def _pick_case(cases: list[ReconstructionCase], n_angles: int, noise_kind: str, noise_level: str) -> ReconstructionCase:
    """Select a reconstruction case by its identifying metadata."""
    for c in cases:
        if c.n_angles == n_angles and c.noise_kind == noise_kind and c.noise_level == noise_level:
            return c
    raise KeyError(f"Case not found: angles={n_angles}, kind={noise_kind}, level={noise_level}")


@dataclass
class SinogramSimulationResult:
    """Outputs of the Exercise 1.1(b) sinogram simulation workflow."""

    sinogram_sets: list[SinogramSet]
    summary_lines: list[str]
    panel_path: Path | None


@dataclass
class ReconstructionExperimentResult:
    """Outputs of the Exercise 1.1(c) reconstruction workflow."""

    cases: list[ReconstructionCase]
    metric_rows: list[dict[str, float | str | int]]
    figure_paths: list[Path]
    metrics_path: Path


# -------- exercise 1.1(b): noisy sinogram simulation --------

def simulate_noisy_sinograms(
    image: np.ndarray,
    angles_list: Iterable[int] = (360, 90, 20),
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 0.05,
    poisson_i0_levels: Iterable[float] = (1e5, 1e3, 1e2),
    attenuation_scale: float = 1000.0,
    seed: int = 42,
) -> list[SinogramSet]:
    """Generate noisy sinograms for the Exercise 1.1 dose experiment.

    Parameters
    ----------
    image:
        Reference CT image in normalized intensity space.
    angles_list:
        Projection counts to simulate.
    gaussian_mu:
        Mean of the additive Gaussian count noise.
    gaussian_sigma:
        Standard deviation of the additive Gaussian count noise.
    poisson_i0_levels:
        Incident photon counts for the Poisson component.
    attenuation_scale:
        Scaling factor used to convert the normalized image to the practical
        attenuation range before projection and noise simulation.
    seed:
        Random seed used to keep the experiment reproducible.

    Returns
    -------
    list[SinogramSet]
        Simulated sinograms grouped by view count.
    """
    rng = np.random.default_rng(seed)
    out: list[SinogramSet] = []
    attenuation_image = scale_image_for_practical(image, attenuation_scale=attenuation_scale)

    for n_angles in angles_list:
        clean, theta = forward_project(image=attenuation_image, n_angles=int(n_angles))
        gaussian_poisson: dict[float, np.ndarray] = {}
        for i0 in poisson_i0_levels:
            gaussian_poisson[float(i0)] = add_gaussian_poisson_noise_practical(
                clean,
                i0=float(i0),
                gaussian_mu=gaussian_mu,
                gaussian_sigma=gaussian_sigma,
                rng=rng,
            )
        out.append(
            SinogramSet(
                n_angles=int(n_angles),
                theta=theta,
                clean=clean,
                gaussian_poisson=gaussian_poisson,
            )
        )
    return out


def save_sinogram_panel(
    sinogram_sets: list[SinogramSet],
    gaussian_mu: float,
    gaussian_sigma: float,
    poisson_i0_levels: Iterable[float],
    out_path: str | Path,
    dpi: int = 150,
    show: bool = True,
) -> Path:
    """Save a multi-panel figure for the Exercise 1.1(b) sinogram comparison.

    Parameters
    ----------
    sinogram_sets:
        Simulated sinogram groups to display.
    gaussian_mu:
        Mean of the Gaussian noise used in the experiment.
    gaussian_sigma:
        Standard deviation of the Gaussian noise used in the experiment.
    poisson_i0_levels:
        I0 values shown across the columns.
    out_path:
        Output path for the saved figure.
    dpi:
        Figure resolution.
    show:
        Whether to display the figure interactively.

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    poisson_levels = [float(v) for v in poisson_i0_levels]
    fig, axes = plt.subplots(len(sinogram_sets), 4, figsize=(15, 10), constrained_layout=True)
    if len(sinogram_sets) == 1:
        axes = np.array([axes])

    for r, s in enumerate(sinogram_sets):
        panel_data = [
            ("Clean", s.clean),
            (
                f"G+P (I0={poisson_levels[0]:.0e}, mu={gaussian_mu}, sigma={gaussian_sigma})",
                s.gaussian_poisson[poisson_levels[0]],
            ),
            (
                f"G+P (I0={poisson_levels[1]:.0e}, mu={gaussian_mu}, sigma={gaussian_sigma})",
                s.gaussian_poisson[poisson_levels[1]],
            ),
            (
                f"G+P (I0={poisson_levels[2]:.0e}, mu={gaussian_mu}, sigma={gaussian_sigma})",
                s.gaussian_poisson[poisson_levels[2]],
            ),
        ]

        for c, (title, arr) in enumerate(panel_data):
            ax = axes[r, c]
            im = ax.imshow(arr, cmap="gray", aspect="auto")
            ax.set_title(f"{title} | angles={s.n_angles}", fontsize=9)
            ax.set_xlabel("Projection angle index")
            ax.set_ylabel("Detector bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = _resolve_out_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# -------- exercise 1.1(c): reconstruction + metrics + analysis figures --------

def reconstruct_fbp(
    sinogram: np.ndarray,
    theta: np.ndarray,
    filter_name: str = "ramp",
    output_size: int | None = None,
) -> np.ndarray:
    """Reconstruct an image with filtered backprojection.

    Parameters
    ----------
    sinogram:
        Input sinogram to reconstruct.
    theta:
        Projection angles associated with ``sinogram``.
    filter_name:
        Filter passed to ``skimage.transform.iradon``.
    output_size:
        Optional output image size.

    Returns
    -------
    np.ndarray
        Float32 reconstructed image.
    """
    if output_size is None:
        output_size = int(sinogram.shape[0])
    rec = iradon(
        sinogram,
        theta=theta,
        circle=True,
        filter_name=filter_name,
        output_size=output_size,
    )
    return rec.astype(np.float32)


def reconstruct_gradient_descent(
    sinogram: np.ndarray,
    theta: np.ndarray,
    output_size: int | None = None,
    n_iters: int = 60,
    step_size: float = 0.02,
    init_mode: str = "fbp",
    init_image: np.ndarray | None = None,
    normalize_gradient: bool = True,
    clip_range: tuple[float, float] | None = None,
    mask_each_iter: bool = True,
    positivity: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Reconstruct an image with practical-style full-batch gradient descent.

    Parameters
    ----------
    sinogram:
        Input sinogram to reconstruct.
    theta:
        Projection angles associated with ``sinogram``.
    output_size:
        Optional reconstructed image size.
    n_iters:
        Number of gradient-descent iterations.
    step_size:
        Gradient-descent step size.
    init_mode:
        Initialization strategy. Supported values are ``"zeros"`` and ``"fbp"``.
    init_image:
        Optional explicit initialization image. When provided, it overrides
        ``init_mode``.
    normalize_gradient:
        Whether to divide the step size by the gradient RMS at each iteration.
    clip_range:
        Optional ``(min, max)`` range applied after each update.
    mask_each_iter:
        Whether to zero pixels outside the reconstruction circle after each update.
    positivity:
        Whether to clip negative voxels after each update.
    show_progress:
        Whether to show a tqdm progress bar.
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

    if init_image is not None:
        x = np.asarray(init_image, dtype=np.float32).copy()
    elif init_mode == "zeros":
        x = np.zeros((n, n), dtype=np.float32)
    elif init_mode == "fbp":
        x = reconstruct_fbp(sino, theta, filter_name="ramp", output_size=n)
    else:
        raise ValueError(f"Unsupported init_mode='{init_mode}'. Use 'zeros' or 'fbp'.")

    if x.shape != (n, n):
        raise ValueError(f"Initial image shape {x.shape} does not match output size {(n, n)}")
    if positivity:
        x = np.maximum(x, 0.0)
    if clip_range is not None:
        lo, hi = clip_range
        x = np.clip(x, float(lo), float(hi))
    if mask_each_iter:
        x = _mask_to_reconstruction_circle(x)

    iterator = _progress(range(int(n_iters)), enabled=show_progress, desc=progress_desc or "GD")
    for _ in iterator:
        ax = radon(x, theta=theta, circle=True).astype(np.float32)
        residual = ax - sino
        grad = iradon(residual, theta=theta, circle=True, filter_name=None, output_size=n).astype(np.float32)
        alpha = float(step_size)
        if normalize_gradient:
            grad_rms = float(np.sqrt(np.mean(grad**2)) + 1e-8)
            alpha = alpha / grad_rms
        x = x - alpha * grad
        if positivity:
            x = np.maximum(x, 0.0)
        if clip_range is not None:
            lo, hi = clip_range
            x = np.clip(x, float(lo), float(hi))
        if mask_each_iter:
            x = _mask_to_reconstruction_circle(x)
    return x


def _compute_metrics(
    reference: np.ndarray,
    recon: np.ndarray,
    metric_mode: str = "practical",
) -> dict[str, float]:
    """Compute scalar comparison metrics between a reference and reconstruction.

    Parameters
    ----------
    reference:
        Ground-truth image.
    recon:
        Reconstructed image to evaluate.
    metric_mode:
        Metric convention. ``"practical"`` compares attenuation-space images
        directly. ``"reporting"`` matches the reporting style used in the
        comparison codebase by masking both images to the reconstruction circle,
        normalizing them to the reference range, clipping the reconstruction to
        ``[0, 1]``, and using ``data_range=1.0`` for PSNR and SSIM.
    """
    ref = np.asarray(reference, dtype=np.float32)
    rec = np.asarray(recon, dtype=np.float32)
    if metric_mode == "reporting":
        ref = _mask_to_reconstruction_circle(ref)
        rec = _mask_to_reconstruction_circle(rec)
        ref_min = float(ref.min())
        ref_max = float(ref.max())
        denom = max(ref_max - ref_min, 1e-6)
        ref = np.clip((ref - ref_min) / denom, 0.0, 1.0)
        rec = np.clip((rec - ref_min) / denom, 0.0, 1.0)
        data_range = 1.0
    elif metric_mode == "practical":
        data_range = float(ref.max() - ref.min())
        if data_range <= 0:
            data_range = 1.0
    else:
        raise ValueError(f"Unsupported metric_mode='{metric_mode}'. Use 'practical' or 'reporting'.")

    mse = float(np.mean((ref - rec) ** 2))
    psnr = float(20.0 * np.log10(data_range) - 10.0 * np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(ref, rec, data_range=data_range))
    mae = float(np.mean(np.abs(ref - rec)))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def run_reconstruction_comparison(
    reference_image: np.ndarray,
    sinogram_sets: list[SinogramSet],
    poisson_i0_levels: Iterable[float],
    fbp_filter: str = "ramp",
    gd_iters: int = 60,
    gd_step_size: float = 0.02,
    gd_init_mode: str = "fbp",
    gd_normalize_gradient: bool = True,
    gd_clip_to_reference_range: bool = True,
    gd_mask_each_iter: bool = True,
    gd_positivity: bool = False,
    metric_mode: str = "reporting",
    show_progress: bool = False,
) -> list[ReconstructionCase]:
    """Run FBP and GD across all simulated Exercise 1.1 sinograms.

    Parameters
    ----------
    reference_image:
        Ground-truth CT image.
    sinogram_sets:
        Simulated sinogram groups to reconstruct.
    poisson_i0_levels:
        I0 levels to reconstruct for each view count.
    fbp_filter:
        Filter passed to the FBP reconstruction.
    gd_iters:
        Iteration count for gradient descent.
    gd_step_size:
        Step size for gradient descent.
    gd_init_mode:
        Initialization strategy for gradient descent.
    gd_normalize_gradient:
        Whether to normalize each gradient-descent update by the gradient RMS.
    gd_clip_to_reference_range:
        Whether to clip the reconstruction to the reference image range after each update.
    gd_mask_each_iter:
        Whether to zero pixels outside the reconstruction circle after each update.
    gd_positivity:
        Whether to enforce positivity in gradient descent.
    metric_mode:
        Metric convention used to score reconstructions.
    show_progress:
        Whether to show per-case progress bars.

    Returns
    -------
    list[ReconstructionCase]
        Reconstruction outputs and metrics for every case.
    """
    i0_levels = [float(v) for v in poisson_i0_levels]
    ref = np.asarray(reference_image, dtype=np.float32)
    clip_range = None
    if gd_clip_to_reference_range:
        clip_range = (float(ref.min()), float(ref.max()))
    cases: list[ReconstructionCase] = []
    for s in sinogram_sets:
        for i0 in i0_levels:
            sino = s.gaussian_poisson[i0]
            rec_fbp = reconstruct_fbp(sino, s.theta, filter_name=fbp_filter, output_size=ref.shape[0])
            rec_gd = reconstruct_gradient_descent(
                sino,
                s.theta,
                output_size=ref.shape[0],
                n_iters=gd_iters,
                step_size=gd_step_size,
                init_mode=gd_init_mode,
                normalize_gradient=gd_normalize_gradient,
                clip_range=clip_range,
                mask_each_iter=gd_mask_each_iter,
                positivity=gd_positivity,
                show_progress=show_progress,
                progress_desc=f"GD {s.n_angles} views | I0={i0:.0e}",
            )
            cases.append(
                ReconstructionCase(
                    n_angles=s.n_angles,
                    noise_kind="gaussian+poisson",
                    noise_level=f"I0={i0:.0e}",
                    sinogram=sino,
                    recon_fbp=rec_fbp,
                    recon_gd=rec_gd,
                    metrics_fbp=_compute_metrics(ref, rec_fbp, metric_mode=metric_mode),
                    metrics_gd=_compute_metrics(ref, rec_gd, metric_mode=metric_mode),
                )
            )
    return cases


def summarize_metrics(cases: list[ReconstructionCase]) -> list[dict[str, float | str | int]]:
    """Convert reconstruction cases into flat metric rows.

    Parameters
    ----------
    cases:
        Reconstruction cases to summarize.

    Returns
    -------
    list[dict[str, float | str | int]]
        Flat rows ready for CSV export or notebook display.
    """
    rows: list[dict[str, float | str | int]] = []
    for c in cases:
        rows.append(
            {
                "n_angles": c.n_angles,
                "noise_kind": c.noise_kind,
                "noise_level": c.noise_level,
                "algorithm": "FBP",
                **c.metrics_fbp,
            }
        )
        rows.append(
            {
                "n_angles": c.n_angles,
                "noise_kind": c.noise_kind,
                "noise_level": c.noise_level,
                "algorithm": "GD",
                **c.metrics_gd,
            }
        )
    return rows


def save_metrics_csv(rows: list[dict[str, float | str | int]], out_path: str | Path) -> Path:
    """Save metric rows to a CSV file.

    Parameters
    ----------
    rows:
        Metric rows to save.
    out_path:
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the saved CSV file.
    """
    ordered_rows = [
        {
            "n_angles": row["n_angles"],
            "noise_kind": row["noise_kind"],
            "noise_level": row["noise_level"],
            "algorithm": row["algorithm"],
            "mse": row["mse"],
            "mae": row["mae"],
            "psnr": row["psnr"],
            "ssim": row["ssim"],
        }
        for row in rows
    ]
    return write_csv_rows(ordered_rows, out_path)


def load_metrics_csv(in_path: str | Path) -> list[dict[str, str | float | int]]:
    """Load previously saved metric rows from a CSV file.

    Parameters
    ----------
    in_path:
        Path to a metrics CSV produced by this module.

    Returns
    -------
    list[dict[str, str | float | int]]
        Parsed metric rows with numeric fields converted when possible.
    """
    path = _resolve_out_path(in_path)
    rows: list[dict[str, str | float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, str | float | int] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = ""
                elif key == "n_angles":
                    parsed[key] = int(value)
                elif key in {"mse", "mae", "psnr", "ssim"}:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def save_reconstruction_panels(
    reference_image: np.ndarray,
    cases: list[ReconstructionCase],
    out_dir: str | Path = "results/ct/figures",
    show: bool = False,
    hardest_case: tuple[int, str, str] | None = None,
    experiment_title: str = "Exercise 1.1(c)",
    file_prefix: str = "exercise_1_1c",
) -> list[Path]:
    """Save reconstruction comparison figures for CT experiments.

    Parameters
    ----------
    reference_image:
        Ground-truth CT image.
    cases:
        Reconstruction cases to visualize.
    out_dir:
        Output directory for the saved figures.
    show:
        Whether to display figures interactively.
    hardest_case:
        Optional explicit case for the five-panel hard-case figure.
    experiment_title:
        Title prefix used in figure suptitles.
    file_prefix:
        Filename prefix used for saved figures.

    Returns
    -------
    list[Path]
        Paths of the saved figures.
    """
    out_root = _resolve_out_path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ref = np.asarray(reference_image, dtype=np.float32)
    image_vmin, image_vmax = _image_display_limits(ref)
    saved: list[Path] = []

    angles_list = sorted({c.n_angles for c in cases}, reverse=True)
    poisson_levels = sorted({c.noise_level for c in cases if c.noise_kind == "gaussian+poisson"})

    if poisson_levels:
        for n_angles in angles_list:
            n_cols = len(poisson_levels)
            fig_recon, axes_recon = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 5.6), constrained_layout=True)
            fig_err, axes_err = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 5.6), constrained_layout=True)
            if n_cols == 1:
                axes_recon = axes_recon.reshape(2, 1)
                axes_err = axes_err.reshape(2, 1)

            fig_recon.suptitle(
                f"{experiment_title} - Reconstruction comparison | angles={n_angles} | columns=I0 | rows=FBP, GD",
                fontsize=12,
            )
            fig_err.suptitle(
                f"{experiment_title} - Error comparison | angles={n_angles} | columns=I0 | rows=|Reference - FBP|, |Reference - GD|",
                fontsize=12,
            )

            for j, lvl in enumerate(poisson_levels):
                c = _pick_case(cases, n_angles=n_angles, noise_kind="gaussian+poisson", noise_level=lvl)
                err_fbp = np.abs(ref - np.asarray(c.recon_fbp, dtype=np.float32))
                err_gd = np.abs(ref - np.asarray(c.recon_gd, dtype=np.float32))
                col_title = _format_i0(lvl)
                err_vmax = max(float(err_fbp.max()), float(err_gd.max()), 1e-6)

                ax = axes_recon[0, j]
                ax.imshow(c.recon_fbp, cmap="gray", vmin=image_vmin, vmax=image_vmax)
                ax.set_title(f"{col_title}\nFBP", fontsize=10)
                ax.axis("off")

                ax = axes_recon[1, j]
                ax.imshow(c.recon_gd, cmap="gray", vmin=image_vmin, vmax=image_vmax)
                ax.set_title(f"{col_title}\nGD (SIRT-like)", fontsize=10)
                ax.axis("off")

                ax = axes_err[0, j]
                ax.imshow(err_fbp, cmap="gray", vmin=0.0, vmax=err_vmax)
                ax.set_title(f"{col_title}\n|Reference - FBP|", fontsize=10)
                ax.axis("off")

                ax = axes_err[1, j]
                ax.imshow(err_gd, cmap="gray", vmin=0.0, vmax=err_vmax)
                ax.set_title(f"{col_title}\n|Reference - GD|", fontsize=10)
                ax.axis("off")

            out_recon = out_root / f"{file_prefix}_reconstruction_compare_angles_{n_angles}.png"
            fig_recon.savefig(out_recon, dpi=150, bbox_inches="tight")
            out_err = out_root / f"{file_prefix}_error_compare_angles_{n_angles}.png"
            fig_err.savefig(out_err, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig_recon)
            plt.close(fig_err)
            saved.extend([out_recon, out_err])

    if hardest_case is None and angles_list and poisson_levels:
        hardest_case = (min(angles_list), "gaussian+poisson", poisson_levels[0])

    if hardest_case is not None:
        n_angles_h, kind_h, lvl_h = hardest_case
        c = _pick_case(cases, n_angles=n_angles_h, noise_kind=kind_h, noise_level=lvl_h)
        err_fbp = np.abs(ref - np.asarray(c.recon_fbp, dtype=np.float32))
        err_gd = np.abs(ref - np.asarray(c.recon_gd, dtype=np.float32))

        fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
        fig.suptitle(
            f"{experiment_title} - Hard case comparison | angles={n_angles_h} | {kind_h} {lvl_h}",
            fontsize=12,
        )
        items = [
            ("Reference", ref),
            ("FBP", c.recon_fbp),
            ("GD (SIRT-like)", c.recon_gd),
            ("|Reference - FBP|", err_fbp),
            ("|Reference - GD|", err_gd),
        ]
        for ax, (title, arr) in zip(axes, items):
            if title.startswith("|Reference"):
                ax.imshow(arr, cmap="gray", vmin=0.0, vmax=max(float(err_fbp.max()), float(err_gd.max()), 1e-6))
            else:
                ax.imshow(arr, cmap="gray", vmin=image_vmin, vmax=image_vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        safe_lvl = str(lvl_h).replace("=", "")
        out = out_root / f"{file_prefix}_hard_case_angles_{n_angles_h}_{kind_h}_{safe_lvl}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        saved.append(out)

    return saved


def load_reference_ct_image(
    image_path: str | Path,
) -> ReferenceImageResult:
    """Load and normalize the reference CT image used throughout Exercise 1.

    Parameters
    ----------
    image_path:
        Path to the CT reference image file.
    Returns
    -------
    ReferenceImageResult
        Prepared image and source metadata.
    """
    path = _resolve_out_path(image_path)
    raw_image = imread(path)
    image = prepare_reference_ct_image(raw_image)
    return ReferenceImageResult(
        image=image,
        image_path=path,
        source_shape=tuple(np.asarray(raw_image).shape),
        source_dtype=str(np.asarray(raw_image).dtype),
    )


def summarize_sinogram_sets(sinogram_sets: list[SinogramSet]) -> list[str]:
    """Create notebook-friendly one-line summaries for sinogram sets.

    Parameters
    ----------
    sinogram_sets:
        Simulated sinogram groups to summarize.

    Returns
    -------
    list[str]
        Human-readable summary lines.
    """
    return [
        f"angles={s.n_angles:>3} | views={len(s.theta):>3} | clean shape={tuple(s.clean.shape)}"
        for s in sinogram_sets
    ]


def run_exercise_1_1_sinogram_experiment(
    image: np.ndarray,
    angles_list: Iterable[int] = (360, 90, 20),
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 0.05,
    poisson_i0_levels: Iterable[float] = (1e5, 1e3, 1e2),
    attenuation_scale: float = 1000.0,
    seed: int = 42,
    panel_out_path: str | Path | None = "results/ct/figures/exercise_1_1/exercise_1_1b_noisy_sinograms.png",
    panel_dpi: int = 150,
    show_panel: bool = False,
) -> SinogramSimulationResult:
    """Run the full Exercise 1.1(b) sinogram simulation workflow.

    Parameters
    ----------
    image:
        Reference CT image in normalized intensity space.
    angles_list:
        Projection counts to simulate.
    gaussian_mu:
        Mean of the Gaussian count noise.
    gaussian_sigma:
        Standard deviation of the Gaussian count noise.
    poisson_i0_levels:
        I0 levels used in the experiment.
    attenuation_scale:
        Scaling factor used to convert the image to practical attenuation units.
    seed:
        Random seed for reproducibility.
    panel_out_path:
        Optional path for the saved sinogram comparison panel.
    panel_dpi:
        Figure resolution for the saved panel.
    show_panel:
        Whether to display the saved panel interactively.

    Returns
    -------
    SinogramSimulationResult
        Simulated sinograms, summary lines, and optional panel path.
    """
    sinogram_sets = simulate_noisy_sinograms(
        image=image,
        angles_list=angles_list,
        gaussian_mu=gaussian_mu,
        gaussian_sigma=gaussian_sigma,
        poisson_i0_levels=poisson_i0_levels,
        attenuation_scale=attenuation_scale,
        seed=seed,
    )
    panel_path = None
    if panel_out_path is not None:
        panel_path = save_sinogram_panel(
            sinogram_sets=sinogram_sets,
            gaussian_mu=gaussian_mu,
            gaussian_sigma=gaussian_sigma,
            poisson_i0_levels=poisson_i0_levels,
            out_path=panel_out_path,
            dpi=panel_dpi,
            show=show_panel,
        )
    return SinogramSimulationResult(
        sinogram_sets=sinogram_sets,
        summary_lines=summarize_sinogram_sets(sinogram_sets),
        panel_path=panel_path,
    )


def run_exercise_1_1_reconstruction_experiment(
    reference_image: np.ndarray,
    sinogram_sets: list[SinogramSet],
    poisson_i0_levels: Iterable[float] = (1e5, 1e3, 1e2),
    attenuation_scale: float = 1000.0,
    fbp_filter: str = "ramp",
    gd_iters: int = 60,
    gd_step_size: float = 0.02,
    gd_init_mode: str = "fbp",
    gd_normalize_gradient: bool = True,
    gd_clip_to_reference_range: bool = True,
    gd_mask_each_iter: bool = True,
    gd_positivity: bool = False,
    metric_mode: str = "reporting",
    figures_out_dir: str | Path = "results/ct/figures/exercise_1_1",
    metrics_out_path: str | Path = "results/ct/metrics/exercise_1_1/exercise_1_1c_metrics.csv",
    hardest_case: tuple[int, str, str] | None = None,
    show_figures: bool = False,
    show_progress: bool = False,
) -> ReconstructionExperimentResult:
    """Run the full Exercise 1.1(c) reconstruction workflow.

    Parameters
    ----------
    reference_image:
        Ground-truth CT image in normalized intensity space. It is converted to
        practical attenuation units internally before reconstruction metrics are
        computed.
    sinogram_sets:
        Simulated sinograms to reconstruct.
    poisson_i0_levels:
        I0 levels included in the experiment.
    attenuation_scale:
        Scaling factor used to convert the reference image to practical attenuation units.
    fbp_filter:
        FBP filter name.
    gd_iters:
        Number of gradient-descent iterations.
    gd_step_size:
        Gradient-descent step size.
    gd_init_mode:
        Initialization strategy for gradient descent.
    gd_normalize_gradient:
        Whether to normalize each gradient-descent update by the gradient RMS.
    gd_clip_to_reference_range:
        Whether to clip the reconstruction to the reference image range after each update.
    gd_mask_each_iter:
        Whether to zero pixels outside the reconstruction circle after each update.
    gd_positivity:
        Whether to enforce positivity in gradient descent.
    metric_mode:
        Metric convention used to score reconstructions.
    figures_out_dir:
        Output directory for saved figures.
    metrics_out_path:
        Output path for the metrics CSV.
    hardest_case:
        Optional explicit hard-case figure selection.
    show_figures:
        Whether to display the generated figures.
    show_progress:
        Whether to show progress bars for iterative methods.

    Returns
    -------
    ReconstructionExperimentResult
        Cases, metric rows, figure paths, and CSV path.
    """
    attenuation_reference = scale_image_for_practical(reference_image, attenuation_scale=attenuation_scale)
    cases = run_reconstruction_comparison(
        reference_image=attenuation_reference,
        sinogram_sets=sinogram_sets,
        poisson_i0_levels=poisson_i0_levels,
        fbp_filter=fbp_filter,
        gd_iters=gd_iters,
        gd_step_size=gd_step_size,
        gd_init_mode=gd_init_mode,
        gd_normalize_gradient=gd_normalize_gradient,
        gd_clip_to_reference_range=gd_clip_to_reference_range,
        gd_mask_each_iter=gd_mask_each_iter,
        gd_positivity=gd_positivity,
        metric_mode=metric_mode,
        show_progress=show_progress,
    )
    figure_paths = save_reconstruction_panels(
        reference_image=attenuation_reference,
        cases=cases,
        out_dir=figures_out_dir,
        show=show_figures,
        hardest_case=hardest_case,
        experiment_title="Exercise 1.1(c)",
        file_prefix="exercise_1_1c",
    )
    metric_rows = summarize_metrics(cases)
    metrics_path = save_metrics_csv(metric_rows, out_path=metrics_out_path)
    return ReconstructionExperimentResult(
        cases=cases,
        metric_rows=metric_rows,
        figure_paths=figure_paths,
        metrics_path=metrics_path,
    )
