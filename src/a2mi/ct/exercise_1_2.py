"""Exercise 1.2 utilities: limited-angle CT simulation and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon

from .exercise_1_1 import (
    ReconstructionCase,
    SinogramSet,
    add_gaussian_poisson_noise_practical,
    run_reconstruction_comparison,
    save_metrics_csv,
    save_reconstruction_panels,
    summarize_metrics,
)


def _resolve_out_path(out_path: str | Path) -> Path:
    p = Path(out_path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[3]
    return project_root / p


@dataclass
class LimitedAngleSinogramSet:
    angle_range: int
    theta: np.ndarray
    clean: np.ndarray
    gaussian_poisson: dict[float, np.ndarray]


@dataclass
class LimitedAngleComparisonResult:
    cases: list[ReconstructionCase]
    metric_rows: list[dict[str, float | str | int]]
    figure_paths: list[Path]
    metrics_path: Path


def make_theta_limited(angle_range: int, step_deg: float = 1.0) -> np.ndarray:
    """Create theta in [0, angle_range) with fixed step."""
    if angle_range <= 0 or angle_range > 180:
        raise ValueError(f"angle_range must be in (0, 180], got {angle_range}")
    if step_deg <= 0:
        raise ValueError(f"step_deg must be > 0, got {step_deg}")
    return np.arange(0.0, float(angle_range), float(step_deg), dtype=np.float32)


def forward_project_limited(
    image: np.ndarray,
    angle_range: int,
    step_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Radon forward projection for limited-angle acquisition."""
    theta = make_theta_limited(angle_range=angle_range, step_deg=step_deg)
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img.shape}")

    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0 - 1.0
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    img = np.where(circle_mask, img, 0.0)

    sino = radon(img, theta=theta, circle=True)
    return sino.astype(np.float32), theta


def simulate_noisy_sinograms_limited(
    image: np.ndarray,
    angle_ranges: Iterable[int] = (180, 120, 40),
    step_deg: float = 1.0,
    poisson_i0_levels: Iterable[float] = (1e5, 1e3, 1e2),
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 5.0,
    scale: float = 1000.0,
    seed: int = 42,
) -> list[LimitedAngleSinogramSet]:
    """Simulate clean and Gaussian+Poisson noisy sinograms for limited-angle ranges."""
    rng = np.random.default_rng(seed)
    out: list[LimitedAngleSinogramSet] = []

    for angle_range in angle_ranges:
        clean, theta = forward_project_limited(image=image, angle_range=int(angle_range), step_deg=step_deg)
        gp: dict[float, np.ndarray] = {}
        for i0 in poisson_i0_levels:
            gp[float(i0)] = add_gaussian_poisson_noise_practical(
                clean,
                i0=float(i0),
                scale=float(scale),
                gaussian_mu=gaussian_mu,
                gaussian_sigma=gaussian_sigma,
                rng=rng,
            )
        out.append(
            LimitedAngleSinogramSet(
                angle_range=int(angle_range),
                theta=theta,
                clean=clean,
                gaussian_poisson=gp,
            )
        )
    return out


def summarize_limited_angle_sinograms(sinogram_sets: list[LimitedAngleSinogramSet]) -> list[str]:
    """Return notebook-friendly summary lines for limited-angle sinograms."""
    lines: list[str] = []
    for s in sinogram_sets:
        lines.append(
            f"range={s.angle_range:>3} deg | views={len(s.theta):>3} | clean shape={tuple(s.clean.shape)}"
        )
    return lines


def to_reconstruction_sinogram_sets(sinogram_sets: list[LimitedAngleSinogramSet]) -> list[SinogramSet]:
    """Adapt limited-angle sets to the shared reconstruction comparison API."""
    return [
        SinogramSet(
            n_angles=int(len(s.theta)),
            theta=np.asarray(s.theta, dtype=np.float32),
            clean=np.asarray(s.clean, dtype=np.float32),
            gaussian_poisson={float(k): np.asarray(v, dtype=np.float32) for k, v in s.gaussian_poisson.items()},
        )
        for s in sinogram_sets
    ]


def save_limited_angle_sinogram_panel(
    sinogram_sets: list[LimitedAngleSinogramSet],
    poisson_i0_levels: Iterable[float],
    gaussian_mu: float,
    gaussian_sigma: float,
    out_path: str | Path,
    dpi: int = 150,
    show: bool = True,
) -> Path:
    """Save panel: clean plus noisy sinograms for each limited-angle range."""
    levels = [float(v) for v in poisson_i0_levels]
    n_cols = 1 + len(levels)
    fig, axes = plt.subplots(
        len(sinogram_sets),
        n_cols,
        figsize=(3.8 * n_cols, 3.6 * len(sinogram_sets)),
        constrained_layout=True,
    )
    if len(sinogram_sets) == 1:
        axes = np.array([axes])

    for r, s in enumerate(sinogram_sets):
        panel = [("Clean", s.clean)] + [(f"G+P I0={level:.0e}", s.gaussian_poisson[level]) for level in levels]
        for c, (title, arr) in enumerate(panel):
            ax = axes[r, c]
            im = ax.imshow(arr, cmap="gray", aspect="auto")
            ax.set_title(
                f"{title} | range={s.angle_range} deg | views={len(s.theta)} | mu={gaussian_mu}, sigma={gaussian_sigma}",
                fontsize=8,
            )
            ax.set_xlabel("View index")
            ax.set_ylabel("Detector bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = _resolve_out_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def run_limited_angle_reconstruction_experiment(
    reference_image: np.ndarray,
    sinogram_sets: list[LimitedAngleSinogramSet],
    poisson_i0_levels: Iterable[float] = (1e5, 1e3, 1e2),
    figures_out_dir: str | Path = "results/ct/figures/exercise_1_2",
    metrics_out_path: str | Path = "results/ct/metrics/exercise_1_2_metrics.csv",
    show: bool = False,
    show_progress: bool = False,
    hardest_case: tuple[int, str, str] | None = None,
    fbp_filter: str = "ramp",
    gd_iters: int = 20,
    gd_step_size: float = 0.8,
    gd_positivity: bool = True,
) -> LimitedAngleComparisonResult:
    """Repeat the Exercise 1.1 reconstruction comparison in the limited-angle setting."""
    reconstruction_sets = to_reconstruction_sinogram_sets(sinogram_sets)
    cases = run_reconstruction_comparison(
        reference_image=reference_image,
        sinogram_sets=reconstruction_sets,
        poisson_i0_levels=poisson_i0_levels,
        fbp_filter=fbp_filter,
        gd_iters=gd_iters,
        gd_step_size=gd_step_size,
        gd_positivity=gd_positivity,
        show_progress=show_progress,
    )
    figure_paths = save_reconstruction_panels(
        reference_image=reference_image,
        cases=cases,
        out_dir=figures_out_dir,
        show=show,
        hardest_case=hardest_case,
        experiment_title="Exercise 1.2(b)",
        file_prefix="exercise_1_2",
    )
    metric_rows = summarize_metrics(cases)
    metrics_path = save_metrics_csv(metric_rows, out_path=metrics_out_path)
    return LimitedAngleComparisonResult(
        cases=cases,
        metric_rows=metric_rows,
        figure_paths=figure_paths,
        metrics_path=metrics_path,
    )



