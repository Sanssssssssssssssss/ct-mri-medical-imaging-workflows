"""Exercise 1.1 utilities: sinogram simulation, reconstruction, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import iradon, radon

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _progress(iterable, enabled: bool = False, desc: str | None = None):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


# -------- common use --------

def _resolve_out_path(out_path: str | Path) -> Path:
    p = Path(out_path)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[3]
    return project_root / p


def _format_i0(level: str) -> str:
    return level if level.startswith("I0=") else level


def prepare_reference_ct_image(raw_image: np.ndarray, attenuation_scale: float = 1000.0) -> np.ndarray:
    """Convert the input CT image to float attenuation values used in the practical."""
    image = np.asarray(raw_image)
    if image.ndim == 3:
        image = image[..., 0]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    image = image / float(attenuation_scale)
    return np.clip(image, 0.0, 1.0 / float(attenuation_scale))


def _image_display_limits(image: np.ndarray) -> tuple[float, float]:
    vmax = float(np.max(np.asarray(image, dtype=np.float32)))
    return 0.0, max(vmax, 1e-6)


def make_theta(n_angles: int) -> np.ndarray:
    """Create projection angles in [0, 360) to match the practical notebook."""
    return np.linspace(0.0, 360.0, int(n_angles), endpoint=False, dtype=np.float32)


def forward_project(image: np.ndarray, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean sinogram with Radon transform."""
    theta = make_theta(n_angles)
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image for forward projection, got shape={img.shape}")
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    # Use half-pixel center for even-sized images to avoid geometric bias.
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = min(h, w) / 2.0 - 1.0
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    img = np.where(circle_mask, img, 0.0)
    sinogram = radon(img, theta=theta, circle=True)
    return sinogram.astype(np.float32), theta


def add_gaussian_noise(
    sinogram: np.ndarray,
    mu: float = 0.0,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add additive Gaussian noise in sinogram domain."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=mu, scale=sigma, size=sinogram.shape)
    return (sinogram + noise).astype(np.float32)


def add_poisson_noise(
    sinogram: np.ndarray,
    i0: float,
    rng: np.random.Generator | None = None,
    min_counts: float = 1.0,
) -> np.ndarray:
    """Add Poisson counting noise using Beer-Lambert physics."""
    if rng is None:
        rng = np.random.default_rng()

    sino = np.asarray(sinogram, dtype=np.float32)
    transmission = np.exp(-sino)
    transmission = np.clip(transmission, 0.0, 1.0)

    expected_counts = np.maximum(float(i0) * transmission, min_counts)
    measured_counts = rng.poisson(expected_counts).astype(np.float32)
    measured_counts = np.maximum(measured_counts, min_counts)

    ratio = measured_counts / float(i0)
    ratio = np.clip(ratio, min_counts / float(i0), 1.0)
    noisy_sino = -np.log(ratio)
    return noisy_sino.astype(np.float32)


def add_gaussian_poisson_noise_practical(
    sinogram: np.ndarray,
    i0: float,
    scale: float = 1000.0,
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 5.0,
    rng: np.random.Generator | None = None,
    min_counts: float = 1.0,
) -> np.ndarray:
    """Practical-2 style noise: scale -> Poisson counts -> Gaussian counts -> log back."""
    if rng is None:
        rng = np.random.default_rng()

    sino = np.asarray(sinogram, dtype=np.float32)
    sino_scaled = sino / float(scale)

    transmission = np.exp(-sino_scaled)
    expected_counts = np.maximum(float(i0) * transmission, min_counts)
    poisson_counts = rng.poisson(expected_counts).astype(np.float32)

    gaussian_counts = rng.normal(loc=gaussian_mu, scale=gaussian_sigma, size=sino.shape).astype(np.float32)
    noisy_counts = poisson_counts + gaussian_counts
    noisy_counts = np.maximum(noisy_counts, min_counts)

    noisy_scaled = -np.log(noisy_counts / float(i0))
    noisy = noisy_scaled * float(scale)
    return noisy.astype(np.float32)


@dataclass
class SinogramSet:
    n_angles: int
    theta: np.ndarray
    clean: np.ndarray
    gaussian_poisson: dict[float, np.ndarray]


@dataclass
class ReconstructionCase:
    n_angles: int
    noise_kind: str
    noise_level: str
    sinogram: np.ndarray
    recon_fbp: np.ndarray
    recon_gd: np.ndarray
    metrics_fbp: dict[str, float]
    metrics_gd: dict[str, float]


def _pick_case(cases: list[ReconstructionCase], n_angles: int, noise_kind: str, noise_level: str) -> ReconstructionCase:
    for c in cases:
        if c.n_angles == n_angles and c.noise_kind == noise_kind and c.noise_level == noise_level:
            return c
    raise KeyError(f"Case not found: angles={n_angles}, kind={noise_kind}, level={noise_level}")


# -------- exercise 1.1(b): noisy sinogram simulation --------

def simulate_noisy_sinograms(
    image: np.ndarray,
    angles_list: Iterable[int] = (360, 90, 20),
    gaussian_mu: float = 0.0,
    gaussian_sigma: float = 5.0,
    poisson_i0_levels: Iterable[float] = (1e5, 1e3, 1e2),
    scale: float = 1000.0,
    seed: int = 42,
) -> list[SinogramSet]:
    """Generate clean and Gaussian+Poisson sinograms for each angle count."""
    rng = np.random.default_rng(seed)
    out: list[SinogramSet] = []

    for n_angles in angles_list:
        clean, theta = forward_project(image=image, n_angles=int(n_angles))
        gaussian_poisson: dict[float, np.ndarray] = {}
        for i0 in poisson_i0_levels:
            gaussian_poisson[float(i0)] = add_gaussian_poisson_noise_practical(
                clean,
                i0=float(i0),
                scale=float(scale),
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
    """Save sinogram comparison panel for exercise 1.1(b)."""
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
    """Reconstruct image by filtered back projection."""
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
    n_iters: int = 80,
    step_size: float = 0.8,
    positivity: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Practical-style least-squares gradient descent using full-batch projections."""
    sino = np.asarray(sinogram, dtype=np.float32)
    if output_size is None:
        output_size = int(sino.shape[0])
    n = int(output_size)

    x = np.zeros((n, n), dtype=np.float32)
    iterator = _progress(range(int(n_iters)), enabled=show_progress, desc=progress_desc or "GD")
    for _ in iterator:
        ax = radon(x, theta=theta, circle=True).astype(np.float32)
        residual = ax - sino
        grad = iradon(residual, theta=theta, circle=True, filter_name=None, output_size=n).astype(np.float32)
        x = x - step_size * grad
        if positivity:
            x = np.maximum(x, 0.0)
    return x


def _compute_metrics(reference: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float32)
    rec = np.asarray(recon, dtype=np.float32)
    mse = float(np.mean((ref - rec) ** 2))
    data_range = float(ref.max() - ref.min())
    if data_range <= 0:
        data_range = 1.0
    psnr = float(20.0 * np.log10(data_range) - 10.0 * np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(ref, rec, data_range=data_range))
    mae = float(np.mean(np.abs(ref - rec)))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def run_reconstruction_comparison(
    reference_image: np.ndarray,
    sinogram_sets: list[SinogramSet],
    poisson_i0_levels: Iterable[float],
    fbp_filter: str = "ramp",
    gd_iters: int = 80,
    gd_step_size: float = 0.8,
    gd_positivity: bool = False,
    show_progress: bool = False,
) -> list[ReconstructionCase]:
    """Run FBP and GD reconstruction over Gaussian+Poisson sinograms."""
    i0_levels = [float(v) for v in poisson_i0_levels]
    ref = np.asarray(reference_image, dtype=np.float32)
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
                    metrics_fbp=_compute_metrics(ref, rec_fbp),
                    metrics_gd=_compute_metrics(ref, rec_gd),
                )
            )
    return cases


def summarize_metrics(cases: list[ReconstructionCase]) -> list[dict[str, float | str | int]]:
    """Flatten case metrics to report-friendly rows."""
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
    """Save summary metrics as CSV."""
    path = _resolve_out_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["n_angles", "noise_kind", "noise_level", "algorithm", "mse", "mae", "psnr", "ssim"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_reconstruction_panels(
    reference_image: np.ndarray,
    cases: list[ReconstructionCase],
    out_dir: str | Path = "results/ct/figures",
    show: bool = False,
    hardest_case: tuple[int, str, str] | None = None,
    experiment_title: str = "Exercise 1.1(c)",
    file_prefix: str = "exercise_1_1c",
) -> list[Path]:
    """Report-friendly figure set for CT reconstruction comparisons."""
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
            fig, axes = plt.subplots(4, n_cols, figsize=(3.6 * n_cols, 10.0), constrained_layout=True)
            if n_cols == 1:
                axes = axes.reshape(4, 1)

            fig.suptitle(
                f"{experiment_title} - Gaussian+Poisson dose sweep | angles={n_angles} | columns=I0 | "
                f"rows=FBP, GD, |err FBP|, |err GD|",
                fontsize=12,
            )

            for j, lvl in enumerate(poisson_levels):
                c = _pick_case(cases, n_angles=n_angles, noise_kind="gaussian+poisson", noise_level=lvl)
                err_fbp = np.abs(ref - np.asarray(c.recon_fbp, dtype=np.float32))
                err_gd = np.abs(ref - np.asarray(c.recon_gd, dtype=np.float32))
                col_title = _format_i0(lvl)
                err_vmax = max(float(err_fbp.max()), float(err_gd.max()), 1e-6)

                ax = axes[0, j]
                ax.imshow(c.recon_fbp, cmap="gray", vmin=image_vmin, vmax=image_vmax)
                ax.set_title(f"{col_title}\nFBP", fontsize=10)
                ax.axis("off")

                ax = axes[1, j]
                ax.imshow(c.recon_gd, cmap="gray", vmin=image_vmin, vmax=image_vmax)
                ax.set_title(f"{col_title}\nGD (SIRT-like)", fontsize=10)
                ax.axis("off")

                ax = axes[2, j]
                ax.imshow(err_fbp, cmap="gray", vmin=0.0, vmax=err_vmax)
                ax.set_title(f"{col_title}\n|Reference - FBP|", fontsize=10)
                ax.axis("off")

                ax = axes[3, j]
                ax.imshow(err_gd, cmap="gray", vmin=0.0, vmax=err_vmax)
                ax.set_title(f"{col_title}\n|Reference - GD|", fontsize=10)
                ax.axis("off")

            out = out_root / f"{file_prefix}_poisson_dose_sweep_angles_{n_angles}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            saved.append(out)

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





