"""Exercise 2.2 MRI utilities: denoising, comparison plots, and notebook helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_bilateral


def _savefig(fig: plt.Figure, out_path: str | Path | None) -> None:
    """Save a Matplotlib figure when an output path is provided.

    Args:
        fig: Figure to save.
        out_path: Target path or `None` to skip saving.
    """
    if out_path is None:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _to_mag(img_coils: np.ndarray) -> np.ndarray:
    """Convert complex coil images to magnitude images.

    Args:
        img_coils: Input coil images.

    Returns:
        Magnitude images with the same shape as the input.
    """
    return np.abs(np.asarray(img_coils))


def save_metrics_rows(rows: list[dict], out_path: str | Path) -> Path:
    """Write metric rows to a CSV file.

    Args:
        rows: Metric rows to save.
        out_path: Output CSV path.

    Returns:
        The saved CSV path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def print_metrics_rows(rows: list[dict], title: str = "Metrics") -> None:
    """Print metric rows in a notebook-friendly format.

    Args:
        rows: Metric rows to print.
        title: Section title shown above the rows.
    """
    print(f"\n{title}")
    for row in rows:
        coil = row.get("coil", "-")
        mse = float(row.get("mse", float("nan")))
        psnr = float(row.get("psnr", float("nan")))
        ssim = float(row.get("ssim", float("nan")))
        print(
            f"method={row.get('method')} | coil={coil}"
            f" | mse={mse:.6f} | psnr={psnr:.4f} | ssim={ssim:.4f}"
        )


def denoise_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian denoising to one image.

    Args:
        img: Input magnitude image.
        sigma: Gaussian standard deviation.

    Returns:
        The Gaussian-smoothed image.
    """
    return gaussian_filter(img, sigma=sigma).astype(np.float32)


def denoise_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply median filtering to one image.

    Args:
        img: Input magnitude image.
        size: Median filter window size.

    Returns:
        The median-filtered image.
    """
    return median_filter(img, size=size).astype(np.float32)


def denoise_bilateral_img(
    img: np.ndarray,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> np.ndarray:
    """Apply bilateral denoising to one image.

    Args:
        img: Input magnitude image.
        sigma_color: Bilateral color-domain smoothing strength.
        sigma_spatial: Bilateral spatial smoothing strength.

    Returns:
        The bilateral-filtered image.
    """
    x = np.asarray(img, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x01 = (x - lo) / (hi - lo)
    y01 = denoise_bilateral(
        x01,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=None,
    ).astype(np.float32)
    return (y01 * (hi - lo) + lo).astype(np.float32)


def denoise_all_coils(
    img_coils: np.ndarray,
    sigma: float = 1.0,
    median_size: int = 3,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> dict[str, np.ndarray]:
    """Apply all image-space denoisers to every coil magnitude image.

    Args:
        img_coils: Coil-first image-space data.
        sigma: Gaussian standard deviation.
        median_size: Median filter window size.
        sigma_color: Bilateral color-domain smoothing strength.
        sigma_spatial: Bilateral spatial smoothing strength.

    Returns:
        A dictionary with `original`, `gaussian`, `median`, and `bilateral`
        stacks, each shaped `(coils, height, width)`.
    """
    original = _to_mag(img_coils).astype(np.float32)
    n_coils = int(original.shape[0])

    gaussian = np.empty_like(original, dtype=np.float32)
    median = np.empty_like(original, dtype=np.float32)
    bilateral = np.empty_like(original, dtype=np.float32)

    for coil_id in range(n_coils):
        coil_img = original[coil_id]
        gaussian[coil_id] = denoise_gaussian(coil_img, sigma=sigma)
        median[coil_id] = denoise_median(coil_img, size=median_size)
        bilateral[coil_id] = denoise_bilateral_img(
            coil_img,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
        )

    return {
        "original": original,
        "gaussian": gaussian,
        "median": median,
        "bilateral": bilateral,
    }


def compute_metrics(reference: np.ndarray, img: np.ndarray) -> tuple[float, float, float]:
    """Compute MSE, PSNR, and SSIM against one reference image.

    Args:
        reference: Reference image.
        img: Image to evaluate.

    Returns:
        A tuple `(mse, psnr, ssim)`.
    """
    ref = np.asarray(reference, dtype=np.float32)
    x = np.asarray(img, dtype=np.float32)
    data_range = float(np.max(ref) - np.min(ref))
    if data_range <= 0:
        data_range = 1.0
    mse = float(mean_squared_error(ref, x))
    if mse <= 1e-15:
        return 0.0, float("inf"), 1.0
    psnr = float(peak_signal_noise_ratio(ref, x, data_range=data_range))
    ssim = float(structural_similarity(ref, x, data_range=data_range))
    return mse, psnr, ssim


def plot_denoise_per_coil(
    denoised: dict[str, np.ndarray],
    coil_id: int,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot one coil across the original and three denoised variants.

    Args:
        denoised: Output from `denoise_all_coils`.
        coil_id: Coil index to visualize.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    keys = ["original", "gaussian", "median", "bilateral"]
    titles = ["Original", "Gaussian", "Median", "Bilateral"]
    imgs = [np.asarray(denoised[key][coil_id], dtype=np.float32) for key in keys]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    for ax, title, img in zip(axes, titles, imgs):
        im = ax.imshow(img, cmap="gray")
        ax.set_title(f"{title} | Coil {coil_id}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_denoise_all_coils(
    denoised: dict[str, np.ndarray],
    out_dir: str | Path | None = None,
    show: bool = False,
) -> list[Path]:
    """Export one denoising comparison figure per coil.

    Args:
        denoised: Output from `denoise_all_coils`.
        out_dir: Optional directory for saved figures.
        show: Whether to display each figure interactively.

    Returns:
        A list of saved figure paths.
    """
    n_coils = int(denoised["original"].shape[0])
    saved: list[Path] = []
    for coil_id in range(n_coils):
        out_path = None if out_dir is None else Path(out_dir) / f"ex2_2_part1_coil_{coil_id}_denoise_compare.png"
        plot_denoise_per_coil(denoised, coil_id=coil_id, out_path=out_path, show=show)
        if out_path is not None:
            saved.append(Path(out_path))
    return saved


def run_exercise_2_2_part1(
    img_coils: np.ndarray,
    sigma: float,
    median_size: int,
    sigma_color: float,
    sigma_spatial: float,
    out_dir: str | Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Run the complete notebook workflow for Exercise 2.2 Part 1.

    Args:
        img_coils: Coil-first image-space data.
        sigma: Gaussian standard deviation.
        median_size: Median filter window size.
        sigma_color: Bilateral color-domain smoothing strength.
        sigma_spatial: Bilateral spatial smoothing strength.
        out_dir: Optional output directory for saved figures.
        show: Whether to display figures interactively.

    Returns:
        A dictionary containing the denoised stacks and saved figure paths.
    """
    denoised = denoise_all_coils(
        img_coils,
        sigma=sigma,
        median_size=median_size,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
    )
    figure_paths = plot_denoise_all_coils(denoised, out_dir=out_dir, show=show)
    return {"denoised": denoised, "figure_paths": figure_paths}


def butterworth_lowpass_filter(shape: tuple[int, int], D0: float = 30.0, n: int = 2) -> np.ndarray:
    """Create a centered 2D Butterworth low-pass filter.

    Args:
        shape: Filter shape as `(height, width)`.
        D0: Cutoff frequency radius.
        n: Butterworth order.

    Returns:
        A 2D Butterworth low-pass mask.
    """
    height, width = int(shape[0]), int(shape[1])
    u = np.arange(height) - height // 2
    v = np.arange(width) - width // 2
    uu, vv = np.meshgrid(u, v, indexing="ij")
    distance = np.sqrt(uu**2 + vv**2)
    return (1.0 / (1.0 + (distance / float(D0)) ** (2 * int(n)))).astype(np.float32)


def run_part2_first_coil(
    kspace_coils_first: np.ndarray,
    coil_id: int = 0,
    D0: float = 30.0,
    n: int = 2,
) -> dict[str, np.ndarray]:
    """Apply centered-k-space Butterworth filtering to one selected coil.

    Args:
        kspace_coils_first: Coil-first k-space array.
        coil_id: Coil index to process.
        D0: Butterworth cutoff frequency radius.
        n: Butterworth order.

    Returns:
        A dictionary containing intermediate k-space arrays and the filtered
        magnitude/phase reconstruction for the selected coil.
    """
    coil_k = np.asarray(kspace_coils_first[coil_id])
    filter_mask = butterworth_lowpass_filter(coil_k.shape, D0=D0, n=n)
    filtered_kspace = coil_k * filter_mask
    img_original = np.fft.ifft2(coil_k)
    img_filtered = np.fft.ifft2(filtered_kspace)
    return {
        "coil_k": coil_k,
        "coil_k_centered": coil_k,
        "H": filter_mask,
        "filtered_k_centered": filtered_kspace,
        "filtered_k": filtered_kspace,
        "img_original": img_original,
        "img_filtered": img_filtered,
        "mag": np.abs(img_filtered),
        "phase": np.angle(img_filtered),
    }


def run_butterworth_all_coils_rsos(
    kspace_coils_first: np.ndarray,
    D0: float = 30.0,
    n: int = 2,
) -> np.ndarray:
    """Apply Butterworth filtering to every coil in k-space, then combine with rSoS.

    Args:
        kspace_coils_first: Coil-first k-space array.
        D0: Butterworth cutoff frequency radius.
        n: Butterworth order.

    Returns:
        The final rSoS image after per-coil Butterworth filtering.

    Raises:
        ValueError: If the input does not have shape `(coils, height, width)`.
    """
    kspace = np.asarray(kspace_coils_first)
    if kspace.ndim != 3:
        raise ValueError(f"Expected k-space shape (coils,H,W), got {kspace.shape}")

    filter_mask = butterworth_lowpass_filter(kspace.shape[-2:], D0=D0, n=n)
    filtered_kspace = kspace * filter_mask[None, :, :]
    img = np.fft.ifft2(filtered_kspace, axes=(-2, -1))
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=0)).astype(np.float32)


def plot_part2_mag_phase(
    part2: dict[str, np.ndarray],
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the filtered magnitude and phase for Exercise 2.2 Part 2.

    Args:
        part2: Output from `run_part2_first_coil`.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    magnitude = np.asarray(part2["mag"], dtype=np.float32)
    phase = np.asarray(part2["phase"], dtype=np.float32)
    magnitude_display = np.log1p(np.maximum(magnitude, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes[0].imshow(magnitude_display, cmap="gray")
    axes[0].set_title("Part 2: Magnitude (k-space Butterworth)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase, cmap="gray")
    axes[1].set_title("Part 2: Phase (k-space Butterworth)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_part2_compare(
    part2: dict[str, np.ndarray],
    denoised: dict[str, np.ndarray],
    coil_id: int = 0,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Compare one Butterworth result against image-space denoisers for one coil.

    Args:
        part2: Output from `run_part2_first_coil`.
        denoised: Output from `denoise_all_coils`.
        coil_id: Coil index to compare.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    original = np.asarray(denoised["original"][coil_id], dtype=np.float32)
    gaussian = np.asarray(denoised["gaussian"][coil_id], dtype=np.float32)
    median = np.asarray(denoised["median"][coil_id], dtype=np.float32)
    bilateral = np.asarray(denoised["bilateral"][coil_id], dtype=np.float32)
    butterworth = np.asarray(np.abs(part2["img_filtered"]), dtype=np.float32)

    imgs = [original, butterworth, gaussian, median, bilateral]
    titles = ["Original", "k-space Butterworth", "Gaussian", "Median", "Bilateral"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for idx, (img, title) in enumerate(zip(imgs, titles)):
        ax = axes[idx]
        display_img = np.log1p(np.maximum(img, 0.0))
        im = ax.imshow(display_img, cmap="gray")
        ax.set_title(f"{title} | Coil {coil_id}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(imgs), len(axes)):
        axes[idx].axis("off")

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def metrics_part2(
    denoised: dict[str, np.ndarray],
    part2: dict[str, np.ndarray],
    coil_id: int = 0,
    reference: np.ndarray | None = None,
) -> list[dict]:
    """Compute Part 2 metrics against one explicit reference image.

    Args:
        denoised: Output from `denoise_all_coils`.
        part2: Output from `run_part2_first_coil`.
        coil_id: Coil index used for the Butterworth comparison.
        reference: Optional explicit reference image. Defaults to coil 0 original.

    Returns:
        Metric rows for the original, Gaussian, median, bilateral, and selected
        k-space Butterworth result.
    """
    ref = np.asarray(denoised["original"][0] if reference is None else reference, dtype=np.float32)
    rows: list[dict] = []
    n_coils = int(denoised["original"].shape[0])

    for current_coil in range(n_coils):
        original = np.asarray(denoised["original"][current_coil], dtype=np.float32)
        mse, psnr, ssim = compute_metrics(ref, original)
        rows.append({"part": "2.2-part2", "coil": current_coil, "method": "original", "mse": mse, "psnr": psnr, "ssim": ssim})

        for method in ("gaussian", "median", "bilateral"):
            img = np.asarray(denoised[method][current_coil], dtype=np.float32)
            mse, psnr, ssim = compute_metrics(ref, img)
            rows.append({"part": "2.2-part2", "coil": current_coil, "method": method, "mse": mse, "psnr": psnr, "ssim": ssim})

        if current_coil == coil_id:
            butterworth = np.asarray(np.abs(part2["img_filtered"]), dtype=np.float32)
            mse, psnr, ssim = compute_metrics(ref, butterworth)
            rows.append({"part": "2.2-part2", "coil": current_coil, "method": "kspace_butterworth", "mse": mse, "psnr": psnr, "ssim": ssim})

    return rows


def run_exercise_2_2_part2(
    kspace_coils_first: np.ndarray,
    denoised: dict[str, np.ndarray],
    coil_id: int,
    D0: float,
    n: int,
    reference: np.ndarray | None,
    mag_phase_out_path: str | Path | None = None,
    compare_out_path: str | Path | None = None,
    metrics_out_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Run the complete notebook workflow for Exercise 2.2 Part 2.

    Args:
        kspace_coils_first: Coil-first k-space array.
        denoised: Output from `denoise_all_coils`.
        coil_id: Coil index used for Butterworth filtering.
        D0: Butterworth cutoff frequency radius.
        n: Butterworth order.
        reference: Explicit reference image for metrics.
        mag_phase_out_path: Optional output path for the magnitude/phase plot.
        compare_out_path: Optional output path for the comparison plot.
        metrics_out_path: Optional output path for the metrics CSV.
        show: Whether to display the generated figures interactively.

    Returns:
        A dictionary containing the filtered data, metric rows, filtered-row
        subset, and saved metrics path.
    """
    part2 = run_part2_first_coil(kspace_coils_first, coil_id=coil_id, D0=D0, n=n)
    plot_part2_mag_phase(part2, out_path=mag_phase_out_path, show=show)
    plot_part2_compare(part2, denoised, coil_id=coil_id, out_path=compare_out_path, show=show)
    rows = metrics_part2(denoised, part2, coil_id=coil_id, reference=reference)
    butter_rows = [row for row in rows if row.get("method") == "kspace_butterworth"]
    metrics_path = None if metrics_out_path is None else save_metrics_rows(rows, metrics_out_path)
    return {
        "part2": part2,
        "rows": rows,
        "butter_rows": butter_rows,
        "metrics_path": metrics_path,
    }


def denoise_combined_image(
    rsos: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    median_size: int = 3,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> np.ndarray:
    """Apply one image-space denoiser directly to an rSoS image.

    Args:
        rsos: Input rSoS image.
        method: One of `gaussian`, `median`, or `bilateral`.
        sigma: Gaussian standard deviation.
        median_size: Median filter window size.
        sigma_color: Bilateral color-domain smoothing strength.
        sigma_spatial: Bilateral spatial smoothing strength.

    Returns:
        The denoised rSoS image.

    Raises:
        ValueError: If `method` is unsupported.
    """
    method_key = method.lower()
    x = np.asarray(rsos, dtype=np.float32)
    if method_key == "gaussian":
        return denoise_gaussian(x, sigma=sigma)
    if method_key == "median":
        return denoise_median(x, size=median_size)
    if method_key == "bilateral":
        return denoise_bilateral_img(x, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    raise ValueError(f"Unknown method: {method}")


def denoise_all_coils_rsos(
    img_coils: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    median_size: int = 3,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> np.ndarray:
    """Apply one denoiser to every coil magnitude image, then combine with rSoS.

    Args:
        img_coils: Coil-first image-space data.
        method: One of `original`, `gaussian`, `median`, or `bilateral`.
        sigma: Gaussian standard deviation.
        median_size: Median filter window size.
        sigma_color: Bilateral color-domain smoothing strength.
        sigma_spatial: Bilateral spatial smoothing strength.

    Returns:
        The final rSoS image after per-coil denoising.

    Raises:
        ValueError: If `method` is unsupported.
    """
    denoised = denoise_all_coils(
        img_coils,
        sigma=sigma,
        median_size=median_size,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
    )
    key = method.lower()
    if key not in denoised:
        raise ValueError(f"Unknown method: {method}")
    return np.sqrt(np.sum(np.square(denoised[key]), axis=0)).astype(np.float32)


def plot_part3_combined_compare(
    rsos: np.ndarray,
    rsos_denoised: np.ndarray,
    method_name: str,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the original rSoS image next to one denoised rSoS result.

    Args:
        rsos: Original rSoS image.
        rsos_denoised: Denoised rSoS image.
        method_name: Label used in the denoised subplot title.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    original = np.asarray(rsos, dtype=np.float32)
    denoised = np.asarray(rsos_denoised, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes[0].imshow(original, cmap="gray")
    axes[0].set_title("rSoS original")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(denoised, cmap="gray")
    axes[1].set_title(f"rSoS denoised ({method_name})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def metrics_part3(
    rsos: np.ndarray,
    rsos_denoised: np.ndarray,
    method_name: str,
    reference: np.ndarray | None = None,
) -> list[dict]:
    """Compute Part 3 metrics against one explicit reference image.

    Args:
        rsos: Original rSoS image.
        rsos_denoised: Denoised rSoS image.
        method_name: Method label stored in the metric rows.
        reference: Explicit reference image. Defaults to the original rSoS image.

    Returns:
        Metric rows for the original rSoS image and the denoised result.
    """
    ref = np.asarray(rsos if reference is None else reference, dtype=np.float32)
    original = np.asarray(rsos, dtype=np.float32)
    denoised = np.asarray(rsos_denoised, dtype=np.float32)

    mse0, psnr0, ssim0 = compute_metrics(ref, original)
    mse, psnr, ssim = compute_metrics(ref, denoised)
    return [
        {"part": "2.2-part3", "method": "original", "mse": mse0, "psnr": psnr0, "ssim": ssim0},
        {"part": "2.2-part3", "method": method_name, "mse": mse, "psnr": psnr, "ssim": ssim},
    ]


def run_exercise_2_2_part3(
    img_coils: np.ndarray,
    kspace_coils_first: np.ndarray,
    rsos: np.ndarray,
    reference: np.ndarray | None,
    bilateral_sigma_color: float,
    bilateral_sigma_spatial: float,
    butter_D0: float,
    butter_n: int,
    bilateral_out_path: str | Path | None = None,
    butterworth_out_path: str | Path | None = None,
    metrics_out_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, object]:
    """Run the complete notebook workflow for Exercise 2.2 Part 3.

    Args:
        img_coils: Coil-first image-space data.
        kspace_coils_first: Coil-first k-space array.
        rsos: Original rSoS image.
        reference: Explicit reference image for metrics.
        bilateral_sigma_color: Bilateral color-domain smoothing strength.
        bilateral_sigma_spatial: Bilateral spatial smoothing strength.
        butter_D0: Butterworth cutoff frequency radius.
        butter_n: Butterworth order.
        bilateral_out_path: Optional output path for the bilateral comparison plot.
        butterworth_out_path: Optional output path for the Butterworth comparison plot.
        metrics_out_path: Optional output path for the metrics CSV.
        show: Whether to display the generated figures interactively.

    Returns:
        A dictionary containing both rSoS results, merged metric rows, and the
        saved metrics path.
    """
    rsos_bilateral = denoise_all_coils_rsos(
        img_coils,
        method="bilateral",
        sigma_color=bilateral_sigma_color,
        sigma_spatial=bilateral_sigma_spatial,
    )
    plot_part3_combined_compare(
        rsos,
        rsos_bilateral,
        method_name="bilateral (per-coil -> rSoS)",
        out_path=bilateral_out_path,
        show=show,
    )

    rsos_butter = run_butterworth_all_coils_rsos(kspace_coils_first, D0=butter_D0, n=butter_n)
    plot_part3_combined_compare(
        rsos,
        rsos_butter,
        method_name="butterworth (per-coil -> rSoS)",
        out_path=butterworth_out_path,
        show=show,
    )

    rows_bilateral = metrics_part3(
        rsos,
        rsos_bilateral,
        method_name="bilateral_per_coil_rsos",
        reference=reference,
    )
    rows_butter = metrics_part3(
        rsos,
        rsos_butter,
        method_name="butterworth_per_coil_rsos",
        reference=reference,
    )
    part3_rows = [rows_bilateral[0], rows_bilateral[1], rows_butter[1]]
    metrics_path = None if metrics_out_path is None else save_metrics_rows(part3_rows, metrics_out_path)
    return {
        "rsos_bilateral": rsos_bilateral,
        "rsos_butter": rsos_butter,
        "rows": part3_rows,
        "butter_rows": [rows_butter[1]],
        "metrics_path": metrics_path,
    }
