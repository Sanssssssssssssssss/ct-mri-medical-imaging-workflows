"""Exercise 2.2 utilities: MRI denoising comparisons."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_bilateral


# -------- common use --------

def _savefig(fig: plt.Figure, out_path: str | Path | None) -> None:
    if out_path is None:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _to_mag(img_coils: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(img_coils))


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def save_metrics_rows(rows: list[dict], out_path: str | Path) -> Path:
    """Save metrics rows to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def print_metrics_rows(rows: list[dict], title: str = "Metrics") -> None:
    """Pretty-print metrics rows in notebook output."""
    print(f"\n{title}")
    for r in rows:
        coil = r.get("coil", "-")
        mse = float(r.get("mse", float("nan")))
        psnr = float(r.get("psnr", float("nan")))
        ssim = float(r.get("ssim", float("nan")))
        print(
            f"method={r.get('method')} | coil={coil}"
            f" | mse={mse:.6f} | psnr={psnr:.4f} | ssim={ssim:.4f}"
        )


# -------- part 1: image-space denoising (gaussian / median / bilateral) --------

def denoise_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(img, sigma=sigma).astype(np.float32)


def denoise_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    return median_filter(img, size=size).astype(np.float32)


def denoise_bilateral_img(
    img: np.ndarray,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> np.ndarray:
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

    y = y01 * (hi - lo) + lo
    return y.astype(np.float32)


def denoise_all_coils(
    img_coils: np.ndarray,
    sigma: float = 1.0,
    median_size: int = 3,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> dict[str, np.ndarray]:
    """Return original/gaussian/median/bilateral magnitude stacks, shape=(coils,H,W)."""
    orig = _to_mag(img_coils).astype(np.float32)
    n_coils = int(orig.shape[0])

    gauss = np.empty_like(orig, dtype=np.float32)
    med = np.empty_like(orig, dtype=np.float32)
    bilat = np.empty_like(orig, dtype=np.float32)

    for c in range(n_coils):
        x = orig[c]
        gauss[c] = denoise_gaussian(x, sigma=sigma)
        med[c] = denoise_median(x, size=median_size)
        bilat[c] = denoise_bilateral_img(x, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

    return {
        "original": orig,
        "gaussian": gauss,
        "median": med,
        "bilateral": bilat,
    }


def compute_metrics(reference: np.ndarray, img: np.ndarray) -> tuple[float, float, float]:
    """Compute MSE, PSNR, SSIM."""
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
    """Show one coil: Original | Gaussian | Median | Bilateral."""
    keys = ["original", "gaussian", "median", "bilateral"]
    titles = ["Original", "Gaussian", "Median", "Bilateral"]
    imgs = [np.asarray(denoised[k][coil_id], dtype=np.float32) for k in keys]

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
    """Export one 1x4 comparison figure per coil."""
    n_coils = int(denoised["original"].shape[0])
    saved: list[Path] = []
    for c in range(n_coils):
        out_path = None
        if out_dir is not None:
            out_path = Path(out_dir) / f"ex2_2_part1_coil_{c}_denoise_compare.png"
        plot_denoise_per_coil(denoised, coil_id=c, out_path=out_path, show=show)
        if out_path is not None:
            saved.append(Path(out_path))
    return saved


# -------- part 2: k-space butterworth low-pass on first coil --------

def butterworth_lowpass_filter(shape: tuple[int, int], D0: float = 30.0, n: int = 2) -> np.ndarray:
    """2D Butterworth low-pass filter for centered k-space coordinates."""
    P, Q = int(shape[0]), int(shape[1])
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing="ij")
    D = np.sqrt(U**2 + V**2)
    H = 1.0 / (1.0 + (D / float(D0)) ** (2 * int(n)))
    return H.astype(np.float32)


def run_part2_first_coil(
    kspace_coils_first: np.ndarray,
    coil_id: int = 0,
    D0: float = 30.0,
    n: int = 2,
) -> dict[str, np.ndarray]:
    coil_k = np.asarray(kspace_coils_first[coil_id])

    # Coursework k-space is already stored with DC at the centre, so the
    # Butterworth mask must be applied in that convention without re-shifting.
    coil_k_centered = coil_k
    H = butterworth_lowpass_filter(coil_k.shape, D0=D0, n=n)

    # Apply the low-pass mask directly in centred k-space.
    filtered_k_centered = coil_k_centered * H

    filtered_k = filtered_k_centered

    img_original = np.fft.ifft2(coil_k)
    img_filtered = np.fft.ifft2(filtered_k)

    return {
        "coil_k": coil_k,
        "coil_k_centered": coil_k_centered,
        "H": H,
        "filtered_k_centered": filtered_k_centered,
        "filtered_k": filtered_k,
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
    k = np.asarray(kspace_coils_first)
    if k.ndim != 3:
        raise ValueError(f"Expected k-space shape (coils,H,W), got {k.shape}")

    H = butterworth_lowpass_filter(k.shape[-2:], D0=D0, n=n)

    # Keep the same centred-k-space convention used everywhere else.
    k_centered = k
    filtered_k_centered = k_centered * H[None, :, :]
    filtered_k = filtered_k_centered

    img = np.fft.ifft2(filtered_k, axes=(-2, -1))
    rsos = np.sqrt(np.sum(np.abs(img) ** 2, axis=0)).astype(np.float32)
    return rsos


def plot_part2_mag_phase(
    part2: dict[str, np.ndarray],
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Show filtered first-coil magnitude and phase images."""
    mag = np.asarray(part2["mag"], dtype=np.float32)
    mag_disp = np.log1p(np.maximum(mag, 0.0))
    phase = np.asarray(part2["phase"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes[0].imshow(mag_disp, cmap="gray")
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
    """Compare k-space Butterworth result against image-space methods for one coil."""
    orig = np.asarray(denoised["original"][coil_id], dtype=np.float32)
    gauss = np.asarray(denoised["gaussian"][coil_id], dtype=np.float32)
    med = np.asarray(denoised["median"][coil_id], dtype=np.float32)
    bilat = np.asarray(denoised["bilateral"][coil_id], dtype=np.float32)
    k_butter = np.asarray(np.abs(part2["img_filtered"]), dtype=np.float32)

    imgs = [orig, k_butter, gauss, med, bilat]
    titles = ["Original", "k-space Butterworth", "Gaussian", "Median", "Bilateral"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = axes[i]
        disp = np.log1p(np.maximum(img, 0.0))
        im = ax.imshow(disp, cmap="gray")
        ax.set_title(f"{title} | Coil {coil_id}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")

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
    """Compute part-2 metrics against one explicit reference image."""
    ref = np.asarray(
        denoised["original"][0] if reference is None else reference,
        dtype=np.float32,
    )
    rows: list[dict] = []
    n_coils = int(denoised["original"].shape[0])
    for c in range(n_coils):
        original_img = np.asarray(denoised["original"][c], dtype=np.float32)
        mse, psnr, ssim = compute_metrics(ref, original_img)
        rows.append({"part": "2.2-part2", "coil": c, "method": "original", "mse": mse, "psnr": psnr, "ssim": ssim})

        for method in ("gaussian", "median", "bilateral"):
            img = np.asarray(denoised[method][c], dtype=np.float32)
            mse, psnr, ssim = compute_metrics(ref, img)
            rows.append(
                {"part": "2.2-part2", "coil": c, "method": method, "mse": mse, "psnr": psnr, "ssim": ssim}
            )

        if c == coil_id:
            kb = np.asarray(np.abs(part2["img_filtered"]), dtype=np.float32)
            mse, psnr, ssim = compute_metrics(ref, kb)
            rows.append(
                {"part": "2.2-part2", "coil": c, "method": "kspace_butterworth", "mse": mse, "psnr": psnr, "ssim": ssim}
            )
    return rows


# -------- part 3: denoise combined image --------

def denoise_combined_image(
    rsos: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    median_size: int = 3,
    sigma_color: float = 0.05,
    sigma_spatial: float = 3.0,
) -> np.ndarray:
    """Denoise a combined rSoS image with one selected method."""
    x = np.asarray(rsos, dtype=np.float32)
    m = method.lower()
    if m == "gaussian":
        return denoise_gaussian(x, sigma=sigma)
    if m == "median":
        return denoise_median(x, size=median_size)
    if m == "bilateral":
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
    """Apply one denoising method to each coil magnitude, then combine with rSoS."""
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
    """Show rSoS original vs rSoS denoised."""
    a = np.asarray(rsos, dtype=np.float32)
    b = np.asarray(rsos_denoised, dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    im0 = axes[0].imshow(a, cmap="gray")
    axes[0].set_title("rSoS original")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(b, cmap="gray")
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
    """Compute part-3 metrics against one explicit reference image."""
    ref = np.asarray(rsos if reference is None else reference, dtype=np.float32)
    original = np.asarray(rsos, dtype=np.float32)
    den = np.asarray(rsos_denoised, dtype=np.float32)
    rows: list[dict] = []

    mse0, psnr0, ssim0 = compute_metrics(ref, original)
    rows.append({"part": "2.2-part3", "method": "original", "mse": mse0, "psnr": psnr0, "ssim": ssim0})

    mse, psnr, ssim = compute_metrics(ref, den)
    rows.append({"part": "2.2-part3", "method": method_name, "mse": mse, "psnr": psnr, "ssim": ssim})
    return rows
