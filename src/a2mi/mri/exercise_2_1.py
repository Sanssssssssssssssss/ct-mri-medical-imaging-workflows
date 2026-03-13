"""Exercise 2.1 MRI utilities: loading, visualization, and coil combination."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_kspace(path: str | Path) -> np.ndarray:
    """Load complex k-space array."""
    k = np.load(Path(path))
    return np.squeeze(k)


def infer_coil_axis(kspace: np.ndarray) -> int:
    """Infer coil axis for a 3D k-space array."""
    if kspace.ndim != 3:
        raise ValueError(f"Expected 3D k-space, got shape={kspace.shape}")
    # For this coursework data, coil dimension is the smallest one (e.g., 6, H, W).
    return int(np.argmin(kspace.shape))


def move_coils_first(kspace: np.ndarray, coil_axis: int) -> np.ndarray:
    """Reorder k-space to (coils, H, W)."""
    return np.moveaxis(kspace, coil_axis, 0)


def kspace_to_image_space(kspace_coils_first: np.ndarray) -> np.ndarray:
    """Inverse FFT from k-space to image space per coil."""
    return np.fft.ifft2(kspace_coils_first, axes=(-2, -1))


def combine_rsos(img_coils: np.ndarray) -> np.ndarray:
    """Root-sum-of-squares combination across coil axis 0."""
    return np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=0))


def _grid(n_items: int, default_cols: int = 3) -> tuple[int, int]:
    cols = min(default_cols, n_items)
    rows = int(np.ceil(n_items / cols))
    return rows, cols


def _savefig(fig: plt.Figure, out_path: str | Path | None) -> None:
    if out_path is None:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_kspace_magnitude_all_coils(
    kspace_coils_first: np.ndarray,
    use_log1p: bool = True,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot k-space magnitude per coil in a compact grid."""
    n_coils = int(kspace_coils_first.shape[0])
    rows, cols = _grid(n_coils, default_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for c in range(n_coils):
        mag = np.abs(kspace_coils_first[c])
        if use_log1p:
            mag = np.log1p(mag)
        im = axes[c].imshow(mag, cmap="gray")
        axes[c].set_title(f"k-space |coil {c}|")
        axes[c].axis("off")
        fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    for i in range(n_coils, len(axes)):
        axes[i].axis("off")

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_single_coil_magnitude_phase(
    img_coils: np.ndarray,
    coil_id: int = 0,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot magnitude and phase of one coil image."""
    mag = np.abs(img_coils[coil_id])
    phase = np.angle(img_coils[coil_id])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes[0].imshow(mag, cmap="gray")
    axes[0].set_title("Magnitude image")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(phase, cmap="gray")
    axes[1].set_title("Phase image")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_image_magnitude_all_coils(
    img_coils: np.ndarray,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot image-space magnitude per coil in a compact grid."""
    n_coils = int(img_coils.shape[0])
    rows, cols = _grid(n_coils, default_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for c in range(n_coils):
        m = np.abs(img_coils[c])
        im = axes[c].imshow(m, cmap="gray")
        axes[c].set_title(f"Coil {c}")
        axes[c].axis("off")
        fig.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    for i in range(n_coils, len(axes)):
        axes[i].axis("off")

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_rsos_image(
    rsos: np.ndarray,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot rSoS-combined image."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    im = ax.imshow(rsos, cmap="gray")
    ax.set_title("rSoS combined image")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)
