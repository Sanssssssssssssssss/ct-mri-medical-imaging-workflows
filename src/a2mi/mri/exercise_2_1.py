"""Exercise 2.1 MRI utilities: loading, visualization, and coil combination."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_kspace(path: str | Path) -> np.ndarray:
    """Load the complex-valued MRI k-space array from disk.

    Args:
        path: Path to the `.npy` file containing the k-space data.

    Returns:
        The loaded k-space array with singleton dimensions removed.
    """
    kspace = np.load(Path(path))
    return np.squeeze(kspace)


def infer_coil_axis(kspace: np.ndarray) -> int:
    """Infer which axis corresponds to coils in a 3D k-space array.

    Args:
        kspace: Complex-valued k-space array.

    Returns:
        The index of the coil axis.

    Raises:
        ValueError: If `kspace` is not 3D.
    """
    if kspace.ndim != 3:
        raise ValueError(f"Expected 3D k-space, got shape={kspace.shape}")
    return int(np.argmin(kspace.shape))


def move_coils_first(kspace: np.ndarray, coil_axis: int) -> np.ndarray:
    """Reorder k-space so the coil axis becomes the first dimension.

    Args:
        kspace: Input k-space array.
        coil_axis: Axis index corresponding to coils.

    Returns:
        K-space with shape `(coils, height, width)`.
    """
    return np.moveaxis(kspace, coil_axis, 0)


def kspace_to_image_space(kspace_coils_first: np.ndarray) -> np.ndarray:
    """Reconstruct image-space data from coil-first k-space.

    Args:
        kspace_coils_first: K-space arranged as `(coils, height, width)`.

    Returns:
        Complex-valued image-space data for each coil.
    """
    return np.fft.ifft2(kspace_coils_first, axes=(-2, -1))


def combine_rsos(img_coils: np.ndarray) -> np.ndarray:
    """Combine multi-coil images with root-sum-of-squares.

    Args:
        img_coils: Complex or magnitude coil images with coil axis first.

    Returns:
        The combined rSoS magnitude image.
    """
    return np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=0))


def prepare_exercise_2_1_data(data_path: str | Path) -> dict[str, np.ndarray | int | tuple[int, ...]]:
    """Prepare the shared MRI arrays used throughout Exercise 2.1 and 2.2.

    Args:
        data_path: Path to the MRI k-space `.npy` file.

    Returns:
        A dictionary containing the raw k-space, inferred coil axis, coil-first
        k-space, image-space coils, rSoS image, and reordered shape metadata.
    """
    k_raw = load_kspace(data_path)
    coil_axis = infer_coil_axis(k_raw)
    kspace_coils_first = move_coils_first(k_raw, coil_axis)
    img_coils = kspace_to_image_space(kspace_coils_first)
    rsos = combine_rsos(img_coils)
    return {
        "k_raw": k_raw,
        "coil_axis": coil_axis,
        "kspace_coils_first": kspace_coils_first,
        "img_coils": img_coils,
        "rsos": rsos,
        "shape": tuple(kspace_coils_first.shape),
    }


def _grid(n_items: int, default_cols: int = 3) -> tuple[int, int]:
    """Choose a compact subplot grid for a given number of panels.

    Args:
        n_items: Number of panels to draw.
        default_cols: Preferred maximum number of columns.

    Returns:
        A `(rows, cols)` tuple.
    """
    cols = min(default_cols, n_items)
    rows = int(np.ceil(n_items / cols))
    return rows, cols


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


def plot_kspace_magnitude_all_coils(
    kspace_coils_first: np.ndarray,
    use_log1p: bool = True,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the k-space magnitude for every coil.

    Args:
        kspace_coils_first: Coil-first k-space array.
        use_log1p: Whether to apply `log1p` before plotting.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    n_coils = int(kspace_coils_first.shape[0])
    rows, cols = _grid(n_coils, default_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for c in range(n_coils):
        magnitude = np.abs(kspace_coils_first[c])
        if use_log1p:
            magnitude = np.log1p(magnitude)
        im = axes[c].imshow(magnitude, cmap="gray")
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
    """Plot the magnitude and phase of one reconstructed coil image.

    Args:
        img_coils: Coil-first reconstructed image data.
        coil_id: Coil index to visualize.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    magnitude = np.abs(img_coils[coil_id])
    phase = np.angle(img_coils[coil_id])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    im0 = axes[0].imshow(magnitude, cmap="gray")
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
    """Plot the reconstructed magnitude image for every coil.

    Args:
        img_coils: Coil-first reconstructed image data.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    n_coils = int(img_coils.shape[0])
    rows, cols = _grid(n_coils, default_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for c in range(n_coils):
        magnitude = np.abs(img_coils[c])
        im = axes[c].imshow(magnitude, cmap="gray")
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
    """Plot the final rSoS-combined image.

    Args:
        rsos: Combined rSoS magnitude image.
        out_path: Optional file path for saving the figure.
        show: Whether to display the figure interactively.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    im = ax.imshow(rsos, cmap="gray")
    ax.set_title("rSoS combined image")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _savefig(fig, out_path)
    if show:
        plt.show()
    plt.close(fig)
