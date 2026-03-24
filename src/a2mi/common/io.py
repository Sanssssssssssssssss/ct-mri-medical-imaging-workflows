"""Common filesystem and figure-output helpers for the :mod:`a2mi` package."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def project_root() -> Path:
    """Return the repository root directory.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Parameters
    ----------
    path:
        Absolute path or project-relative path.

    Returns
    -------
    Path
        Absolute filesystem path.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def save_figure(fig: Figure, out_path: str | Path | None, dpi: int = 150) -> Path | None:
    """Save a Matplotlib figure if an output path is provided.

    Parameters
    ----------
    fig:
        Figure to save.
    out_path:
        Output path, or ``None`` to skip saving.
    dpi:
        Figure resolution.

    Returns
    -------
    Path | None
        Absolute saved path, or ``None`` when saving is skipped.
    """
    if out_path is None:
        return None
    path = resolve_project_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def write_csv_rows(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Write a list of dictionaries to CSV.

    Parameters
    ----------
    rows:
        Rows to write. The first row defines the field order.
    out_path:
        Output CSV path.

    Returns
    -------
    Path
        Absolute path of the saved CSV file.
    """
    path = resolve_project_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
