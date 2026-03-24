"""Tests for shared filesystem and export helpers."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt

from a2mi.common import project_root, resolve_project_path, save_figure, write_csv_rows


def test_project_root_and_resolve_project_path() -> None:
    """Project-relative helpers should resolve to absolute paths."""
    root = project_root()
    assert root.is_absolute()
    resolved = resolve_project_path("README.md")
    assert resolved == root / "README.md"


def test_save_figure_and_write_csv_rows(tmp_path) -> None:
    """Figure and CSV helpers should create parent directories and files."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig_path = save_figure(fig, tmp_path / "figs" / "line.png")
    plt.close(fig)
    assert fig_path is not None
    assert fig_path.exists()

    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    csv_path = write_csv_rows(rows, tmp_path / "metrics" / "rows.csv")
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))
    assert reader == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_save_figure_accepts_none_output() -> None:
    """Saving should be skipped cleanly when no path is provided."""
    fig, _ = plt.subplots()
    assert save_figure(fig, None) is None
    plt.close(fig)
