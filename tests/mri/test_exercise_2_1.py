"""Tests for MRI Exercise 2.1 helpers."""

from __future__ import annotations

import numpy as np
import pytest

from a2mi.mri.exercise_2_1 import (
    combine_rsos,
    infer_coil_axis,
    kspace_to_image_space,
    load_kspace,
    move_coils_first,
    plot_image_magnitude_all_coils,
    plot_image_magnitude_all_coils_adaptive,
    plot_kspace_magnitude_all_coils,
    plot_rsos_image,
    plot_single_coil_magnitude_phase,
    prepare_exercise_2_1_data,
)


def test_load_prepare_and_transform(tmp_path, synthetic_mri_image_coils: np.ndarray, synthetic_kspace_coils: np.ndarray) -> None:
    """MRI loading and preparation helpers should preserve coil-first data."""
    np.save(tmp_path / "kspace.npy", synthetic_kspace_coils)
    loaded = load_kspace(tmp_path / "kspace.npy")
    assert loaded.shape == synthetic_kspace_coils.shape

    coil_axis = infer_coil_axis(loaded)
    assert coil_axis == 0
    moved = move_coils_first(loaded, coil_axis)
    np.testing.assert_allclose(moved, synthetic_kspace_coils)

    img = kspace_to_image_space(moved)
    assert img.shape == synthetic_mri_image_coils.shape
    rsos = combine_rsos(img)
    assert rsos.shape == synthetic_mri_image_coils.shape[1:]

    prepared = prepare_exercise_2_1_data(tmp_path / "kspace.npy")
    assert prepared["coil_axis"] == 0
    assert prepared["shape"] == synthetic_kspace_coils.shape


def test_infer_coil_axis_validation() -> None:
    """Coil-axis inference should reject non-3D arrays."""
    with pytest.raises(ValueError):
        infer_coil_axis(np.zeros((4, 4), dtype=np.complex64))


def test_plotting_helpers_save_outputs(tmp_path, synthetic_kspace_coils: np.ndarray) -> None:
    """Exercise 2.1 plotting helpers should save figures."""
    img_coils = kspace_to_image_space(synthetic_kspace_coils)
    rsos = combine_rsos(img_coils)

    paths = [
        tmp_path / "kspace.png",
        tmp_path / "single.png",
        tmp_path / "all.png",
        tmp_path / "all_adaptive.png",
        tmp_path / "rsos.png",
    ]
    plot_kspace_magnitude_all_coils(synthetic_kspace_coils, out_path=paths[0], show=False)
    plot_single_coil_magnitude_phase(img_coils, out_path=paths[1], show=False)
    plot_image_magnitude_all_coils(img_coils, out_path=paths[2], show=False)
    plot_image_magnitude_all_coils_adaptive(img_coils, out_path=paths[3], show=False)
    plot_rsos_image(rsos, out_path=paths[4], show=False)
    assert all(path.exists() for path in paths)
