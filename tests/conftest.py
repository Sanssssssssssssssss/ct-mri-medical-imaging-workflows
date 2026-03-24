"""Shared pytest fixtures for the :mod:`a2mi` test suite."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")


@pytest.fixture
def synthetic_ct_image() -> np.ndarray:
    """Create a small normalized CT-like phantom image."""
    size = 32
    yy, xx = np.ogrid[:size, :size]
    cy = cx = (size - 1) / 2.0
    radius = size / 4.0
    image = np.zeros((size, size), dtype=np.float32)
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    image[mask] = 1.0
    image[8:12, 20:24] = 0.5
    return image


@pytest.fixture
def synthetic_mri_image_coils() -> np.ndarray:
    """Create a small stack of synthetic complex MRI coil images."""
    size = 32
    yy, xx = np.mgrid[:size, :size]
    base = np.exp(-(((yy - 16.0) ** 2 + (xx - 16.0) ** 2) / 90.0)).astype(np.float32)
    coils = []
    for weight in (1.0, 0.75, 0.5):
        phase = np.exp(1j * weight * (xx / size))
        coils.append((base * weight * phase).astype(np.complex64))
    return np.stack(coils, axis=0)


@pytest.fixture
def synthetic_kspace_coils(synthetic_mri_image_coils: np.ndarray) -> np.ndarray:
    """Create synthetic k-space data from the MRI coil images."""
    return np.fft.fft2(synthetic_mri_image_coils, axes=(-2, -1))
