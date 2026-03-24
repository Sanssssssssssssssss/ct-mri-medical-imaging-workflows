"""Smoke tests for the installable :mod:`a2mi` package."""

from __future__ import annotations

import numpy as np

import a2mi


def test_top_level_package_exports_subpackages() -> None:
    """The package should expose the CT and MRI subpackages."""
    assert hasattr(a2mi, "ct")
    assert hasattr(a2mi, "mri")


def test_combine_rsos_matches_expected_pixelwise_norm() -> None:
    """rSoS should compute the pixelwise Euclidean norm over coils."""
    coils = np.array(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[4.0, 0.0], [0.0, 12.0]],
        ],
        dtype=np.float32,
    )
    combined = a2mi.mri.combine_rsos(coils)
    expected = np.array([[5.0, 4.0], [0.0, 12.0]], dtype=np.float32)
    np.testing.assert_allclose(combined, expected)


def test_make_theta_spans_full_circle_without_endpoint_overlap() -> None:
    """CT theta generation should return evenly spaced angles in [0, 360)."""
    theta = a2mi.ct.make_theta(4)
    np.testing.assert_allclose(theta, np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32))
