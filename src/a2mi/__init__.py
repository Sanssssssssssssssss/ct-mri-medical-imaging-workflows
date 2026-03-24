"""Top-level package for the medical imaging coursework codebase.

The package is intentionally small and notebook-friendly: the public API is
grouped into the :mod:`a2mi.ct` and :mod:`a2mi.mri` subpackages so notebooks
and scripts can import stable helpers without relying on file-relative paths.
"""

from __future__ import annotations

from . import ct, mri

__all__ = ["ct", "mri", "__version__"]

__version__ = "0.1.0"
