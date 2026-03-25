"""Sphinx configuration for the a2mi documentation site."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import a2mi

project = "a2mi"
author = "X"
copyright = "2026, X"
release = a2mi.__version__

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

modindex_common_prefix = ["a2mi."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scikit-image": ("https://scikit-image.org/docs/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "a2mi documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_sidebars = {
    "index": [],
    "**": ["page-toc"],
}
html_theme_options = {
    "navigation_with_keys": True,
    "show_prev_next": True,
    "header_links_before_dropdown": 4,
    "navbar_align": "left",
    "secondary_sidebar_items": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Sanssssssssssssssss/A2_Coursework",
            "icon": "fa-brands fa-github",
        }
    ],
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
