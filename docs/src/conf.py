# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Read version from pyproject.toml
pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"

if sys.version_info >= (3, 11):
    import tomllib

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
else:
    import re

    content = pyproject_path.read_text()
    name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    authors_match = re.search(r"authors\s*=\s*\[\{([^\}]+)\}", content)
    author_name_match = (
        re.search(r'name\s*=\s*"([^"]+)"', authors_match.group(1)) if authors_match else None
    )

    pyproject = {
        "project": {
            "name": name_match.group(1) if name_match else "torch-neuronx",
            "version": version_match.group(1) if version_match else "0.0.0",
            "authors": [
                {"name": author_name_match.group(1) if author_name_match else "AWS Neuron"}
            ],
        }
    }

project = pyproject["project"]["name"]
version = pyproject["project"]["version"]
release = version
author = pyproject["project"]["authors"][0]["name"]
copyright = f"2024, {author}"

# General configuration
extensions = []
templates_path = ["_templates"]
exclude_patterns = []

# HTML output options
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
