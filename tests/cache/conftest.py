"""
pytest configuration for cache tests.

Sets up fresh temp cache directory before torch_neuronx is imported.
This ensures persistent caching tests have process-isolated caches.
"""

import os
import tempfile

import pytest

_CACHE_TMPDIR = tempfile.mkdtemp(prefix="neff_cache_test_")
os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = _CACHE_TMPDIR


@pytest.fixture(scope="session")
def cache_dir():
    """Provide access to the isolated cache directory."""
    return _CACHE_TMPDIR
