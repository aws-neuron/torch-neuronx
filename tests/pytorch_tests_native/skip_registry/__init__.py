"""Auto-discovery for skip/xfail registries.

Drop a new *_registry.py file in this folder with SKIP_GROUPS and/or XFAIL_GROUPS
lists, and they will be automatically combined.
"""

import importlib
import pkgutil
from pathlib import Path

from tests.pytorch_tests_native.neuron_test_registry import SkipGroup

SKIP_GROUPS: list[SkipGroup] = []
XFAIL_GROUPS: list[SkipGroup] = []


def _load_registries():
    """Auto-discover and load all *_registry.py modules."""
    package_dir = Path(__file__).parent

    for _, name, _ in pkgutil.iter_modules([str(package_dir)]):
        if name.endswith("_registry"):
            module = importlib.import_module(f".{name}", __package__)
            if hasattr(module, "SKIP_GROUPS"):
                SKIP_GROUPS.extend(module.SKIP_GROUPS)
            if hasattr(module, "XFAIL_GROUPS"):
                XFAIL_GROUPS.extend(module.XFAIL_GROUPS)


_load_registries()
