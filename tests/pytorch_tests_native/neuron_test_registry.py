"""Neuron skip/xfail registry for upstream PyTorch tests.

Single source of truth for all tests that need to be skipped or xfailed
when running the upstream PyTorch test suite on Neuron. Applied automatically
via conftest.py using pytest_collection_modifyitems — no patches to upstream code.

Organization:
    Each SkipGroup represents ONE root cause. When the root cause is fixed,
    delete the entire group and all its tests are re-enabled.

    Five levels of granularity per group:
    - folders:  skip entire directories (e.g., "distributed/", "test/inductor/")
    - files:    skip entire test files (e.g., "test_sparse.py")
    - classes:  skip entire test classes (e.g., "TestSparseOps")
    - tests:    exact test IDs (e.g., "TestClass::test_method")
    - patterns: glob patterns against test name (e.g., "*_neuron_complex*")
"""

import fnmatch
from dataclasses import dataclass, field

import pytest


@dataclass
class SkipGroup:
    """A group of tests to skip/xfail, tied to a single root cause."""

    reason: str
    tests: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)


# =============================================================================
# SKIP: Tests that should not run with Neuron backend
# Each group = one root cause. Fix the cause, delete the group.
# =============================================================================

SKIP_GROUPS: list[SkipGroup] = []

XFAIL_GROUPS: list[SkipGroup] = []


# Pull in all registries from skip_registry folder
def _load_skip_registry():
    try:
        from tests.pytorch_tests_native import skip_registry

        SKIP_GROUPS.extend(skip_registry.SKIP_GROUPS)
        XFAIL_GROUPS.extend(skip_registry.XFAIL_GROUPS)
    except ImportError:
        pass


_load_skip_registry()


# =============================================================================
# Matching logic (used by conftest.py)
# =============================================================================


def _group_matches(group: SkipGroup, nodeid: str, filename: str) -> bool:
    """Check if a test matches any entry in a group."""
    # Folder match: anchored to test/ directory (e.g., "distributed" matches "test/distributed/")
    filepath = nodeid.split("::")[0]
    if any(f"test/{f}" in filepath for f in group.folders):
        return True
    if filename in group.files:
        return True
    # Extract class name from nodeid (e.g., "path/test.py::TestClass::test_method")
    parts = nodeid.split("::")
    if len(parts) >= 2 and parts[1] in group.classes:
        return True
    if any(t in nodeid for t in group.tests):
        return True
    test_name = parts[-1] if len(parts) > 1 else nodeid
    return any(fnmatch.fnmatch(test_name, p) for p in group.patterns)


def apply_markers(item: pytest.Item) -> None:
    """Apply skip/xfail markers to a pytest item based on the registry."""
    filename = item.fspath.basename
    nodeid = item.nodeid

    for group in SKIP_GROUPS:
        if _group_matches(group, nodeid, filename):
            item.add_marker(pytest.mark.skip(reason=group.reason))
            return

    for group in XFAIL_GROUPS:
        if _group_matches(group, nodeid, filename):
            item.add_marker(pytest.mark.xfail(reason=group.reason, strict=False))
            return
