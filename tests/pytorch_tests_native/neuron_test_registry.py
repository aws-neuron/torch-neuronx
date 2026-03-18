"""Neuron skip/xfail registry for upstream PyTorch tests.

Single source of truth for all tests that need to be skipped or xfailed
when running the upstream PyTorch test suite on Neuron. Applied automatically
via conftest.py using pytest_collection_modifyitems — no patches to upstream code.

Organization:
    Each SkipGroup represents ONE root cause. When the root cause is fixed,
    delete the entire group and all its tests are re-enabled.

    Three levels of granularity per group:
    - folders:  skip entire directories (e.g., "distributed/")
    - tests:    contiguous component match against nodeid split by :: and /
                Supports any combination: "test_method", "TestClass",
                "test_sparse.py", "test_sparse.py::TestClass",
                "distributed::test_comm.py::TestComm", etc.
    - patterns: glob patterns against test function name
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


def _split_nodeid(nodeid: str) -> list[str]:
    """Split a nodeid into components (path segments + class + test)."""
    return nodeid.replace("/", "::").split("::")


def _is_contiguous_match(skip_entry: list, pytest_nodeid: list) -> bool:
    """Check if skip_entry appears as a contiguous subsequence in pytest_nodeid.

    Example:
        pytest_nodeid: "tests/.../test_fsdp.py::TestFSDP::test_something"
                       → ["tests", ..., "test_fsdp.py", "TestFSDP", "test_something"]
        skip_entry:    "test_fsdp.py::TestFSDP" → ["test_fsdp.py", "TestFSDP"]
        Result: True (contiguous match)

    Pytest nodeids are always fully qualified, so they will always have more
    components than any skip entry. If skip_entry is longer, returns False.
    """
    return any(
        pytest_nodeid[i : i + len(skip_entry)] == skip_entry
        for i in range(len(pytest_nodeid) - len(skip_entry) + 1)
    )


def _group_matches(group: SkipGroup, nodeid: str) -> bool:
    """Check if a test matches any entry in a group.

    Matching order (first match wins):
    1. folders — checks if "test/{folder}" is in the file path
    2. tests   — splits nodeid into components by / and ::, then
                 checks if the entry is a contiguous subsequence.
                 Supports any combination: "test_name", "TestClass",
                 "test_file.py", "test_file.py::TestClass", etc.
    3. patterns — glob match against the test function name only

    Note: pytest nodeids are always fully qualified paths like
    "tests/pytorch_tests_native/test/distributed/file.py::TestClass::test_method"
    so they will always be longer than or equal to any skip entry we define.
    """
    filepath = nodeid.split("::")[0]
    if any(f"test/{f}" in filepath for f in group.folders):
        return True
    pytest_nodeid_parts = _split_nodeid(nodeid)
    if any(_is_contiguous_match(_split_nodeid(t), pytest_nodeid_parts) for t in group.tests):
        return True
    return any(fnmatch.fnmatch(pytest_nodeid_parts[-1], p) for p in group.patterns)


def apply_markers(item: pytest.Item) -> None:
    """Apply skip/xfail markers to a pytest item based on the registry."""
    nodeid = item.nodeid

    for group in SKIP_GROUPS:
        if _group_matches(group, nodeid):
            item.add_marker(pytest.mark.skip(reason=group.reason))
            return

    for group in XFAIL_GROUPS:
        if _group_matches(group, nodeid):
            item.add_marker(pytest.mark.xfail(reason=group.reason, strict=False))
            return
